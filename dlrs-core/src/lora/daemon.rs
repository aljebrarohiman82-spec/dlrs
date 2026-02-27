//! Auto ZK Sharing Daemon
//!
//! Runs a background loop that:
//! 1. Periodically announces local adapters to the P2P network
//! 2. Responds to capability challenges from remote peers
//! 3. Auto-evolves adapters based on feedback
//! 4. Prunes/replicates based on fitness

use super::manager::LoraManager;
use crate::network::{DlrsMessage, MessageType, SwarmCommand, SwarmEvent2};
use crate::zk::CapabilityProof;
use log::{info, warn};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{self, Duration};

/// Configuration for the auto-sharing daemon
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// How often to announce seeds (seconds)
    pub announce_interval_secs: u64,
    /// How often to run evolution cycle (seconds)
    pub evolution_interval_secs: u64,
    /// How often to prune/replicate (seconds)
    pub lifecycle_interval_secs: u64,
    /// Local peer ID string
    pub peer_id: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            announce_interval_secs: 30,
            evolution_interval_secs: 60,
            lifecycle_interval_secs: 120,
            peer_id: "local".to_string(),
        }
    }
}

/// The auto-sharing daemon
pub struct AutoShareDaemon {
    pub config: DaemonConfig,
    manager: Arc<Mutex<LoraManager>>,
    cmd_tx: mpsc::Sender<SwarmCommand>,
}

impl AutoShareDaemon {
    pub fn new(
        config: DaemonConfig,
        manager: Arc<Mutex<LoraManager>>,
        cmd_tx: mpsc::Sender<SwarmCommand>,
    ) -> Self {
        Self {
            config,
            manager,
            cmd_tx,
        }
    }

    /// Start the daemon — spawns background tasks and processes incoming events
    pub async fn run(self, mut evt_rx: mpsc::Receiver<SwarmEvent2>) {
        let manager = self.manager.clone();
        let cmd_tx = self.cmd_tx.clone();
        let config = self.config.clone();

        // Task 1: Periodic announcements
        let mgr_announce = manager.clone();
        let cmd_announce = cmd_tx.clone();
        let peer_id_announce = config.peer_id.clone();
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(config.announce_interval_secs));
            loop {
                interval.tick().await;
                let mgr = mgr_announce.lock().await;
                let adapters = mgr.list();
                for (id, _name) in &adapters {
                    if let Some(adapter) = mgr.get(id) {
                        let msg = DlrsMessage::new(
                            &peer_id_announce,
                            MessageType::SeedAnnounce {
                                seed_id: adapter.id().to_string(),
                                commitment_root: adapter.commitment().root_commitment.clone(),
                                domains: adapter.config.domains.clone(),
                                fitness: adapter.seed.fitness,
                                rank: adapter.seed.lrim.rank,
                            },
                        );
                        if cmd_announce.send(SwarmCommand::Publish(msg)).await.is_err() {
                            warn!("Announce channel closed");
                            return;
                        }
                    }
                }
                if !adapters.is_empty() {
                    info!("Announced {} adapters to network", adapters.len());
                }
            }
        });

        // Task 2: Periodic lifecycle (prune + replicate)
        let mgr_lifecycle = manager.clone();
        let lifecycle_secs = self.config.lifecycle_interval_secs;
        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(lifecycle_secs));
            loop {
                interval.tick().await;
                let mut mgr = mgr_lifecycle.lock().await;
                let pruned = mgr.prune();
                if !pruned.is_empty() {
                    info!("Pruned {} low-fitness adapters", pruned.len());
                }
                if mgr.evolution_config.auto_replicate {
                    let replicated = mgr.auto_replicate();
                    if !replicated.is_empty() {
                        info!("Replicated {} high-fitness adapters", replicated.len());
                    }
                }
                // Auto-save
                if let Err(e) = mgr.save() {
                    warn!("Failed to auto-save: {}", e);
                }
            }
        });

        // Task 3: Process incoming network events
        let mgr_events = manager.clone();
        let cmd_events = cmd_tx.clone();
        let peer_id_events = self.config.peer_id.clone();
        tokio::spawn(async move {
            while let Some(evt) = evt_rx.recv().await {
                match evt {
                    SwarmEvent2::MessageReceived(msg) => {
                        Self::handle_message(
                            &mgr_events,
                            &cmd_events,
                            &peer_id_events,
                            msg,
                        )
                        .await;
                    }
                    SwarmEvent2::PeerDiscovered(peer) => {
                        info!("New peer joined: {}", peer);
                    }
                    SwarmEvent2::PeerLost(peer) => {
                        info!("Peer left: {}", peer);
                    }
                }
            }
        });
    }

    /// Handle an incoming network message
    async fn handle_message(
        manager: &Arc<Mutex<LoraManager>>,
        cmd_tx: &mpsc::Sender<SwarmCommand>,
        local_peer_id: &str,
        msg: DlrsMessage,
    ) {
        match &msg.message_type {
            MessageType::CapabilityRequest {
                request_id,
                domain,
                challenge_vector,
                threshold,
            } => {
                info!(
                    "Capability request from {}: domain='{}', threshold={:.2}",
                    msg.from, domain, threshold
                );
                let mgr = manager.lock().await;
                let candidates = mgr.find_by_domain(domain);
                for adapter in candidates {
                    let domain_vec =
                        nalgebra::DVector::from_vec(challenge_vector.clone());
                    let score = adapter.seed.lrim.capability_in_domain(&domain_vec);
                    let normalized = (score / (score + 1.0)).min(1.0);

                    if normalized >= *threshold {
                        let response = DlrsMessage::new(
                            local_peer_id,
                            MessageType::CapabilityResponse {
                                request_id: request_id.clone(),
                                seed_id: adapter.id().to_string(),
                                score: normalized,
                                proof_hash: CapabilityProof::generate(
                                    &adapter.seed.lrim,
                                    &adapter.commitment().root_commitment,
                                    domain,
                                    &domain_vec,
                                    24,
                                )
                                .proof_hash,
                            },
                        );
                        let _ = cmd_tx.send(SwarmCommand::Publish(response)).await;
                        info!(
                            "Responded to capability request {}: score={:.3}",
                            request_id, normalized
                        );
                    }
                }
            }
            MessageType::SeedAnnounce {
                seed_id,
                domains,
                fitness,
                rank,
                ..
            } => {
                info!(
                    "Remote seed: id={}..{}, domains={:?}, fitness={:.3}, rank={}",
                    &seed_id[..8.min(seed_id.len())],
                    &seed_id[seed_id.len().saturating_sub(4)..],
                    domains,
                    fitness,
                    rank
                );
            }
            MessageType::Ping { .. } => {
                let pong = DlrsMessage::new(
                    local_peer_id,
                    MessageType::Pong {
                        timestamp: chrono::Utc::now(),
                    },
                );
                let _ = cmd_tx.send(SwarmCommand::Publish(pong)).await;
            }
            _ => {
                info!("Received message type: {:?}", msg.message_type);
            }
        }
    }
}
