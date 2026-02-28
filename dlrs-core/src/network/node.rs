//! DlrsNode — a peer in the DLRS network
//!
//! Manages local seeds and communicates with the P2P network
//! via libp2p gossipsub.

use super::protocol::{DlrsMessage, MessageType};
use crate::seed::DnaSeed;
use crate::storage::SeedStore;
use log::{info, warn};
use std::collections::HashMap;

/// Configuration for a DLRS network node
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Display name for this node
    pub name: String,
    /// Maximum number of seeds this node will host
    pub max_seeds: usize,
    /// Whether to announce seeds automatically
    pub auto_announce: bool,
    /// Port to listen on (0 = random)
    pub listen_port: u16,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            name: "dlrs-node".to_string(),
            max_seeds: 1000,
            auto_announce: true,
            listen_port: 0,
        }
    }
}

/// A node in the DLRS P2P network
pub struct DlrsNode {
    /// Node configuration
    pub config: NodeConfig,
    /// Local peer ID (generated on creation)
    pub peer_id: String,
    /// Local seed store
    pub store: SeedStore,
    /// Pending capability requests
    pending_requests: HashMap<String, PendingRequest>,
    /// Message inbox (received from network)
    inbox: Vec<DlrsMessage>,
}

/// A pending outbound request
#[allow(dead_code)]
struct PendingRequest {
    request_id: String,
    domain: String,
    responses: Vec<(String, f64)>, // (seed_id, score)
}

impl DlrsNode {
    /// Create a new DLRS node
    pub fn new(config: NodeConfig) -> Self {
        let peer_id = uuid::Uuid::new_v4().to_string();
        info!("Creating DLRS node '{}' with peer_id={}", config.name, peer_id);
        Self {
            config,
            peer_id,
            store: SeedStore::new(),
            pending_requests: HashMap::new(),
            inbox: Vec::new(),
        }
    }

    /// Add a seed to this node
    pub fn add_seed(&mut self, seed: DnaSeed) -> Result<(), String> {
        if self.store.count() >= self.config.max_seeds {
            return Err("Node seed capacity reached".to_string());
        }
        let id = seed.id.clone();
        self.store.put(seed);
        info!("Added seed {} to node {}", id, self.config.name);
        Ok(())
    }

    /// Create a seed announcement message
    pub fn announce_seed(&self, seed_id: &str) -> Option<DlrsMessage> {
        self.store.get(seed_id).map(|seed| {
            DlrsMessage::new(
                &self.peer_id,
                MessageType::SeedAnnounce {
                    seed_id: seed.id.clone(),
                    commitment_root: seed.commitment.root_commitment.clone(),
                    domains: seed.domains.clone(),
                    fitness: seed.fitness,
                    rank: seed.lrim.rank,
                },
            )
        })
    }

    /// Create announcement messages for all local seeds
    pub fn announce_all_seeds(&self) -> Vec<DlrsMessage> {
        self.store
            .list_ids()
            .iter()
            .filter_map(|id| self.announce_seed(id))
            .collect()
    }

    /// Handle an incoming message
    pub fn handle_message(&mut self, msg: DlrsMessage) {
        match &msg.message_type {
            MessageType::Ping { .. } => {
                info!("Ping from {}", msg.from);
                // In a real implementation, send a Pong back
            }
            MessageType::Pong { .. } => {
                info!("Pong from {}", msg.from);
            }
            MessageType::SeedAnnounce { seed_id, domains, fitness, .. } => {
                info!(
                    "Seed announcement from {}: seed={}, domains={:?}, fitness={:.3}",
                    msg.from, seed_id, domains, fitness
                );
            }
            MessageType::CapabilityRequest { request_id, domain, challenge_vector, threshold } => {
                info!("Capability request {} for domain '{}'", request_id, domain);
                // Find best local seed for this domain
                if let Some(best) = self.find_best_seed_for_domain(domain) {
                    let domain_vec = nalgebra::DVector::from_vec(challenge_vector.clone());
                    let score = best.lrim.capability_in_domain(&domain_vec);
                    let normalized = (score / (score + 1.0)).min(1.0);
                    if normalized >= *threshold {
                        info!("Responding to request {} with score {:.3}", request_id, normalized);
                    }
                }
            }
            MessageType::CapabilityResponse { request_id, seed_id, score, .. } => {
                if let Some(pending) = self.pending_requests.get_mut(request_id) {
                    pending.responses.push((seed_id.clone(), *score));
                    info!("Got capability response for {}: seed={}, score={:.3}", request_id, seed_id, score);
                } else {
                    warn!("Received response for unknown request {}", request_id);
                }
            }
            MessageType::MergeRequest { request_id, initiator_seed_id, target_seed_id } => {
                info!(
                    "Merge request {}: {} wants to merge with {}",
                    request_id, initiator_seed_id, target_seed_id
                );
            }
            MessageType::MergeResponse { request_id, accepted, .. } => {
                info!("Merge response for {}: accepted={}", request_id, accepted);
            }
        }
        self.inbox.push(msg);
    }

    /// Find the best local seed for a given domain
    fn find_best_seed_for_domain(&self, domain: &str) -> Option<DnaSeed> {
        self.store
            .list_all()
            .into_iter()
            .filter(|s| s.domains.iter().any(|d| d == domain))
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the number of messages received
    pub fn inbox_count(&self) -> usize {
        self.inbox.len()
    }

    /// Get the peer ID
    pub fn id(&self) -> &str {
        &self.peer_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_test_seed(name: &str, domains: Vec<String>) -> DnaSeed {
        let k = DMatrix::new_random(10, 8);
        DnaSeed::new(name, &k, 3, domains)
    }

    #[test]
    fn test_create_node() {
        let node = DlrsNode::new(NodeConfig::default());
        assert_eq!(node.store.count(), 0);
        assert!(!node.peer_id.is_empty());
    }

    #[test]
    fn test_add_and_announce_seed() {
        let mut node = DlrsNode::new(NodeConfig::default());
        let seed = make_test_seed("test-seed", vec!["ai".to_string()]);
        let seed_id = seed.id.clone();

        node.add_seed(seed).unwrap();
        assert_eq!(node.store.count(), 1);

        let msg = node.announce_seed(&seed_id).unwrap();
        match msg.message_type {
            MessageType::SeedAnnounce { seed_id: id, .. } => {
                assert_eq!(id, seed_id);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_max_seeds_limit() {
        let config = NodeConfig {
            max_seeds: 1,
            ..NodeConfig::default()
        };
        let mut node = DlrsNode::new(config);
        let seed1 = make_test_seed("seed1", vec!["ai".to_string()]);
        let seed2 = make_test_seed("seed2", vec!["ai".to_string()]);

        assert!(node.add_seed(seed1).is_ok());
        assert!(node.add_seed(seed2).is_err());
    }

    #[test]
    fn test_handle_ping() {
        let mut node = DlrsNode::new(NodeConfig::default());
        let ping = DlrsMessage::new(
            "remote-peer",
            MessageType::Ping { timestamp: chrono::Utc::now() },
        );
        node.handle_message(ping);
        assert_eq!(node.inbox_count(), 1);
    }

    #[test]
    fn test_announce_all() {
        let mut node = DlrsNode::new(NodeConfig::default());
        node.add_seed(make_test_seed("s1", vec!["ai".into()])).unwrap();
        node.add_seed(make_test_seed("s2", vec!["crypto".into()])).unwrap();
        let announcements = node.announce_all_seeds();
        assert_eq!(announcements.len(), 2);
    }
}
