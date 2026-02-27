//! Real libp2p swarm — gossipsub-based P2P for DLRS seed sharing
//!
//! Runs a tokio-based event loop that:
//! - Discovers peers via mDNS
//! - Publishes seed announcements on gossipsub
//! - Responds to capability challenges
//! - Receives and routes messages to the local node

use super::protocol::{DlrsMessage, DLRS_TOPIC};
use libp2p::{
    gossipsub, mdns, noise,
    swarm::SwarmEvent,
    tcp, yamux, Multiaddr, PeerId,
};
use log::{error, info, warn};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tokio::sync::mpsc;

/// Combined behaviour: gossipsub for messaging + mDNS for discovery
#[derive(libp2p::swarm::NetworkBehaviour)]
pub struct DlrsBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
}

/// Commands that can be sent to the swarm task
#[derive(Debug)]
pub enum SwarmCommand {
    /// Publish a message to the network
    Publish(DlrsMessage),
    /// Stop the swarm
    Shutdown,
}

/// Events received from the network
#[derive(Debug, Clone)]
pub enum SwarmEvent2 {
    /// A message was received from a peer
    MessageReceived(DlrsMessage),
    /// A new peer was discovered
    PeerDiscovered(String),
    /// A peer disconnected
    PeerLost(String),
}

/// Configuration for the P2P swarm
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Port to listen on (0 = random)
    pub listen_port: u16,
    /// Gossipsub heartbeat interval
    pub heartbeat_secs: u64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            listen_port: 0,
            heartbeat_secs: 1,
        }
    }
}

/// Build and run the libp2p swarm, returning channels for communication
pub async fn run_swarm(
    config: SwarmConfig,
) -> Result<
    (
        PeerId,
        mpsc::Sender<SwarmCommand>,
        mpsc::Receiver<SwarmEvent2>,
    ),
    Box<dyn std::error::Error>,
> {
    // Build the swarm
    let mut swarm = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            // Gossipsub config
            let message_id_fn = |message: &gossipsub::Message| {
                let mut s = DefaultHasher::new();
                message.data.hash(&mut s);
                gossipsub::MessageId::from(s.finish().to_string())
            };
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(config.heartbeat_secs))
                .validation_mode(gossipsub::ValidationMode::Strict)
                .message_id_fn(message_id_fn)
                .build()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error + Send + Sync>)?;

            let mdns = mdns::tokio::Behaviour::new(
                mdns::Config::default(),
                key.public().to_peer_id(),
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

            Ok(DlrsBehaviour { gossipsub, mdns })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    let local_peer_id = *swarm.local_peer_id();
    info!("Local peer ID: {}", local_peer_id);

    // Subscribe to the DLRS topic
    let topic = gossipsub::IdentTopic::new(DLRS_TOPIC);
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;

    // Listen on all interfaces
    let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", config.listen_port).parse()?;
    swarm.listen_on(listen_addr)?;

    // Create communication channels
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<SwarmCommand>(256);
    let (evt_tx, evt_rx) = mpsc::channel::<SwarmEvent2>(256);

    // Spawn the event loop
    tokio::spawn(async move {
        use futures::StreamExt;
        loop {
            tokio::select! {
                // Handle commands from the application
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        SwarmCommand::Publish(msg) => {
                            match msg.to_bytes() {
                                Ok(data) => {
                                    if let Err(e) = swarm
                                        .behaviour_mut()
                                        .gossipsub
                                        .publish(topic.clone(), data)
                                    {
                                        warn!("Failed to publish: {}", e);
                                    }
                                }
                                Err(e) => error!("Failed to serialize message: {}", e),
                            }
                        }
                        SwarmCommand::Shutdown => {
                            info!("Swarm shutting down");
                            break;
                        }
                    }
                }
                // Handle swarm events
                event = swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(DlrsBehaviourEvent::Gossipsub(
                            gossipsub::Event::Message { message, .. },
                        )) => {
                            match DlrsMessage::from_bytes(&message.data) {
                                Ok(msg) => {
                                    let _ = evt_tx
                                        .send(SwarmEvent2::MessageReceived(msg))
                                        .await;
                                }
                                Err(e) => {
                                    warn!("Failed to decode gossipsub message: {}", e);
                                }
                            }
                        }
                        SwarmEvent::Behaviour(DlrsBehaviourEvent::Mdns(
                            mdns::Event::Discovered(peers),
                        )) => {
                            for (peer_id, _addr) in peers {
                                info!("mDNS discovered peer: {}", peer_id);
                                swarm
                                    .behaviour_mut()
                                    .gossipsub
                                    .add_explicit_peer(&peer_id);
                                let _ = evt_tx
                                    .send(SwarmEvent2::PeerDiscovered(peer_id.to_string()))
                                    .await;
                            }
                        }
                        SwarmEvent::Behaviour(DlrsBehaviourEvent::Mdns(
                            mdns::Event::Expired(peers),
                        )) => {
                            for (peer_id, _addr) in peers {
                                info!("mDNS peer expired: {}", peer_id);
                                swarm
                                    .behaviour_mut()
                                    .gossipsub
                                    .remove_explicit_peer(&peer_id);
                                let _ = evt_tx
                                    .send(SwarmEvent2::PeerLost(peer_id.to_string()))
                                    .await;
                            }
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("Listening on {}", address);
                        }
                        _ => {}
                    }
                }
            }
        }
    });

    Ok((local_peer_id, cmd_tx, evt_rx))
}
