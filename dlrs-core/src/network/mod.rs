//! Network module — P2P networking for DLRS seed distribution
//!
//! Uses libp2p to create a gossip-based network where seeds can be
//! discovered, shared, and verified across peers.

mod protocol;
mod node;
pub mod swarm;

pub use protocol::{DlrsMessage, MessageType, DLRS_TOPIC};
pub use node::{DlrsNode, NodeConfig};
pub use swarm::{run_swarm, SwarmCommand, SwarmConfig, SwarmEvent2};
