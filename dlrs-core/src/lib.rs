//! DLRS — Distributed Low-Rank Space
//!
//! A framework treating knowledge as low-rank matrix decompositions
//! distributed across a trustless network with zero-knowledge proofs.

pub mod seed;
pub mod zk;
pub mod network;
pub mod storage;
pub mod lora;

pub use seed::{DnaSeed, LowRankIdentity, MutationRules, ReplicationPolicy, Lineage};
pub use zk::{ZkCommitment, CapabilityProof};
pub use network::{DlrsMessage, DlrsNode, NodeConfig};
pub use storage::SeedStore;
pub use lora::{LoraAdapter, LoraConfig, LoraManager, AutoShareDaemon};
