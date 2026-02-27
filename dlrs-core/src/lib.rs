//! DLRS — Distributed Low-Rank Space
//!
//! A framework treating knowledge as low-rank matrix decompositions
//! distributed across a trustless network with zero-knowledge proofs.

pub mod seed;
pub mod zk;
pub mod network;
pub mod storage;

pub use seed::{DnaSeed, LowRankIdentity, MutationRules, ReplicationPolicy};
pub use zk::{ZkCommitment, CapabilityProof};
pub use storage::SeedStore;
