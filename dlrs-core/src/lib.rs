//! DLRS — Distributed Low-Rank Space
//!
//! A framework treating knowledge as low-rank matrix decompositions
//! distributed across a trustless network with zero-knowledge proofs
//! and TEE hardware security.

pub mod seed;
pub mod zk;
pub mod network;
pub mod storage;
pub mod lora;
pub mod tee;

pub use seed::{DnaSeed, LowRankIdentity, MutationRules, ReplicationPolicy, Lineage};
pub use zk::{ZkCommitment, CapabilityProof};
pub use network::{DlrsMessage, DlrsNode, NodeConfig};
pub use storage::{SeedStore, BackupManager, BackupManifest, SnapshotMeta};
pub use lora::{
    LoraAdapter, LoraConfig, LoraManager, AutoShareDaemon,
    Dataset, TrainConfig, TrainHistory, train, auto_train, auto_tune,
    SharingRegistry, SharePolicy,
};
pub use tee::{
    TeeEnclave, TeeBackend, TeeError, SecurityLevel,
    AttestationPolicy, AttestationVerdict,
    SealedStorage, SecureChannel,
};
