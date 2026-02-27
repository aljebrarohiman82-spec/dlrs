//! DNA Seed — the atomic unit of Distributed Low-Rank Space
//!
//! A seed encodes: knowledge (low-rank matrix) + program (expression/mutation/replication)
//! + proof (ZK commitment) + lineage (Merkle ancestry).

mod lrim;
mod dna;
mod mutation;
mod replication;
mod lineage;

pub use lrim::LowRankIdentity;
pub use dna::DnaSeed;
pub use mutation::MutationRules;
pub use replication::ReplicationPolicy;
pub use lineage::Lineage;
