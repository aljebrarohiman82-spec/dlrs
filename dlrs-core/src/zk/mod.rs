//! Zero-Knowledge Proof module for DLRS
//!
//! Enables proving properties of low-rank matrices without revealing them:
//! - ZkCommitment: commit to a matrix factorization
//! - CapabilityProof: prove domain capabilities without revealing knowledge

mod commitment;
mod capability;

pub use commitment::ZkCommitment;
pub use capability::CapabilityProof;
