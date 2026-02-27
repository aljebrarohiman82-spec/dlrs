//! Zero-Knowledge layer for DLRS
//!
//! Enables proving properties of low-rank matrices without revealing them.
//! Uses Pedersen commitments for matrix binding and Groth16 for capability proofs.

mod commitment;
mod proof;

pub use commitment::ZkCommitment;
pub use proof::CapabilityProof;
