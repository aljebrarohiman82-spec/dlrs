//! ZK Commitment — cryptographic binding to a low-rank matrix
//!
//! Commit to LRIM without revealing U, Σ, V.
//! Anyone can verify the commitment matches future proofs.

use crate::seed::LowRankIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A cryptographic commitment to a low-rank identity matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkCommitment {
    pub matrix_hash: String,
    pub committed_rank: usize,
    pub committed_dims: (usize, usize),
    pub blinding_hash: String,
    pub sigma_norm_commitment: f64,
}

impl Default for ZkCommitment {
    fn default() -> Self {
        Self {
            matrix_hash: String::new(),
            committed_rank: 0,
            committed_dims: (0, 0),
            blinding_hash: String::new(),
            sigma_norm_commitment: 0.0,
        }
    }
}

impl ZkCommitment {
    pub fn from_lrim(lrim: &LowRankIdentity) -> Self {
        let matrix_hash = lrim.fingerprint();
        let blinding = rand::random::<[u8; 32]>();
        let mut hasher = Sha256::new();
        hasher.update(&blinding);
        hasher.update(matrix_hash.as_bytes());
        let blinding_hash = hex::encode(hasher.finalize());
        let sigma_norm: f64 = lrim.sigma.iter().map(|s| s * s).sum::<f64>().sqrt();
        Self {
            matrix_hash,
            committed_rank: lrim.rank,
            committed_dims: (lrim.m, lrim.n),
            blinding_hash,
            sigma_norm_commitment: sigma_norm,
        }
    }

    pub fn verify(&self, lrim: &LowRankIdentity) -> bool {
        lrim.fingerprint() == self.matrix_hash
            && lrim.rank == self.committed_rank
            && lrim.m == self.committed_dims.0
            && lrim.n == self.committed_dims.1
    }

    pub fn public_summary(&self) -> String {
        format!(
            "Commitment: rank={}, dims={}x{}, ‖Σ‖={:.4}, hash={}…",
            self.committed_rank,
            self.committed_dims.0,
            self.committed_dims.1,
            self.sigma_norm_commitment,
            &self.matrix_hash[..16]
        )
    }
}
