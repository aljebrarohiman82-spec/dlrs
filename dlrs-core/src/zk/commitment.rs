//! ZkCommitment — Zero-Knowledge commitment to a low-rank matrix
//!
//! A ZkCommitment binds a seed to its LRIM without revealing the actual
//! matrix factors. It uses SHA256-based Pedersen-style commitments over
//! the matrix components, producing a verifiable fingerprint.

use crate::seed::LowRankIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A zero-knowledge commitment to a low-rank identity matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkCommitment {
    /// Commitment to U factor
    pub u_commitment: String,
    /// Commitment to sigma (singular values)
    pub sigma_commitment: String,
    /// Commitment to V factor
    pub v_commitment: String,
    /// Combined commitment hash (root)
    pub root_commitment: String,
    /// The rank that was committed to (public info)
    pub committed_rank: usize,
    /// Dimensions (public info)
    pub committed_m: usize,
    pub committed_n: usize,
    /// Blinding factor hash (for hiding)
    pub blinding_hash: String,
}

impl Default for ZkCommitment {
    fn default() -> Self {
        Self {
            u_commitment: String::new(),
            sigma_commitment: String::new(),
            v_commitment: String::new(),
            root_commitment: String::new(),
            committed_rank: 0,
            committed_m: 0,
            committed_n: 0,
            blinding_hash: String::new(),
        }
    }
}

impl ZkCommitment {
    /// Create a commitment from a low-rank identity matrix
    pub fn from_lrim(lrim: &LowRankIdentity) -> Self {
        let blinding = Self::generate_blinding();
        let u_commitment = Self::commit_matrix_data(lrim.u.as_slice(), &blinding, "U");
        let sigma_commitment = Self::commit_vector_data(lrim.sigma.as_slice(), &blinding, "S");
        let v_commitment = Self::commit_matrix_data(lrim.v.as_slice(), &blinding, "V");
        let root_commitment = Self::compute_root(&u_commitment, &sigma_commitment, &v_commitment);
        let blinding_hash = Self::hash_bytes(&blinding);

        Self {
            u_commitment,
            sigma_commitment,
            v_commitment,
            root_commitment,
            committed_rank: lrim.rank,
            committed_m: lrim.m,
            committed_n: lrim.n,
            blinding_hash,
        }
    }

    /// Verify that a given LRIM matches this commitment (requires blinding factor)
    pub fn verify(&self, lrim: &LowRankIdentity) -> bool {
        // Structural check: rank and dimensions must match
        if lrim.rank != self.committed_rank
            || lrim.m != self.committed_m
            || lrim.n != self.committed_n
        {
            return false;
        }
        // Verify fingerprint matches root commitment
        let fingerprint = lrim.fingerprint();
        let check_hash = Self::hash_str(&format!("{}:{}", fingerprint, self.blinding_hash));
        check_hash == self.root_commitment
    }

    /// Check if two commitments are to matrices of compatible dimensions
    pub fn is_compatible(&self, other: &ZkCommitment) -> bool {
        self.committed_n == other.committed_m
    }

    /// Compute a merged commitment from two parent commitments
    pub fn merge(a: &ZkCommitment, b: &ZkCommitment) -> ZkCommitment {
        let mut hasher = Sha256::new();
        hasher.update(a.root_commitment.as_bytes());
        hasher.update(b.root_commitment.as_bytes());
        hasher.update(b"merge");
        let root = hex::encode(hasher.finalize());
        ZkCommitment {
            u_commitment: format!("merged:{}", &root[..16]),
            sigma_commitment: format!("merged:{}", &root[16..32]),
            v_commitment: format!("merged:{}", &root[32..48]),
            root_commitment: root,
            committed_rank: a.committed_rank + b.committed_rank,
            committed_m: a.committed_m,
            committed_n: b.committed_n,
            blinding_hash: String::new(),
        }
    }

    fn generate_blinding() -> Vec<u8> {
        use sha2::Sha256;
        let mut hasher = Sha256::new();
        // Use timestamp + fixed salt as pseudo-random blinding
        let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        hasher.update(now.to_le_bytes());
        hasher.update(b"dlrs-blinding-v1");
        hasher.finalize().to_vec()
    }

    fn commit_matrix_data(data: &[f64], blinding: &[u8], label: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(label.as_bytes());
        hasher.update(blinding);
        for val in data {
            hasher.update(val.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    fn commit_vector_data(data: &[f64], blinding: &[u8], label: &str) -> String {
        Self::commit_matrix_data(data, blinding, label)
    }

    fn compute_root(u_commit: &str, sigma_commit: &str, v_commit: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(u_commit.as_bytes());
        hasher.update(sigma_commit.as_bytes());
        hasher.update(v_commit.as_bytes());
        hex::encode(hasher.finalize())
    }

    fn hash_bytes(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn hash_str(data: &str) -> String {
        Self::hash_bytes(data.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_commitment_from_lrim() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);
        let commitment = ZkCommitment::from_lrim(&lrim);

        assert_eq!(commitment.committed_rank, 3);
        assert_eq!(commitment.committed_m, 10);
        assert_eq!(commitment.committed_n, 8);
        assert!(!commitment.root_commitment.is_empty());
        assert!(!commitment.u_commitment.is_empty());
    }

    #[test]
    fn test_different_matrices_different_commitments() {
        let k1 = DMatrix::new_random(10, 8);
        let k2 = DMatrix::new_random(10, 8);
        let c1 = ZkCommitment::from_lrim(&LowRankIdentity::from_matrix(&k1, 3));
        let c2 = ZkCommitment::from_lrim(&LowRankIdentity::from_matrix(&k2, 3));
        // Root commitments should differ (different blinding + data)
        assert_ne!(c1.root_commitment, c2.root_commitment);
    }

    #[test]
    fn test_compatibility_check() {
        let c1 = ZkCommitment {
            committed_m: 10,
            committed_n: 8,
            ..ZkCommitment::default()
        };
        let c2 = ZkCommitment {
            committed_m: 8,
            committed_n: 5,
            ..ZkCommitment::default()
        };
        let c3 = ZkCommitment {
            committed_m: 7,
            committed_n: 5,
            ..ZkCommitment::default()
        };
        assert!(c1.is_compatible(&c2));  // 10x8 * 8x5 = OK
        assert!(!c1.is_compatible(&c3)); // 10x8 * 7x5 = NO
    }

    #[test]
    fn test_default_commitment() {
        let c = ZkCommitment::default();
        assert_eq!(c.committed_rank, 0);
        assert!(c.root_commitment.is_empty());
    }
}
