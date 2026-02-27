//! Low-Rank Identity Matrix (LRIM)
//!
//! Every entity in DLRS is represented by: Identity = U · Σ · Vᵀ
//! where U = capability basis, Σ = strength, V = domain projection

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Low-Rank Identity Matrix — the mathematical core of every entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankIdentity {
    /// Capability basis vectors (m × r)
    pub u: DMatrix<f64>,
    /// Capability strengths (diagonal, length r)
    pub sigma: DVector<f64>,
    /// Domain projections (n × r)
    pub v: DMatrix<f64>,
    /// Intrinsic rank
    pub rank: usize,
    /// Dimensions
    pub m: usize,
    pub n: usize,
}

impl LowRankIdentity {
    /// Create a new LRIM from raw factors
    pub fn new(u: DMatrix<f64>, sigma: DVector<f64>, v: DMatrix<f64>) -> Self {
        let rank = sigma.len();
        let m = u.nrows();
        let n = v.nrows();
        assert_eq!(u.ncols(), rank, "U columns must equal rank");
        assert_eq!(v.ncols(), rank, "V columns must equal rank");
        Self { u, sigma, v, rank, m, n }
    }

    /// Create LRIM from a full matrix via truncated SVD
    pub fn from_matrix(k: &DMatrix<f64>, target_rank: usize) -> Self {
        let svd = k.clone().svd(true, true);
        let u_full = svd.u.expect("SVD must produce U");
        let v_full = svd.v_t.expect("SVD must produce Vt").transpose();
        let s_full = svd.singular_values;

        let r = target_rank.min(s_full.len());

        let u = u_full.columns(0, r).into_owned();
        let sigma = s_full.rows(0, r).into_owned();
        let v = v_full.columns(0, r).into_owned();

        Self::new(u, sigma, v)
    }

    /// Reconstruct the approximate matrix K ≈ U · diag(Σ) · Vᵀ
    pub fn reconstruct(&self) -> DMatrix<f64> {
        let sigma_mat = DMatrix::from_diagonal(&self.sigma);
        &self.u * sigma_mat * self.v.transpose()
    }

    /// Compute reconstruction error (Frobenius norm)
    pub fn reconstruction_error(&self, original: &DMatrix<f64>) -> f64 {
        let approx = self.reconstruct();
        let diff = original - approx;
        diff.norm()
    }

    /// Compression ratio: original_params / low_rank_params
    pub fn compression_ratio(&self) -> f64 {
        let original = (self.m * self.n) as f64;
        let compressed = ((self.m + self.n) * self.rank + self.rank) as f64;
        original / compressed
    }

    /// Fingerprint: SHA256 hash of the factorization (for commitments)
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for val in self.u.iter() {
            hasher.update(val.to_le_bytes());
        }
        for val in self.sigma.iter() {
            hasher.update(val.to_le_bytes());
        }
        for val in self.v.iter() {
            hasher.update(val.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Compute capability score in a given domain direction
    pub fn capability_in_domain(&self, domain_vector: &DVector<f64>) -> f64 {
        // Project domain onto V space, weight by sigma
        let projection = self.v.transpose() * domain_vector;
        let weighted: f64 = projection.iter()
            .zip(self.sigma.iter())
            .map(|(p, s)| p * s)
            .sum();
        weighted.abs()
    }

    /// Merge two LRIMs into a higher-rank composite
    pub fn merge(a: &Self, b: &Self) -> Self {
        assert_eq!(a.m, b.m, "Dimension m must match");
        assert_eq!(a.n, b.n, "Dimension n must match");

        // Reconstruct both and re-factorize at combined rank
        let combined = a.reconstruct() + b.reconstruct();
        let target_rank = (a.rank + b.rank).min(a.m.min(a.n));
        Self::from_matrix(&combined, target_rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_factorize_and_reconstruct() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);

        assert_eq!(lrim.rank, 3);
        assert_eq!(lrim.m, 10);
        assert_eq!(lrim.n, 8);

        let error = lrim.reconstruction_error(&k);
        println!("Reconstruction error at rank 3: {:.6}", error);

        let lrim_full = LowRankIdentity::from_matrix(&k, 8);
        let error_full = lrim_full.reconstruction_error(&k);
        assert!(error_full < 1e-10, "Full rank error should be ~0");
    }

    #[test]
    fn test_compression_ratio() {
        let k = DMatrix::new_random(1000, 500);
        let lrim = LowRankIdentity::from_matrix(&k, 16);
        let ratio = lrim.compression_ratio();
        println!("1000x500 at rank 16: {:.1}x compression", ratio);
        assert!(ratio > 10.0, "Should achieve significant compression");
    }

    #[test]
    fn test_merge() {
        let a = DMatrix::new_random(10, 8);
        let b = DMatrix::new_random(10, 8);
        let lrim_a = LowRankIdentity::from_matrix(&a, 3);
        let lrim_b = LowRankIdentity::from_matrix(&b, 3);
        let merged = LowRankIdentity::merge(&lrim_a, &lrim_b);
        assert!(merged.rank <= 6);
    }
}
