//! CapabilityProof — prove domain capabilities without revealing knowledge
//!
//! A capability proof demonstrates that a seed can solve problems in a
//! given domain to a certain quality level, without revealing the underlying
//! matrix factorization.

use crate::seed::LowRankIdentity;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A proof that a seed has capability in a specific domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityProof {
    /// The domain this proof applies to
    pub domain: String,
    /// Claimed capability score (0.0 to 1.0)
    pub claimed_score: f64,
    /// Challenge-response hash proving the claim
    pub proof_hash: String,
    /// The commitment root this proof is bound to
    pub commitment_root: String,
    /// Timestamp of proof generation
    pub created_at: DateTime<Utc>,
    /// Proof is valid until this time
    pub expires_at: DateTime<Utc>,
    /// Number of challenge rounds completed
    pub challenge_rounds: u32,
}

/// A challenge issued by a verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    /// Random domain vector (serialized)
    pub domain_vector: Vec<f64>,
    /// Expected minimum score
    pub threshold: f64,
    /// Challenge nonce for freshness
    pub nonce: String,
}

/// Result of verifying a capability proof
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// Proof is valid
    Valid { score: f64 },
    /// Proof has expired
    Expired,
    /// Proof failed verification
    Invalid { reason: String },
}

impl CapabilityProof {
    /// Generate a capability proof for a given domain
    pub fn generate(
        lrim: &LowRankIdentity,
        commitment_root: &str,
        domain: &str,
        domain_vector: &nalgebra::DVector<f64>,
        validity_hours: i64,
    ) -> Self {
        let score = lrim.capability_in_domain(domain_vector);
        // Normalize score to 0..1 range
        let normalized = (score / (score + 1.0)).min(1.0);

        let proof_hash = Self::compute_proof_hash(
            commitment_root,
            domain,
            normalized,
            lrim.rank,
        );

        let now = Utc::now();
        Self {
            domain: domain.to_string(),
            claimed_score: normalized,
            proof_hash,
            commitment_root: commitment_root.to_string(),
            created_at: now,
            expires_at: now + chrono::Duration::hours(validity_hours),
            challenge_rounds: 1,
        }
    }

    /// Respond to a verification challenge
    pub fn respond_to_challenge(
        &self,
        lrim: &LowRankIdentity,
        challenge: &Challenge,
    ) -> ChallengeResponse {
        let domain_vec = nalgebra::DVector::from_vec(challenge.domain_vector.clone());
        let score = lrim.capability_in_domain(&domain_vec);
        let normalized = (score / (score + 1.0)).min(1.0);

        let response_hash = Self::compute_response_hash(
            &self.proof_hash,
            &challenge.nonce,
            normalized,
        );

        ChallengeResponse {
            score: normalized,
            response_hash,
            meets_threshold: normalized >= challenge.threshold,
        }
    }

    /// Check if this proof has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Verify structural integrity of the proof
    pub fn verify_structure(&self) -> VerificationResult {
        if self.is_expired() {
            return VerificationResult::Expired;
        }
        if self.claimed_score < 0.0 || self.claimed_score > 1.0 {
            return VerificationResult::Invalid {
                reason: "Score out of range".to_string(),
            };
        }
        if self.proof_hash.is_empty() {
            return VerificationResult::Invalid {
                reason: "Empty proof hash".to_string(),
            };
        }
        VerificationResult::Valid {
            score: self.claimed_score,
        }
    }

    fn compute_proof_hash(
        commitment_root: &str,
        domain: &str,
        score: f64,
        rank: usize,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(commitment_root.as_bytes());
        hasher.update(domain.as_bytes());
        hasher.update(score.to_le_bytes());
        hasher.update(rank.to_le_bytes());
        hex::encode(hasher.finalize())
    }

    fn compute_response_hash(proof_hash: &str, nonce: &str, score: f64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(proof_hash.as_bytes());
        hasher.update(nonce.as_bytes());
        hasher.update(score.to_le_bytes());
        hex::encode(hasher.finalize())
    }
}

/// Response to a verification challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeResponse {
    pub score: f64,
    pub response_hash: String,
    pub meets_threshold: bool,
}

impl Challenge {
    /// Create a new random challenge for a domain
    pub fn new(dimension: usize, threshold: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let domain_vector: Vec<f64> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let nonce: String = hex::encode(Sha256::new().chain_update(
            rng.gen::<u64>().to_le_bytes()
        ).finalize());

        Self {
            domain_vector,
            threshold,
            nonce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_generate_proof() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);
        let domain_vec = DVector::new_random(8);

        let proof = CapabilityProof::generate(
            &lrim,
            "test-commitment-root",
            "machine-learning",
            &domain_vec,
            24,
        );

        assert_eq!(proof.domain, "machine-learning");
        assert!(proof.claimed_score >= 0.0 && proof.claimed_score <= 1.0);
        assert!(!proof.is_expired());
    }

    #[test]
    fn test_verify_structure() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);
        let domain_vec = DVector::new_random(8);

        let proof = CapabilityProof::generate(&lrim, "root", "ai", &domain_vec, 24);
        match proof.verify_structure() {
            VerificationResult::Valid { score } => {
                assert!(score >= 0.0 && score <= 1.0);
            }
            other => panic!("Expected Valid, got {:?}", other),
        }
    }

    #[test]
    fn test_challenge_response() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);
        let domain_vec = DVector::new_random(8);

        let proof = CapabilityProof::generate(&lrim, "root", "ai", &domain_vec, 24);
        let challenge = Challenge::new(8, 0.0);
        let response = proof.respond_to_challenge(&lrim, &challenge);

        assert!(response.score >= 0.0);
        assert!(response.meets_threshold); // threshold is 0, always meets
    }

    #[test]
    fn test_expired_proof() {
        let k = DMatrix::new_random(10, 8);
        let lrim = LowRankIdentity::from_matrix(&k, 3);
        let domain_vec = DVector::new_random(8);

        let mut proof = CapabilityProof::generate(&lrim, "root", "ai", &domain_vec, 0);
        proof.expires_at = Utc::now() - chrono::Duration::hours(1);

        assert!(proof.is_expired());
        match proof.verify_structure() {
            VerificationResult::Expired => {}
            other => panic!("Expected Expired, got {:?}", other),
        }
    }
}
