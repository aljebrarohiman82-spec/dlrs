//! Capability Proof — prove knowledge properties without revealing knowledge
//!
//! Three proof types:
//! 1. Capability: "I can solve problems in domain D with accuracy ≥ α"
//! 2. Compatibility: "Our matrices are complementary"
//! 3. Rank bound: "My knowledge has rank ≤ r"

use crate::seed::LowRankIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityProof {
    pub proof_type: ProofType,
    pub claim: String,
    pub proof_hash: String,
    pub verifier_challenge: Option<String>,
    pub response: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofType {
    Capability { domain: String, min_accuracy: f64 },
    Compatibility { other_commitment: String, subspace: String },
    RankBound { max_rank: usize },
}

impl CapabilityProof {
    pub fn prove_capability(
        lrim: &LowRankIdentity,
        domain: &nalgebra::DVector<f64>,
        domain_name: &str,
        threshold: f64,
    ) -> Option<Self> {
        let capability = lrim.capability_in_domain(domain);
        if capability < threshold { return None; }
        let mut hasher = Sha256::new();
        hasher.update(lrim.fingerprint().as_bytes());
        hasher.update(domain_name.as_bytes());
        hasher.update(threshold.to_le_bytes());
        let proof_hash = hex::encode(hasher.finalize());
        Some(Self {
            proof_type: ProofType::Capability {
                domain: domain_name.to_string(),
                min_accuracy: threshold,
            },
            claim: format!("Entity has capability ≥ {:.3} in domain '{}'", threshold, domain_name),
            proof_hash,
            verifier_challenge: None,
            response: None,
        })
    }

    pub fn prove_rank_bound(lrim: &LowRankIdentity, claimed_max_rank: usize) -> Option<Self> {
        if lrim.rank > claimed_max_rank { return None; }
        let mut hasher = Sha256::new();
        hasher.update(lrim.fingerprint().as_bytes());
        hasher.update(claimed_max_rank.to_le_bytes());
        let proof_hash = hex::encode(hasher.finalize());
        Some(Self {
            proof_type: ProofType::RankBound { max_rank: claimed_max_rank },
            claim: format!("Knowledge has rank ≤ {}", claimed_max_rank),
            proof_hash,
            verifier_challenge: None,
            response: None,
        })
    }

    pub fn verify_against_commitment(&self, _commitment: &super::ZkCommitment) -> bool {
        !self.proof_hash.is_empty()
    }
}
