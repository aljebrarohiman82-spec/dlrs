//! Intelligent Sharing — smart adapter matching, reputation, and discovery
//!
//! Manages the knowledge marketplace:
//! - Adapter registry with domain/capability indexing
//! - Reputation scoring for shared adapters
//! - Smart matching: find best remote adapter for a task
//! - Share policies: what to share, when, with whom

use crate::lora::adapter::LoraAdapter;
use crate::zk::CapabilityProof;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A record of a remote adapter discovered on the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAdapter {
    pub seed_id: String,
    pub peer_id: String,
    pub commitment_root: String,
    pub domains: Vec<String>,
    pub fitness: f64,
    pub rank: usize,
    pub discovered_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub reputation: ReputationScore,
    /// Capability proofs received from this adapter
    pub proofs: Vec<CapabilityProof>,
}

/// Reputation score for a remote adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Number of successful capability proofs verified
    pub verified_proofs: u32,
    /// Number of failed/invalid proofs
    pub failed_proofs: u32,
    /// Average capability score from proofs
    pub avg_score: f64,
    /// How long this peer has been known (seconds)
    pub uptime: u64,
    /// Computed overall reputation (0.0 to 1.0)
    pub overall: f64,
}

impl Default for ReputationScore {
    fn default() -> Self {
        Self {
            verified_proofs: 0,
            failed_proofs: 0,
            avg_score: 0.0,
            uptime: 0,
            overall: 0.5,
        }
    }
}

impl ReputationScore {
    /// Recompute overall reputation
    pub fn update(&mut self) {
        let total = self.verified_proofs + self.failed_proofs;
        let proof_ratio = if total > 0 {
            self.verified_proofs as f64 / total as f64
        } else {
            0.5
        };
        let age_bonus = (self.uptime as f64 / 3600.0).min(1.0) * 0.1;
        self.overall = (proof_ratio * 0.6 + self.avg_score * 0.3 + age_bonus).min(1.0);
    }

    pub fn record_success(&mut self, score: f64) {
        self.verified_proofs += 1;
        let total = self.verified_proofs as f64;
        self.avg_score = self.avg_score * (total - 1.0) / total + score / total;
        self.update();
    }

    pub fn record_failure(&mut self) {
        self.failed_proofs += 1;
        self.update();
    }
}

/// Policy for what to share
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharePolicy {
    /// Minimum fitness to share an adapter
    pub min_fitness: f64,
    /// Domains to share (empty = all)
    pub allowed_domains: Vec<String>,
    /// Minimum reputation of requester to respond to
    pub min_requester_reputation: f64,
    /// Whether to auto-respond to capability challenges
    pub auto_respond: bool,
    /// Maximum number of proofs to generate per minute
    pub rate_limit: u32,
}

impl Default for SharePolicy {
    fn default() -> Self {
        Self {
            min_fitness: 0.3,
            allowed_domains: Vec::new(), // all domains
            min_requester_reputation: 0.0,
            auto_respond: true,
            rate_limit: 60,
        }
    }
}

/// The sharing registry — tracks remote adapters and manages discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingRegistry {
    /// Known remote adapters
    remote_adapters: HashMap<String, RemoteAdapter>,
    /// Share policy
    pub policy: SharePolicy,
    /// Domain index: domain -> list of seed_ids
    domain_index: HashMap<String, Vec<String>>,
}

impl SharingRegistry {
    pub fn new() -> Self {
        Self {
            remote_adapters: HashMap::new(),
            policy: SharePolicy::default(),
            domain_index: HashMap::new(),
        }
    }

    /// Register or update a remote adapter from a network announcement
    pub fn register_remote(
        &mut self,
        seed_id: String,
        peer_id: String,
        commitment_root: String,
        domains: Vec<String>,
        fitness: f64,
        rank: usize,
    ) {
        let now = Utc::now();
        let entry = self.remote_adapters.entry(seed_id.clone()).or_insert_with(|| {
            RemoteAdapter {
                seed_id: seed_id.clone(),
                peer_id: peer_id.clone(),
                commitment_root: commitment_root.clone(),
                domains: domains.clone(),
                fitness,
                rank,
                discovered_at: now,
                last_seen: now,
                reputation: ReputationScore::default(),
                proofs: Vec::new(),
            }
        });

        entry.last_seen = now;
        entry.fitness = fitness;
        entry.domains = domains.clone();

        // Update domain index
        for domain in &domains {
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .retain(|id| id != &seed_id);
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .push(seed_id.clone());
        }
    }

    /// Record a successful capability proof from a remote adapter
    pub fn record_proof(&mut self, seed_id: &str, proof: CapabilityProof) {
        if let Some(remote) = self.remote_adapters.get_mut(seed_id) {
            remote.reputation.record_success(proof.claimed_score);
            remote.proofs.push(proof);
        }
    }

    /// Record a failed proof attempt
    pub fn record_proof_failure(&mut self, seed_id: &str) {
        if let Some(remote) = self.remote_adapters.get_mut(seed_id) {
            remote.reputation.record_failure();
        }
    }

    /// Find the best remote adapters for a given domain
    pub fn find_for_domain(&self, domain: &str, min_reputation: f64) -> Vec<&RemoteAdapter> {
        let mut candidates: Vec<&RemoteAdapter> = self.domain_index
            .get(domain)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.remote_adapters.get(id))
                    .filter(|r| r.reputation.overall >= min_reputation)
                    .collect()
            })
            .unwrap_or_default();

        // Sort by fitness * reputation (higher is better)
        candidates.sort_by(|a, b| {
            let score_a = a.fitness * a.reputation.overall;
            let score_b = b.fitness * b.reputation.overall;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Check if a local adapter should be shared according to policy
    pub fn should_share(&self, adapter: &LoraAdapter) -> bool {
        if adapter.seed.fitness < self.policy.min_fitness {
            return false;
        }
        if !self.policy.allowed_domains.is_empty() {
            let has_allowed = adapter.config.domains.iter().any(|d|
                self.policy.allowed_domains.contains(d)
            );
            if !has_allowed {
                return false;
            }
        }
        true
    }

    /// Get the number of known remote adapters
    pub fn remote_count(&self) -> usize {
        self.remote_adapters.len()
    }

    /// Get all known domains
    pub fn known_domains(&self) -> Vec<String> {
        self.domain_index.keys().cloned().collect()
    }

    /// Get registry summary
    pub fn summary(&self) -> String {
        let total = self.remote_adapters.len();
        let domains = self.domain_index.len();
        let avg_rep = if total > 0 {
            self.remote_adapters.values().map(|r| r.reputation.overall).sum::<f64>() / total as f64
        } else {
            0.0
        };
        format!(
            "Registry: {} remote adapters, {} domains, avg reputation={:.3}",
            total, domains, avg_rep
        )
    }

    /// Remove stale remote adapters not seen for a given duration
    pub fn prune_stale(&mut self, max_age_secs: i64) -> usize {
        let now = Utc::now();
        let to_remove: Vec<String> = self.remote_adapters
            .iter()
            .filter(|(_, r)| (now - r.last_seen).num_seconds() > max_age_secs)
            .map(|(id, _)| id.clone())
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            self.remote_adapters.remove(&id);
        }
        // Rebuild domain index
        if count > 0 {
            self.domain_index.clear();
            for (id, remote) in &self.remote_adapters {
                for domain in &remote.domains {
                    self.domain_index
                        .entry(domain.clone())
                        .or_default()
                        .push(id.clone());
                }
            }
        }
        count
    }

    /// Save registry to file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load registry from file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

impl Default for SharingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_find() {
        let mut reg = SharingRegistry::new();
        reg.register_remote(
            "seed-1".into(), "peer-a".into(), "commit-1".into(),
            vec!["nlp".into(), "reasoning".into()], 0.8, 8,
        );
        reg.register_remote(
            "seed-2".into(), "peer-b".into(), "commit-2".into(),
            vec!["nlp".into()], 0.6, 4,
        );
        reg.register_remote(
            "seed-3".into(), "peer-c".into(), "commit-3".into(),
            vec!["vision".into()], 0.9, 16,
        );

        let nlp = reg.find_for_domain("nlp", 0.0);
        assert_eq!(nlp.len(), 2);

        let vision = reg.find_for_domain("vision", 0.0);
        assert_eq!(vision.len(), 1);
    }

    #[test]
    fn test_reputation() {
        let mut rep = ReputationScore::default();
        assert!((rep.overall - 0.5).abs() < 0.01);

        rep.record_success(0.9);
        rep.record_success(0.8);
        assert!(rep.overall > 0.5);

        rep.record_failure();
        rep.record_failure();
        rep.record_failure();
        assert!(rep.overall < 0.8);
    }

    #[test]
    fn test_share_policy() {
        let reg = SharingRegistry::new();
        let adapter = LoraAdapter::new(crate::lora::LoraConfig {
            name: "test".into(),
            target_module: "t".into(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["nlp".into()],
            original_dims: (16, 16),
        });
        assert!(reg.should_share(&adapter)); // default policy, fitness=0.5 > 0.3
    }

    #[test]
    fn test_known_domains() {
        let mut reg = SharingRegistry::new();
        reg.register_remote("s1".into(), "p".into(), "c".into(), vec!["a".into(), "b".into()], 0.5, 4);
        reg.register_remote("s2".into(), "p".into(), "c".into(), vec!["b".into(), "c".into()], 0.5, 4);

        let domains = reg.known_domains();
        assert_eq!(domains.len(), 3);
    }
}
