//! Seed network synchronization
//!
//! Gossip protocol for distributing DNA seeds across peers.

use crate::seed::DnaSeed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    pub id: String,
    pub address: String,
    pub domains: Vec<String>,
    pub last_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedFingerprint {
    pub seed_id: String,
    pub name: String,
    pub rank: usize,
    pub fitness: f64,
    pub epoch: u64,
    pub domains: Vec<String>,
    pub commitment_hash: String,
}

impl From<&DnaSeed> for SeedFingerprint {
    fn from(seed: &DnaSeed) -> Self {
        Self {
            seed_id: seed.id.clone(),
            name: seed.name.clone(),
            rank: seed.lrim.rank,
            fitness: seed.fitness,
            epoch: seed.epoch,
            domains: seed.domains.clone(),
            commitment_hash: seed.commitment.matrix_hash.clone(),
        }
    }
}

pub struct SeedNetwork {
    pub seeds: HashMap<String, DnaSeed>,
    pub peers: HashMap<String, Peer>,
    pub interested_domains: Vec<String>,
    pub min_fitness_threshold: f64,
    pub max_seeds: usize,
}

impl SeedNetwork {
    pub fn new(domains: Vec<String>) -> Self {
        Self {
            seeds: HashMap::new(),
            peers: HashMap::new(),
            interested_domains: domains,
            min_fitness_threshold: 0.3,
            max_seeds: 1000,
        }
    }

    pub fn add_local_seed(&mut self, seed: DnaSeed) {
        self.seeds.insert(seed.id.clone(), seed);
    }

    pub fn generate_fingerprints(&self) -> Vec<SeedFingerprint> {
        self.seeds.values().map(SeedFingerprint::from).collect()
    }

    pub fn want_seeds(&self, remote_fingerprints: &[SeedFingerprint]) -> Vec<String> {
        remote_fingerprints.iter()
            .filter(|fp| {
                !self.seeds.contains_key(&fp.seed_id)
                && fp.fitness >= self.min_fitness_threshold
                && fp.domains.iter().any(|d| self.interested_domains.contains(d))
            })
            .map(|fp| fp.seed_id.clone())
            .collect()
    }

    pub fn accept_seed(&mut self, seed: DnaSeed) -> bool {
        if seed.fitness < self.min_fitness_threshold { return false; }
        if !seed.commitment.verify(&seed.lrim) { return false; }
        if self.seeds.len() >= self.max_seeds {
            if let Some(worst_id) = self.seeds.values()
                .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .map(|s| s.id.clone())
            {
                if self.seeds[&worst_id].fitness < seed.fitness {
                    self.seeds.remove(&worst_id);
                } else { return false; }
            }
        }
        self.seeds.insert(seed.id.clone(), seed);
        true
    }

    pub fn top_seeds(&self, domain: &str, limit: usize) -> Vec<&DnaSeed> {
        let mut domain_seeds: Vec<&DnaSeed> = self.seeds.values()
            .filter(|s| s.domains.iter().any(|d| d == domain))
            .collect();
        domain_seeds.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        domain_seeds.truncate(limit);
        domain_seeds
    }

    pub fn stats(&self) -> String {
        format!(
            "Network: {} seeds, {} peers, domains: {:?}",
            self.seeds.len(), self.peers.len(), self.interested_domains
        )
    }
}
