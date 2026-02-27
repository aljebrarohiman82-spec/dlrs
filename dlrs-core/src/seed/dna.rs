//! DnaSeed — the programmable DNA of knowledge
//!
//! A seed = compressed knowledge + program + proof + lineage.
//! It is simultaneously data, code, and verification.

use super::{LowRankIdentity, MutationRules, ReplicationPolicy, Lineage};
use crate::zk::ZkCommitment;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Expression instruction — how the seed "unfolds" into action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instruction {
    /// Apply the knowledge matrix to input, produce output
    Transform { input_domain: String, output_domain: String },
    /// Filter: only activate if input matches criteria
    Gate { condition: String, threshold: f64 },
    /// Compose with another seed's output
    Chain { next_seed_id: String },
    /// Emit a signal to the network
    Signal { channel: String, payload_type: String },
}

/// The DNA Seed — minimal self-contained unit of DLRS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaSeed {
    pub id: String,
    pub name: String,
    pub lrim: LowRankIdentity,
    pub express: Vec<Instruction>,
    pub mutation: MutationRules,
    pub replication: ReplicationPolicy,
    pub commitment: ZkCommitment,
    pub lineage: Lineage,
    pub epoch: u64,
    pub fitness: f64,
    pub domains: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub mutated_at: Option<DateTime<Utc>>,
}

impl DnaSeed {
    pub fn new(
        name: impl Into<String>,
        knowledge: &nalgebra::DMatrix<f64>,
        rank: usize,
        domains: Vec<String>,
    ) -> Self {
        let lrim = LowRankIdentity::from_matrix(knowledge, rank);
        let commitment = ZkCommitment::from_lrim(&lrim);
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            lrim, express: Vec::new(),
            mutation: MutationRules::default(),
            replication: ReplicationPolicy::default(),
            commitment, lineage: Lineage::genesis(),
            epoch: 0, fitness: 0.5,
            domains, created_at: Utc::now(), mutated_at: None,
        }
    }

    pub fn express_on(&self, input: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        let k = self.lrim.reconstruct();
        &k * input
    }

    pub fn evolve(&mut self, feedback: &nalgebra::DMatrix<f64>, learning_rate: f64) {
        if !self.mutation.can_mutate(self.fitness) { return; }
        let current = self.lrim.reconstruct();
        let error = feedback - &current;
        let sigma_inv: nalgebra::DVector<f64> = self.lrim.sigma.map(|s| {
            if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }
        });
        for i in 0..self.lrim.rank {
            let si = sigma_inv[i];
            for row in 0..self.lrim.m {
                let grad: f64 = (0..self.lrim.n)
                    .map(|col| error[(row, col)] * self.lrim.v[(col, i)])
                    .sum();
                self.lrim.u[(row, i)] += learning_rate * grad * si;
            }
            for col in 0..self.lrim.n {
                let grad: f64 = (0..self.lrim.m)
                    .map(|row| error[(row, col)] * self.lrim.u[(row, i)])
                    .sum();
                self.lrim.v[(col, i)] += learning_rate * grad * si;
            }
        }
        let new_error = self.lrim.reconstruction_error(feedback);
        let old_error = (feedback - current).norm();
        if new_error < old_error {
            self.fitness = (self.fitness + 0.01).min(1.0);
        } else {
            self.fitness = (self.fitness - 0.005).max(0.0);
        }
        self.commitment = ZkCommitment::from_lrim(&self.lrim);
        self.mutated_at = Some(Utc::now());
        self.epoch += 1;
        self.lineage.record_mutation(self.epoch, new_error);
    }

    pub fn replicate(&self) -> Option<DnaSeed> {
        if !self.replication.should_replicate(self.fitness, self.epoch) { return None; }
        let mut child = self.clone();
        child.id = Uuid::new_v4().to_string();
        child.name = format!("{}_gen{}", self.name, self.epoch + 1);
        child.lineage = self.lineage.spawn_child(&self.id);
        child.epoch = 0;
        child.fitness = self.fitness * 0.9;
        child.created_at = Utc::now();
        child.mutated_at = None;
        Some(child)
    }

    pub fn merge_with(&self, other: &DnaSeed) -> DnaSeed {
        let merged_lrim = LowRankIdentity::merge(&self.lrim, &other.lrim);
        let mut domains = self.domains.clone();
        for d in &other.domains {
            if !domains.contains(d) { domains.push(d.clone()); }
        }
        let mut seed = DnaSeed {
            id: Uuid::new_v4().to_string(),
            name: format!("{}⊕{}", self.name, other.name),
            lrim: merged_lrim, express: Vec::new(),
            mutation: MutationRules::default(),
            replication: ReplicationPolicy::default(),
            commitment: ZkCommitment::default(),
            lineage: Lineage::merge_lineages(&self.lineage, &other.lineage),
            epoch: 0, fitness: (self.fitness + other.fitness) / 2.0,
            domains, created_at: Utc::now(), mutated_at: None,
        };
        seed.commitment = ZkCommitment::from_lrim(&seed.lrim);
        seed
    }

    pub fn summary(&self) -> String {
        format!(
            "DnaSeed '{}' | rank={} | dims={}x{} | fitness={:.3} | epoch={} | domains={:?} | compression={:.1}x",
            self.name, self.lrim.rank, self.lrim.m, self.lrim.n,
            self.fitness, self.epoch, self.domains, self.lrim.compression_ratio()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_create_seed() {
        let k = DMatrix::new_random(50, 30);
        let seed = DnaSeed::new("test-knowledge", &k, 8, vec!["ai".into(), "crypto".into()]);
        println!("{}", seed.summary());
        assert_eq!(seed.lrim.rank, 8);
        assert_eq!(seed.epoch, 0);
    }

    #[test]
    fn test_evolve_seed() {
        let k = DMatrix::new_random(20, 15);
        let mut seed = DnaSeed::new("evolvable", &k, 4, vec!["test".into()]);
        let target = DMatrix::new_random(20, 15);
        let initial_fitness = seed.fitness;
        for _ in 0..10 { seed.evolve(&target, 0.01); }
        println!("Fitness: {:.3} -> {:.3}, Epoch: {}", initial_fitness, seed.fitness, seed.epoch);
        assert_eq!(seed.epoch, 10);
    }
}
