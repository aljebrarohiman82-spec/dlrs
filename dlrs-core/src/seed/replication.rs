//! Replication policy — when and how seeds copy themselves

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationPolicy {
    pub min_fitness: f64,
    pub min_epochs: u64,
    pub max_children: usize,
    pub children_produced: usize,
    pub enabled: bool,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self {
            min_fitness: 0.7,
            min_epochs: 5,
            max_children: 3,
            children_produced: 0,
            enabled: true,
        }
    }
}

impl ReplicationPolicy {
    pub fn should_replicate(&self, fitness: f64, epoch: u64) -> bool {
        self.enabled
            && fitness >= self.min_fitness
            && epoch >= self.min_epochs
            && self.children_produced < self.max_children
    }

    pub fn sterile() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    pub fn viral() -> Self {
        Self {
            min_fitness: 0.5,
            min_epochs: 2,
            max_children: 10,
            children_produced: 0,
            enabled: true,
        }
    }
}
