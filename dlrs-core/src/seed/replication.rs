//! ReplicationPolicy — governs when and how seeds reproduce
//!
//! Replication is how knowledge propagates through the network.
//! Policies control fitness thresholds, population limits, and offspring variation.

use serde::{Deserialize, Serialize};

/// Strategy for selecting when replication occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationTrigger {
    /// Replicate when fitness exceeds a threshold
    FitnessThreshold { threshold: f64 },
    /// Replicate after a fixed number of epochs
    EpochInterval { interval: u64 },
    /// Replicate when both fitness and epoch criteria are met
    FitnessAndEpoch { fitness_threshold: f64, min_epochs: u64 },
}

/// Policy governing seed replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationPolicy {
    /// When replication is triggered
    pub trigger: ReplicationTrigger,
    /// Maximum number of children a single seed can produce
    pub max_offspring: u32,
    /// Number of offspring already produced
    pub offspring_count: u32,
    /// Fitness decay applied to children (child_fitness = parent_fitness * decay)
    pub child_fitness_decay: f64,
    /// Whether the parent survives replication
    pub parent_survives: bool,
    /// Minimum fitness the parent must maintain to replicate
    pub min_parent_fitness: f64,
}

impl Default for ReplicationPolicy {
    fn default() -> Self {
        Self {
            trigger: ReplicationTrigger::FitnessAndEpoch {
                fitness_threshold: 0.7,
                min_epochs: 5,
            },
            max_offspring: 10,
            offspring_count: 0,
            child_fitness_decay: 0.9,
            parent_survives: true,
            min_parent_fitness: 0.3,
        }
    }
}

impl ReplicationPolicy {
    /// Create a policy that replicates aggressively
    pub fn prolific() -> Self {
        Self {
            trigger: ReplicationTrigger::FitnessThreshold { threshold: 0.5 },
            max_offspring: 100,
            offspring_count: 0,
            child_fitness_decay: 0.95,
            parent_survives: true,
            min_parent_fitness: 0.1,
        }
    }

    /// Create a policy that replicates rarely but produces strong offspring
    pub fn selective() -> Self {
        Self {
            trigger: ReplicationTrigger::FitnessAndEpoch {
                fitness_threshold: 0.9,
                min_epochs: 20,
            },
            max_offspring: 3,
            offspring_count: 0,
            child_fitness_decay: 0.98,
            parent_survives: true,
            min_parent_fitness: 0.8,
        }
    }

    /// Determine whether the seed should replicate given current state
    pub fn should_replicate(&self, fitness: f64, epoch: u64) -> bool {
        if self.offspring_count >= self.max_offspring {
            return false;
        }
        if fitness < self.min_parent_fitness {
            return false;
        }

        match &self.trigger {
            ReplicationTrigger::FitnessThreshold { threshold } => {
                fitness >= *threshold
            }
            ReplicationTrigger::EpochInterval { interval } => {
                epoch > 0 && epoch % interval == 0
            }
            ReplicationTrigger::FitnessAndEpoch { fitness_threshold, min_epochs } => {
                fitness >= *fitness_threshold && epoch >= *min_epochs
            }
        }
    }

    /// Compute the fitness for a child seed
    pub fn child_fitness(&self, parent_fitness: f64) -> f64 {
        parent_fitness * self.child_fitness_decay
    }

    /// Record that a replication occurred
    pub fn record_replication(&mut self) {
        self.offspring_count += 1;
    }

    /// Check if this policy has capacity for more offspring
    pub fn can_replicate_more(&self) -> bool {
        self.offspring_count < self.max_offspring
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = ReplicationPolicy::default();
        // Needs fitness >= 0.7 AND epoch >= 5
        assert!(!policy.should_replicate(0.5, 10)); // fitness too low
        assert!(!policy.should_replicate(0.8, 3));  // epoch too low
        assert!(policy.should_replicate(0.8, 10));  // both criteria met
    }

    #[test]
    fn test_max_offspring() {
        let mut policy = ReplicationPolicy::default();
        policy.max_offspring = 2;
        policy.offspring_count = 2;
        assert!(!policy.should_replicate(0.9, 100));
    }

    #[test]
    fn test_fitness_threshold_trigger() {
        let policy = ReplicationPolicy::prolific();
        assert!(policy.should_replicate(0.6, 0));
        assert!(!policy.should_replicate(0.05, 0)); // below min_parent_fitness
    }

    #[test]
    fn test_epoch_interval_trigger() {
        let policy = ReplicationPolicy {
            trigger: ReplicationTrigger::EpochInterval { interval: 10 },
            ..ReplicationPolicy::default()
        };
        assert!(!policy.should_replicate(0.5, 5));
        assert!(policy.should_replicate(0.5, 10));
        assert!(policy.should_replicate(0.5, 20));
    }

    #[test]
    fn test_child_fitness_decay() {
        let policy = ReplicationPolicy::default();
        let child_fit = policy.child_fitness(0.8);
        assert!((child_fit - 0.72).abs() < 1e-10);
    }
}
