//! MutationRules — controls when and how seeds mutate
//!
//! Mutation is the engine of evolution. These rules govern whether a seed
//! is allowed to mutate based on fitness, epoch, and configurable thresholds.

use serde::{Deserialize, Serialize};

/// Strategy for how mutation rate changes over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationStrategy {
    /// Fixed mutation rate regardless of fitness
    Constant,
    /// Higher mutation when fitness is low (explore), lower when high (exploit)
    AdaptiveFitness,
    /// Mutation rate decays with each epoch
    EpochDecay { decay_rate: f64 },
    /// Simulated annealing: high exploration early, convergence later
    Annealing { initial_temp: f64, cooling_rate: f64 },
}

/// Rules governing seed mutation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRules {
    /// Minimum fitness required to allow mutation (avoid wasting compute on dead seeds)
    pub min_fitness: f64,
    /// Maximum fitness at which mutation is still allowed (don't break what works)
    pub max_fitness: f64,
    /// Base probability of mutation per epoch
    pub base_rate: f64,
    /// Maximum magnitude of perturbation applied to matrix factors
    pub max_perturbation: f64,
    /// Strategy for adapting mutation over time
    pub strategy: MutationStrategy,
    /// Number of epochs between allowed mutations (cooldown)
    pub cooldown_epochs: u64,
    /// Track the last epoch at which mutation occurred
    pub last_mutation_epoch: u64,
}

impl Default for MutationRules {
    fn default() -> Self {
        Self {
            min_fitness: 0.05,
            max_fitness: 0.98,
            base_rate: 0.8,
            max_perturbation: 0.1,
            strategy: MutationStrategy::AdaptiveFitness,
            cooldown_epochs: 0,
            last_mutation_epoch: 0,
        }
    }
}

impl MutationRules {
    /// Create strict rules (high bar for mutation)
    pub fn strict() -> Self {
        Self {
            min_fitness: 0.3,
            max_fitness: 0.95,
            base_rate: 0.3,
            max_perturbation: 0.01,
            strategy: MutationStrategy::EpochDecay { decay_rate: 0.99 },
            cooldown_epochs: 5,
            last_mutation_epoch: 0,
        }
    }

    /// Create aggressive rules (mutate freely)
    pub fn aggressive() -> Self {
        Self {
            min_fitness: 0.0,
            max_fitness: 1.0,
            base_rate: 1.0,
            max_perturbation: 0.5,
            strategy: MutationStrategy::Constant,
            cooldown_epochs: 0,
            last_mutation_epoch: 0,
        }
    }

    /// Determine whether mutation is allowed given the current fitness
    pub fn can_mutate(&self, fitness: f64) -> bool {
        fitness >= self.min_fitness && fitness <= self.max_fitness
    }

    /// Compute effective mutation rate given fitness and epoch
    pub fn effective_rate(&self, fitness: f64, epoch: u64) -> f64 {
        if !self.can_mutate(fitness) {
            return 0.0;
        }

        match &self.strategy {
            MutationStrategy::Constant => self.base_rate,
            MutationStrategy::AdaptiveFitness => {
                // Low fitness → high rate, high fitness → low rate
                let fitness_factor = 1.0 - fitness;
                self.base_rate * (0.2 + 0.8 * fitness_factor)
            }
            MutationStrategy::EpochDecay { decay_rate } => {
                self.base_rate * decay_rate.powf(epoch as f64)
            }
            MutationStrategy::Annealing { initial_temp, cooling_rate } => {
                let temp = initial_temp * cooling_rate.powf(epoch as f64);
                self.base_rate * (temp / initial_temp).min(1.0)
            }
        }
    }

    /// Compute perturbation magnitude for current state
    pub fn perturbation_magnitude(&self, fitness: f64, epoch: u64) -> f64 {
        self.max_perturbation * self.effective_rate(fitness, epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rules() {
        let rules = MutationRules::default();
        assert!(rules.can_mutate(0.5));
        assert!(!rules.can_mutate(0.01)); // below min
        assert!(!rules.can_mutate(0.99)); // above max
    }

    #[test]
    fn test_aggressive_rules() {
        let rules = MutationRules::aggressive();
        assert!(rules.can_mutate(0.0));
        assert!(rules.can_mutate(1.0));
    }

    #[test]
    fn test_adaptive_rate() {
        let rules = MutationRules::default();
        let rate_low = rules.effective_rate(0.1, 0);
        let rate_high = rules.effective_rate(0.9, 0);
        assert!(rate_low > rate_high, "Low fitness should have higher mutation rate");
    }

    #[test]
    fn test_epoch_decay() {
        let rules = MutationRules::strict();
        let rate_early = rules.effective_rate(0.5, 0);
        let rate_late = rules.effective_rate(0.5, 100);
        assert!(rate_early > rate_late, "Later epochs should have lower rate");
    }

    #[test]
    fn test_annealing() {
        let rules = MutationRules {
            strategy: MutationStrategy::Annealing {
                initial_temp: 1.0,
                cooling_rate: 0.95,
            },
            ..MutationRules::default()
        };
        let rate_0 = rules.effective_rate(0.5, 0);
        let rate_50 = rules.effective_rate(0.5, 50);
        assert!(rate_0 > rate_50);
    }
}
