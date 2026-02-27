//! Mutation rules — governs how a seed evolves

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRules {
    pub min_fitness_to_mutate: f64,
    pub max_learning_rate: f64,
    pub perturbation_prob: f64,
    pub max_rank_delta: usize,
    pub frozen: bool,
}

impl Default for MutationRules {
    fn default() -> Self {
        Self {
            min_fitness_to_mutate: 0.1,
            max_learning_rate: 0.05,
            perturbation_prob: 0.1,
            max_rank_delta: 2,
            frozen: false,
        }
    }
}

impl MutationRules {
    pub fn can_mutate(&self, current_fitness: f64) -> bool {
        !self.frozen && current_fitness >= self.min_fitness_to_mutate
    }

    pub fn conservative() -> Self {
        Self {
            min_fitness_to_mutate: 0.7,
            max_learning_rate: 0.01,
            perturbation_prob: 0.01,
            max_rank_delta: 1,
            frozen: false,
        }
    }

    pub fn aggressive() -> Self {
        Self {
            min_fitness_to_mutate: 0.0,
            max_learning_rate: 0.1,
            perturbation_prob: 0.3,
            max_rank_delta: 4,
            frozen: false,
        }
    }
}
