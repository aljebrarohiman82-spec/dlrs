//! LoRA Adapter — a single low-rank adapter backed by DLRS
//!
//! Each adapter wraps a DnaSeed and provides LoRA-specific operations:
//! apply to weights, merge adapters, export to standard formats.

use crate::seed::DnaSeed;
use crate::zk::ZkCommitment;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Configuration for a LoRA adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Human-readable name
    pub name: String,
    /// Target module path (e.g. "attention.query", "ffn.up")
    pub target_module: String,
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f64,
    /// Domains this adapter is trained for
    pub domains: Vec<String>,
    /// Original weight dimensions (rows, cols)
    pub original_dims: (usize, usize),
}

/// Export format for LoRA adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdapterFormat {
    /// Raw JSON (U, sigma, V factors)
    DlrsJson,
    /// Flat weight delta matrix W = alpha * U * diag(sigma) * V^T
    WeightDelta,
}

/// A LoRA adapter — wraps a DnaSeed with LoRA-specific semantics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    /// The underlying DNA seed
    pub seed: DnaSeed,
    /// LoRA-specific configuration
    pub config: LoraConfig,
}

impl LoraAdapter {
    /// Create a new random LoRA adapter
    pub fn new(config: LoraConfig) -> Self {
        let (m, n) = config.original_dims;
        // Initialize with small random perturbation (like LoRA init)
        let init_matrix = DMatrix::new_random(m, n) * 0.01;
        let seed = DnaSeed::new(
            &config.name,
            &init_matrix,
            config.rank,
            config.domains.clone(),
        );
        Self { seed, config }
    }

    /// Create from an existing weight delta matrix
    pub fn from_weight_delta(config: LoraConfig, delta: &DMatrix<f64>) -> Self {
        let seed = DnaSeed::new(
            &config.name,
            delta,
            config.rank,
            config.domains.clone(),
        );
        Self { seed, config }
    }

    /// Apply this adapter to a base weight matrix: W' = W + alpha * ΔW
    pub fn apply_to_weights(&self, base_weights: &DMatrix<f64>) -> DMatrix<f64> {
        let delta = self.weight_delta();
        base_weights + delta * self.scaling_factor()
    }

    /// Get the weight delta (reconstructed from low-rank factors)
    pub fn weight_delta(&self) -> DMatrix<f64> {
        self.seed.lrim.reconstruct()
    }

    /// Scaling factor: alpha / rank
    pub fn scaling_factor(&self) -> f64 {
        self.config.alpha / self.config.rank as f64
    }

    /// Compute the effective contribution magnitude
    pub fn contribution_norm(&self) -> f64 {
        let delta = self.weight_delta();
        delta.norm() * self.scaling_factor()
    }

    /// Express the adapter on an input vector (forward pass through ΔW)
    pub fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        self.seed.express_on(input) * self.scaling_factor()
    }

    /// Evolve the adapter with a target weight delta (fine-tuning)
    pub fn evolve(&mut self, target_delta: &DMatrix<f64>, learning_rate: f64) {
        self.seed.evolve(target_delta, learning_rate);
    }

    /// Merge two adapters into one (adapter composition)
    pub fn merge_with(&self, other: &LoraAdapter) -> LoraAdapter {
        let merged_seed = self.seed.merge_with(&other.seed);
        let mut domains = self.config.domains.clone();
        for d in &other.config.domains {
            if !domains.contains(d) {
                domains.push(d.clone());
            }
        }
        let config = LoraConfig {
            name: format!("{}+{}", self.config.name, other.config.name),
            target_module: self.config.target_module.clone(),
            rank: merged_seed.lrim.rank,
            alpha: (self.config.alpha + other.config.alpha) / 2.0,
            domains,
            original_dims: self.config.original_dims,
        };
        LoraAdapter {
            seed: merged_seed,
            config,
        }
    }

    /// Export to the specified format
    pub fn export(&self, format: &AdapterFormat) -> Result<String, serde_json::Error> {
        match format {
            AdapterFormat::DlrsJson => serde_json::to_string_pretty(self),
            AdapterFormat::WeightDelta => {
                let delta = self.weight_delta() * self.scaling_factor();
                let export = WeightDeltaExport {
                    name: self.config.name.clone(),
                    target_module: self.config.target_module.clone(),
                    rows: self.config.original_dims.0,
                    cols: self.config.original_dims.1,
                    data: delta.as_slice().to_vec(),
                };
                serde_json::to_string_pretty(&export)
            }
        }
    }

    /// Get a summary of this adapter
    pub fn summary(&self) -> String {
        format!(
            "LoRA '{}' | module={} | rank={} | alpha={:.1} | dims={}x{} | fitness={:.3} | compression={:.1}x | domains={:?}",
            self.config.name,
            self.config.target_module,
            self.config.rank,
            self.config.alpha,
            self.config.original_dims.0,
            self.config.original_dims.1,
            self.seed.fitness,
            self.seed.lrim.compression_ratio(),
            self.config.domains,
        )
    }

    /// Get the ZK commitment for this adapter
    pub fn commitment(&self) -> &ZkCommitment {
        &self.seed.commitment
    }

    /// Get the adapter ID
    pub fn id(&self) -> &str {
        &self.seed.id
    }
}

#[derive(Serialize, Deserialize)]
struct WeightDeltaExport {
    name: String,
    target_module: String,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LoraConfig {
        LoraConfig {
            name: "test-lora".to_string(),
            target_module: "attention.query".to_string(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["nlp".to_string()],
            original_dims: (64, 64),
        }
    }

    #[test]
    fn test_create_adapter() {
        let adapter = LoraAdapter::new(test_config());
        assert_eq!(adapter.config.rank, 4);
        assert_eq!(adapter.seed.lrim.rank, 4);
        println!("{}", adapter.summary());
    }

    #[test]
    fn test_apply_to_weights() {
        let adapter = LoraAdapter::new(test_config());
        let base = DMatrix::new_random(64, 64);
        let modified = adapter.apply_to_weights(&base);
        // Modified should differ from base
        assert!((modified - &base).norm() > 0.0);
    }

    #[test]
    fn test_forward_pass() {
        let adapter = LoraAdapter::new(test_config());
        let input = DVector::new_random(64);
        let output = adapter.forward(&input);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_evolve_adapter() {
        let mut adapter = LoraAdapter::new(test_config());
        let target = DMatrix::new_random(64, 64) * 0.01;
        let initial_fitness = adapter.seed.fitness;
        for _ in 0..5 {
            adapter.evolve(&target, 0.01);
        }
        assert_eq!(adapter.seed.epoch, 5);
        println!("Fitness: {:.3} -> {:.3}", initial_fitness, adapter.seed.fitness);
    }

    #[test]
    fn test_merge_adapters() {
        let a = LoraAdapter::new(LoraConfig {
            name: "adapter-a".into(),
            domains: vec!["nlp".into()],
            ..test_config()
        });
        let b = LoraAdapter::new(LoraConfig {
            name: "adapter-b".into(),
            domains: vec!["vision".into()],
            ..test_config()
        });
        let merged = a.merge_with(&b);
        assert!(merged.config.domains.contains(&"nlp".to_string()));
        assert!(merged.config.domains.contains(&"vision".to_string()));
        assert_eq!(merged.config.name, "adapter-a+adapter-b");
    }

    #[test]
    fn test_export_json() {
        let adapter = LoraAdapter::new(test_config());
        let json = adapter.export(&AdapterFormat::DlrsJson).unwrap();
        assert!(json.contains("test-lora"));
    }

    #[test]
    fn test_export_weight_delta() {
        let adapter = LoraAdapter::new(test_config());
        let json = adapter.export(&AdapterFormat::WeightDelta).unwrap();
        assert!(json.contains("attention.query"));
    }
}
