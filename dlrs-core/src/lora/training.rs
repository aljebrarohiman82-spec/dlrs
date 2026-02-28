//! Training Engine — local LoRA training with dataset, loss, and training loop
//!
//! Provides real training capabilities:
//! - DataSample / Dataset abstraction
//! - Multiple loss functions (MSE, Cosine, Huber)
//! - Full training loop with batching, scheduling, early stopping
//! - Training history and metrics

use crate::lora::adapter::LoraAdapter;
use log::info;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A single training sample: input -> target output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

/// A training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub samples: Vec<DataSample>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Dataset {
    pub fn new(name: impl Into<String>, input_dim: usize, output_dim: usize) -> Self {
        Self {
            name: name.into(),
            samples: Vec::new(),
            input_dim,
            output_dim,
        }
    }

    pub fn add_sample(&mut self, input: Vec<f64>, target: Vec<f64>) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(target.len(), self.output_dim);
        self.samples.push(DataSample { input, target });
    }

    /// Generate a synthetic dataset for testing (random input->target pairs)
    pub fn synthetic(name: &str, input_dim: usize, output_dim: usize, n_samples: usize) -> Self {
        let mut ds = Self::new(name, input_dim, output_dim);
        for _ in 0..n_samples {
            let input: Vec<f64> = (0..input_dim).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
            let target: Vec<f64> = (0..output_dim).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
            ds.add_sample(input, target);
        }
        ds
    }

    /// Build the target matrix from all samples (output_dim x input_dim)
    /// This represents the ideal W such that W * input ≈ target for all samples
    pub fn build_target_matrix(&self) -> DMatrix<f64> {
        if self.samples.is_empty() {
            return DMatrix::zeros(self.output_dim, self.input_dim);
        }
        // Least squares: T * X^T * (X * X^T)^-1
        // For simplicity, accumulate outer products
        let n = self.samples.len() as f64;
        let mut target_matrix = DMatrix::zeros(self.output_dim, self.input_dim);
        for sample in &self.samples {
            let input = DVector::from_vec(sample.input.clone());
            let target = DVector::from_vec(sample.target.clone());
            // Rank-1 update: target * input^T
            for i in 0..self.output_dim {
                for j in 0..self.input_dim {
                    target_matrix[(i, j)] += target[i] * input[j] / n;
                }
            }
        }
        target_matrix
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Save dataset to JSON file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load dataset from JSON file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// Loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Huber loss (smooth L1)
    Huber { delta: f64 },
    /// Cosine similarity loss
    Cosine,
}

impl LossFunction {
    pub fn compute(&self, predicted: &DMatrix<f64>, target: &DMatrix<f64>) -> f64 {
        let diff = target - predicted;
        match self {
            LossFunction::MSE => {
                diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64
            }
            LossFunction::MAE => {
                diff.iter().map(|x| x.abs()).sum::<f64>() / diff.len() as f64
            }
            LossFunction::Huber { delta } => {
                diff.iter().map(|x| {
                    if x.abs() <= *delta {
                        0.5 * x * x
                    } else {
                        delta * (x.abs() - 0.5 * delta)
                    }
                }).sum::<f64>() / diff.len() as f64
            }
            LossFunction::Cosine => {
                let p_norm = predicted.norm();
                let t_norm = target.norm();
                if p_norm < 1e-10 || t_norm < 1e-10 {
                    return 1.0;
                }
                let dot: f64 = predicted.iter().zip(target.iter()).map(|(a, b)| a * b).sum();
                1.0 - (dot / (p_norm * t_norm))
            }
        }
    }

    pub fn name(&self) -> &str {
        match self {
            LossFunction::MSE => "MSE",
            LossFunction::MAE => "MAE",
            LossFunction::Huber { .. } => "Huber",
            LossFunction::Cosine => "Cosine",
        }
    }
}

/// Learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LrSchedule {
    /// Fixed learning rate
    Constant,
    /// Linear decay from lr to min_lr
    LinearDecay { min_lr: f64 },
    /// Cosine annealing
    CosineAnnealing { min_lr: f64 },
    /// Step decay: multiply by factor every step_size epochs
    StepDecay { factor: f64, step_size: u32 },
}

impl LrSchedule {
    pub fn get_lr(&self, base_lr: f64, epoch: u32, total_epochs: u32) -> f64 {
        match self {
            LrSchedule::Constant => base_lr,
            LrSchedule::LinearDecay { min_lr } => {
                let progress = epoch as f64 / total_epochs as f64;
                base_lr + (min_lr - base_lr) * progress
            }
            LrSchedule::CosineAnnealing { min_lr } => {
                let progress = epoch as f64 / total_epochs as f64;
                min_lr + (base_lr - min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            LrSchedule::StepDecay { factor, step_size } => {
                base_lr * factor.powf((epoch / step_size) as f64)
            }
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: u32,
    pub learning_rate: f64,
    pub loss_fn: LossFunction,
    pub lr_schedule: LrSchedule,
    /// Stop early if loss doesn't improve for this many epochs
    pub patience: u32,
    /// Minimum improvement to reset patience counter
    pub min_delta: f64,
    /// Log every N epochs
    pub log_interval: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.01,
            loss_fn: LossFunction::MSE,
            lr_schedule: LrSchedule::CosineAnnealing { min_lr: 0.001 },
            patience: 20,
            min_delta: 1e-6,
            log_interval: 10,
        }
    }
}

/// Record of a single training epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: u32,
    pub loss: f64,
    pub learning_rate: f64,
    pub fitness: f64,
}

/// Complete training history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainHistory {
    pub records: Vec<EpochRecord>,
    pub best_loss: f64,
    pub best_epoch: u32,
    pub final_loss: f64,
    pub stopped_early: bool,
}

/// Train a LoRA adapter on a dataset
pub fn train(
    adapter: &mut LoraAdapter,
    dataset: &Dataset,
    config: &TrainConfig,
) -> TrainHistory {
    let target_matrix = dataset.build_target_matrix();
    let mut records = Vec::new();
    let mut best_loss = f64::MAX;
    let mut best_epoch = 0u32;
    let mut patience_counter = 0u32;

    info!(
        "Training '{}' on '{}' ({} samples) for {} epochs | loss={} | lr={:.4} | schedule={:?}",
        adapter.config.name,
        dataset.name,
        dataset.len(),
        config.epochs,
        config.loss_fn.name(),
        config.learning_rate,
        config.lr_schedule,
    );

    for epoch in 0..config.epochs {
        let lr = config.lr_schedule.get_lr(config.learning_rate, epoch, config.epochs);

        // Evolution step (gradient descent on LRIM factors)
        adapter.evolve(&target_matrix, lr);

        // Compute loss
        let predicted = adapter.weight_delta() * adapter.scaling_factor();
        let loss = config.loss_fn.compute(&predicted, &target_matrix);

        let record = EpochRecord {
            epoch,
            loss,
            learning_rate: lr,
            fitness: adapter.seed.fitness,
        };
        records.push(record);

        // Early stopping check
        if loss < best_loss - config.min_delta {
            best_loss = loss;
            best_epoch = epoch;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }

        if epoch % config.log_interval == 0 || epoch == config.epochs - 1 {
            info!(
                "  epoch={:>4} | loss={:.6} | lr={:.5} | fitness={:.4}",
                epoch, loss, lr, adapter.seed.fitness
            );
        }

        if patience_counter >= config.patience && config.patience > 0 {
            info!("  Early stopping at epoch {} (no improvement for {} epochs)", epoch, config.patience);
            return TrainHistory {
                final_loss: loss,
                best_loss,
                best_epoch,
                records,
                stopped_early: true,
            };
        }
    }

    let final_loss = records.last().map(|r| r.loss).unwrap_or(0.0);
    info!(
        "Training complete: best_loss={:.6} at epoch {} | final_loss={:.6} | fitness={:.4}",
        best_loss, best_epoch, final_loss, adapter.seed.fitness
    );

    TrainHistory {
        final_loss,
        best_loss,
        best_epoch,
        records,
        stopped_early: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraConfig;

    fn test_adapter() -> LoraAdapter {
        LoraAdapter::new(LoraConfig {
            name: "train-test".into(),
            target_module: "test".into(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["test".into()],
            original_dims: (16, 16),
        })
    }

    #[test]
    fn test_dataset_creation() {
        let ds = Dataset::synthetic("test-data", 16, 16, 50);
        assert_eq!(ds.len(), 50);
        assert_eq!(ds.input_dim, 16);
        let matrix = ds.build_target_matrix();
        assert_eq!(matrix.nrows(), 16);
        assert_eq!(matrix.ncols(), 16);
    }

    #[test]
    fn test_loss_functions() {
        let a = DMatrix::new_random(4, 4);
        let b = DMatrix::new_random(4, 4);
        let mse = LossFunction::MSE.compute(&a, &b);
        let mae = LossFunction::MAE.compute(&a, &b);
        let huber = LossFunction::Huber { delta: 1.0 }.compute(&a, &b);
        let cosine = LossFunction::Cosine.compute(&a, &b);
        assert!(mse > 0.0);
        assert!(mae > 0.0);
        assert!(huber > 0.0);
        assert!(cosine >= 0.0 && cosine <= 2.0);
        // Self-comparison should be 0
        let self_mse = LossFunction::MSE.compute(&a, &a);
        assert!(self_mse < 1e-10);
    }

    #[test]
    fn test_lr_schedules() {
        let base_lr = 0.1;
        // Cosine should start at base_lr and decrease
        let lr_0 = LrSchedule::CosineAnnealing { min_lr: 0.001 }.get_lr(base_lr, 0, 100);
        let lr_50 = LrSchedule::CosineAnnealing { min_lr: 0.001 }.get_lr(base_lr, 50, 100);
        let lr_100 = LrSchedule::CosineAnnealing { min_lr: 0.001 }.get_lr(base_lr, 100, 100);
        assert!(lr_0 > lr_50);
        assert!(lr_50 > lr_100);
    }

    #[test]
    fn test_training_loop() {
        let mut adapter = test_adapter();
        let ds = Dataset::synthetic("train", 16, 16, 20);
        let config = TrainConfig {
            epochs: 30,
            learning_rate: 0.01,
            patience: 0, // disable early stopping
            log_interval: 100, // quiet
            ..TrainConfig::default()
        };
        let history = train(&mut adapter, &ds, &config);
        assert_eq!(history.records.len(), 30);
        assert!(history.final_loss < history.records[0].loss || history.records[0].loss < 1e-10);
    }

    #[test]
    fn test_early_stopping() {
        let mut adapter = test_adapter();
        // Train with a zero target (easy to converge)
        let mut ds = Dataset::new("zeros", 16, 16);
        for _ in 0..10 {
            ds.add_sample(vec![0.0; 16], vec![0.0; 16]);
        }
        let config = TrainConfig {
            epochs: 1000,
            patience: 5,
            log_interval: 1000,
            ..TrainConfig::default()
        };
        let history = train(&mut adapter, &ds, &config);
        assert!(history.stopped_early);
        assert!(history.records.len() < 1000);
    }
}
