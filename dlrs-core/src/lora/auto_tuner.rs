//! AI-Assisted Auto-Tuner — intelligent hyperparameter optimization
//!
//! Automatically selects:
//! - Optimal rank via energy analysis of singular values
//! - Learning rate via loss landscape probing
//! - When to stop, merge, or replicate
//! - Best adapter for a given task

use crate::lora::adapter::{LoraAdapter, LoraConfig};
use crate::lora::training::{train, Dataset, LossFunction, LrSchedule, TrainConfig, TrainHistory};
use log::info;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Result of rank analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankAnalysis {
    /// Singular values of the target matrix
    pub singular_values: Vec<f64>,
    /// Cumulative energy ratios
    pub energy_ratios: Vec<f64>,
    /// Recommended rank (captures 95% energy)
    pub recommended_rank: usize,
    /// Compression ratio at recommended rank
    pub compression_ratio: f64,
}

/// Result of learning rate search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrSearchResult {
    /// Tested learning rates and their final losses
    pub results: Vec<(f64, f64)>,
    /// Best learning rate found
    pub best_lr: f64,
    /// Loss at best lr
    pub best_loss: f64,
}

/// Auto-tuning recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneRecommendation {
    pub rank: usize,
    pub learning_rate: f64,
    pub epochs: u32,
    pub loss_fn: LossFunction,
    pub lr_schedule: LrSchedule,
    pub alpha: f64,
    pub explanation: String,
}

/// Analyze a target matrix to find optimal rank
pub fn analyze_rank(target: &DMatrix<f64>, energy_threshold: f64) -> RankAnalysis {
    let svd = target.clone().svd(false, false);
    let sv = svd.singular_values;
    let total_energy: f64 = sv.iter().map(|s| s * s).sum();

    let mut cumulative = 0.0;
    let mut energy_ratios = Vec::new();
    let mut recommended_rank = sv.len();

    for (i, s) in sv.iter().enumerate() {
        cumulative += s * s;
        let ratio = cumulative / total_energy;
        energy_ratios.push(ratio);
        if ratio >= energy_threshold && recommended_rank == sv.len() {
            recommended_rank = i + 1;
        }
    }

    let m = target.nrows();
    let n = target.ncols();
    let original = (m * n) as f64;
    let compressed = ((m + n) * recommended_rank + recommended_rank) as f64;

    info!(
        "Rank analysis: {} singular values, recommended rank={} ({:.1}% energy), compression={:.1}x",
        sv.len(),
        recommended_rank,
        energy_threshold * 100.0,
        original / compressed,
    );

    RankAnalysis {
        singular_values: sv.as_slice().to_vec(),
        energy_ratios,
        recommended_rank,
        compression_ratio: original / compressed,
    }
}

/// Search for the best learning rate by training short probes
pub fn lr_search(
    config: &LoraConfig,
    dataset: &Dataset,
    lr_candidates: &[f64],
    probe_epochs: u32,
) -> LrSearchResult {
    let mut results = Vec::new();
    let mut best_lr = lr_candidates[0];
    let mut best_loss = f64::MAX;

    for &lr in lr_candidates {
        let mut adapter = LoraAdapter::new(config.clone());
        let train_config = TrainConfig {
            epochs: probe_epochs,
            learning_rate: lr,
            patience: 0,
            log_interval: probe_epochs + 1, // quiet
            loss_fn: LossFunction::MSE,
            lr_schedule: LrSchedule::Constant,
            ..TrainConfig::default()
        };
        let history = train(&mut adapter, dataset, &train_config);
        results.push((lr, history.best_loss));

        if history.best_loss < best_loss {
            best_loss = history.best_loss;
            best_lr = lr;
        }
    }

    info!("LR search: best lr={:.5} with loss={:.6}", best_lr, best_loss);

    LrSearchResult {
        results,
        best_lr,
        best_loss,
    }
}

/// Generate a full tuning recommendation for a dataset
pub fn auto_tune(
    dataset: &Dataset,
    target_module: &str,
    domains: Vec<String>,
) -> TuneRecommendation {
    let target_matrix = dataset.build_target_matrix();

    // Step 1: Analyze rank
    let rank_analysis = analyze_rank(&target_matrix, 0.95);
    let rank = rank_analysis.recommended_rank.max(1);

    // Step 2: Choose alpha (common heuristic: alpha = 2 * rank)
    let alpha = (rank as f64) * 2.0;

    // Step 3: Create probe config and search LR
    let probe_config = LoraConfig {
        name: "probe".into(),
        target_module: target_module.into(),
        rank,
        alpha,
        domains: domains.clone(),
        original_dims: (dataset.output_dim, dataset.input_dim),
    };

    let lr_candidates = vec![0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001];
    let lr_result = lr_search(&probe_config, dataset, &lr_candidates, 20);

    // Step 4: Determine epochs based on dataset size
    let epochs = if dataset.len() < 50 {
        200
    } else if dataset.len() < 500 {
        100
    } else {
        50
    };

    // Step 5: Choose loss function
    let loss_fn = if dataset.output_dim > 100 {
        LossFunction::Cosine // high-dim benefits from cosine
    } else {
        LossFunction::MSE
    };

    // Step 6: LR schedule
    let lr_schedule = LrSchedule::CosineAnnealing {
        min_lr: lr_result.best_lr * 0.01,
    };

    let explanation = format!(
        "Rank {} captures {:.1}% of target energy ({:.1}x compression). \
         LR {:.5} selected from {} candidates (probe loss {:.6}). \
         {} loss with cosine annealing for {} epochs. Alpha={:.0}.",
        rank,
        rank_analysis.energy_ratios.get(rank.saturating_sub(1)).copied().unwrap_or(1.0) * 100.0,
        rank_analysis.compression_ratio,
        lr_result.best_lr,
        lr_candidates.len(),
        lr_result.best_loss,
        loss_fn.name(),
        epochs,
        alpha,
    );

    info!("Auto-tune recommendation: {}", explanation);

    TuneRecommendation {
        rank,
        learning_rate: lr_result.best_lr,
        epochs,
        loss_fn,
        lr_schedule,
        alpha,
        explanation,
    }
}

/// Run full auto-tuned training: analyze -> recommend -> train
pub fn auto_train(
    name: &str,
    target_module: &str,
    dataset: &Dataset,
    domains: Vec<String>,
) -> (LoraAdapter, TrainHistory, TuneRecommendation) {
    let rec = auto_tune(dataset, target_module, domains.clone());

    let config = LoraConfig {
        name: name.into(),
        target_module: target_module.into(),
        rank: rec.rank,
        alpha: rec.alpha,
        domains,
        original_dims: (dataset.output_dim, dataset.input_dim),
    };

    let mut adapter = LoraAdapter::new(config);

    let train_config = TrainConfig {
        epochs: rec.epochs,
        learning_rate: rec.learning_rate,
        loss_fn: rec.loss_fn.clone(),
        lr_schedule: rec.lr_schedule.clone(),
        patience: 30,
        min_delta: 1e-7,
        log_interval: (rec.epochs / 5).max(1),
    };

    let history = train(&mut adapter, dataset, &train_config);

    (adapter, history, rec)
}

/// Select the best adapter from a set for a given task (domain + test data)
pub fn select_best(
    adapters: &[&LoraAdapter],
    test_data: &Dataset,
    domain: &str,
) -> Option<(usize, f64)> {
    if adapters.is_empty() {
        return None;
    }

    let target = test_data.build_target_matrix();
    let mut best_idx = 0;
    let mut best_score = f64::MAX;

    for (i, adapter) in adapters.iter().enumerate() {
        // Check domain match
        let domain_match = adapter.config.domains.iter().any(|d| d == domain);
        let domain_penalty = if domain_match { 1.0 } else { 2.0 };

        let predicted = adapter.weight_delta() * adapter.scaling_factor();
        let loss = LossFunction::MSE.compute(&predicted, &target);
        let score = loss * domain_penalty / (adapter.seed.fitness + 0.01);

        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }

    Some((best_idx, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_analysis() {
        let m = DMatrix::new_random(32, 32);
        let analysis = analyze_rank(&m, 0.95);
        assert!(analysis.recommended_rank > 0);
        assert!(analysis.recommended_rank <= 32);
        assert!(analysis.compression_ratio > 0.0);
    }

    #[test]
    fn test_auto_tune() {
        let ds = Dataset::synthetic("test", 16, 16, 30);
        let rec = auto_tune(&ds, "attn.q", vec!["test".into()]);
        assert!(rec.rank > 0);
        assert!(rec.learning_rate > 0.0);
        assert!(rec.epochs > 0);
    }

    #[test]
    fn test_auto_train() {
        let ds = Dataset::synthetic("auto-test", 16, 16, 20);
        let (adapter, history, rec) = auto_train("auto-lora", "attn.q", &ds, vec!["test".into()]);
        assert!(adapter.seed.epoch > 0);
        assert!(history.best_loss < f64::MAX);
        assert!(!rec.explanation.is_empty());
    }

    #[test]
    fn test_select_best() {
        let a = LoraAdapter::new(LoraConfig {
            name: "a".into(),
            target_module: "t".into(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["nlp".into()],
            original_dims: (16, 16),
        });
        let b = LoraAdapter::new(LoraConfig {
            name: "b".into(),
            target_module: "t".into(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["vision".into()],
            original_dims: (16, 16),
        });
        let test_ds = Dataset::synthetic("test", 16, 16, 10);
        let result = select_best(&[&a, &b], &test_ds, "nlp");
        assert!(result.is_some());
    }
}
