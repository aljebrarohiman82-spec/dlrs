//! LoRA Manager — orchestrates multiple LoRA adapters
//!
//! Provides lifecycle management: create, evolve, merge, prune, persist,
//! and auto-share adapters via ZK proofs.

use super::adapter::{LoraAdapter, LoraConfig};
use crate::storage::SeedStore;
use crate::zk::CapabilityProof;
use log::info;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Manager for a collection of LoRA adapters
#[derive(Serialize, Deserialize)]
pub struct LoraManager {
    /// All managed adapters, keyed by adapter ID
    adapters: HashMap<String, LoraAdapter>,
    /// Adapter store path for persistence
    store_path: Option<String>,
    /// Auto-evolution settings
    pub evolution_config: EvolutionConfig,
}

/// Configuration for automatic evolution cycles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Learning rate for evolution
    pub learning_rate: f64,
    /// Number of evolution steps per cycle
    pub steps_per_cycle: u32,
    /// Minimum fitness to keep an adapter alive
    pub prune_threshold: f64,
    /// Whether to auto-replicate high-fitness adapters
    pub auto_replicate: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            steps_per_cycle: 10,
            prune_threshold: 0.1,
            auto_replicate: true,
        }
    }
}

/// Statistics about the manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerStats {
    pub total_adapters: usize,
    pub avg_fitness: f64,
    pub best_fitness: f64,
    pub total_domains: usize,
    pub total_parameters: usize,
}

impl LoraManager {
    /// Create a new empty manager
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            store_path: None,
            evolution_config: EvolutionConfig::default(),
        }
    }

    /// Create with a persistence path
    pub fn with_store(path: impl Into<String>) -> Self {
        Self {
            adapters: HashMap::new(),
            store_path: Some(path.into()),
            evolution_config: EvolutionConfig::default(),
        }
    }

    /// Create a new adapter and register it
    pub fn create_adapter(&mut self, config: LoraConfig) -> String {
        let adapter = LoraAdapter::new(config);
        let id = adapter.id().to_string();
        info!("Created adapter: {}", adapter.summary());
        self.adapters.insert(id.clone(), adapter);
        id
    }

    /// Create from an existing weight delta
    pub fn create_from_delta(&mut self, config: LoraConfig, delta: &DMatrix<f64>) -> String {
        let adapter = LoraAdapter::from_weight_delta(config, delta);
        let id = adapter.id().to_string();
        info!("Created adapter from delta: {}", adapter.summary());
        self.adapters.insert(id.clone(), adapter);
        id
    }

    /// Get an adapter by ID
    pub fn get(&self, id: &str) -> Option<&LoraAdapter> {
        self.adapters.get(id)
    }

    /// Get a mutable adapter by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut LoraAdapter> {
        self.adapters.get_mut(id)
    }

    /// Remove an adapter
    pub fn remove(&mut self, id: &str) -> Option<LoraAdapter> {
        self.adapters.remove(id)
    }

    /// List all adapter IDs and names
    pub fn list(&self) -> Vec<(String, String)> {
        self.adapters
            .iter()
            .map(|(id, a)| (id.clone(), a.config.name.clone()))
            .collect()
    }

    /// Get all adapters for a specific target module
    pub fn adapters_for_module(&self, module: &str) -> Vec<&LoraAdapter> {
        self.adapters
            .values()
            .filter(|a| a.config.target_module == module)
            .collect()
    }

    /// Find adapters by domain
    pub fn find_by_domain(&self, domain: &str) -> Vec<&LoraAdapter> {
        self.adapters
            .values()
            .filter(|a| a.config.domains.iter().any(|d| d == domain))
            .collect()
    }

    /// Evolve a specific adapter with target data
    pub fn evolve_adapter(&mut self, id: &str, target: &DMatrix<f64>) -> Option<f64> {
        let lr = self.evolution_config.learning_rate;
        if let Some(adapter) = self.adapters.get_mut(id) {
            let old_fitness = adapter.seed.fitness;
            for _ in 0..self.evolution_config.steps_per_cycle {
                adapter.evolve(target, lr);
            }
            let new_fitness = adapter.seed.fitness;
            info!(
                "Evolved '{}': fitness {:.3} -> {:.3} ({} steps)",
                adapter.config.name,
                old_fitness,
                new_fitness,
                self.evolution_config.steps_per_cycle,
            );
            Some(new_fitness)
        } else {
            None
        }
    }

    /// Run one evolution cycle on all adapters with per-adapter targets
    pub fn evolution_cycle(&mut self, targets: &HashMap<String, DMatrix<f64>>) {
        let lr = self.evolution_config.learning_rate;
        let steps = self.evolution_config.steps_per_cycle;
        let ids: Vec<String> = self.adapters.keys().cloned().collect();

        for id in &ids {
            if let Some(target) = targets.get(id) {
                if let Some(adapter) = self.adapters.get_mut(id) {
                    for _ in 0..steps {
                        adapter.evolve(target, lr);
                    }
                }
            }
        }
    }

    /// Merge two adapters, keeping both originals
    pub fn merge_adapters(&mut self, id_a: &str, id_b: &str) -> Option<String> {
        let a = self.adapters.get(id_a)?.clone();
        let b = self.adapters.get(id_b)?.clone();
        let merged = a.merge_with(&b);
        let id = merged.id().to_string();
        info!("Merged adapters: {}", merged.summary());
        self.adapters.insert(id.clone(), merged);
        Some(id)
    }

    /// Apply an adapter to base weights
    pub fn apply_adapter(
        &self,
        adapter_id: &str,
        base_weights: &DMatrix<f64>,
    ) -> Option<DMatrix<f64>> {
        self.adapters
            .get(adapter_id)
            .map(|a| a.apply_to_weights(base_weights))
    }

    /// Apply multiple adapters to base weights (stacked LoRA)
    pub fn apply_stack(
        &self,
        adapter_ids: &[&str],
        base_weights: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let mut result = base_weights.clone();
        for id in adapter_ids {
            if let Some(adapter) = self.adapters.get(*id) {
                let delta = adapter.weight_delta() * adapter.scaling_factor();
                result += delta;
            }
        }
        result
    }

    /// Generate a ZK capability proof for an adapter
    pub fn generate_proof(
        &self,
        adapter_id: &str,
        domain: &str,
        domain_vector: &DVector<f64>,
    ) -> Option<CapabilityProof> {
        let adapter = self.adapters.get(adapter_id)?;
        let proof = CapabilityProof::generate(
            &adapter.seed.lrim,
            &adapter.seed.commitment.root_commitment,
            domain,
            domain_vector,
            24,
        );
        Some(proof)
    }

    /// Prune adapters below fitness threshold
    pub fn prune(&mut self) -> Vec<String> {
        let threshold = self.evolution_config.prune_threshold;
        let to_remove: Vec<String> = self
            .adapters
            .iter()
            .filter(|(_, a)| a.seed.fitness < threshold)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &to_remove {
            if let Some(a) = self.adapters.remove(id) {
                info!("Pruned adapter '{}' (fitness={:.3})", a.config.name, a.seed.fitness);
            }
        }
        to_remove
    }

    /// Auto-replicate high-fitness adapters
    pub fn auto_replicate(&mut self) -> Vec<String> {
        let mut new_ids = Vec::new();
        let to_replicate: Vec<(String, LoraAdapter)> = self
            .adapters
            .iter()
            .filter(|(_, a)| a.seed.replication.should_replicate(a.seed.fitness, a.seed.epoch))
            .map(|(id, a)| (id.clone(), a.clone()))
            .collect();

        for (_parent_id, adapter) in to_replicate {
            if let Some(child_seed) = adapter.seed.replicate() {
                let child = LoraAdapter {
                    seed: child_seed,
                    config: LoraConfig {
                        name: format!("{}_child", adapter.config.name),
                        ..adapter.config.clone()
                    },
                };
                let id = child.id().to_string();
                info!("Replicated adapter: {}", child.summary());
                self.adapters.insert(id.clone(), child);
                new_ids.push(id);
            }
        }
        new_ids
    }

    /// Get manager statistics
    pub fn stats(&self) -> ManagerStats {
        let total = self.adapters.len();
        let avg_fitness = if total > 0 {
            self.adapters.values().map(|a| a.seed.fitness).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let best_fitness = self
            .adapters
            .values()
            .map(|a| a.seed.fitness)
            .fold(0.0f64, f64::max);
        let mut domains = std::collections::HashSet::new();
        let mut total_params = 0usize;
        for a in self.adapters.values() {
            for d in &a.config.domains {
                domains.insert(d.clone());
            }
            let (m, n) = a.config.original_dims;
            total_params += (m + n) * a.config.rank + a.config.rank;
        }
        ManagerStats {
            total_adapters: total,
            avg_fitness,
            best_fitness,
            total_domains: domains.len(),
            total_parameters: total_params,
        }
    }

    /// Export to SeedStore for network sharing
    pub fn to_seed_store(&self) -> SeedStore {
        let mut store = SeedStore::new();
        for adapter in self.adapters.values() {
            store.put(adapter.seed.clone());
        }
        store
    }

    /// Save manager state to file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(path) = &self.store_path {
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
            info!("Saved manager state to {}", path);
        }
        Ok(())
    }

    /// Load manager state from file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let mut manager: Self = serde_json::from_str(&json)?;
        manager.store_path = Some(path.to_string());
        info!("Loaded manager with {} adapters from {}", manager.adapters.len(), path);
        Ok(manager)
    }

    /// Number of adapters
    pub fn count(&self) -> usize {
        self.adapters.len()
    }
}

impl Default for LoraManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(name: &str) -> LoraConfig {
        LoraConfig {
            name: name.to_string(),
            target_module: "attn.q".to_string(),
            rank: 4,
            alpha: 8.0,
            domains: vec!["nlp".to_string()],
            original_dims: (32, 32),
        }
    }

    #[test]
    fn test_create_and_list() {
        let mut mgr = LoraManager::new();
        let id = mgr.create_adapter(test_config("adapter-1"));
        assert_eq!(mgr.count(), 1);
        let list = mgr.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].0, id);
    }

    #[test]
    fn test_evolve_and_stats() {
        let mut mgr = LoraManager::new();
        let id = mgr.create_adapter(test_config("evolvable"));
        let target = DMatrix::new_random(32, 32) * 0.01;
        mgr.evolve_adapter(&id, &target);
        let stats = mgr.stats();
        assert_eq!(stats.total_adapters, 1);
        println!("Stats: {:?}", stats);
    }

    #[test]
    fn test_merge() {
        let mut mgr = LoraManager::new();
        let a = mgr.create_adapter(test_config("lora-a"));
        let b = mgr.create_adapter(LoraConfig {
            name: "lora-b".into(),
            domains: vec!["vision".into()],
            ..test_config("lora-b")
        });
        let merged_id = mgr.merge_adapters(&a, &b).unwrap();
        assert_eq!(mgr.count(), 3); // a, b, merged
        let merged = mgr.get(&merged_id).unwrap();
        assert!(merged.config.domains.contains(&"nlp".to_string()));
        assert!(merged.config.domains.contains(&"vision".to_string()));
    }

    #[test]
    fn test_apply_stack() {
        let mut mgr = LoraManager::new();
        let a = mgr.create_adapter(test_config("stack-a"));
        let b = mgr.create_adapter(test_config("stack-b"));
        let base = DMatrix::new_random(32, 32);
        let stacked = mgr.apply_stack(&[&a, &b], &base);
        assert_ne!(stacked, base);
    }

    #[test]
    fn test_prune() {
        let mut mgr = LoraManager::new();
        let id = mgr.create_adapter(test_config("weak"));
        mgr.get_mut(&id).unwrap().seed.fitness = 0.01;
        let pruned = mgr.prune();
        assert_eq!(pruned.len(), 1);
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_find_by_domain() {
        let mut mgr = LoraManager::new();
        mgr.create_adapter(test_config("nlp-adapter"));
        mgr.create_adapter(LoraConfig {
            name: "vision-adapter".into(),
            domains: vec!["vision".into()],
            ..test_config("x")
        });
        assert_eq!(mgr.find_by_domain("nlp").len(), 1);
        assert_eq!(mgr.find_by_domain("vision").len(), 1);
    }

    #[test]
    fn test_to_seed_store() {
        let mut mgr = LoraManager::new();
        mgr.create_adapter(test_config("s1"));
        mgr.create_adapter(test_config("s2"));
        let store = mgr.to_seed_store();
        assert_eq!(store.count(), 2);
    }
}
