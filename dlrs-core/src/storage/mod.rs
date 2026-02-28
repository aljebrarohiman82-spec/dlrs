//! Storage module — seed persistence and retrieval
//!
//! Provides in-memory and file-based storage for DNA seeds,
//! plus versioned backup and rollback.

pub mod backup;

pub use backup::{BackupManager, BackupManifest, SnapshotMeta};

use crate::seed::DnaSeed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// In-memory seed store with JSON serialization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedStore {
    seeds: HashMap<String, DnaSeed>,
}

impl SeedStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self {
            seeds: HashMap::new(),
        }
    }

    /// Store a seed (upsert)
    pub fn put(&mut self, seed: DnaSeed) {
        self.seeds.insert(seed.id.clone(), seed);
    }

    /// Retrieve a seed by ID
    pub fn get(&self, id: &str) -> Option<&DnaSeed> {
        self.seeds.get(id)
    }

    /// Retrieve a mutable reference to a seed
    pub fn get_mut(&mut self, id: &str) -> Option<&mut DnaSeed> {
        self.seeds.get_mut(id)
    }

    /// Remove a seed by ID
    pub fn remove(&mut self, id: &str) -> Option<DnaSeed> {
        self.seeds.remove(id)
    }

    /// Check if a seed exists
    pub fn contains(&self, id: &str) -> bool {
        self.seeds.contains_key(id)
    }

    /// Get the number of seeds
    pub fn count(&self) -> usize {
        self.seeds.len()
    }

    /// List all seed IDs
    pub fn list_ids(&self) -> Vec<String> {
        self.seeds.keys().cloned().collect()
    }

    /// List all seeds
    pub fn list_all(&self) -> Vec<DnaSeed> {
        self.seeds.values().cloned().collect()
    }

    /// Find seeds by domain
    pub fn find_by_domain(&self, domain: &str) -> Vec<&DnaSeed> {
        self.seeds
            .values()
            .filter(|s| s.domains.iter().any(|d| d == domain))
            .collect()
    }

    /// Find seeds above a fitness threshold
    pub fn find_fit(&self, min_fitness: f64) -> Vec<&DnaSeed> {
        self.seeds
            .values()
            .filter(|s| s.fitness >= min_fitness)
            .collect()
    }

    /// Get the fittest seed overall
    pub fn fittest(&self) -> Option<&DnaSeed> {
        self.seeds
            .values()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Serialize the entire store to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a store from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from a file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let store = Self::from_json(&json)?;
        Ok(store)
    }

    /// Prune seeds below a fitness threshold
    pub fn prune(&mut self, min_fitness: f64) -> usize {
        let to_remove: Vec<String> = self
            .seeds
            .iter()
            .filter(|(_, s)| s.fitness < min_fitness)
            .map(|(id, _)| id.clone())
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            self.seeds.remove(&id);
        }
        count
    }
}

impl Default for SeedStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn make_seed(name: &str, domains: Vec<String>) -> DnaSeed {
        let k = DMatrix::new_random(10, 8);
        DnaSeed::new(name, &k, 3, domains)
    }

    #[test]
    fn test_put_and_get() {
        let mut store = SeedStore::new();
        let seed = make_seed("test", vec!["ai".into()]);
        let id = seed.id.clone();

        store.put(seed);
        assert_eq!(store.count(), 1);
        assert!(store.contains(&id));

        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.name, "test");
    }

    #[test]
    fn test_remove() {
        let mut store = SeedStore::new();
        let seed = make_seed("to-remove", vec![]);
        let id = seed.id.clone();

        store.put(seed);
        assert_eq!(store.count(), 1);

        let removed = store.remove(&id).unwrap();
        assert_eq!(removed.name, "to-remove");
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_find_by_domain() {
        let mut store = SeedStore::new();
        store.put(make_seed("s1", vec!["ai".into(), "crypto".into()]));
        store.put(make_seed("s2", vec!["ai".into()]));
        store.put(make_seed("s3", vec!["biology".into()]));

        let ai_seeds = store.find_by_domain("ai");
        assert_eq!(ai_seeds.len(), 2);

        let bio_seeds = store.find_by_domain("biology");
        assert_eq!(bio_seeds.len(), 1);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut store = SeedStore::new();
        store.put(make_seed("s1", vec!["ai".into()]));
        store.put(make_seed("s2", vec!["crypto".into()]));

        let json = store.to_json().unwrap();
        let restored = SeedStore::from_json(&json).unwrap();

        assert_eq!(restored.count(), 2);
    }

    #[test]
    fn test_prune() {
        let mut store = SeedStore::new();
        let mut low_fit = make_seed("low", vec!["test".into()]);
        low_fit.fitness = 0.1;
        let mut high_fit = make_seed("high", vec!["test".into()]);
        high_fit.fitness = 0.9;

        store.put(low_fit);
        store.put(high_fit);

        let pruned = store.prune(0.5);
        assert_eq!(pruned, 1);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_fittest() {
        let mut store = SeedStore::new();
        let mut s1 = make_seed("weak", vec![]);
        s1.fitness = 0.3;
        let mut s2 = make_seed("strong", vec![]);
        s2.fitness = 0.95;

        store.put(s1);
        store.put(s2);

        let best = store.fittest().unwrap();
        assert_eq!(best.name, "strong");
    }
}
