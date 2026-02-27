//! Local storage for DNA seeds
//!
//! Persistent store with JSON serialization.
//! Open the app → see all your seeds → sync with network.

use crate::seed::DnaSeed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct SeedStore {
    pub seeds: HashMap<String, DnaSeed>,
    pub path: PathBuf,
    pub metadata: StoreMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub owner: String,
    pub created_at: String,
    pub total_seeds_ever: u64,
    pub total_evolutions: u64,
    pub total_replications: u64,
}

impl SeedStore {
    pub fn open(path: impl AsRef<Path>, owner: &str) -> Self {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            if let Ok(data) = std::fs::read_to_string(&path) {
                if let Ok(store) = serde_json::from_str(&data) {
                    return store;
                }
            }
        }
        Self {
            seeds: HashMap::new(),
            path,
            metadata: StoreMetadata {
                owner: owner.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                total_seeds_ever: 0,
                total_evolutions: 0,
                total_replications: 0,
            },
        }
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    pub fn add(&mut self, seed: DnaSeed) {
        self.metadata.total_seeds_ever += 1;
        self.seeds.insert(seed.id.clone(), seed);
    }

    pub fn get(&self, id: &str) -> Option<&DnaSeed> {
        self.seeds.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut DnaSeed> {
        self.seeds.get_mut(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<DnaSeed> {
        self.seeds.remove(id)
    }

    pub fn list_by_fitness(&self) -> Vec<&DnaSeed> {
        let mut seeds: Vec<&DnaSeed> = self.seeds.values().collect();
        seeds.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        seeds
    }

    pub fn list_by_domain(&self, domain: &str) -> Vec<&DnaSeed> {
        self.seeds.values()
            .filter(|s| s.domains.iter().any(|d| d == domain))
            .collect()
    }

    pub fn summary(&self) -> String {
        let total = self.seeds.len();
        let avg_fitness = if total > 0 {
            self.seeds.values().map(|s| s.fitness).sum::<f64>() / total as f64
        } else { 0.0 };
        let domains: Vec<String> = {
            let mut d: Vec<String> = self.seeds.values()
                .flat_map(|s| s.domains.clone()).collect();
            d.sort(); d.dedup(); d
        };
        format!(
            "SeedStore '{}' | {} seeds | avg fitness {:.3} | domains: {:?}",
            self.metadata.owner, total, avg_fitness, domains
        )
    }
}
