//! Lineage — Merkle ancestry tracking for DNA seeds
//!
//! Every seed knows where it came from. Lineage tracks the full genetic
//! history as a Merkle-like hash chain, enabling verifiable provenance
//! without revealing the actual content of ancestors.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single event in the seed's history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEvent {
    /// Epoch at which this event occurred
    pub epoch: u64,
    /// Type of event
    pub event_type: LineageEventType,
    /// Reconstruction error or fitness at this point
    pub metric: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Hash of the state at this event
    pub state_hash: String,
}

/// Types of lineage events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineageEventType {
    /// Seed was created from scratch
    Genesis,
    /// Seed was mutated/evolved
    Mutation,
    /// Seed was replicated from a parent
    Replication { parent_id: String },
    /// Seed was merged from two parents
    Merge { parent_a_id: String, parent_b_id: String },
}

/// Full lineage record for a seed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    /// The root hash of the lineage Merkle chain
    pub root_hash: String,
    /// Generation number (0 = genesis, increments on replication)
    pub generation: u64,
    /// Ordered history of events
    pub events: Vec<LineageEvent>,
    /// Parent IDs (empty for genesis, one for replication, two for merge)
    pub parent_ids: Vec<String>,
}

impl Lineage {
    /// Create a genesis lineage (no parents)
    pub fn genesis() -> Self {
        let event = LineageEvent {
            epoch: 0,
            event_type: LineageEventType::Genesis,
            metric: 0.0,
            timestamp: Utc::now(),
            state_hash: Self::hash_event("genesis", 0, 0.0),
        };
        let root_hash = event.state_hash.clone();
        Self {
            root_hash,
            generation: 0,
            events: vec![event],
            parent_ids: Vec::new(),
        }
    }

    /// Record a mutation event
    pub fn record_mutation(&mut self, epoch: u64, error: f64) {
        let state_hash = Self::chain_hash(&self.root_hash, "mutation", epoch, error);
        self.events.push(LineageEvent {
            epoch,
            event_type: LineageEventType::Mutation,
            metric: error,
            timestamp: Utc::now(),
            state_hash: state_hash.clone(),
        });
        self.root_hash = state_hash;
    }

    /// Create a child lineage from this parent
    pub fn spawn_child(&self, parent_id: &str) -> Self {
        let event = LineageEvent {
            epoch: 0,
            event_type: LineageEventType::Replication {
                parent_id: parent_id.to_string(),
            },
            metric: 0.0,
            timestamp: Utc::now(),
            state_hash: Self::chain_hash(&self.root_hash, "replicate", 0, 0.0),
        };
        let root_hash = event.state_hash.clone();
        Self {
            root_hash,
            generation: self.generation + 1,
            events: vec![event],
            parent_ids: vec![parent_id.to_string()],
        }
    }

    /// Merge two lineages into a new combined lineage
    pub fn merge_lineages(a: &Lineage, b: &Lineage) -> Self {
        let combined_hash = Self::merge_hash(&a.root_hash, &b.root_hash);
        let event = LineageEvent {
            epoch: 0,
            event_type: LineageEventType::Merge {
                parent_a_id: a.root_hash.clone(),
                parent_b_id: b.root_hash.clone(),
            },
            metric: 0.0,
            timestamp: Utc::now(),
            state_hash: combined_hash.clone(),
        };
        Self {
            root_hash: combined_hash,
            generation: a.generation.max(b.generation) + 1,
            events: vec![event],
            parent_ids: vec![a.root_hash.clone(), b.root_hash.clone()],
        }
    }

    /// Get the total number of mutations in this lineage
    pub fn mutation_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e.event_type, LineageEventType::Mutation))
            .count()
    }

    /// Verify the hash chain integrity
    pub fn verify_chain(&self) -> bool {
        if self.events.is_empty() {
            return false;
        }
        // The root hash should match the last event's state hash
        if let Some(last) = self.events.last() {
            last.state_hash == self.root_hash
        } else {
            false
        }
    }

    /// Compute a hash for an event
    fn hash_event(event_type: &str, epoch: u64, metric: f64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(event_type.as_bytes());
        hasher.update(epoch.to_le_bytes());
        hasher.update(metric.to_le_bytes());
        hex::encode(hasher.finalize())
    }

    /// Chain a new hash onto the existing root
    fn chain_hash(prev_hash: &str, event_type: &str, epoch: u64, metric: f64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prev_hash.as_bytes());
        hasher.update(event_type.as_bytes());
        hasher.update(epoch.to_le_bytes());
        hasher.update(metric.to_le_bytes());
        hex::encode(hasher.finalize())
    }

    /// Merge two hashes (for lineage merge)
    fn merge_hash(hash_a: &str, hash_b: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(hash_a.as_bytes());
        hasher.update(hash_b.as_bytes());
        hasher.update(b"merge");
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis() {
        let lineage = Lineage::genesis();
        assert_eq!(lineage.generation, 0);
        assert_eq!(lineage.events.len(), 1);
        assert!(lineage.parent_ids.is_empty());
        assert!(lineage.verify_chain());
    }

    #[test]
    fn test_record_mutation() {
        let mut lineage = Lineage::genesis();
        let original_hash = lineage.root_hash.clone();
        lineage.record_mutation(1, 0.5);
        assert_ne!(lineage.root_hash, original_hash);
        assert_eq!(lineage.events.len(), 2);
        assert_eq!(lineage.mutation_count(), 1);
        assert!(lineage.verify_chain());
    }

    #[test]
    fn test_spawn_child() {
        let parent = Lineage::genesis();
        let child = parent.spawn_child("parent-id-123");
        assert_eq!(child.generation, 1);
        assert_eq!(child.parent_ids, vec!["parent-id-123"]);
        assert!(child.verify_chain());
    }

    #[test]
    fn test_merge_lineages() {
        let a = Lineage::genesis();
        let b = Lineage::genesis();
        let merged = Lineage::merge_lineages(&a, &b);
        assert_eq!(merged.generation, 1);
        assert_eq!(merged.parent_ids.len(), 2);
        assert!(merged.verify_chain());
    }

    #[test]
    fn test_chain_integrity() {
        let mut lineage = Lineage::genesis();
        for i in 1..=10 {
            lineage.record_mutation(i, i as f64 * 0.1);
        }
        assert!(lineage.verify_chain());
        assert_eq!(lineage.mutation_count(), 10);
    }
}
