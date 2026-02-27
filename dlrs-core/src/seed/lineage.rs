//! Lineage — Merkle-tree ancestry tracking for DNA seeds

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEvent {
    pub epoch: u64,
    pub event_type: LineageEventType,
    pub timestamp: DateTime<Utc>,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineageEventType {
    Genesis,
    Mutation { error_after: f64 },
    Replication { child_id: String },
    Merge { parent_a: String, parent_b: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub events: Vec<LineageEvent>,
    pub root_hash: String,
}

impl Lineage {
    pub fn genesis() -> Self {
        let event = LineageEvent {
            epoch: 0,
            event_type: LineageEventType::Genesis,
            timestamp: Utc::now(),
            hash: Self::hash_event("genesis", 0),
        };
        let root_hash = event.hash.clone();
        Self { events: vec![event], root_hash }
    }

    pub fn record_mutation(&mut self, epoch: u64, error: f64) {
        let event = LineageEvent {
            epoch,
            event_type: LineageEventType::Mutation { error_after: error },
            timestamp: Utc::now(),
            hash: Self::hash_chain(&self.root_hash, &format!("mutate:{epoch}:{error}")),
        };
        self.root_hash = event.hash.clone();
        self.events.push(event);
    }

    pub fn spawn_child(&self, parent_id: &str) -> Self {
        let mut child = Self::genesis();
        child.events[0].event_type = LineageEventType::Replication {
            child_id: parent_id.to_string(),
        };
        child.root_hash = Self::hash_chain(&self.root_hash, &format!("spawn:{parent_id}"));
        child
    }

    pub fn merge_lineages(a: &Lineage, b: &Lineage) -> Self {
        let event = LineageEvent {
            epoch: 0,
            event_type: LineageEventType::Merge {
                parent_a: a.root_hash.clone(),
                parent_b: b.root_hash.clone(),
            },
            timestamp: Utc::now(),
            hash: Self::hash_chain(&a.root_hash, &b.root_hash),
        };
        let root_hash = event.hash.clone();
        Self { events: vec![event], root_hash }
    }

    pub fn generation_count(&self) -> usize {
        self.events.len()
    }

    fn hash_event(data: &str, epoch: u64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.update(epoch.to_le_bytes());
        hex::encode(hasher.finalize())
    }

    fn hash_chain(prev: &str, data: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prev.as_bytes());
        hasher.update(data.as_bytes());
        hex::encode(hasher.finalize())
    }
}
