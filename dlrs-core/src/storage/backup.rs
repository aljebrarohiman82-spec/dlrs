//! Backup System — versioned snapshots with rollback for permanent archive
//!
//! Provides:
//! - Automatic versioned snapshots (timestamped)
//! - Rollback to any previous version
//! - Permanent archive directory
//! - Integrity verification via SHA256

use chrono::{DateTime, Utc};
use log::info;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Metadata for a single backup snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub checksum: String,
    pub size_bytes: u64,
    pub adapter_count: usize,
    pub description: String,
    pub filename: String,
}

/// Backup manifest tracking all snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub snapshots: Vec<SnapshotMeta>,
    pub next_version: u64,
    pub backup_dir: String,
}

impl BackupManifest {
    pub fn new(backup_dir: impl Into<String>) -> Self {
        Self {
            snapshots: Vec::new(),
            next_version: 1,
            backup_dir: backup_dir.into(),
        }
    }

    /// Load an existing manifest or create a new one
    pub fn load_or_create(backup_dir: &str) -> Self {
        let manifest_path = Path::new(backup_dir).join("manifest.json");
        if manifest_path.exists() {
            if let Ok(json) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<BackupManifest>(&json) {
                    info!("Loaded backup manifest with {} snapshots", manifest.snapshots.len());
                    return manifest;
                }
            }
        }
        Self::new(backup_dir)
    }

    /// Save the manifest
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.backup_dir)?;
        let path = Path::new(&self.backup_dir).join("manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// The backup manager
pub struct BackupManager {
    pub manifest: BackupManifest,
}

impl BackupManager {
    /// Create or load a backup manager for the given directory
    pub fn new(backup_dir: &str) -> Self {
        Self {
            manifest: BackupManifest::load_or_create(backup_dir),
        }
    }

    /// Create a snapshot from JSON data
    pub fn create_snapshot(
        &mut self,
        data: &str,
        adapter_count: usize,
        description: &str,
    ) -> Result<SnapshotMeta, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.manifest.backup_dir)?;

        let version = self.manifest.next_version;
        let now = Utc::now();
        let filename = format!("snapshot_v{:04}_{}.json", version, now.format("%Y%m%d_%H%M%S"));
        let filepath = Path::new(&self.manifest.backup_dir).join(&filename);

        // Compute checksum
        let checksum = hex::encode(Sha256::digest(data.as_bytes()));

        // Write snapshot file
        std::fs::write(&filepath, data)?;

        let meta = SnapshotMeta {
            version,
            timestamp: now,
            checksum,
            size_bytes: data.len() as u64,
            adapter_count,
            description: description.to_string(),
            filename,
        };

        self.manifest.snapshots.push(meta.clone());
        self.manifest.next_version += 1;
        self.manifest.save()?;

        info!(
            "Created snapshot v{}: {} ({} adapters, {} bytes)",
            version, description, adapter_count, data.len()
        );

        Ok(meta)
    }

    /// Load a snapshot by version number
    pub fn load_snapshot(&self, version: u64) -> Result<String, Box<dyn std::error::Error>> {
        let meta = self.manifest.snapshots
            .iter()
            .find(|s| s.version == version)
            .ok_or_else(|| format!("Snapshot v{} not found", version))?;

        let filepath = Path::new(&self.manifest.backup_dir).join(&meta.filename);
        let data = std::fs::read_to_string(&filepath)?;

        // Verify checksum
        let checksum = hex::encode(Sha256::digest(data.as_bytes()));
        if checksum != meta.checksum {
            return Err(format!(
                "Checksum mismatch for v{}: expected {}, got {}",
                version, meta.checksum, checksum
            ).into());
        }

        info!("Loaded snapshot v{}: {} bytes, checksum OK", version, data.len());
        Ok(data)
    }

    /// Load the latest snapshot
    pub fn load_latest(&self) -> Result<String, Box<dyn std::error::Error>> {
        let latest = self.manifest.snapshots
            .last()
            .ok_or("No snapshots available")?;
        self.load_snapshot(latest.version)
    }

    /// List all available snapshots
    pub fn list_snapshots(&self) -> &[SnapshotMeta] {
        &self.manifest.snapshots
    }

    /// Get the number of snapshots
    pub fn snapshot_count(&self) -> usize {
        self.manifest.snapshots.len()
    }

    /// Verify integrity of all snapshots
    pub fn verify_all(&self) -> Vec<(u64, bool)> {
        let mut results = Vec::new();
        for meta in &self.manifest.snapshots {
            let filepath = Path::new(&self.manifest.backup_dir).join(&meta.filename);
            let ok = if let Ok(data) = std::fs::read_to_string(&filepath) {
                let checksum = hex::encode(Sha256::digest(data.as_bytes()));
                checksum == meta.checksum
            } else {
                false
            };
            results.push((meta.version, ok));
        }
        results
    }

    /// Remove old snapshots keeping only the latest N
    pub fn retain_latest(&mut self, keep: usize) -> Result<usize, Box<dyn std::error::Error>> {
        if self.manifest.snapshots.len() <= keep {
            return Ok(0);
        }

        let remove_count = self.manifest.snapshots.len() - keep;
        let to_remove: Vec<SnapshotMeta> = self.manifest.snapshots
            .drain(..remove_count)
            .collect();

        for meta in &to_remove {
            let filepath = Path::new(&self.manifest.backup_dir).join(&meta.filename);
            let _ = std::fs::remove_file(filepath);
        }

        self.manifest.save()?;
        info!("Removed {} old snapshots, keeping latest {}", remove_count, keep);
        Ok(remove_count)
    }

    /// Get the backup directory path
    pub fn backup_dir(&self) -> &str {
        &self.manifest.backup_dir
    }

    /// Get total backup size in bytes
    pub fn total_size(&self) -> u64 {
        self.manifest.snapshots.iter().map(|s| s.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dlrs-backup-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_create_and_load_snapshot() {
        let dir = test_dir();
        let mut mgr = BackupManager::new(dir.to_str().unwrap());

        let data = r#"{"test": "data", "count": 42}"#;
        let meta = mgr.create_snapshot(data, 3, "test snapshot").unwrap();
        assert_eq!(meta.version, 1);
        assert_eq!(meta.adapter_count, 3);

        let loaded = mgr.load_snapshot(1).unwrap();
        assert_eq!(loaded, data);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multiple_snapshots() {
        let dir = test_dir();
        let mut mgr = BackupManager::new(dir.to_str().unwrap());

        mgr.create_snapshot("v1", 1, "first").unwrap();
        mgr.create_snapshot("v2", 2, "second").unwrap();
        mgr.create_snapshot("v3", 3, "third").unwrap();

        assert_eq!(mgr.snapshot_count(), 3);

        let latest = mgr.load_latest().unwrap();
        assert_eq!(latest, "v3");

        let v1 = mgr.load_snapshot(1).unwrap();
        assert_eq!(v1, "v1");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_integrity() {
        let dir = test_dir();
        let mut mgr = BackupManager::new(dir.to_str().unwrap());

        mgr.create_snapshot("valid data", 1, "test").unwrap();
        let results = mgr.verify_all();
        assert_eq!(results.len(), 1);
        assert!(results[0].1); // should be valid

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_retain_latest() {
        let dir = test_dir();
        let mut mgr = BackupManager::new(dir.to_str().unwrap());

        for i in 0..5 {
            mgr.create_snapshot(&format!("data-{}", i), i, &format!("v{}", i)).unwrap();
        }
        assert_eq!(mgr.snapshot_count(), 5);

        let removed = mgr.retain_latest(2).unwrap();
        assert_eq!(removed, 3);
        assert_eq!(mgr.snapshot_count(), 2);

        // Should still load latest
        let latest = mgr.load_latest().unwrap();
        assert_eq!(latest, "data-4");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
