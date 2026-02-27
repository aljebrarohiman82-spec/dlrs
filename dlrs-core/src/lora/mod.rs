//! LoRA Manager — manage low-rank adapters as DLRS seeds
//!
//! LoRA (Low-Rank Adaptation) adapters are represented as DLRS seeds.
//! This module provides loading, creating, evolving, merging, and
//! exporting LoRA adapters backed by the LRIM engine.

mod adapter;
mod manager;
pub mod daemon;

pub use adapter::{LoraAdapter, LoraConfig, AdapterFormat};
pub use manager::{LoraManager, EvolutionConfig, ManagerStats};
pub use daemon::{AutoShareDaemon, DaemonConfig};
