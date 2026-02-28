//! LoRA Manager — manage low-rank adapters as DLRS seeds
//!
//! LoRA (Low-Rank Adaptation) adapters are represented as DLRS seeds.
//! This module provides loading, creating, evolving, merging, and
//! exporting LoRA adapters backed by the LRIM engine.

mod adapter;
mod manager;
pub mod daemon;
pub mod training;
pub mod auto_tuner;
pub mod sharing;

pub use adapter::{LoraAdapter, LoraConfig, AdapterFormat};
pub use manager::{LoraManager, EvolutionConfig, ManagerStats};
pub use daemon::{AutoShareDaemon, DaemonConfig};
pub use training::{Dataset, DataSample, LossFunction, LrSchedule, TrainConfig, TrainHistory, EpochRecord, train};
pub use auto_tuner::{RankAnalysis, LrSearchResult, TuneRecommendation, analyze_rank, lr_search, auto_tune, auto_train, select_best};
pub use sharing::{SharingRegistry, RemoteAdapter, ReputationScore, SharePolicy};
