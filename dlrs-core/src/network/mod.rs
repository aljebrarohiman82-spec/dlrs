//! Network layer — gossip-based seed distribution
//!
//! Seeds propagate through the network based on fitness.
//! High-fitness seeds spread; low-fitness seeds are pruned.

pub mod sync;
pub use sync::SeedNetwork;
