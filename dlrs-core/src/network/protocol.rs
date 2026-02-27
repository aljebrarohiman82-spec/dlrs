//! DLRS network protocol messages
//!
//! Defines the message types exchanged between peers in the DLRS network.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Types of messages in the DLRS protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Announce a seed's existence (broadcast commitment, not data)
    SeedAnnounce {
        seed_id: String,
        commitment_root: String,
        domains: Vec<String>,
        fitness: f64,
        rank: usize,
    },
    /// Request capability proof for a domain
    CapabilityRequest {
        request_id: String,
        domain: String,
        challenge_vector: Vec<f64>,
        threshold: f64,
    },
    /// Respond with a capability proof
    CapabilityResponse {
        request_id: String,
        seed_id: String,
        score: f64,
        proof_hash: String,
    },
    /// Request to merge two seeds (collaborative compute)
    MergeRequest {
        request_id: String,
        initiator_seed_id: String,
        target_seed_id: String,
    },
    /// Accept or reject a merge request
    MergeResponse {
        request_id: String,
        accepted: bool,
        merged_commitment: Option<String>,
    },
    /// Ping for peer discovery and liveness
    Ping { timestamp: DateTime<Utc> },
    /// Pong response
    Pong { timestamp: DateTime<Utc> },
}

/// A complete DLRS network message with envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlrsMessage {
    /// Unique message ID
    pub id: String,
    /// Sender peer ID
    pub from: String,
    /// Message payload
    pub message_type: MessageType,
    /// When the message was created
    pub timestamp: DateTime<Utc>,
    /// Protocol version
    pub version: u32,
}

impl DlrsMessage {
    /// Create a new message
    pub fn new(from: impl Into<String>, message_type: MessageType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            from: from.into(),
            message_type,
            timestamp: Utc::now(),
            version: 1,
        }
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

/// DLRS gossipsub topic name
pub const DLRS_TOPIC: &str = "dlrs/seeds/v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_roundtrip() {
        let msg = DlrsMessage::new(
            "peer-123",
            MessageType::SeedAnnounce {
                seed_id: "seed-abc".to_string(),
                commitment_root: "hash123".to_string(),
                domains: vec!["ai".to_string(), "crypto".to_string()],
                fitness: 0.85,
                rank: 4,
            },
        );

        let bytes = msg.to_bytes().unwrap();
        let decoded = DlrsMessage::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.id, msg.id);
        assert_eq!(decoded.from, "peer-123");
        assert_eq!(decoded.version, 1);
    }

    #[test]
    fn test_ping_pong() {
        let ping = DlrsMessage::new("peer-a", MessageType::Ping { timestamp: Utc::now() });
        let pong = DlrsMessage::new("peer-b", MessageType::Pong { timestamp: Utc::now() });

        assert!(ping.to_bytes().is_ok());
        assert!(pong.to_bytes().is_ok());
    }

    #[test]
    fn test_capability_request_response() {
        let req = DlrsMessage::new(
            "verifier",
            MessageType::CapabilityRequest {
                request_id: "req-1".to_string(),
                domain: "machine-learning".to_string(),
                challenge_vector: vec![0.1, 0.2, 0.3],
                threshold: 0.5,
            },
        );
        let bytes = req.to_bytes().unwrap();
        let decoded = DlrsMessage::from_bytes(&bytes).unwrap();
        match decoded.message_type {
            MessageType::CapabilityRequest { domain, threshold, .. } => {
                assert_eq!(domain, "machine-learning");
                assert!((threshold - 0.5).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong message type"),
        }
    }
}
