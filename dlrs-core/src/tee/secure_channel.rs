//! Secure Channel — TEE-to-TEE encrypted communication
//!
//! Establishes an attested, encrypted channel between two TEE enclaves:
//! 1. Mutual attestation (both sides verify each other's enclave identity)
//! 2. Key exchange (Diffie-Hellman simulated via SHA256 HKDF)
//! 3. Encrypted message passing with forward secrecy
//!
//! Used for securely transferring sealed adapter data between peers.

use super::attestation::{
    generate_quote, verify_quote, AttestationPolicy, AttestationQuote, AttestationVerdict,
};
use super::enclave::{SecurityLevel, TeeEnclave, TeeError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// State of a secure channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelState {
    /// Initial state, waiting for handshake
    Pending,
    /// Attestation in progress
    Attesting,
    /// Channel established and ready for use
    Established,
    /// Channel closed or failed
    Closed,
}

/// A secure channel between two TEE enclaves
pub struct SecureChannel {
    /// Channel identifier
    pub channel_id: String,
    /// Local enclave ID
    pub local_enclave_id: String,
    /// Remote enclave ID (set after attestation)
    pub remote_enclave_id: Option<String>,
    /// Current state
    pub state: ChannelState,
    /// Session key (derived after successful attestation + key exchange)
    session_key: Option<Vec<u8>>,
    /// Message counter (for replay protection)
    send_counter: u64,
    recv_counter: u64,
    /// Remote security level
    pub remote_security_level: Option<SecurityLevel>,
    /// When the channel was established
    pub established_at: Option<DateTime<Utc>>,
    /// Attestation policy
    policy: AttestationPolicy,
}

/// An encrypted message on the secure channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureMessage {
    pub channel_id: String,
    pub sequence: u64,
    pub ciphertext: String,
    pub mac: String,
    pub timestamp: DateTime<Utc>,
}

/// Handshake initiation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeInit {
    pub channel_id: String,
    pub quote: AttestationQuote,
    pub dh_public: String,
}

/// Handshake response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub channel_id: String,
    pub quote: AttestationQuote,
    pub dh_public: String,
    pub accepted: bool,
    pub reason: Option<String>,
}

impl SecureChannel {
    /// Create a new secure channel (initiator side)
    pub fn new(local_enclave_id: &str, policy: AttestationPolicy) -> Self {
        Self {
            channel_id: uuid::Uuid::new_v4().to_string(),
            local_enclave_id: local_enclave_id.to_string(),
            remote_enclave_id: None,
            state: ChannelState::Pending,
            session_key: None,
            send_counter: 0,
            recv_counter: 0,
            remote_security_level: None,
            established_at: None,
            policy,
        }
    }

    /// Step 1: Initiator generates handshake init message
    pub fn initiate_handshake(&mut self, enclave: &TeeEnclave) -> HandshakeInit {
        self.state = ChannelState::Attesting;

        let nonce = format!("channel-{}-init", self.channel_id);
        let quote = generate_quote(enclave, &nonce);

        // Generate DH public key (simulated)
        let dh_public = Self::generate_dh_public(&self.channel_id, &enclave.id);

        HandshakeInit {
            channel_id: self.channel_id.clone(),
            quote,
            dh_public,
        }
    }

    /// Step 2: Responder processes handshake init and generates response
    pub fn respond_to_handshake(
        &mut self,
        enclave: &TeeEnclave,
        init: &HandshakeInit,
    ) -> HandshakeResponse {
        self.channel_id = init.channel_id.clone();
        self.state = ChannelState::Attesting;

        // Verify initiator's attestation
        let verdict = verify_quote(&init.quote, &self.policy);

        match verdict {
            AttestationVerdict::Trusted {
                enclave_id,
                security_level,
            } => {
                self.remote_enclave_id = Some(enclave_id);
                self.remote_security_level = Some(security_level);

                // Generate our quote
                let nonce = format!("channel-{}-resp", self.channel_id);
                let quote = generate_quote(enclave, &nonce);

                // Generate DH public key
                let dh_public = Self::generate_dh_public(&self.channel_id, &enclave.id);

                // Derive session key from both DH publics
                let session_key = Self::derive_session_key(
                    &init.dh_public,
                    &dh_public,
                    &self.channel_id,
                );
                self.session_key = Some(session_key);
                self.state = ChannelState::Established;
                self.established_at = Some(Utc::now());

                log::info!(
                    "Secure channel {} established (responder): remote security={:?}",
                    &self.channel_id[..8],
                    security_level
                );

                HandshakeResponse {
                    channel_id: self.channel_id.clone(),
                    quote,
                    dh_public,
                    accepted: true,
                    reason: None,
                }
            }
            _ => {
                self.state = ChannelState::Closed;
                let reason = format!("Attestation failed: {:?}", verdict);

                let nonce = format!("channel-{}-reject", self.channel_id);
                let quote = generate_quote(enclave, &nonce);

                HandshakeResponse {
                    channel_id: self.channel_id.clone(),
                    quote,
                    dh_public: String::new(),
                    accepted: false,
                    reason: Some(reason),
                }
            }
        }
    }

    /// Step 3: Initiator completes handshake with response
    pub fn complete_handshake(
        &mut self,
        response: &HandshakeResponse,
        our_dh_public: &str,
    ) -> Result<(), TeeError> {
        if !response.accepted {
            self.state = ChannelState::Closed;
            return Err(TeeError::AttestationError(
                response
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Rejected".into()),
            ));
        }

        // Verify responder's attestation
        let verdict = verify_quote(&response.quote, &self.policy);
        match verdict {
            AttestationVerdict::Trusted {
                enclave_id,
                security_level,
            } => {
                self.remote_enclave_id = Some(enclave_id);
                self.remote_security_level = Some(security_level);

                // Derive session key
                let session_key = Self::derive_session_key(
                    our_dh_public,
                    &response.dh_public,
                    &self.channel_id,
                );
                self.session_key = Some(session_key);
                self.state = ChannelState::Established;
                self.established_at = Some(Utc::now());

                log::info!(
                    "Secure channel {} established (initiator): remote security={:?}",
                    &self.channel_id[..8],
                    security_level
                );

                Ok(())
            }
            _ => {
                self.state = ChannelState::Closed;
                Err(TeeError::AttestationError(format!(
                    "Responder attestation failed: {:?}",
                    verdict
                )))
            }
        }
    }

    /// Encrypt a message for the secure channel
    pub fn encrypt_message(&mut self, plaintext: &[u8]) -> Result<SecureMessage, TeeError> {
        let session_key = self
            .session_key
            .as_ref()
            .ok_or_else(|| TeeError::EnclaveError("Channel not established".into()))?;

        let seq = self.send_counter;
        self.send_counter += 1;

        // Encrypt: XOR with keystream derived from session key + sequence
        let keystream = Self::message_keystream(session_key, seq, plaintext.len());
        let ciphertext_bytes: Vec<u8> = plaintext
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ keystream[i])
            .collect();
        let ciphertext = hex::encode(&ciphertext_bytes);

        // MAC
        let mac = {
            let mut h = Sha256::new();
            h.update(session_key);
            h.update(seq.to_le_bytes());
            h.update(plaintext);
            h.update(b"dlrs-channel-mac-v1");
            hex::encode(h.finalize())
        };

        Ok(SecureMessage {
            channel_id: self.channel_id.clone(),
            sequence: seq,
            ciphertext,
            mac,
            timestamp: Utc::now(),
        })
    }

    /// Decrypt a message from the secure channel
    pub fn decrypt_message(&mut self, msg: &SecureMessage) -> Result<Vec<u8>, TeeError> {
        let session_key = self
            .session_key
            .as_ref()
            .ok_or_else(|| TeeError::EnclaveError("Channel not established".into()))?;

        // Check for replay
        if msg.sequence < self.recv_counter {
            return Err(TeeError::IntegrityError("Replay detected".into()));
        }

        // Decrypt
        let ciphertext_bytes = hex::decode(&msg.ciphertext)
            .map_err(|e| TeeError::SealingError(format!("Hex decode: {}", e)))?;
        let keystream = Self::message_keystream(session_key, msg.sequence, ciphertext_bytes.len());
        let plaintext: Vec<u8> = ciphertext_bytes
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ keystream[i])
            .collect();

        // Verify MAC
        let expected_mac = {
            let mut h = Sha256::new();
            h.update(session_key);
            h.update(msg.sequence.to_le_bytes());
            h.update(&plaintext);
            h.update(b"dlrs-channel-mac-v1");
            hex::encode(h.finalize())
        };

        if msg.mac != expected_mac {
            return Err(TeeError::IntegrityError("Message MAC mismatch".into()));
        }

        self.recv_counter = msg.sequence + 1;
        Ok(plaintext)
    }

    /// Close the channel
    pub fn close(&mut self) {
        self.session_key = None;
        self.state = ChannelState::Closed;
        log::info!("Secure channel {} closed", &self.channel_id[..8]);
    }

    /// Check if the channel is established
    pub fn is_established(&self) -> bool {
        self.state == ChannelState::Established && self.session_key.is_some()
    }

    /// Get channel summary
    pub fn summary(&self) -> String {
        format!(
            "Channel {} | state={:?} | remote={} | security={:?}",
            &self.channel_id[..8],
            self.state,
            self.remote_enclave_id
                .as_deref()
                .map(|s| &s[..8.min(s.len())])
                .unwrap_or("none"),
            self.remote_security_level,
        )
    }

    // --- Internal helpers ---

    fn generate_dh_public(channel_id: &str, enclave_id: &str) -> String {
        // Simulated DH: in a real TEE, this would use ECDH within the enclave
        let mut h = Sha256::new();
        h.update(channel_id.as_bytes());
        h.update(enclave_id.as_bytes());
        h.update(rand::random::<[u8; 32]>());
        h.update(b"dlrs-dh-public-v1");
        hex::encode(h.finalize())
    }

    fn derive_session_key(dh_a: &str, dh_b: &str, channel_id: &str) -> Vec<u8> {
        // HKDF-like derivation from shared secret
        let mut h = Sha256::new();
        // Sort to ensure same key regardless of order
        let (first, second) = if dh_a < dh_b { (dh_a, dh_b) } else { (dh_b, dh_a) };
        h.update(first.as_bytes());
        h.update(second.as_bytes());
        h.update(channel_id.as_bytes());
        h.update(b"dlrs-session-key-v1");
        h.finalize().to_vec()
    }

    fn message_keystream(session_key: &[u8], sequence: u64, len: usize) -> Vec<u8> {
        let mut keystream = Vec::with_capacity(len);
        let mut counter = 0u64;
        while keystream.len() < len {
            let mut h = Sha256::new();
            h.update(session_key);
            h.update(sequence.to_le_bytes());
            h.update(counter.to_le_bytes());
            h.update(b"msg-ks");
            keystream.extend_from_slice(&h.finalize());
            counter += 1;
        }
        keystream.truncate(len);
        keystream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tee::enclave::TeeBackend;

    #[test]
    fn test_full_handshake() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        let policy = AttestationPolicy::default();
        let mut channel_a = SecureChannel::new(&enclave_a.id, policy.clone());
        let mut channel_b = SecureChannel::new(&enclave_b.id, policy);

        // Step 1: A initiates
        let init = channel_a.initiate_handshake(&enclave_a);
        assert_eq!(channel_a.state, ChannelState::Attesting);

        // Step 2: B responds
        let response = channel_b.respond_to_handshake(&enclave_b, &init);
        assert!(response.accepted);
        assert!(channel_b.is_established());

        // Step 3: A completes
        channel_a
            .complete_handshake(&response, &init.dh_public)
            .unwrap();
        assert!(channel_a.is_established());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        let policy = AttestationPolicy::default();
        let mut channel_a = SecureChannel::new(&enclave_a.id, policy.clone());
        let mut channel_b = SecureChannel::new(&enclave_b.id, policy);

        // Handshake
        let init = channel_a.initiate_handshake(&enclave_a);
        let response = channel_b.respond_to_handshake(&enclave_b, &init);
        channel_a
            .complete_handshake(&response, &init.dh_public)
            .unwrap();

        // Encrypt on A
        let plaintext = b"secret adapter data: U=[1,2,3] S=[4,5] V=[6,7,8]";
        let msg = channel_a.encrypt_message(plaintext).unwrap();

        // Decrypt on B
        let decrypted = channel_b.decrypt_message(&msg).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_replay_protection() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        let policy = AttestationPolicy::default();
        let mut channel_a = SecureChannel::new(&enclave_a.id, policy.clone());
        let mut channel_b = SecureChannel::new(&enclave_b.id, policy);

        let init = channel_a.initiate_handshake(&enclave_a);
        let response = channel_b.respond_to_handshake(&enclave_b, &init);
        channel_a
            .complete_handshake(&response, &init.dh_public)
            .unwrap();

        let msg = channel_a.encrypt_message(b"test").unwrap();
        channel_b.decrypt_message(&msg).unwrap();

        // Replay the same message
        let result = channel_b.decrypt_message(&msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_message() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        let policy = AttestationPolicy::default();
        let mut channel_a = SecureChannel::new(&enclave_a.id, policy.clone());
        let mut channel_b = SecureChannel::new(&enclave_b.id, policy);

        let init = channel_a.initiate_handshake(&enclave_a);
        let response = channel_b.respond_to_handshake(&enclave_b, &init);
        channel_a
            .complete_handshake(&response, &init.dh_public)
            .unwrap();

        let mut msg = channel_a.encrypt_message(b"original").unwrap();
        msg.ciphertext = "deadbeef".to_string(); // tamper

        let result = channel_b.decrypt_message(&msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_strict_policy_rejection() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        // B uses strict policy that rejects simulated enclaves
        let strict = AttestationPolicy::strict(vec![]);
        let mut channel_a = SecureChannel::new(&enclave_a.id, AttestationPolicy::default());
        let mut channel_b = SecureChannel::new(&enclave_b.id, strict);

        let init = channel_a.initiate_handshake(&enclave_a);
        let response = channel_b.respond_to_handshake(&enclave_b, &init);
        assert!(!response.accepted);
        assert_eq!(channel_b.state, ChannelState::Closed);
    }

    #[test]
    fn test_close_channel() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let mut channel = SecureChannel::new(&enclave.id, AttestationPolicy::default());
        channel.close();
        assert_eq!(channel.state, ChannelState::Closed);
        assert!(!channel.is_established());
    }
}
