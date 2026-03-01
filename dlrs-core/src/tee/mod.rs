//! TEE (Trusted Execution Environment) — Hardware security for DLRS
//!
//! Provides secure enclave support for protecting LoRA adapter secrets:
//! - **Enclave**: TEE abstraction (Intel SGX / ARM TrustZone / Simulated)
//! - **Attestation**: Remote attestation protocol for verifying enclave identity
//! - **Sealed Storage**: Persistent encrypted storage backed by TEE sealing keys
//! - **Secure Channel**: Attested, encrypted TEE-to-TEE communication

pub mod enclave;
pub mod attestation;
pub mod sealed_storage;
pub mod secure_channel;

pub use enclave::{
    TeeEnclave, TeeBackend, TeeError, SecurityLevel,
    EnclaveStatus, EnclaveMeasurement, EnclaveProofResult,
};
pub use attestation::{
    AttestationQuote, AttestationPolicy, AttestationVerdict,
    EnclaveReport, generate_quote, verify_quote, mutual_attestation,
};
pub use sealed_storage::{SealedStorage, SealedAdapterRecord, SealedIndex};
pub use secure_channel::{
    SecureChannel, SecureMessage, ChannelState,
    HandshakeInit, HandshakeResponse,
};
