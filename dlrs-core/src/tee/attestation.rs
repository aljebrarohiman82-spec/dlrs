//! Remote Attestation — verify enclave identity and integrity across the network
//!
//! Implements a remote attestation protocol:
//! 1. Challenger sends a nonce
//! 2. Enclave produces a quote (measurement + nonce signature)
//! 3. Challenger verifies quote against a trust policy
//!
//! This allows peers to verify that a remote enclave is running trusted code
//! before sharing sensitive adapter data.

use super::enclave::{EnclaveMeasurement, SecurityLevel, TeeBackend, TeeEnclave, TeeError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// An attestation quote produced by an enclave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationQuote {
    /// The enclave measurement
    pub measurement: EnclaveMeasurement,
    /// Nonce provided by the challenger (for freshness)
    pub nonce: String,
    /// Quote signature (hash binding measurement + nonce + enclave state)
    pub signature: String,
    /// TEE backend used
    pub backend: TeeBackend,
    /// Security level of the enclave
    pub security_level: SecurityLevel,
    /// Timestamp of quote generation
    pub timestamp: DateTime<Utc>,
    /// Additional enclave data (e.g., sealed item count)
    pub enclave_data: EnclaveReport,
}

/// Additional data included in an attestation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveReport {
    pub enclave_id: String,
    pub sealed_adapters: usize,
    pub uptime_secs: i64,
}

/// Policy for accepting or rejecting attestation quotes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationPolicy {
    /// Trusted MRENCLAVE values (empty = accept any)
    pub trusted_enclaves: Vec<String>,
    /// Trusted MRSIGNER values (empty = accept any)
    pub trusted_signers: Vec<String>,
    /// Minimum security version number
    pub min_svn: u16,
    /// Whether to accept software-simulated enclaves
    pub allow_simulated: bool,
    /// Maximum age of quote in seconds
    pub max_quote_age_secs: i64,
    /// Minimum security level required
    pub min_security_level: SecurityLevel,
}

impl Default for AttestationPolicy {
    fn default() -> Self {
        Self {
            trusted_enclaves: Vec::new(),
            trusted_signers: Vec::new(),
            min_svn: 0,
            allow_simulated: true, // permissive default for development
            max_quote_age_secs: 3600,
            min_security_level: SecurityLevel::Software,
        }
    }
}

impl AttestationPolicy {
    /// Strict policy: require hardware TEE, known signer
    pub fn strict(trusted_signers: Vec<String>) -> Self {
        Self {
            trusted_enclaves: Vec::new(),
            trusted_signers,
            min_svn: 1,
            allow_simulated: false,
            max_quote_age_secs: 300,
            min_security_level: SecurityLevel::Hardware,
        }
    }
}

/// Result of verifying an attestation quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttestationVerdict {
    /// Quote is valid and trusted
    Trusted {
        enclave_id: String,
        security_level: SecurityLevel,
    },
    /// Quote is valid but from an untrusted enclave
    Untrusted { reason: String },
    /// Quote is invalid or corrupted
    Invalid { reason: String },
    /// Quote has expired
    Expired,
}

impl AttestationVerdict {
    pub fn is_trusted(&self) -> bool {
        matches!(self, AttestationVerdict::Trusted { .. })
    }
}

/// Generate an attestation quote from an enclave
pub fn generate_quote(enclave: &TeeEnclave, nonce: &str) -> AttestationQuote {
    let measurement = enclave.get_measurement().clone();
    let status = enclave.status();

    let report = EnclaveReport {
        enclave_id: enclave.id.clone(),
        sealed_adapters: status.sealed_items / 3, // 3 keys per adapter (U/S/V)
        uptime_secs: (Utc::now() - status.created_at).num_seconds(),
    };

    // Compute quote signature: H(measurement || nonce || report)
    let signature = {
        let mut h = Sha256::new();
        h.update(measurement.mrenclave.as_bytes());
        h.update(measurement.mrsigner.as_bytes());
        h.update(measurement.isv_svn.to_le_bytes());
        h.update(nonce.as_bytes());
        h.update(report.enclave_id.as_bytes());
        h.update(report.sealed_adapters.to_le_bytes());
        h.update(b"dlrs-attestation-quote-v1");
        hex::encode(h.finalize())
    };

    AttestationQuote {
        measurement,
        nonce: nonce.to_string(),
        signature,
        backend: enclave.backend,
        security_level: enclave.security_level(),
        timestamp: Utc::now(),
        enclave_data: report,
    }
}

/// Verify an attestation quote against a policy
pub fn verify_quote(quote: &AttestationQuote, policy: &AttestationPolicy) -> AttestationVerdict {
    // 1. Check quote freshness
    let age = (Utc::now() - quote.timestamp).num_seconds();
    if age > policy.max_quote_age_secs {
        return AttestationVerdict::Expired;
    }

    // 2. Verify signature integrity
    let expected_sig = {
        let mut h = Sha256::new();
        h.update(quote.measurement.mrenclave.as_bytes());
        h.update(quote.measurement.mrsigner.as_bytes());
        h.update(quote.measurement.isv_svn.to_le_bytes());
        h.update(quote.nonce.as_bytes());
        h.update(quote.enclave_data.enclave_id.as_bytes());
        h.update(quote.enclave_data.sealed_adapters.to_le_bytes());
        h.update(b"dlrs-attestation-quote-v1");
        hex::encode(h.finalize())
    };

    if quote.signature != expected_sig {
        return AttestationVerdict::Invalid {
            reason: "Quote signature mismatch".into(),
        };
    }

    // 3. Check simulated enclave policy
    if !policy.allow_simulated && quote.security_level == SecurityLevel::Software {
        return AttestationVerdict::Untrusted {
            reason: "Software-simulated enclave not allowed by policy".into(),
        };
    }

    // 4. Check security level
    let level_ok = match policy.min_security_level {
        SecurityLevel::Hardware => quote.security_level == SecurityLevel::Hardware,
        SecurityLevel::Software => true,
        SecurityLevel::Degraded => true,
    };
    if !level_ok {
        return AttestationVerdict::Untrusted {
            reason: format!(
                "Security level {:?} below minimum {:?}",
                quote.security_level, policy.min_security_level
            ),
        };
    }

    // 5. Check SVN
    if quote.measurement.isv_svn < policy.min_svn {
        return AttestationVerdict::Untrusted {
            reason: format!(
                "SVN {} below minimum {}",
                quote.measurement.isv_svn, policy.min_svn
            ),
        };
    }

    // 6. Check trusted enclaves (if list is non-empty)
    if !policy.trusted_enclaves.is_empty()
        && !policy
            .trusted_enclaves
            .contains(&quote.measurement.mrenclave)
    {
        return AttestationVerdict::Untrusted {
            reason: "MRENCLAVE not in trusted list".into(),
        };
    }

    // 7. Check trusted signers (if list is non-empty)
    if !policy.trusted_signers.is_empty()
        && !policy
            .trusted_signers
            .contains(&quote.measurement.mrsigner)
    {
        return AttestationVerdict::Untrusted {
            reason: "MRSIGNER not in trusted list".into(),
        };
    }

    AttestationVerdict::Trusted {
        enclave_id: quote.enclave_data.enclave_id.clone(),
        security_level: quote.security_level,
    }
}

/// Run a full attestation handshake between two enclaves
pub fn mutual_attestation(
    local: &TeeEnclave,
    remote_quote: &AttestationQuote,
    policy: &AttestationPolicy,
) -> Result<(AttestationQuote, AttestationVerdict), TeeError> {
    // Verify the remote enclave's quote
    let verdict = verify_quote(remote_quote, policy);

    // Generate our own quote using the remote's nonce as freshness binding
    let our_quote = generate_quote(local, &remote_quote.nonce);

    Ok((our_quote, verdict))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_verify_quote() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let quote = generate_quote(&enclave, "test-nonce-123");

        assert_eq!(quote.nonce, "test-nonce-123");
        assert_eq!(quote.backend, TeeBackend::Simulated);
        assert!(!quote.signature.is_empty());

        let policy = AttestationPolicy::default();
        let verdict = verify_quote(&quote, &policy);
        assert!(verdict.is_trusted());
    }

    #[test]
    fn test_strict_policy_rejects_simulated() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let quote = generate_quote(&enclave, "nonce");

        let policy = AttestationPolicy::strict(vec![]);
        let verdict = verify_quote(&quote, &policy);
        assert!(!verdict.is_trusted());
    }

    #[test]
    fn test_expired_quote() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let mut quote = generate_quote(&enclave, "nonce");
        quote.timestamp = Utc::now() - chrono::Duration::hours(2);

        let policy = AttestationPolicy::default(); // max 3600s
        let verdict = verify_quote(&quote, &policy);
        assert!(matches!(verdict, AttestationVerdict::Expired));
    }

    #[test]
    fn test_tampered_quote() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let mut quote = generate_quote(&enclave, "nonce");
        quote.measurement.isv_svn = 99; // tamper

        let policy = AttestationPolicy::default();
        let verdict = verify_quote(&quote, &policy);
        assert!(matches!(verdict, AttestationVerdict::Invalid { .. }));
    }

    #[test]
    fn test_mutual_attestation() {
        let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

        let quote_b = generate_quote(&enclave_b, "nonce-from-a");
        let policy = AttestationPolicy::default();

        let (quote_a, verdict) = mutual_attestation(&enclave_a, &quote_b, &policy).unwrap();
        assert!(verdict.is_trusted());
        assert!(!quote_a.signature.is_empty());
    }

    #[test]
    fn test_svn_check() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let quote = generate_quote(&enclave, "nonce");

        let mut policy = AttestationPolicy::default();
        policy.min_svn = 99; // higher than enclave's SVN of 1

        let verdict = verify_quote(&quote, &policy);
        assert!(!verdict.is_trusted());
    }
}
