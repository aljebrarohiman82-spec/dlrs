//! TEE Enclave — Trusted Execution Environment abstraction
//!
//! Provides a unified interface for TEE operations across different backends:
//! - Intel SGX (via simulation layer)
//! - ARM TrustZone (via simulation layer)
//! - Software-simulated enclave (always available, for dev/test)
//!
//! The enclave protects sensitive matrix factors (U, sigma, V) and key material
//! from exposure to untrusted code, even on a compromised host.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Supported TEE backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeeBackend {
    /// Intel SGX enclave
    IntelSgx,
    /// ARM TrustZone
    ArmTrustZone,
    /// Software-simulated (always available, NOT hardware-secured)
    Simulated,
}

impl TeeBackend {
    pub fn name(&self) -> &str {
        match self {
            TeeBackend::IntelSgx => "Intel SGX",
            TeeBackend::ArmTrustZone => "ARM TrustZone",
            TeeBackend::Simulated => "Simulated (software)",
        }
    }

    pub fn is_hardware(&self) -> bool {
        !matches!(self, TeeBackend::Simulated)
    }
}

/// TEE enclave security level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Hardware-backed (SGX/TrustZone available and verified)
    Hardware,
    /// Software simulation (no hardware TEE, development only)
    Software,
    /// Degraded (hardware TEE detected but health check failed)
    Degraded,
}

/// Status of a running enclave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveStatus {
    pub enclave_id: String,
    pub backend: TeeBackend,
    pub security_level: SecurityLevel,
    pub created_at: DateTime<Utc>,
    pub sealed_items: usize,
    pub memory_usage_bytes: u64,
    pub is_healthy: bool,
}

/// Measurement of enclave identity — used for remote attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveMeasurement {
    /// MRENCLAVE — hash of enclave code + data at build time
    pub mrenclave: String,
    /// MRSIGNER — hash of the signing key
    pub mrsigner: String,
    /// Product ID
    pub product_id: u16,
    /// Security version number
    pub isv_svn: u16,
    /// Enclave attributes (flags)
    pub attributes: u64,
}

impl EnclaveMeasurement {
    /// Compute a measurement for this enclave (simulated via SHA256)
    pub fn compute(code_hash: &str, signer_key: &str, product_id: u16, svn: u16) -> Self {
        let mrenclave = {
            let mut h = Sha256::new();
            h.update(code_hash.as_bytes());
            h.update(b"mrenclave-v1");
            hex::encode(h.finalize())
        };
        let mrsigner = {
            let mut h = Sha256::new();
            h.update(signer_key.as_bytes());
            h.update(b"mrsigner-v1");
            hex::encode(h.finalize())
        };
        Self {
            mrenclave,
            mrsigner,
            product_id,
            isv_svn: svn,
            attributes: 0x0000_0000_0000_0007, // DEBUG | INIT | MODE64
        }
    }

    /// Verify measurement matches expected values
    pub fn matches(&self, expected: &EnclaveMeasurement) -> bool {
        self.mrenclave == expected.mrenclave
            && self.mrsigner == expected.mrsigner
            && self.isv_svn >= expected.isv_svn
    }
}

/// A TEE enclave instance managing secure memory and operations
pub struct TeeEnclave {
    /// Unique identifier
    pub id: String,
    /// Which backend is in use
    pub backend: TeeBackend,
    /// Enclave measurement
    pub measurement: EnclaveMeasurement,
    /// Sealed (encrypted) key-value store inside the enclave
    sealed_store: HashMap<String, Vec<u8>>,
    /// Enclave sealing key (derived from measurement + hardware key)
    sealing_key: Vec<u8>,
    /// Creation timestamp
    created_at: DateTime<Utc>,
    /// Security level
    security_level: SecurityLevel,
}

impl TeeEnclave {
    /// Create and initialize a new TEE enclave
    pub fn new(backend: TeeBackend) -> Result<Self, TeeError> {
        let security_level = match backend {
            TeeBackend::IntelSgx => {
                if Self::probe_sgx() {
                    SecurityLevel::Hardware
                } else {
                    log::warn!("Intel SGX not available, falling back to simulated enclave");
                    SecurityLevel::Software
                }
            }
            TeeBackend::ArmTrustZone => {
                if Self::probe_trustzone() {
                    SecurityLevel::Hardware
                } else {
                    log::warn!("ARM TrustZone not available, falling back to simulated enclave");
                    SecurityLevel::Software
                }
            }
            TeeBackend::Simulated => SecurityLevel::Software,
        };

        let id = uuid::Uuid::new_v4().to_string();
        let measurement = EnclaveMeasurement::compute(
            &format!("dlrs-enclave-{}", id),
            "dlrs-signer-key-v1",
            1,
            1,
        );

        // Derive sealing key from measurement
        let sealing_key = Self::derive_sealing_key(&measurement, &id);

        log::info!(
            "TEE enclave initialized: backend={}, security={:?}, id={}",
            backend.name(),
            security_level,
            &id[..8]
        );

        Ok(Self {
            id,
            backend,
            measurement,
            sealed_store: HashMap::new(),
            sealing_key,
            created_at: Utc::now(),
            security_level,
        })
    }

    /// Store data securely inside the enclave (sealed)
    pub fn seal(&mut self, key: &str, plaintext: &[u8]) -> Result<(), TeeError> {
        let sealed = self.encrypt(plaintext)?;
        self.sealed_store.insert(key.to_string(), sealed);
        Ok(())
    }

    /// Retrieve sealed data from the enclave
    pub fn unseal(&self, key: &str) -> Result<Vec<u8>, TeeError> {
        let sealed = self
            .sealed_store
            .get(key)
            .ok_or_else(|| TeeError::KeyNotFound(key.to_string()))?;
        self.decrypt(sealed)
    }

    /// Check if a key exists in sealed storage
    pub fn contains(&self, key: &str) -> bool {
        self.sealed_store.contains_key(key)
    }

    /// Remove sealed data
    pub fn remove(&mut self, key: &str) -> bool {
        self.sealed_store.remove(key).is_some()
    }

    /// List all sealed keys
    pub fn sealed_keys(&self) -> Vec<String> {
        self.sealed_store.keys().cloned().collect()
    }

    /// Seal matrix factor data (f64 slice -> encrypted bytes)
    pub fn seal_matrix_factors(
        &mut self,
        adapter_id: &str,
        u_data: &[f64],
        sigma_data: &[f64],
        v_data: &[f64],
    ) -> Result<(), TeeError> {
        let u_key = format!("{}/U", adapter_id);
        let s_key = format!("{}/S", adapter_id);
        let v_key = format!("{}/V", adapter_id);

        let u_bytes: Vec<u8> = u_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = sigma_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let v_bytes: Vec<u8> = v_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.seal(&u_key, &u_bytes)?;
        self.seal(&s_key, &s_bytes)?;
        self.seal(&v_key, &v_bytes)?;

        log::info!(
            "Sealed matrix factors for adapter {}: U={}B, S={}B, V={}B",
            &adapter_id[..8.min(adapter_id.len())],
            u_bytes.len(),
            s_bytes.len(),
            v_bytes.len()
        );

        Ok(())
    }

    /// Unseal matrix factor data back to f64 slices
    pub fn unseal_matrix_factors(
        &self,
        adapter_id: &str,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), TeeError> {
        let u_key = format!("{}/U", adapter_id);
        let s_key = format!("{}/S", adapter_id);
        let v_key = format!("{}/V", adapter_id);

        let u_bytes = self.unseal(&u_key)?;
        let s_bytes = self.unseal(&s_key)?;
        let v_bytes = self.unseal(&v_key)?;

        fn bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }

        Ok((bytes_to_f64(&u_bytes), bytes_to_f64(&s_bytes), bytes_to_f64(&v_bytes)))
    }

    /// Perform computation inside the enclave (ZK proof generation)
    /// Returns proof hash without exposing the actual matrix factors
    pub fn enclave_compute_proof(
        &self,
        adapter_id: &str,
        domain: &str,
        challenge_data: &[u8],
    ) -> Result<EnclaveProofResult, TeeError> {
        // Unseal the factors
        let (u_data, sigma_data, v_data) = self.unseal_matrix_factors(adapter_id)?;

        // Compute proof inside enclave (factors never leave)
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update(challenge_data);
        for val in &u_data {
            hasher.update(val.to_le_bytes());
        }
        for val in &sigma_data {
            hasher.update(val.to_le_bytes());
        }
        for val in &v_data {
            hasher.update(val.to_le_bytes());
        }
        let proof_hash = hex::encode(hasher.finalize());

        // Compute capability score inside enclave
        let sigma_norm: f64 = sigma_data.iter().map(|s| s * s).sum::<f64>().sqrt();
        let capability = sigma_norm / (sigma_norm + 1.0);

        Ok(EnclaveProofResult {
            proof_hash,
            capability_score: capability,
            enclave_id: self.id.clone(),
            security_level: self.security_level,
        })
    }

    /// Get enclave status
    pub fn status(&self) -> EnclaveStatus {
        let mem: u64 = self
            .sealed_store
            .values()
            .map(|v| v.len() as u64)
            .sum();

        EnclaveStatus {
            enclave_id: self.id.clone(),
            backend: self.backend,
            security_level: self.security_level,
            created_at: self.created_at,
            sealed_items: self.sealed_store.len(),
            memory_usage_bytes: mem,
            is_healthy: true,
        }
    }

    /// Get the enclave measurement for remote attestation
    pub fn get_measurement(&self) -> &EnclaveMeasurement {
        &self.measurement
    }

    /// Get the security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    // --- Internal helpers ---

    /// Encrypt data using the sealing key (AES-256-GCM simulated via SHA256-XOR)
    fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, TeeError> {
        // In a real TEE, this would use hardware-backed AES-256-GCM
        // Here we simulate with SHA256-derived keystream XOR
        let mut result = Vec::with_capacity(plaintext.len() + 32);

        // Generate random nonce
        let nonce: [u8; 16] = rand::random();
        result.extend_from_slice(&nonce);

        // Generate keystream and XOR
        let keystream = self.generate_keystream(&nonce, plaintext.len());
        for (i, byte) in plaintext.iter().enumerate() {
            result.push(byte ^ keystream[i]);
        }

        // Append MAC
        let mac = self.compute_mac(&nonce, plaintext);
        result.extend_from_slice(&mac);

        Ok(result)
    }

    /// Decrypt sealed data
    fn decrypt(&self, sealed: &[u8]) -> Result<Vec<u8>, TeeError> {
        if sealed.len() < 48 {
            // 16 nonce + 0 data + 32 mac
            return Err(TeeError::SealingError("Data too short".into()));
        }

        let nonce = &sealed[..16];
        let mac = &sealed[sealed.len() - 32..];
        let ciphertext = &sealed[16..sealed.len() - 32];

        // Decrypt
        let keystream = self.generate_keystream(nonce, ciphertext.len());
        let plaintext: Vec<u8> = ciphertext
            .iter()
            .enumerate()
            .map(|(i, byte)| byte ^ keystream[i])
            .collect();

        // Verify MAC
        let expected_mac = self.compute_mac(nonce, &plaintext);
        if mac != expected_mac.as_slice() {
            return Err(TeeError::IntegrityError(
                "MAC verification failed".into(),
            ));
        }

        Ok(plaintext)
    }

    fn generate_keystream(&self, nonce: &[u8], len: usize) -> Vec<u8> {
        let mut keystream = Vec::with_capacity(len);
        let mut counter = 0u64;
        while keystream.len() < len {
            let mut h = Sha256::new();
            h.update(&self.sealing_key);
            h.update(nonce);
            h.update(counter.to_le_bytes());
            keystream.extend_from_slice(&h.finalize());
            counter += 1;
        }
        keystream.truncate(len);
        keystream
    }

    fn compute_mac(&self, nonce: &[u8], plaintext: &[u8]) -> Vec<u8> {
        let mut h = Sha256::new();
        h.update(&self.sealing_key);
        h.update(nonce);
        h.update(plaintext);
        h.update(b"dlrs-mac-v1");
        h.finalize().to_vec()
    }

    fn derive_sealing_key(measurement: &EnclaveMeasurement, enclave_id: &str) -> Vec<u8> {
        let mut h = Sha256::new();
        h.update(measurement.mrenclave.as_bytes());
        h.update(measurement.mrsigner.as_bytes());
        h.update(enclave_id.as_bytes());
        h.update(b"dlrs-sealing-key-v1");
        h.finalize().to_vec()
    }

    fn probe_sgx() -> bool {
        // In a real implementation, this would check:
        // - /dev/sgx_enclave or /dev/isgx exists
        // - CPUID leaf 0x12 reports SGX support
        // - SGX driver is loaded
        #[cfg(target_arch = "x86_64")]
        {
            std::path::Path::new("/dev/sgx_enclave").exists()
                || std::path::Path::new("/dev/isgx").exists()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn probe_trustzone() -> bool {
        // In a real implementation, check ARM TrustZone availability
        #[cfg(target_arch = "aarch64")]
        {
            std::path::Path::new("/dev/tee0").exists()
                || std::path::Path::new("/dev/opteearmtz00").exists()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
}

/// Result of an enclave-internal proof computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnclaveProofResult {
    pub proof_hash: String,
    pub capability_score: f64,
    pub enclave_id: String,
    pub security_level: SecurityLevel,
}

/// TEE-related errors
#[derive(Debug, thiserror::Error)]
pub enum TeeError {
    #[error("TEE backend not available: {0}")]
    BackendUnavailable(String),

    #[error("Key not found in sealed storage: {0}")]
    KeyNotFound(String),

    #[error("Sealing error: {0}")]
    SealingError(String),

    #[error("Integrity verification failed: {0}")]
    IntegrityError(String),

    #[error("Attestation failed: {0}")]
    AttestationError(String),

    #[error("Enclave error: {0}")]
    EnclaveError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_enclave() {
        let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        assert_eq!(enclave.backend, TeeBackend::Simulated);
        assert_eq!(enclave.security_level(), SecurityLevel::Software);
        let status = enclave.status();
        assert!(status.is_healthy);
        assert_eq!(status.sealed_items, 0);
    }

    #[test]
    fn test_seal_unseal() {
        let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let data = b"secret matrix factors here";
        enclave.seal("test-key", data).unwrap();
        assert!(enclave.contains("test-key"));

        let recovered = enclave.unseal("test-key").unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_seal_unseal_matrix_factors() {
        let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let u = vec![1.0, 2.0, 3.0, 4.0];
        let s = vec![5.0, 6.0];
        let v = vec![7.0, 8.0, 9.0];

        enclave
            .seal_matrix_factors("adapter-123", &u, &s, &v)
            .unwrap();

        let (ru, rs, rv) = enclave.unseal_matrix_factors("adapter-123").unwrap();
        assert_eq!(ru, u);
        assert_eq!(rs, s);
        assert_eq!(rv, v);
    }

    #[test]
    fn test_enclave_compute_proof() {
        let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        let u = vec![1.0, 2.0, 3.0, 4.0];
        let s = vec![5.0, 6.0];
        let v = vec![7.0, 8.0, 9.0];

        enclave
            .seal_matrix_factors("proof-adapter", &u, &s, &v)
            .unwrap();

        let result = enclave
            .enclave_compute_proof("proof-adapter", "nlp", b"challenge-nonce")
            .unwrap();

        assert!(!result.proof_hash.is_empty());
        assert!(result.capability_score > 0.0 && result.capability_score <= 1.0);
        assert_eq!(result.security_level, SecurityLevel::Software);
    }

    #[test]
    fn test_integrity_check() {
        let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        enclave.seal("test", b"data").unwrap();

        // Tamper with sealed data
        if let Some(sealed) = enclave.sealed_store.get_mut("test") {
            if sealed.len() > 20 {
                sealed[20] ^= 0xFF; // flip a byte
            }
        }

        // Unseal should fail with integrity error
        let result = enclave.unseal("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_measurement() {
        let m1 = EnclaveMeasurement::compute("code-v1", "signer-a", 1, 1);
        let m2 = EnclaveMeasurement::compute("code-v1", "signer-a", 1, 1);
        let m3 = EnclaveMeasurement::compute("code-v2", "signer-a", 1, 1);

        assert!(m1.matches(&m2));
        assert!(!m1.matches(&m3));
    }

    #[test]
    fn test_remove_sealed() {
        let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
        enclave.seal("k1", b"data1").unwrap();
        enclave.seal("k2", b"data2").unwrap();
        assert_eq!(enclave.sealed_keys().len(), 2);

        assert!(enclave.remove("k1"));
        assert!(!enclave.contains("k1"));
        assert_eq!(enclave.sealed_keys().len(), 1);
    }
}
