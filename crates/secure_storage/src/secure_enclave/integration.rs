//! # Secure Enclave Integration Layer
//!
//! Integration layer for combining Secure Enclaves with HSM and MPC systems
//! for ultra-secure financial operations in `TallyIO` platform.

use super::{EnclaveOperation, SecureEnclaveSystem};
use crate::error::{SecureStorageError, SecureStorageResult};
use crate::types::KeyId;

use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// HSM-Enclave integration manager
pub struct HsmEnclaveIntegration {
    /// Enclave system reference
    enclave_system: Arc<SecureEnclaveSystem>,
    /// HSM provider (when available)
    #[cfg(feature = "hsm")]
    hsm_provider: Option<Arc<dyn crate::hsm::HsmProvider>>,
    /// Integration configuration
    config: HsmIntegrationConfig,
}

impl std::fmt::Debug for HsmEnclaveIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("HsmEnclaveIntegration");
        debug_struct
            .field("enclave_system", &"<SecureEnclaveSystem>")
            .field("config", &self.config);

        #[cfg(feature = "hsm")]
        debug_struct.field("hsm_provider", &self.hsm_provider.is_some());

        debug_struct.finish()
    }
}

/// HSM integration configuration
#[derive(Debug, Clone)]
pub struct HsmIntegrationConfig {
    /// Use HSM for key generation
    pub hsm_key_generation: bool,
    /// Use HSM for signing operations
    pub hsm_signing: bool,
    /// Fallback to enclave if HSM fails
    pub enclave_fallback: bool,
    /// Maximum HSM operation timeout
    pub hsm_timeout: Duration,
    /// Key size threshold for HSM usage
    pub hsm_key_size_threshold: usize,
}

impl Default for HsmIntegrationConfig {
    fn default() -> Self {
        Self {
            hsm_key_generation: true,
            hsm_signing: true,
            enclave_fallback: true,
            hsm_timeout: Duration::from_millis(100),
            hsm_key_size_threshold: 2048, // Use HSM for 2048-bit keys and above
        }
    }
}

impl HsmEnclaveIntegration {
    /// Create new HSM-Enclave integration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(
        enclave_system: Arc<SecureEnclaveSystem>,
        config: HsmIntegrationConfig,
    ) -> SecureStorageResult<Self> {
        info!("Initializing HSM-Enclave integration");

        Ok(Self {
            enclave_system,
            #[cfg(feature = "hsm")]
            hsm_provider: None,
            config,
        })
    }

    /// Initialize HSM provider
    ///
    /// # Errors
    ///
    /// Returns error if HSM initialization fails
    #[cfg(feature = "hsm")]
    pub fn initialize_hsm(
        &mut self,
        hsm_provider: Arc<dyn crate::hsm::HsmProvider>,
    ) -> SecureStorageResult<()> {
        self.hsm_provider = Some(hsm_provider);
        info!("HSM provider initialized successfully");
        Ok(())
    }

    /// Generate key using optimal method (HSM or Enclave)
    ///
    /// # Errors
    ///
    /// Returns error if key generation fails
    pub async fn generate_key(
        &self,
        key_id: &KeyId,
        key_size: usize,
    ) -> SecureStorageResult<Vec<u8>> {
        let start = Instant::now();

        // Determine optimal method based on configuration and key size
        let use_hsm =
            self.config.hsm_key_generation && key_size >= self.config.hsm_key_size_threshold;

        let result = if use_hsm {
            match self.generate_key_hsm(key_id, key_size) {
                Ok(key) => Ok(key),
                Err(e) => {
                    if self.config.enclave_fallback {
                        warn!("HSM key generation failed, falling back to enclave: {}", e);
                        self.generate_key_enclave(key_id, key_size).await
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            self.generate_key_enclave(key_id, key_size).await
        };

        let elapsed = start.elapsed();
        debug!("Key generation completed in {:?}", elapsed);

        result
    }

    /// Generate key using HSM
    #[allow(clippy::used_underscore_binding)]
    fn generate_key_hsm(&self, _key_id: &KeyId, key_size: usize) -> SecureStorageResult<Vec<u8>> {
        #[cfg(feature = "hsm")]
        {
            self.hsm_provider.as_ref().map_or_else(
                || {
                    Err(SecureStorageError::InvalidInput {
                        field: "hsm_provider".to_string(),
                        reason: "HSM provider not initialized".to_string(),
                    })
                },
                |_hsm| {
                    // In production, this would use actual HSM key generation
                    debug!("Generating key using HSM");

                    // Placeholder implementation
                    let key_data = vec![0x00AA_u8; key_size / 8]; // Convert bits to bytes
                    Ok(key_data)
                },
            )
        }
        #[cfg(not(feature = "hsm"))]
        {
            let _ = (key_size, &self.config); // Suppress unused warnings
            Err(SecureStorageError::InvalidInput {
                field: "hsm_feature".to_string(),
                reason: "HSM feature not enabled".to_string(),
            })
        }
    }

    /// Generate key using Enclave
    async fn generate_key_enclave(
        &self,
        key_id: &KeyId,
        key_size: usize,
    ) -> SecureStorageResult<Vec<u8>> {
        debug!("Generating key using Secure Enclave");

        let operation_id = format!("keygen_{key_id}");
        let key_size_bytes = key_size / 8;

        let result = self
            .enclave_system
            .execute_secure_operation(EnclaveOperation::KeyGeneration, operation_id, move || {
                // In production, this would use proper cryptographic key generation
                let key_data = vec![0xBB; key_size_bytes];
                Ok(key_data)
            })
            .await?;

        Ok(result.result)
    }

    /// Sign data using optimal method
    ///
    /// # Errors
    ///
    /// Returns error if signing fails
    pub async fn sign_data(&self, key_id: &KeyId, data: &[u8]) -> SecureStorageResult<Vec<u8>> {
        let start = Instant::now();

        let use_hsm = self.config.hsm_signing;

        let result = if use_hsm {
            match self.sign_data_hsm(key_id, data) {
                Ok(signature) => Ok(signature),
                Err(e) => {
                    if self.config.enclave_fallback {
                        warn!("HSM signing failed, falling back to enclave: {}", e);
                        self.sign_data_enclave(key_id, data).await
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            self.sign_data_enclave(key_id, data).await
        };

        let elapsed = start.elapsed();
        debug!("Data signing completed in {:?}", elapsed);

        result
    }

    /// Sign data using HSM
    #[allow(clippy::used_underscore_binding)]
    fn sign_data_hsm(&self, _key_id: &KeyId, _data: &[u8]) -> SecureStorageResult<Vec<u8>> {
        #[cfg(feature = "hsm")]
        {
            self.hsm_provider.as_ref().map_or_else(
                || {
                    Err(SecureStorageError::InvalidInput {
                        field: "hsm_provider".to_string(),
                        reason: "HSM provider not initialized".to_string(),
                    })
                },
                |_hsm| {
                    debug!("Signing data using HSM");

                    // Placeholder implementation
                    let signature = vec![0xCC; 64]; // 64-byte signature
                    Ok(signature)
                },
            )
        }
        #[cfg(not(feature = "hsm"))]
        {
            let _ = (&self.config, _key_id, _data); // Suppress unused warnings
            Err(SecureStorageError::InvalidInput {
                field: "hsm_feature".to_string(),
                reason: "HSM feature not enabled".to_string(),
            })
        }
    }

    /// Sign data using Enclave
    async fn sign_data_enclave(&self, key_id: &KeyId, data: &[u8]) -> SecureStorageResult<Vec<u8>> {
        debug!("Signing data using Secure Enclave");

        let operation_id = format!("sign_{key_id}");
        let data_clone = data.to_vec();

        let result = self
            .enclave_system
            .execute_secure_operation(
                EnclaveOperation::DigitalSignature,
                operation_id,
                move || {
                    // In production, this would use proper cryptographic signing
                    let mut signature = vec![0xDD; 64];
                    // Mix in some data for uniqueness
                    if !data_clone.is_empty() {
                        signature[0] = data_clone[0];
                    }
                    Ok(signature)
                },
            )
            .await?;

        Ok(result.result)
    }

    /// Get integration statistics
    #[must_use]
    pub const fn get_integration_stats(&self) -> HsmIntegrationStats {
        HsmIntegrationStats {
            hsm_available: self.is_hsm_available(),
            enclave_available: true, // Always available if we have the system
            fallback_enabled: self.config.enclave_fallback,
            hsm_operations_total: 0,     // Would track in production
            enclave_operations_total: 0, // Would track in production
        }
    }

    /// Check if HSM is available
    #[must_use]
    pub const fn is_hsm_available(&self) -> bool {
        #[cfg(feature = "hsm")]
        {
            self.hsm_provider.is_some()
        }
        #[cfg(not(feature = "hsm"))]
        {
            false
        }
    }
}

/// HSM integration statistics
#[derive(Debug, Clone)]
pub struct HsmIntegrationStats {
    /// HSM is available
    pub hsm_available: bool,
    /// Enclave is available
    pub enclave_available: bool,
    /// Fallback is enabled
    pub fallback_enabled: bool,
    /// Total HSM operations
    pub hsm_operations_total: u64,
    /// Total enclave operations
    pub enclave_operations_total: u64,
}

/// MPC-Enclave integration manager
#[derive(Debug)]
pub struct MpcEnclaveIntegration {
    /// Enclave system reference
    enclave_system: Arc<SecureEnclaveSystem>,
    /// MPC system reference
    mpc_system: Option<Arc<crate::mpc::MpcSystem>>,
    /// Integration configuration
    config: MpcIntegrationConfig,
}

/// MPC integration configuration
#[derive(Debug, Clone)]
pub struct MpcIntegrationConfig {
    /// Threshold for MPC operations (minimum parties required)
    pub threshold: u32,
    /// Total number of parties
    pub total_parties: u32,
    /// Use MPC for threshold signatures
    pub mpc_threshold_signatures: bool,
    /// Use MPC for distributed key generation
    pub mpc_key_generation: bool,
    /// Maximum MPC operation timeout
    pub mpc_timeout: Duration,
    /// Minimum value threshold for MPC operations
    pub mpc_value_threshold: u64,
}

impl Default for MpcIntegrationConfig {
    fn default() -> Self {
        Self {
            threshold: 3,
            total_parties: 5,
            mpc_threshold_signatures: true,
            mpc_key_generation: true,
            mpc_timeout: Duration::from_millis(500),
            mpc_value_threshold: 1_000_000, // Use MPC for operations > 1M units
        }
    }
}

impl MpcEnclaveIntegration {
    /// Create new MPC-Enclave integration
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(
        enclave_system: Arc<SecureEnclaveSystem>,
        config: MpcIntegrationConfig,
    ) -> SecureStorageResult<Self> {
        info!("Initializing MPC-Enclave integration");

        // Validate MPC configuration
        if config.threshold > config.total_parties {
            return Err(SecureStorageError::InvalidInput {
                field: "mpc_config".to_string(),
                reason: "Threshold cannot exceed total parties".to_string(),
            });
        }

        if config.threshold == 0 {
            return Err(SecureStorageError::InvalidInput {
                field: "mpc_threshold".to_string(),
                reason: "Threshold must be at least 1".to_string(),
            });
        }

        Ok(Self {
            enclave_system,
            mpc_system: None,
            config,
        })
    }

    /// Initialize MPC system
    ///
    /// # Errors
    ///
    /// Returns error if MPC initialization fails
    pub fn initialize_mpc(&mut self) -> SecureStorageResult<()> {
        let threshold_config = crate::mpc::ThresholdConfig {
            threshold: self.config.threshold,
            total_parties: self.config.total_parties,
            timeout: self.config.mpc_timeout,
            proactive_security: true,
            refresh_interval: Duration::from_secs(3600), // 1 hour
        };

        let party_id = crate::mpc::PartyId::new(1); // Would be configurable in production
        let mpc_system = Arc::new(crate::mpc::MpcSystem::new(party_id, threshold_config)?);
        self.mpc_system = Some(mpc_system);
        info!("MPC system initialized successfully");
        Ok(())
    }

    /// Execute threshold signature using MPC + Enclave
    ///
    /// # Errors
    ///
    /// Returns error if threshold signature fails
    pub async fn threshold_sign(
        &self,
        key_id: &KeyId,
        message: &[u8],
        value: u64,
    ) -> SecureStorageResult<Vec<u8>> {
        let start = Instant::now();

        // Determine if MPC should be used based on value threshold
        let use_mpc =
            self.config.mpc_threshold_signatures && value >= self.config.mpc_value_threshold;

        let result = if use_mpc {
            self.threshold_sign_mpc(key_id, message)
        } else {
            self.threshold_sign_enclave(key_id, message).await
        };

        let elapsed = start.elapsed();
        debug!("Threshold signature completed in {:?}", elapsed);

        result
    }

    /// Execute threshold signature using MPC
    fn threshold_sign_mpc(&self, _key_id: &KeyId, _message: &[u8]) -> SecureStorageResult<Vec<u8>> {
        self.mpc_system.as_ref().map_or_else(
            || {
                Err(SecureStorageError::InvalidInput {
                    field: "mpc_system".to_string(),
                    reason: "MPC system not initialized".to_string(),
                })
            },
            |_mpc| {
                debug!("Executing threshold signature using MPC");

                // In production, this would:
                // 1. Coordinate with other MPC parties
                // 2. Execute threshold signature protocol
                // 3. Aggregate partial signatures
                // 4. Verify final signature

                // Placeholder implementation
                let signature = vec![0xEE; 64];
                Ok(signature)
            },
        )
    }

    /// Execute threshold signature using Enclave
    async fn threshold_sign_enclave(
        &self,
        key_id: &KeyId,
        message: &[u8],
    ) -> SecureStorageResult<Vec<u8>> {
        debug!("Executing threshold signature using Secure Enclave");

        let operation_id = format!("threshold_sign_{key_id}");
        let message_clone = message.to_vec();

        let result = self
            .enclave_system
            .execute_secure_operation(
                EnclaveOperation::ThresholdSignature,
                operation_id,
                move || {
                    // In production, this would use proper threshold cryptography
                    let mut signature = vec![0xFF; 64];
                    if !message_clone.is_empty() {
                        signature[0] = message_clone[0];
                    }
                    Ok(signature)
                },
            )
            .await?;

        Ok(result.result)
    }

    /// Get MPC integration statistics
    #[must_use]
    pub const fn get_mpc_stats(&self) -> MpcIntegrationStats {
        MpcIntegrationStats {
            mpc_available: self.mpc_system.is_some(),
            enclave_available: true,
            threshold: self.config.threshold,
            total_parties: self.config.total_parties,
            mpc_operations_total: 0,     // Would track in production
            enclave_operations_total: 0, // Would track in production
        }
    }
}

/// MPC integration statistics
#[derive(Debug, Clone)]
pub struct MpcIntegrationStats {
    /// MPC is available
    pub mpc_available: bool,
    /// Enclave is available
    pub enclave_available: bool,
    /// MPC threshold
    pub threshold: u32,
    /// Total MPC parties
    pub total_parties: u32,
    /// Total MPC operations
    pub mpc_operations_total: u64,
    /// Total enclave operations
    pub enclave_operations_total: u64,
}
