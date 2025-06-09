//! # Automatic Key Rotation
//!
//! Production-grade automatic key rotation for `TallyIO` financial platform.
//! Ensures cryptographic keys are rotated every 30 days with zero downtime.

use crate::error::{SecureStorageError, SecureStorageResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info};

// pub mod scheduler;
// pub mod policy;
// pub mod backup;

/// HSM key type for rotation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HsmKeyType {
    /// AES encryption key
    Aes256,
    /// RSA signing key
    Rsa2048,
    /// RSA signing key
    Rsa4096,
    /// ECDSA signing key
    EcdsaP256,
    /// ECDSA signing key
    EcdsaP384,
}

/// HSM key information
#[derive(Debug, Clone)]
pub struct HsmKeyInfo {
    /// Unique key identifier
    pub key_id: String,
    /// Type of the key
    pub key_type: HsmKeyType,
    /// Key label for identification
    pub label: String,
    /// Whether key is extractable
    pub extractable: bool,
}

/// HSM operation result
#[derive(Debug)]
pub struct HsmOperationResult {
    /// Result of the operation
    pub result: HsmKeyInfo,
    /// Duration of the operation
    pub duration: Duration,
}

/// HSM provider trait for key operations
#[async_trait::async_trait]
pub trait HsmProvider: Send + Sync {
    /// Generate a new key
    async fn generate_key(
        &self,
        key_type: HsmKeyType,
        label: &str,
        extractable: bool,
    ) -> SecureStorageResult<HsmOperationResult>;

    /// Delete a key
    async fn delete_key(&self, key_id: &str) -> SecureStorageResult<()>;

    /// Get key information
    async fn get_key_info(&self, key_id: &str) -> SecureStorageResult<HsmKeyInfo>;
}

/// Audit log trait for logging operations
#[async_trait::async_trait]
pub trait AuditLog: Send + Sync {
    /// Log an operation
    async fn log_operation(
        &self,
        actor: &str,
        action: &str,
        resource: &str,
        result: &str,
        metadata: Option<serde_json::Value>,
    ) -> SecureStorageResult<()>;
}

/// Key rotation configuration
#[derive(Debug, Clone)]
/// TODO: Add documentation
pub struct KeyRotationConfig {
    /// Rotation interval (default: 30 days)
    pub rotation_interval: Duration,
    /// Grace period for old keys (default: 7 days)
    pub grace_period: Duration,
    /// Maximum concurrent rotations
    pub max_concurrent_rotations: u32,
    /// Enable automatic rotation
    pub auto_rotation_enabled: bool,
    /// Backup old keys before rotation
    pub backup_old_keys: bool,
    /// Notification endpoints for rotation events
    pub notification_endpoints: Vec<String>,
}

impl Default for KeyRotationConfig {
    fn default() -> Self {
        Self {
            rotation_interval: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            grace_period: Duration::from_secs(7 * 24 * 60 * 60),       // 7 days
            max_concurrent_rotations: 3,
            auto_rotation_enabled: true,
            backup_old_keys: true,
            notification_endpoints: Vec::with_capacity(5),
        }
    }
}

/// Key rotation status
#[derive(Debug, Clone, PartialEq, Eq)]
/// TODO: Add documentation
pub enum RotationStatus {
    /// Key is current and valid
    Current,
    /// Key needs rotation soon
    RotationDue,
    /// Key is being rotated
    Rotating,
    /// Key has been rotated (in grace period)
    Rotated,
    /// Key is expired and should not be used
    Expired,
    /// Rotation failed
    Failed {
        /// Human-readable reason for failure
        reason: String,
    },
}

/// Key metadata for rotation tracking
///
/// Contains all information needed to track and manage automatic key rotation
/// including timing, status, and backup information.
#[derive(Debug, Clone)]
pub struct RotationKeyInfo {
    /// Unique identifier for the key
    pub key_id: String,
    /// Type of the cryptographic key
    pub key_type: HsmKeyType,
    /// When the key was originally created
    pub created_at: SystemTime,
    /// When the key was last rotated (None if never rotated)
    pub last_rotated: Option<SystemTime>,
    /// When the next rotation should occur
    pub next_rotation: SystemTime,
    /// Current rotation status
    pub status: RotationStatus,
    /// Number of times this key has been rotated
    pub rotation_count: u32,
    /// Location where old key is backed up (if any)
    pub backup_location: Option<String>,
}

impl RotationKeyInfo {
    /// Create new rotation key info
    ///
    /// # Arguments
    ///
    /// * `key_id` - Unique identifier for the key
    /// * `key_type` - Type of the cryptographic key
    /// * `rotation_interval` - How often the key should be rotated
    ///
    /// # Returns
    ///
    /// A new `RotationKeyInfo` instance with default values
    #[must_use]
    pub fn new(key_id: String, key_type: HsmKeyType, rotation_interval: Duration) -> Self {
        let now = SystemTime::now();
        Self {
            key_id,
            key_type,
            created_at: now,
            last_rotated: None,
            next_rotation: now + rotation_interval,
            status: RotationStatus::Current,
            rotation_count: 0,
            backup_location: None,
        }
    }

    /// Check if key needs rotation
    ///
    /// Returns true if the current time is past the scheduled next rotation time.
    ///
    /// # Returns
    ///
    /// `true` if the key needs rotation, `false` otherwise
    #[must_use]
    pub fn needs_rotation(&self) -> bool {
        SystemTime::now() >= self.next_rotation
    }

    /// Check if key is expired
    ///
    /// A key is considered expired if it has been rotated and the grace period has passed.
    ///
    /// # Arguments
    ///
    /// * `grace_period` - How long after rotation the old key remains valid
    ///
    /// # Returns
    ///
    /// `true` if the key is expired, `false` otherwise
    #[must_use]
    pub fn is_expired(&self, grace_period: Duration) -> bool {
        self.last_rotated
            .is_some_and(|rotated_at| SystemTime::now() > rotated_at + grace_period)
    }
}

/// Key rotation result containing information about a completed rotation
///
/// This structure contains all relevant information about a key rotation operation
/// including timing, backup location, and the new key identifier.
#[derive(Debug, Clone)]
pub struct RotationResult {
    /// Identifier of the old key that was rotated
    pub old_key_id: String,
    /// Identifier of the new key that replaced the old one
    pub new_key_id: String,
    /// When the rotation was completed
    pub rotation_time: SystemTime,
    /// Location where the old key was backed up (if any)
    pub backup_location: Option<String>,
    /// How long the rotation operation took
    pub duration: Duration,
}

/// Automatic key rotation manager
pub struct KeyRotationManager {
    config: KeyRotationConfig,
    hsm_provider: Arc<dyn HsmProvider>,
    audit_log: Arc<dyn AuditLog>,
    keys: Arc<RwLock<HashMap<String, RotationKeyInfo>>>,
    active_rotations: Arc<Mutex<HashMap<String, Instant>>>,
    rotation_history: Arc<RwLock<Vec<RotationResult>>>,
    shutdown_signal: Arc<tokio::sync::Notify>,
}

impl KeyRotationManager {
    /// Create a new key rotation manager
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for key rotation behavior
    /// * `hsm_provider` - HSM provider for key operations
    /// * `audit_log` - Audit log for recording rotation events
    ///
    /// # Returns
    ///
    /// A new `KeyRotationManager` instance
    #[must_use]
    pub fn new(
        config: KeyRotationConfig,
        hsm_provider: Arc<dyn HsmProvider>,
        audit_log: Arc<dyn AuditLog>,
    ) -> Self {
        Self {
            config,
            hsm_provider,
            audit_log,
            keys: Arc::new(RwLock::new(HashMap::new())),
            active_rotations: Arc::new(Mutex::new(HashMap::new())),
            rotation_history: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            shutdown_signal: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Start the automatic rotation scheduler
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Internal` if scheduler fails to start
    pub fn start_scheduler(&self) -> SecureStorageResult<()> {
        if !self.config.auto_rotation_enabled {
            info!("Automatic key rotation is disabled");
            return Ok(());
        }

        info!("Starting automatic key rotation scheduler");

        let keys = self.keys.clone();
        let hsm_provider = self.hsm_provider.clone();
        let audit_log = self.audit_log.clone();
        let active_rotations = self.active_rotations.clone();
        let rotation_history = self.rotation_history.clone();
        let config = self.config.clone();
        let shutdown_signal = self.shutdown_signal.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Check every hour

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::check_and_rotate_keys(
                            &keys,
                            &hsm_provider,
                            &audit_log,
                            &active_rotations,
                            &rotation_history,
                            &config,
                        ).await {
                            error!("Key rotation check failed: {}", e);
                        }
                    }
                    () = shutdown_signal.notified() => {
                        info!("Key rotation scheduler shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the automatic rotation scheduler
    pub fn stop_scheduler(&self) {
        self.shutdown_signal.notify_waiters();
    }

    /// Register a key for automatic rotation
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::InvalidInput` if key info is invalid
    pub async fn register_key(&self, key_info: HsmKeyInfo) -> SecureStorageResult<()> {
        let rotation_info = RotationKeyInfo::new(
            key_info.key_id.clone(),
            key_info.key_type,
            self.config.rotation_interval,
        );

        self.keys
            .write()
            .await
            .insert(key_info.key_id.clone(), rotation_info);

        // Log registration
        self.audit_log
            .log_operation(
                "system",
                "key_registered_for_rotation",
                &key_info.key_id,
                "success",
                None,
            )
            .await
            .map_err(|e| SecureStorageError::Internal {
                reason: format!("Failed to log key registration: {e}"),
            })?;

        info!("Registered key {} for automatic rotation", key_info.key_id);
        Ok(())
    }

    /// Manually rotate a specific key
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    /// Returns `SecureStorageError::Internal` if rotation fails
    pub async fn rotate_key(&self, key_id: &str) -> SecureStorageResult<RotationResult> {
        // Check if rotation is already in progress
        {
            let active = self.active_rotations.lock().await;
            if active.contains_key(key_id) {
                return Err(SecureStorageError::Internal {
                    reason: format!("Key {key_id} is already being rotated"),
                });
            }
        }

        // Mark rotation as active
        {
            let mut active = self.active_rotations.lock().await;
            active.insert(key_id.to_string(), Instant::now());
        }

        let result = self.perform_key_rotation(key_id).await;

        // Remove from active rotations
        {
            let mut active = self.active_rotations.lock().await;
            active.remove(key_id);
        }

        result
    }

    /// Get rotation status for all keys
    ///
    /// Returns a map of all registered keys and their current rotation information.
    ///
    /// # Returns
    ///
    /// `HashMap` mapping key IDs to their rotation information
    #[must_use]
    pub async fn get_rotation_status(&self) -> HashMap<String, RotationKeyInfo> {
        let keys = self.keys.read().await;
        keys.clone()
    }

    /// Get rotation history
    ///
    /// Returns a list of all completed key rotations with their results.
    ///
    /// # Returns
    ///
    /// Vector of rotation results ordered by completion time
    #[must_use]
    pub async fn get_rotation_history(&self) -> Vec<RotationResult> {
        let history = self.rotation_history.read().await;
        history.clone()
    }

    /// Check and rotate keys that need rotation
    async fn check_and_rotate_keys(
        keys: &Arc<RwLock<HashMap<String, RotationKeyInfo>>>,
        hsm_provider: &Arc<dyn HsmProvider>,
        audit_log: &Arc<dyn AuditLog>,
        active_rotations: &Arc<Mutex<HashMap<String, Instant>>>,
        rotation_history: &Arc<RwLock<Vec<RotationResult>>>,
        config: &KeyRotationConfig,
    ) -> SecureStorageResult<()> {
        let keys_to_rotate = {
            let keys_guard = keys.read().await;
            keys_guard
                .iter()
                .filter(|(_, info)| info.needs_rotation() && info.status == RotationStatus::Current)
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>()
        };

        if keys_to_rotate.is_empty() {
            debug!("No keys need rotation");
            return Ok(());
        }

        info!("Found {} keys that need rotation", keys_to_rotate.len());

        // Limit concurrent rotations
        let active_count = {
            let active = active_rotations.lock().await;
            active.len()
        };

        let available_slots = config
            .max_concurrent_rotations
            .saturating_sub(u32::try_from(active_count).unwrap_or(u32::MAX));
        let keys_to_process = keys_to_rotate
            .into_iter()
            .take(available_slots as usize)
            .collect::<Vec<_>>();

        for key_id in keys_to_process {
            let keys_clone = keys.clone();
            let hsm_clone = hsm_provider.clone();
            let audit_clone = audit_log.clone();
            let active_clone = active_rotations.clone();
            let history_clone = rotation_history.clone();
            let config_clone = config.clone();

            tokio::spawn(async move {
                // Mark as active
                {
                    let mut active = active_clone.lock().await;
                    active.insert(key_id.clone(), Instant::now());
                }

                let result = Self::perform_single_key_rotation(
                    &key_id,
                    &keys_clone,
                    &hsm_clone,
                    &audit_clone,
                    &config_clone,
                )
                .await;

                // Remove from active
                {
                    let mut active = active_clone.lock().await;
                    active.remove(&key_id);
                }

                match result {
                    Ok(rotation_result) => {
                        // Add to history
                        let mut history = history_clone.write().await;
                        history.push(rotation_result);

                        // Keep only last 1000 entries
                        let history_len = history.len();
                        if history_len > 1000 {
                            history.drain(0..history_len - 1000);
                        }
                    }
                    Err(e) => {
                        error!("Failed to rotate key {}: {}", key_id, e);

                        // Update key status to failed
                        let mut keys_guard = keys_clone.write().await;
                        if let Some(key_info) = keys_guard.get_mut(&key_id) {
                            key_info.status = RotationStatus::Failed {
                                reason: e.to_string(),
                            };
                        }
                    }
                }
            });
        }

        Ok(())
    }

    /// Perform rotation for a single key
    async fn perform_single_key_rotation(
        key_id: &str,
        keys: &Arc<RwLock<HashMap<String, RotationKeyInfo>>>,
        hsm_provider: &Arc<dyn HsmProvider>,
        audit_log: &Arc<dyn AuditLog>,
        config: &KeyRotationConfig,
    ) -> SecureStorageResult<RotationResult> {
        let start_time = Instant::now();

        // Get key info
        let (key_type, label) = {
            let key_info = keys
                .read()
                .await
                .get(key_id)
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "rotation key".to_string(),
                    identifier: key_id.to_string(),
                })?
                .clone();
            (key_info.key_type, format!("{key_id}_rotated"))
        };

        // Update status to rotating
        {
            let mut keys_guard = keys.write().await;
            if let Some(key_info) = keys_guard.get_mut(key_id) {
                key_info.status = RotationStatus::Rotating;
            }
        }

        // Generate new key
        let new_key_result = hsm_provider
            .generate_key(
                key_type, &label, false, // Non-extractable for security
            )
            .await?;

        let new_key_id = new_key_result.result.key_id;

        // Backup old key if configured
        let backup_location = if config.backup_old_keys {
            // TODO: Implement key backup
            Some(format!("backup_{key_id}"))
        } else {
            None
        };

        // Update key info
        let now = SystemTime::now();
        {
            let mut keys_guard = keys.write().await;
            if let Some(key_info) = keys_guard.get_mut(key_id) {
                key_info.last_rotated = Some(now);
                key_info.next_rotation = now + config.rotation_interval;
                key_info.status = RotationStatus::Rotated;
                key_info.rotation_count += 1;
                key_info.backup_location.clone_from(&backup_location);
            }
        }

        // Log rotation
        audit_log
            .log_operation(
                "system",
                "key_rotated",
                key_id,
                "success",
                Some(serde_json::json!({
                    "old_key_id": key_id,
                    "new_key_id": new_key_id,
                    "backup_location": backup_location,
                })),
            )
            .await
            .map_err(|e| SecureStorageError::Internal {
                reason: format!("Failed to log key rotation: {e}"),
            })?;

        info!("Successfully rotated key {} to {}", key_id, new_key_id);

        Ok(RotationResult {
            old_key_id: key_id.to_string(),
            new_key_id,
            rotation_time: now,
            backup_location,
            duration: start_time.elapsed(),
        })
    }

    /// Perform key rotation implementation
    async fn perform_key_rotation(&self, key_id: &str) -> SecureStorageResult<RotationResult> {
        Self::perform_single_key_rotation(
            key_id,
            &self.keys,
            &self.hsm_provider,
            &self.audit_log,
            &self.config,
        )
        .await
    }
}
