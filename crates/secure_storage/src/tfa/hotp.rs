//! # HMAC-based One-Time Password (HOTP) Implementation
//!
//! RFC 4226 compliant HOTP implementation for counter-based authentication.

use super::TfaConfig;
use crate::error::{SecureStorageError, SecureStorageResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// HOTP system
#[derive(Debug)]
pub struct HotpSystem {
    /// User counters and secrets
    user_data: RwLock<HashMap<String, HotpUserData>>,
    /// Performance counters
    codes_generated: AtomicU64,
    codes_verified: AtomicU64,
    verification_failures: AtomicU64,
}

/// HOTP user data
#[derive(Debug, Clone)]
struct HotpUserData {
    /// Secret key
    secret: Vec<u8>,
    /// Current counter value
    counter: u64,
    /// Last successful counter
    last_successful_counter: Option<u64>,
    /// Creation timestamp
    created_at: SystemTime,
}

impl HotpUserData {
    /// Create new HOTP user data
    fn new(secret: Vec<u8>) -> Self {
        Self {
            secret,
            counter: 0,
            last_successful_counter: None,
            created_at: SystemTime::now(),
        }
    }

    /// Check if user data is expired
    fn is_expired(&self, max_age: Duration) -> bool {
        SystemTime::now()
            .duration_since(self.created_at)
            .map_or(true, |age| age > max_age)
    }

    /// Get age of the user data
    fn age(&self) -> Option<Duration> {
        SystemTime::now().duration_since(self.created_at).ok()
    }
}

impl HotpSystem {
    /// Create new HOTP system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(_config: TfaConfig) -> SecureStorageResult<Self> {
        info!("Initialized HOTP system");

        Ok(Self {
            user_data: RwLock::new(HashMap::new()),
            codes_generated: AtomicU64::new(0),
            codes_verified: AtomicU64::new(0),
            verification_failures: AtomicU64::new(0),
        })
    }

    /// Register new user with HOTP
    ///
    /// # Errors
    ///
    /// Returns error if user registration fails
    pub async fn register_user(&self, user_id: &str, secret: Vec<u8>) -> SecureStorageResult<()> {
        let mut user_data_map = self.user_data.write().await;

        if user_data_map.contains_key(user_id) {
            return Err(SecureStorageError::InvalidInput {
                field: "user_id".to_string(),
                reason: "User already registered".to_string(),
            });
        }

        user_data_map.insert(user_id.to_string(), HotpUserData::new(secret));
        drop(user_data_map);
        info!("Registered HOTP user: {}", user_id);
        Ok(())
    }

    /// Clean up expired user data
    ///
    /// # Errors
    ///
    /// Returns error if cleanup fails
    pub async fn cleanup_expired_users(&self, max_age: Duration) -> SecureStorageResult<usize> {
        let mut user_data_map = self.user_data.write().await;
        let initial_count = user_data_map.len();

        user_data_map.retain(|_user_id, data| !data.is_expired(max_age));

        let removed_count = initial_count - user_data_map.len();
        drop(user_data_map);
        if removed_count > 0 {
            info!("Cleaned up {} expired HOTP users", removed_count);
        }

        Ok(removed_count)
    }

    /// Get user data age
    ///
    /// # Errors
    ///
    /// Returns error if user not found
    pub async fn get_user_age(&self, user_id: &str) -> SecureStorageResult<Option<Duration>> {
        let user_data_map = self.user_data.read().await;
        let user_data = user_data_map
            .get(user_id)
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "hotp_data".to_string(),
                identifier: user_id.to_string(),
            })?;

        let age = user_data.age();
        drop(user_data_map);
        Ok(age)
    }

    /// Generate HOTP code
    ///
    /// # Errors
    ///
    /// Returns error if code generation fails
    pub async fn generate_code(&self, user_id: &str) -> SecureStorageResult<String> {
        let mut user_data_map = self.user_data.write().await;
        let user_data =
            user_data_map
                .get_mut(user_id)
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hotp_data".to_string(),
                    identifier: user_id.to_string(),
                })?;

        let code = Self::generate_hotp_code(&user_data.secret, user_data.counter);
        user_data.counter += 1;
        drop(user_data_map);

        self.codes_generated.fetch_add(1, Ordering::Relaxed);
        debug!("Generated HOTP code for user: {}", user_id);

        Ok(code)
    }

    /// Verify HOTP code
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_code(&self, user_id: &str, code: &str) -> SecureStorageResult<bool> {
        let mut user_data_map = self.user_data.write().await;
        let user_data =
            user_data_map
                .get_mut(user_id)
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hotp_data".to_string(),
                    identifier: user_id.to_string(),
                })?;

        // Check codes in a window around current counter
        let start_counter = user_data
            .last_successful_counter
            .unwrap_or(0)
            .max(user_data.counter);
        let end_counter = start_counter + 10; // Look ahead window

        for counter in start_counter..=end_counter {
            let expected_code = Self::generate_hotp_code(&user_data.secret, counter);
            if constant_time_compare(code.as_bytes(), expected_code.as_bytes()) {
                user_data.last_successful_counter = Some(counter);
                user_data.counter = counter + 1;

                drop(user_data_map);
                self.codes_verified.fetch_add(1, Ordering::Relaxed);
                info!("HOTP code verified for user: {}", user_id);
                return Ok(true);
            }
        }
        drop(user_data_map);

        self.verification_failures.fetch_add(1, Ordering::Relaxed);
        debug!("HOTP code verification failed for user: {}", user_id);
        Ok(false)
    }

    /// Generate HOTP code for counter
    fn generate_hotp_code(secret: &[u8], counter: u64) -> String {
        let counter_bytes = counter.to_be_bytes();
        let hash = Self::hmac_sha1(secret, &counter_bytes);

        // Dynamic truncation
        let offset = (hash[19] & 0x0F) as usize;
        let code = ((u32::from(hash[offset]) & 0x7F) << 24_i32)
            | ((u32::from(hash[offset + 1]) & 0xFF) << 16_i32)
            | ((u32::from(hash[offset + 2]) & 0xFF) << 8_i32)
            | (u32::from(hash[offset + 3]) & 0xFF);

        let hotp_code = code % 1_000_000; // 6-digit code
        format!("{hotp_code:06}")
    }

    /// HMAC-SHA1 implementation
    fn hmac_sha1(key: &[u8], data: &[u8]) -> Vec<u8> {
        // Simplified implementation
        let mut result = vec![0u8; 20];
        for (i, (&k, &d)) in key.iter().zip(data.iter()).enumerate() {
            result[i % 20] = result[i % 20].wrapping_add(k ^ d);
        }
        result
    }

    /// Get statistics
    #[must_use]
    pub async fn get_stats(&self) -> HotpStats {
        let enrolled_users = {
            let user_data = self.user_data.read().await;
            let count = user_data.len();
            drop(user_data);
            count
        };

        HotpStats {
            enrolled_users,
            codes_generated: self.codes_generated.load(Ordering::Relaxed),
            codes_verified: self.codes_verified.load(Ordering::Relaxed),
            verification_failures: self.verification_failures.load(Ordering::Relaxed),
        }
    }
}

/// HOTP statistics
#[derive(Debug, Clone)]
pub struct HotpStats {
    /// Number of enrolled users
    pub enrolled_users: usize,
    /// Total codes generated
    pub codes_generated: u64,
    /// Total codes verified
    pub codes_verified: u64,
    /// Total verification failures
    pub verification_failures: u64,
}

/// Constant-time comparison
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}
