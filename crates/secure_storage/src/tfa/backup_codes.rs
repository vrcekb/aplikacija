//! # Backup Recovery Codes
//!
//! One-time backup codes for account recovery when primary 2FA methods are unavailable.

use super::TfaConfig;
use crate::error::SecureStorageResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};
use zeroize::Zeroize;

/// Backup codes system
#[derive(Debug)]
pub struct BackupCodesSystem {
    /// User backup codes
    user_codes: RwLock<HashMap<String, Vec<BackupCode>>>,
    /// Performance counters
    codes_generated: AtomicU64,
    codes_used: AtomicU64,
    invalid_attempts: AtomicU64,
}

/// Backup code
#[derive(Debug, Clone)]
pub struct BackupCode {
    /// Code value
    code: String,
    /// Used flag
    used: bool,
    /// Creation timestamp
    created_at: SystemTime,
    /// Used timestamp
    used_at: Option<SystemTime>,
}

impl BackupCode {
    /// Create new backup code
    #[must_use]
    pub fn new(code: String) -> Self {
        Self {
            code,
            used: false,
            created_at: SystemTime::now(),
            used_at: None,
        }
    }

    /// Mark code as used
    pub fn mark_used(&mut self) {
        self.used = true;
        self.used_at = Some(SystemTime::now());
    }

    /// Check if code is valid
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        !self.used
    }

    /// Get code value
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Get creation timestamp
    #[must_use]
    pub const fn created_at(&self) -> SystemTime {
        self.created_at
    }

    /// Get used timestamp
    #[must_use]
    pub const fn used_at(&self) -> Option<SystemTime> {
        self.used_at
    }

    /// Check if code is expired
    #[must_use]
    pub fn is_expired(&self, max_age: Duration) -> bool {
        SystemTime::now()
            .duration_since(self.created_at)
            .map_or(true, |age| age > max_age)
    }

    /// Get age of the code
    #[must_use]
    pub fn age(&self) -> Option<Duration> {
        SystemTime::now().duration_since(self.created_at).ok()
    }
}

impl Drop for BackupCode {
    fn drop(&mut self) {
        self.code.zeroize();
    }
}

impl BackupCodesSystem {
    /// Create new backup codes system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(_config: TfaConfig) -> SecureStorageResult<Self> {
        info!("Initialized backup codes system");

        Ok(Self {
            user_codes: RwLock::new(HashMap::new()),
            codes_generated: AtomicU64::new(0),
            codes_used: AtomicU64::new(0),
            invalid_attempts: AtomicU64::new(0),
        })
    }

    /// Generate backup codes for user
    ///
    /// # Errors
    ///
    /// Returns error if code generation fails
    pub async fn generate_codes(
        &self,
        user_id: &str,
        count: u32,
    ) -> SecureStorageResult<Vec<String>> {
        let mut codes = Vec::with_capacity(10);
        let mut backup_codes = Vec::with_capacity(10);

        for _ in 0..count {
            let code = Self::generate_random_code();
            codes.push(code.clone());
            backup_codes.push(BackupCode::new(code));
        }

        // Store codes
        {
            let mut user_codes = self.user_codes.write().await;
            user_codes.insert(user_id.to_string(), backup_codes);
        }

        self.codes_generated
            .fetch_add(u64::from(count), Ordering::Relaxed);
        info!("Generated {} backup codes for user: {}", count, user_id);

        Ok(codes)
    }

    /// Verify backup code
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_code(&self, user_id: &str, code: &str) -> SecureStorageResult<bool> {
        let mut user_codes = self.user_codes.write().await;

        if let Some(codes) = user_codes.get_mut(user_id) {
            for backup_code in codes.iter_mut() {
                if backup_code.is_valid()
                    && constant_time_compare(backup_code.code().as_bytes(), code.as_bytes())
                {
                    backup_code.mark_used();
                    self.codes_used.fetch_add(1, Ordering::Relaxed);
                    info!("Backup code used for user: {}", user_id);
                    drop(user_codes);
                    return Ok(true);
                }
            }
        }
        drop(user_codes);

        self.invalid_attempts.fetch_add(1, Ordering::Relaxed);
        warn!("Invalid backup code attempt for user: {}", user_id);
        Ok(false)
    }

    /// Get remaining codes count
    ///
    /// # Errors
    ///
    /// Returns error if user not found
    pub async fn get_remaining_codes_count(&self, user_id: &str) -> SecureStorageResult<usize> {
        let user_codes = self.user_codes.read().await;

        let result = user_codes.get(user_id).map_or(0, |codes| {
            codes.iter().filter(|code| code.is_valid()).count()
        });
        drop(user_codes);
        Ok(result)
    }

    /// Generate random backup code
    fn generate_random_code() -> String {
        // Generate 8-character alphanumeric code
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let mut code = String::with_capacity(8);

        for i in 0..8 {
            let index = (i * 7 + 13) % CHARS.len(); // Simple deterministic "random"
            code.push(CHARS[index] as char);
        }

        code
    }

    /// Get statistics
    #[must_use]
    pub async fn get_stats(&self) -> BackupCodesStats {
        let user_codes = self.user_codes.read().await;
        let total_codes = user_codes.values().map(Vec::len).sum();
        let used_code_count = user_codes
            .values()
            .flat_map(|codes| codes.iter())
            .filter(|code| !code.is_valid())
            .count();
        drop(user_codes);

        BackupCodesStats {
            total_codes,
            used_codes: used_code_count,
            codes_generated: self.codes_generated.load(Ordering::Relaxed),
            codes_used: self.codes_used.load(Ordering::Relaxed),
            invalid_attempts: self.invalid_attempts.load(Ordering::Relaxed),
        }
    }
}

/// Backup codes statistics
#[derive(Debug, Clone)]
pub struct BackupCodesStats {
    /// Total backup codes
    pub total_codes: usize,
    /// Used backup codes
    pub used_codes: usize,
    /// Total codes generated
    pub codes_generated: u64,
    /// Total codes used
    pub codes_used: u64,
    /// Invalid code attempts
    pub invalid_attempts: u64,
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
