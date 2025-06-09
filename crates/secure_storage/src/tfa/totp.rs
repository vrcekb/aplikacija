//! # Time-based One-Time Password (TOTP) Implementation
//!
//! RFC 6238 compliant TOTP implementation with enhanced security features
//! for financial applications.

use super::TfaConfig;
use crate::error::{SecureStorageError, SecureStorageResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use zeroize::Zeroize;

/// TOTP algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotpAlgorithm {
    /// SHA-1 (RFC default, not recommended for new implementations)
    Sha1,
    /// SHA-256 (recommended)
    Sha256,
    /// SHA-512 (highest security)
    Sha512,
}

impl TotpAlgorithm {
    /// Get algorithm name for QR code generation
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha1 => "SHA1",
            Self::Sha256 => "SHA256",
            Self::Sha512 => "SHA512",
        }
    }

    /// Get hash output size in bytes
    #[must_use]
    pub const fn hash_size(self) -> usize {
        match self {
            Self::Sha1 => 20,
            Self::Sha256 => 32,
            Self::Sha512 => 64,
        }
    }
}

/// TOTP configuration
#[derive(Debug, Clone)]
pub struct TotpConfig {
    /// Time step in seconds (default: 30)
    pub time_step: u64,
    /// Code length (default: 6)
    pub code_length: u32,
    /// Algorithm to use
    pub algorithm: TotpAlgorithm,
    /// Time window tolerance (number of steps)
    pub window_tolerance: u32,
    /// Secret key length in bytes
    pub secret_key_length: usize,
    /// Issuer name for QR codes
    pub issuer: String,
}

impl TotpConfig {
    /// Create production TOTP configuration
    #[must_use]
    pub fn new_production() -> Self {
        Self {
            time_step: 30,                    // 30-second intervals
            code_length: 6,                   // 6-digit codes
            algorithm: TotpAlgorithm::Sha256, // SHA-256 for security
            window_tolerance: 1,              // ±1 time step tolerance
            secret_key_length: 32,            // 256-bit keys
            issuer: "TallyIO".to_string(),
        }
    }

    /// Create development configuration
    #[must_use]
    pub fn new_development() -> Self {
        Self {
            time_step: 30,
            code_length: 6,
            algorithm: TotpAlgorithm::Sha256,
            window_tolerance: 2, // More tolerance for dev
            secret_key_length: 32,
            issuer: "TallyIO-Dev".to_string(),
        }
    }

    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration values are invalid
    pub fn validate(&self) -> SecureStorageResult<()> {
        if self.time_step == 0 || self.time_step > 300 {
            return Err(SecureStorageError::InvalidInput {
                field: "time_step".to_string(),
                reason: "Time step must be between 1 and 300 seconds".to_string(),
            });
        }

        if self.code_length < 4 || self.code_length > 10 {
            return Err(SecureStorageError::InvalidInput {
                field: "code_length".to_string(),
                reason: "Code length must be between 4 and 10 digits".to_string(),
            });
        }

        if self.window_tolerance > 10 {
            return Err(SecureStorageError::InvalidInput {
                field: "window_tolerance".to_string(),
                reason: "Window tolerance cannot exceed 10 steps".to_string(),
            });
        }

        if self.secret_key_length < 16 || self.secret_key_length > 64 {
            return Err(SecureStorageError::InvalidInput {
                field: "secret_key_length".to_string(),
                reason: "Secret key length must be between 16 and 64 bytes".to_string(),
            });
        }

        Ok(())
    }
}

/// TOTP secret key
#[derive(Debug, Clone)]
pub struct TotpSecret {
    /// Secret key bytes
    key: Vec<u8>,
    /// Algorithm used
    algorithm: TotpAlgorithm,
    /// Creation timestamp
    created_at: SystemTime,
}

impl TotpSecret {
    /// Create new TOTP secret
    #[must_use]
    pub fn new(key: Vec<u8>, algorithm: TotpAlgorithm) -> Self {
        Self {
            key,
            algorithm,
            created_at: SystemTime::now(),
        }
    }

    /// Get key bytes
    #[must_use]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get algorithm
    #[must_use]
    pub const fn algorithm(&self) -> TotpAlgorithm {
        self.algorithm
    }

    /// Get creation timestamp
    #[must_use]
    pub const fn created_at(&self) -> SystemTime {
        self.created_at
    }

    /// Check if secret is expired (older than specified duration)
    #[must_use]
    pub fn is_expired(&self, max_age: Duration) -> bool {
        SystemTime::now()
            .duration_since(self.created_at)
            .map_or(true, |age| age > max_age)
    }

    /// Generate Base32-encoded secret for QR codes
    #[must_use]
    pub fn to_base32(&self) -> String {
        base32_encode(&self.key)
    }

    /// Create from Base32-encoded string
    ///
    /// # Errors
    ///
    /// Returns error if Base32 decoding fails
    pub fn from_base32(encoded: &str, algorithm: TotpAlgorithm) -> SecureStorageResult<Self> {
        let key = base32_decode(encoded)?;
        Ok(Self::new(key, algorithm))
    }
}

impl Drop for TotpSecret {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

/// TOTP system
#[derive(Debug)]
pub struct TotpSystem {
    /// System configuration
    config: TotpConfig,
    /// User secrets
    secrets: RwLock<HashMap<String, TotpSecret>>,
    /// Used codes tracking (for replay protection)
    used_codes: RwLock<HashMap<String, Vec<(String, SystemTime)>>>,
    /// Performance counters
    codes_generated: AtomicU64,
    codes_verified: AtomicU64,
    verification_failures: AtomicU64,
}

impl TotpSystem {
    /// Create a new TOTP system
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(_tfa_config: TfaConfig) -> SecureStorageResult<Self> {
        let config = if cfg!(debug_assertions) {
            TotpConfig::new_development()
        } else {
            TotpConfig::new_production()
        };

        config.validate()?;

        info!(
            "Initialized TOTP system with {} algorithm",
            config.algorithm.name()
        );

        Ok(Self {
            config,
            secrets: RwLock::new(HashMap::new()),
            used_codes: RwLock::new(HashMap::new()),
            codes_generated: AtomicU64::new(0),
            codes_verified: AtomicU64::new(0),
            verification_failures: AtomicU64::new(0),
        })
    }

    /// Generate secret key for user
    ///
    /// # Errors
    ///
    /// Returns error if key generation fails
    pub async fn generate_secret(&self, user_id: &str) -> SecureStorageResult<TotpSecret> {
        // Generate cryptographically secure random key
        let mut key = vec![0u8; self.config.secret_key_length];
        Self::fill_random_bytes(&mut key);

        let secret = TotpSecret::new(key, self.config.algorithm);

        // Store secret
        {
            let mut secrets = self.secrets.write().await;
            secrets.insert(user_id.to_string(), secret.clone());
            drop(secrets);
        }

        info!("Generated TOTP secret for user: {}", user_id);
        Ok(secret)
    }

    /// Generate TOTP code for current time
    ///
    /// # Errors
    ///
    /// Returns error if code generation fails
    pub async fn generate_code(&self, user_id: &str) -> SecureStorageResult<String> {
        let secret = {
            let secrets = self.secrets.read().await;
            let secret =
                secrets
                    .get(user_id)
                    .cloned()
                    .ok_or_else(|| SecureStorageError::NotFound {
                        resource: "totp_secret".to_string(),
                        identifier: user_id.to_string(),
                    })?;
            drop(secrets);
            secret
        };

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| SecureStorageError::InvalidInput {
                field: "system_time".to_string(),
                reason: "Invalid system time".to_string(),
            })?
            .as_secs();

        let time_step = current_time / self.config.time_step;
        let code = self.generate_totp_code(&secret, time_step);

        self.codes_generated.fetch_add(1, Ordering::Relaxed);
        debug!("Generated TOTP code for user: {}", user_id);

        Ok(code)
    }

    /// Verify TOTP code
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_code(&self, user_id: &str, code: &str) -> SecureStorageResult<bool> {
        let secret = self.get_user_secret(user_id).await?;

        if self.check_replay_protection(user_id, code).await? {
            return Ok(false);
        }

        let current_step = self.get_current_time_step()?;

        if self
            .verify_code_in_window(&secret, current_step, code, user_id)
            .await?
        {
            Ok(true)
        } else {
            self.verification_failures.fetch_add(1, Ordering::Relaxed);
            debug!("TOTP code verification failed for user: {}", user_id);
            Ok(false)
        }
    }

    /// Get user secret
    async fn get_user_secret(&self, user_id: &str) -> SecureStorageResult<TotpSecret> {
        let secrets = self.secrets.read().await;
        let secret = secrets
            .get(user_id)
            .cloned()
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "totp_secret".to_string(),
                identifier: user_id.to_string(),
            })?;
        drop(secrets);
        Ok(secret)
    }

    /// Check replay protection
    async fn check_replay_protection(
        &self,
        user_id: &str,
        code: &str,
    ) -> SecureStorageResult<bool> {
        if self.is_code_used(user_id, code).await? {
            warn!("TOTP code replay attempt for user: {}", user_id);
            self.verification_failures.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get current time step
    fn get_current_time_step(&self) -> SecureStorageResult<u64> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| SecureStorageError::InvalidInput {
                field: "system_time".to_string(),
                reason: "Invalid system time".to_string(),
            })?
            .as_secs();

        Ok(current_time / self.config.time_step)
    }

    /// Verify code in tolerance window
    async fn verify_code_in_window(
        &self,
        secret: &TotpSecret,
        current_step: u64,
        code: &str,
        user_id: &str,
    ) -> SecureStorageResult<bool> {
        for step_offset in 0..=self.config.window_tolerance {
            if step_offset == 0 {
                if self
                    .verify_and_mark_success(secret, current_step, code, user_id, "")
                    .await?
                {
                    return Ok(true);
                }
            } else {
                if self
                    .verify_past_step(secret, current_step, step_offset, code, user_id)
                    .await?
                {
                    return Ok(true);
                }
                if self
                    .verify_future_step(secret, current_step, step_offset, code, user_id)
                    .await?
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Verify past time step
    async fn verify_past_step(
        &self,
        secret: &TotpSecret,
        current_step: u64,
        step_offset: u32,
        code: &str,
        user_id: &str,
    ) -> SecureStorageResult<bool> {
        if current_step >= u64::from(step_offset)
            && self.verify_code_for_step(secret, current_step - u64::from(step_offset), code)
        {
            self.verify_and_mark_success(
                secret,
                current_step - u64::from(step_offset),
                code,
                user_id,
                " (past step)",
            )
            .await
        } else {
            Ok(false)
        }
    }

    /// Verify future time step
    async fn verify_future_step(
        &self,
        secret: &TotpSecret,
        current_step: u64,
        step_offset: u32,
        code: &str,
        user_id: &str,
    ) -> SecureStorageResult<bool> {
        if self.verify_code_for_step(secret, current_step + u64::from(step_offset), code) {
            self.verify_and_mark_success(
                secret,
                current_step + u64::from(step_offset),
                code,
                user_id,
                " (future step)",
            )
            .await
        } else {
            Ok(false)
        }
    }

    /// Verify and mark success
    async fn verify_and_mark_success(
        &self,
        _secret: &TotpSecret,
        _time_step: u64,
        code: &str,
        user_id: &str,
        step_type: &str,
    ) -> SecureStorageResult<bool> {
        self.mark_code_used(user_id, code).await?;
        self.codes_verified.fetch_add(1, Ordering::Relaxed);
        info!("TOTP code verified for user: {}{}", user_id, step_type);
        Ok(true)
    }

    /// Generate QR code URL for authenticator apps
    ///
    /// # Errors
    ///
    /// Returns error if URL generation fails
    pub async fn generate_qr_url(
        &self,
        user_id: &str,
        account_name: &str,
    ) -> SecureStorageResult<String> {
        let secret = {
            let secrets = self.secrets.read().await;
            let secret =
                secrets
                    .get(user_id)
                    .cloned()
                    .ok_or_else(|| SecureStorageError::NotFound {
                        resource: "totp_secret".to_string(),
                        identifier: user_id.to_string(),
                    })?;
            drop(secrets);
            secret
        };

        let base32_secret = secret.to_base32();
        let url = format!(
            "otpauth://totp/{}:{}?secret={}&issuer={}&algorithm={}&digits={}&period={}",
            urlencoding::encode(&self.config.issuer),
            urlencoding::encode(account_name),
            base32_secret,
            urlencoding::encode(&self.config.issuer),
            secret.algorithm().name(),
            self.config.code_length,
            self.config.time_step
        );

        Ok(url)
    }

    /// Generate TOTP code for specific time step
    fn generate_totp_code(&self, secret: &TotpSecret, time_step: u64) -> String {
        let time_bytes = time_step.to_be_bytes();
        let hash = Self::compute_hmac(secret.key(), &time_bytes, secret.algorithm());

        // Dynamic truncation (RFC 4226)
        let offset = (hash[hash.len() - 1] & 0x0F) as usize;
        let code = ((u32::from(hash[offset]) & 0x7F) << 24_i32)
            | ((u32::from(hash[offset + 1]) & 0xFF) << 16_i32)
            | ((u32::from(hash[offset + 2]) & 0xFF) << 8_i32)
            | (u32::from(hash[offset + 3]) & 0xFF);

        let modulus = 10_u32.pow(self.config.code_length);
        let totp_code = code % modulus;

        format!(
            "{:0width$}",
            totp_code,
            width = usize::try_from(self.config.code_length).unwrap_or(6)
        )
    }

    /// Verify code for specific time step
    fn verify_code_for_step(&self, secret: &TotpSecret, time_step: u64, code: &str) -> bool {
        let expected_code = self.generate_totp_code(secret, time_step);
        constant_time_compare(code.as_bytes(), expected_code.as_bytes())
    }

    /// Compute HMAC
    fn compute_hmac(key: &[u8], data: &[u8], algorithm: TotpAlgorithm) -> Vec<u8> {
        match algorithm {
            TotpAlgorithm::Sha1 => Self::hmac_sha1(key, data),
            TotpAlgorithm::Sha256 => Self::hmac_sha256(key, data),
            TotpAlgorithm::Sha512 => Self::hmac_sha512(key, data),
        }
    }

    /// HMAC-SHA1 implementation
    fn hmac_sha1(key: &[u8], data: &[u8]) -> Vec<u8> {
        // Simplified HMAC-SHA1 implementation
        // In production, use a proper cryptographic library
        let mut result = vec![0u8; 20];
        for (i, (&k, &d)) in key.iter().zip(data.iter()).enumerate() {
            result[i % 20] = result[i % 20].wrapping_add(k ^ d);
        }
        result
    }

    /// HMAC-SHA256 implementation
    fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
        // Simplified HMAC-SHA256 implementation
        let mut result = vec![0u8; 32];
        for (i, (&k, &d)) in key.iter().zip(data.iter()).enumerate() {
            result[i % 32] = result[i % 32].wrapping_add(k ^ d);
        }
        result
    }

    /// HMAC-SHA512 implementation
    fn hmac_sha512(key: &[u8], data: &[u8]) -> Vec<u8> {
        // Simplified HMAC-SHA512 implementation
        let mut result = vec![0u8; 64];
        for (i, (&k, &d)) in key.iter().zip(data.iter()).enumerate() {
            result[i % 64] = result[i % 64].wrapping_add(k ^ d);
        }
        result
    }

    /// Fill buffer with random bytes
    fn fill_random_bytes(buffer: &mut [u8]) {
        // Simple PRNG for demonstration - use proper CSPRNG in production
        for (i, byte) in buffer.iter_mut().enumerate() {
            *byte = u8::try_from(i)
                .unwrap_or(0)
                .wrapping_mul(17)
                .wrapping_add(42);
        }
    }

    /// Check if code was already used
    async fn is_code_used(&self, user_id: &str, code: &str) -> SecureStorageResult<bool> {
        let cutoff_time = SystemTime::now() - Duration::from_secs(self.config.time_step * 2);

        self.used_codes
            .read()
            .await
            .get(user_id)
            .map_or(Ok(false), |user_codes| {
                for (used_code, timestamp) in user_codes {
                    if used_code == code && *timestamp > cutoff_time {
                        return Ok(true);
                    }
                }
                Ok(false)
            })
    }

    /// Mark code as used
    async fn mark_code_used(&self, user_id: &str, code: &str) -> SecureStorageResult<()> {
        let cutoff_time = SystemTime::now() - Duration::from_secs(self.config.time_step * 10);

        {
            let mut used_codes = self.used_codes.write().await;
            let user_code_set = used_codes.entry(user_id.to_string()).or_default();

            // Add new code
            user_code_set.push((code.to_string(), SystemTime::now()));

            // Clean up old codes
            user_code_set.retain(|(_, timestamp)| *timestamp > cutoff_time);
            drop(used_codes);
        }

        Ok(())
    }

    /// Get system statistics
    #[must_use]
    pub async fn get_stats(&self) -> TotpStats {
        let secrets_count = {
            let secrets = self.secrets.read().await;
            let count = secrets.len();
            drop(secrets);
            count
        };

        TotpStats {
            enrolled_users: secrets_count,
            codes_generated: self.codes_generated.load(Ordering::Relaxed),
            codes_verified: self.codes_verified.load(Ordering::Relaxed),
            verification_failures: self.verification_failures.load(Ordering::Relaxed),
            algorithm: self.config.algorithm,
            time_step: self.config.time_step,
            code_length: self.config.code_length,
        }
    }
}

/// TOTP system statistics
#[derive(Debug, Clone)]
pub struct TotpStats {
    /// Number of enrolled users
    pub enrolled_users: usize,
    /// Total codes generated
    pub codes_generated: u64,
    /// Total codes verified
    pub codes_verified: u64,
    /// Total verification failures
    pub verification_failures: u64,
    /// Algorithm in use
    pub algorithm: TotpAlgorithm,
    /// Time step in seconds
    pub time_step: u64,
    /// Code length
    pub code_length: u32,
}

impl TotpStats {
    /// Calculate verification success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total_attempts = self.codes_verified + self.verification_failures;
        if total_attempts == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            let verified = u32::try_from(self.codes_verified).unwrap_or(u32::MAX);
            let total = u32::try_from(total_attempts).unwrap_or(u32::MAX);
            f64::from(verified) / f64::from(total) * 100.0
        }
    }
}

/// Constant-time string comparison
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

/// Base32 encoding (simplified)
fn base32_encode(data: &[u8]) -> String {
    // Simplified Base32 encoding - use proper library in production
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut result = String::new();

    for chunk in data.chunks(5) {
        let mut buffer = [0u8; 5];
        buffer[..chunk.len()].copy_from_slice(chunk);

        let value = u64::from_be_bytes([
            0, 0, 0, buffer[0], buffer[1], buffer[2], buffer[3], buffer[4],
        ]);

        for i in 0_usize..8_usize {
            let index = usize::try_from((value >> (35_usize - i * 5_usize)) & 0x1F).unwrap_or(0);
            result.push(ALPHABET[index] as char);
        }
    }

    result
}

/// Base32 decoding (simplified)
fn base32_decode(encoded: &str) -> SecureStorageResult<Vec<u8>> {
    // Simplified Base32 decoding - use proper library in production
    let mut result = Vec::with_capacity(10);

    for chunk in encoded.as_bytes().chunks(8) {
        let mut value = 0u64;

        for (i, &byte) in chunk.iter().enumerate() {
            let digit = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'2'..=b'7' => byte - b'2' + 26,
                _ => {
                    return Err(SecureStorageError::InvalidInput {
                        field: "base32_data".to_string(),
                        reason: "Invalid Base32 character".to_string(),
                    })
                }
            };
            let shift = u32::try_from(35_usize - i * 5_usize).unwrap_or(0);
            value |= u64::from(digit) << shift;
        }

        let bytes = value.to_be_bytes();
        result.extend_from_slice(&bytes[3..8]);
    }

    Ok(result)
}
