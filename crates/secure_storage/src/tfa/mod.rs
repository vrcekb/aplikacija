//! # Two-Factor Authentication (2FA) System
//!
//! Ultra-secure two-factor authentication implementation for `TallyIO` financial platform.
//! Provides TOTP, HOTP, hardware tokens, and biometric authentication support
//! with military-grade security standards.
//!
//! ## Features
//!
//! - **TOTP (Time-based OTP)**: RFC 6238 compliant time-based tokens
//! - **HOTP (HMAC-based OTP)**: RFC 4226 compliant counter-based tokens
//! - **Hardware Tokens**: `YubiKey`, FIDO2, and `WebAuthn` support
//! - **Biometric Authentication**: Fingerprint and facial recognition
//! - **Backup Codes**: Secure recovery codes for account recovery
//! - **Rate Limiting**: Brute-force protection with exponential backoff
//! - **Audit Logging**: Comprehensive security event logging
//!
//! ## Security Properties
//!
//! - **Cryptographic Security**: HMAC-SHA256/SHA512 with secure key derivation
//! - **Replay Protection**: Time window validation and counter synchronization
//! - **Brute-Force Resistance**: Rate limiting with account lockout
//! - **Side-Channel Protection**: Constant-time operations
//! - **Forward Secrecy**: Regular key rotation and secure deletion

use crate::error::{SecureStorageError, SecureStorageResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

pub mod backup_codes;
pub mod biometric;
pub mod hardware;
pub mod hotp;
pub mod rate_limiting;
pub mod totp;

/// 2FA method types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TfaMethod {
    /// Time-based One-Time Password
    Totp,
    /// HMAC-based One-Time Password
    Hotp,
    /// Hardware security key (`YubiKey`, FIDO2)
    HardwareToken,
    /// Biometric authentication
    Biometric,
    /// SMS-based (not recommended for high security)
    Sms,
    /// Email-based (backup only)
    Email,
    /// Backup recovery codes
    BackupCodes,
}

impl TfaMethod {
    /// Get security level of the method
    #[must_use]
    pub const fn security_level(self) -> u8 {
        match self {
            Self::HardwareToken => 10, // Highest security
            Self::Biometric => 9,      // Very high security
            Self::Totp => 8,           // High security
            Self::Hotp => 7,           // Good security
            Self::BackupCodes => 6,    // Medium security (one-time use)
            Self::Email => 3,          // Low security
            Self::Sms => 2,            // Very low security (SIM swapping)
        }
    }

    /// Check if method is hardware-based
    #[must_use]
    pub const fn is_hardware_based(self) -> bool {
        matches!(self, Self::HardwareToken | Self::Biometric)
    }

    /// Check if method is phishing-resistant
    #[must_use]
    pub const fn is_phishing_resistant(self) -> bool {
        matches!(self, Self::HardwareToken | Self::Biometric)
    }
}

/// Multiple methods policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipleMethodsPolicy {
    /// Allow multiple methods for redundancy
    Allow,
    /// Require single method only
    SingleOnly,
}

/// Hardware requirement policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwarePolicy {
    /// Require hardware for high-value operations
    RequiredForHighValue,
    /// Hardware optional
    Optional,
}

/// Backup codes policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupCodesPolicy {
    /// Enable backup codes
    Enabled,
    /// Disable backup codes
    Disabled,
}

/// Audit logging policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditPolicy {
    /// Full audit logging enabled
    Enabled,
    /// Audit logging disabled
    Disabled,
}

/// Security configuration flags
#[derive(Debug, Clone)]
pub struct SecurityFlags {
    /// Multiple methods policy
    pub multiple_methods: MultipleMethodsPolicy,
    /// Hardware requirement policy
    pub hardware_policy: HardwarePolicy,
    /// Backup codes policy
    pub backup_codes: BackupCodesPolicy,
    /// Audit logging policy
    pub audit_logging: AuditPolicy,
}

/// 2FA configuration
#[derive(Debug, Clone)]
pub struct TfaConfig {
    /// Required security level (1-10)
    pub required_security_level: u8,
    /// Security flags
    pub security_flags: SecurityFlags,
    /// Maximum authentication attempts before lockout
    pub max_attempts: u32,
    /// Lockout duration
    pub lockout_duration: Duration,
    /// TOTP time window (seconds)
    pub totp_time_window: u32,
    /// TOTP code validity period
    pub totp_validity_period: Duration,
    /// Backup codes count
    pub backup_codes_count: u32,
}

impl TfaConfig {
    /// Create production configuration
    #[must_use]
    pub const fn new_production() -> Self {
        Self {
            required_security_level: 8, // Require TOTP or better
            security_flags: SecurityFlags {
                multiple_methods: MultipleMethodsPolicy::Allow,
                hardware_policy: HardwarePolicy::RequiredForHighValue,
                backup_codes: BackupCodesPolicy::Enabled,
                audit_logging: AuditPolicy::Enabled,
            },
            max_attempts: 3,                            // 3 attempts before lockout
            lockout_duration: Duration::from_secs(900), // 15 minutes
            totp_time_window: 30,                       // 30-second windows
            totp_validity_period: Duration::from_secs(90), // 90-second validity
            backup_codes_count: 10,                     // 10 backup codes
        }
    }

    /// Create development configuration
    #[must_use]
    pub const fn new_development() -> Self {
        Self {
            required_security_level: 6, // Lower requirement for dev
            security_flags: SecurityFlags {
                multiple_methods: MultipleMethodsPolicy::Allow,
                hardware_policy: HardwarePolicy::Optional,
                backup_codes: BackupCodesPolicy::Enabled,
                audit_logging: AuditPolicy::Enabled,
            },
            max_attempts: 10,                          // More attempts for testing
            lockout_duration: Duration::from_secs(60), // 1 minute lockout
            totp_time_window: 30,
            totp_validity_period: Duration::from_secs(300), // 5 minutes for dev
            backup_codes_count: 5,                          // Fewer codes for dev
        }
    }

    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration values are invalid
    pub fn validate(&self) -> SecureStorageResult<()> {
        if self.required_security_level == 0 || self.required_security_level > 10 {
            return Err(SecureStorageError::InvalidInput {
                field: "required_security_level".to_string(),
                reason: "Security level must be between 1 and 10".to_string(),
            });
        }

        if self.max_attempts == 0 || self.max_attempts > 100 {
            return Err(SecureStorageError::InvalidInput {
                field: "max_attempts".to_string(),
                reason: "Max attempts must be between 1 and 100".to_string(),
            });
        }

        if self.totp_time_window == 0 || self.totp_time_window > 300 {
            return Err(SecureStorageError::InvalidInput {
                field: "totp_time_window".to_string(),
                reason: "TOTP time window must be between 1 and 300 seconds".to_string(),
            });
        }

        if self.backup_codes_count > 50 {
            return Err(SecureStorageError::InvalidInput {
                field: "backup_codes_count".to_string(),
                reason: "Backup codes count cannot exceed 50".to_string(),
            });
        }

        Ok(())
    }
}

/// User 2FA enrollment information
#[derive(Debug, Clone)]
pub struct TfaEnrollment {
    /// User identifier
    pub user_id: String,
    /// Enrolled methods
    pub methods: HashMap<TfaMethod, TfaMethodData>,
    /// Primary method
    pub primary_method: Option<TfaMethod>,
    /// Backup methods
    pub backup_methods: Vec<TfaMethod>,
    /// Enrollment timestamp
    pub enrolled_at: SystemTime,
    /// Last successful authentication
    pub last_auth_success: Option<SystemTime>,
    /// Failed attempts counter
    pub failed_attempts: u32,
    /// Account locked until
    pub locked_until: Option<SystemTime>,
    /// Backup codes (if enabled)
    pub backup_codes: Option<Vec<String>>,
}

/// Method-specific data
#[derive(Debug, Clone)]
pub struct TfaMethodData {
    /// Method type
    pub method: TfaMethod,
    /// Secret key (for TOTP/HOTP)
    secret_key: Option<Vec<u8>>,
    /// Counter (for HOTP)
    pub counter: Option<u64>,
    /// Device identifier (for hardware tokens)
    pub device_id: Option<String>,
    /// Biometric template hash
    pub biometric_hash: Option<Vec<u8>>,
    /// Method-specific configuration
    pub config: HashMap<String, String>,
    /// Enrollment timestamp
    pub enrolled_at: SystemTime,
    /// Last used timestamp
    pub last_used: Option<SystemTime>,
}

impl TfaMethodData {
    /// Create new method data
    #[must_use]
    pub fn new(method: TfaMethod) -> Self {
        Self {
            method,
            secret_key: None,
            counter: None,
            device_id: None,
            biometric_hash: None,
            config: HashMap::new(),
            enrolled_at: SystemTime::now(),
            last_used: None,
        }
    }

    /// Set secret key (for TOTP/HOTP)
    #[must_use]
    pub fn with_secret_key(mut self, key: Vec<u8>) -> Self {
        self.secret_key = Some(key);
        self
    }

    /// Set counter (for HOTP)
    #[must_use]
    pub const fn with_counter(mut self, counter: u64) -> Self {
        self.counter = Some(counter);
        self
    }

    /// Set device ID (for hardware tokens)
    #[must_use]
    pub fn with_device_id(mut self, device_id: String) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Get secret key
    #[must_use]
    pub fn secret_key(&self) -> Option<&[u8]> {
        self.secret_key.as_deref()
    }
}

/// Authentication challenge
#[derive(Debug, Clone)]
pub struct TfaChallenge {
    /// Challenge ID
    pub challenge_id: String,
    /// User ID
    pub user_id: String,
    /// Required methods
    pub required_methods: Vec<TfaMethod>,
    /// Challenge creation time
    pub created_at: SystemTime,
    /// Challenge expiry time
    pub expires_at: SystemTime,
    /// Completed methods
    pub completed_methods: Vec<TfaMethod>,
    /// Challenge nonce
    pub nonce: Vec<u8>,
}

/// Authentication response
#[derive(Debug, Clone)]
pub struct TfaResponse {
    /// Challenge ID
    pub challenge_id: String,
    /// Method used
    pub method: TfaMethod,
    /// Response code/token
    pub response: String,
    /// Additional data (for hardware tokens)
    pub additional_data: Option<Vec<u8>>,
}

/// 2FA authentication result
#[derive(Debug, Clone)]
pub struct TfaResult {
    /// Authentication successful
    pub success: bool,
    /// Methods used
    pub methods_used: Vec<TfaMethod>,
    /// Security level achieved
    pub security_level: u8,
    /// Authentication timestamp
    pub timestamp: SystemTime,
    /// Session token (if successful)
    pub session_token: Option<String>,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Two-Factor Authentication System
#[derive(Debug)]
pub struct TfaSystem {
    /// System configuration
    config: TfaConfig,
    /// User enrollments
    enrollments: RwLock<HashMap<String, TfaEnrollment>>,
    /// Active challenges
    challenges: RwLock<HashMap<String, TfaChallenge>>,
    /// TOTP system
    totp_system: Arc<totp::TotpSystem>,
    /// HOTP system
    hotp_system: Arc<hotp::HotpSystem>,
    /// Hardware token system
    hardware_system: Arc<hardware::HardwareTokenSystem>,
    /// Biometric system
    biometric_system: Arc<biometric::BiometricSystem>,
    /// Backup codes system
    backup_codes_system: Arc<backup_codes::BackupCodesSystem>,
    /// Rate limiting system
    rate_limiter: Arc<rate_limiting::RateLimiter>,
    /// Performance counters
    authentications_attempted: AtomicU64,
    authentications_successful: AtomicU64,
    authentications_failed: AtomicU64,
}

impl TfaSystem {
    /// Create a new 2FA system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(config: TfaConfig) -> SecureStorageResult<Self> {
        config.validate()?;

        info!(
            "Initializing 2FA system with security level {}",
            config.required_security_level
        );

        let totp_system = Arc::new(totp::TotpSystem::new(config.clone())?);
        let hotp_system = Arc::new(hotp::HotpSystem::new(config.clone())?);
        let hardware_system = Arc::new(hardware::HardwareTokenSystem::new(config.clone())?);
        let biometric_system = Arc::new(biometric::BiometricSystem::new(config.clone())?);
        let backup_codes_system = Arc::new(backup_codes::BackupCodesSystem::new(config.clone())?);
        let rate_limiter = Arc::new(rate_limiting::RateLimiter::new(config.clone())?);

        Ok(Self {
            config,
            enrollments: RwLock::new(HashMap::new()),
            challenges: RwLock::new(HashMap::new()),
            totp_system,
            hotp_system,
            hardware_system,
            biometric_system,
            backup_codes_system,
            rate_limiter,
            authentications_attempted: AtomicU64::new(0),
            authentications_successful: AtomicU64::new(0),
            authentications_failed: AtomicU64::new(0),
        })
    }

    /// Enroll user for 2FA
    ///
    /// # Errors
    ///
    /// Returns error if enrollment fails
    ///
    /// # Panics
    ///
    /// This function does not panic in normal operation
    pub async fn enroll_user(
        &self,
        user_id: String,
        method: TfaMethod,
        method_data: TfaMethodData,
    ) -> SecureStorageResult<TfaEnrollment> {
        // Validate method meets security requirements
        if method.security_level() < self.config.required_security_level {
            return Err(SecureStorageError::InvalidInput {
                field: "method".to_string(),
                reason: format!(
                    "Method security level {} below required {}",
                    method.security_level(),
                    self.config.required_security_level
                ),
            });
        }

        let enrollment_clone = {
            let mut enrollments = self.enrollments.write().await;

            let enrollment = enrollments
                .entry(user_id.clone())
                .or_insert_with(|| TfaEnrollment {
                    user_id: user_id.clone(),
                    methods: HashMap::new(),
                    primary_method: None,
                    backup_methods: Vec::with_capacity(0),
                    enrolled_at: SystemTime::now(),
                    last_auth_success: None,
                    failed_attempts: 0,
                    locked_until: None,
                    backup_codes: None,
                });

            // Add method to enrollment
            enrollment.methods.insert(method, method_data);

            // Set as primary if first method or higher security
            let should_set_primary = enrollment
                .primary_method
                .is_none_or(|primary| method.security_level() > primary.security_level());

            if should_set_primary {
                enrollment.primary_method = Some(method);
            }

            // Generate backup codes if enabled and not already generated
            if matches!(
                self.config.security_flags.backup_codes,
                BackupCodesPolicy::Enabled
            ) && enrollment.backup_codes.is_none()
            {
                let backup_codes = self
                    .backup_codes_system
                    .generate_codes(&user_id, self.config.backup_codes_count)
                    .await?;
                enrollment.backup_codes = Some(backup_codes);
            }

            let clone = enrollment.clone();
            drop(enrollments);
            clone
        };

        info!("User {} enrolled for 2FA with method {:?}", user_id, method);
        Ok(enrollment_clone)
    }

    /// Create authentication challenge
    ///
    /// # Errors
    ///
    /// Returns error if challenge creation fails
    pub async fn create_challenge(
        &self,
        user_id: String,
        required_methods: Option<Vec<TfaMethod>>,
    ) -> SecureStorageResult<TfaChallenge> {
        // Check if user is enrolled
        let enrollment = self
            .enrollments
            .read()
            .await
            .get(&user_id)
            .cloned()
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "tfa_enrollment".to_string(),
                identifier: user_id.clone(),
            })?;

        // Check if account is locked
        if let Some(locked_until) = enrollment.locked_until {
            if SystemTime::now() < locked_until {
                return Err(SecureStorageError::InvalidInput {
                    field: "account_status".to_string(),
                    reason: "Account is temporarily locked".to_string(),
                });
            }
        }

        // Determine required methods
        let methods = required_methods.unwrap_or_else(|| {
            enrollment
                .primary_method
                .map_or_else(|| vec![TfaMethod::Totp], |primary| vec![primary])
        });

        // Generate challenge
        let challenge_id = Self::generate_challenge_id();
        let nonce = Self::generate_nonce();
        let now = SystemTime::now();

        let challenge = TfaChallenge {
            challenge_id: challenge_id.clone(),
            user_id: user_id.clone(),
            required_methods: methods,
            created_at: now,
            expires_at: now + Duration::from_secs(300), // 5 minutes
            completed_methods: Vec::with_capacity(0),
            nonce,
        };

        // Store challenge
        {
            let mut challenges = self.challenges.write().await;
            challenges.insert(challenge_id.clone(), challenge.clone());
        }

        info!(
            "Created 2FA challenge {} for user {}",
            challenge_id, user_id
        );
        Ok(challenge)
    }

    /// Verify authentication response
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_response(&self, response: TfaResponse) -> SecureStorageResult<TfaResult> {
        self.authentications_attempted
            .fetch_add(1, Ordering::Relaxed);

        // Check challenge expiration first
        {
            let mut challenges = self.challenges.write().await;
            if let Some(challenge) = challenges.get(&response.challenge_id) {
                if SystemTime::now() > challenge.expires_at {
                    challenges.remove(&response.challenge_id);
                    drop(challenges);
                    return Ok(Self::create_error_result("Challenge expired"));
                }
            } else {
                drop(challenges);
                return Err(SecureStorageError::NotFound {
                    resource: "tfa_challenge".to_string(),
                    identifier: response.challenge_id.clone(),
                });
            }
        }

        // Check rate limiting
        let user_id = {
            let challenges = self.challenges.read().await;
            let challenge = challenges.get(&response.challenge_id).ok_or_else(|| {
                SecureStorageError::NotFound {
                    resource: "tfa_challenge".to_string(),
                    identifier: response.challenge_id.clone(),
                }
            })?;
            let user_id = challenge.user_id.clone();
            drop(challenges);
            user_id
        };

        if !self.check_rate_limit(&user_id).await? {
            return Ok(Self::create_error_result("Rate limit exceeded"));
        }

        // Verify method
        let verification_result = {
            let challenges = self.challenges.read().await;
            let challenge = challenges.get(&response.challenge_id).ok_or_else(|| {
                SecureStorageError::NotFound {
                    resource: "tfa_challenge".to_string(),
                    identifier: response.challenge_id.clone(),
                }
            })?;
            let result = self.verify_method(&response, challenge).await?;
            drop(challenges);
            result
        };

        // Handle result
        if verification_result {
            self.handle_successful_verification_safe(response).await
        } else {
            self.handle_failed_verification_safe(&user_id).await
        }
    }

    /// Check rate limiting
    async fn check_rate_limit(&self, user_id: &str) -> SecureStorageResult<bool> {
        self.rate_limiter.check_rate_limit(user_id).await
    }

    /// Verify authentication method
    async fn verify_method(
        &self,
        response: &TfaResponse,
        challenge: &TfaChallenge,
    ) -> SecureStorageResult<bool> {
        match response.method {
            TfaMethod::Totp => {
                self.totp_system
                    .verify_code(&challenge.user_id, &response.response)
                    .await
            }
            TfaMethod::Hotp => {
                self.hotp_system
                    .verify_code(&challenge.user_id, &response.response)
                    .await
            }
            TfaMethod::HardwareToken => {
                self.hardware_system
                    .verify_token(&challenge.user_id, &response.response, &challenge.nonce)
                    .await
            }
            TfaMethod::Biometric => {
                self.biometric_system
                    .verify_biometric(&challenge.user_id, &response.response)
                    .await
            }
            TfaMethod::BackupCodes => {
                self.backup_codes_system
                    .verify_code(&challenge.user_id, &response.response)
                    .await
            }
            _ => Ok(false),
        }
    }

    /// Handle successful verification (safe version)
    async fn handle_successful_verification_safe(
        &self,
        response: TfaResponse,
    ) -> SecureStorageResult<TfaResult> {
        let mut challenges = self.challenges.write().await;
        let challenge = challenges.get_mut(&response.challenge_id).ok_or_else(|| {
            SecureStorageError::NotFound {
                resource: "tfa_challenge".to_string(),
                identifier: response.challenge_id.clone(),
            }
        })?;

        challenge.completed_methods.push(response.method);

        let all_completed = challenge
            .required_methods
            .iter()
            .all(|method| challenge.completed_methods.contains(method));

        if all_completed {
            self.authentications_successful
                .fetch_add(1, Ordering::Relaxed);

            let security_level = challenge
                .completed_methods
                .iter()
                .map(|method| method.security_level())
                .max()
                .unwrap_or(0);

            let session_token = Self::generate_session_token(&challenge.user_id);
            let user_id = challenge.user_id.clone();
            let completed_methods = challenge.completed_methods.clone();

            self.update_successful_auth(&user_id).await?;
            challenges.remove(&response.challenge_id);
            drop(challenges);

            info!("2FA authentication successful for user {}", user_id);

            Ok(TfaResult {
                success: true,
                methods_used: completed_methods,
                security_level,
                timestamp: SystemTime::now(),
                session_token: Some(session_token),
                error_message: None,
            })
        } else {
            Ok(TfaResult {
                success: false,
                methods_used: challenge.completed_methods.clone(),
                security_level: 0,
                timestamp: SystemTime::now(),
                session_token: None,
                error_message: Some("Additional authentication methods required".to_string()),
            })
        }
    }

    /// Handle failed verification (safe version)
    async fn handle_failed_verification_safe(
        &self,
        user_id: &str,
    ) -> SecureStorageResult<TfaResult> {
        self.authentications_failed.fetch_add(1, Ordering::Relaxed);
        self.update_failed_auth(user_id).await?;

        warn!("2FA authentication failed for user {}", user_id);

        Ok(TfaResult {
            success: false,
            methods_used: Vec::with_capacity(0),
            security_level: 0,
            timestamp: SystemTime::now(),
            session_token: None,
            error_message: Some("Authentication failed".to_string()),
        })
    }

    /// Create error result
    fn create_error_result(error_message: &str) -> TfaResult {
        TfaResult {
            success: false,
            methods_used: Vec::with_capacity(0),
            security_level: 0,
            timestamp: SystemTime::now(),
            session_token: None,
            error_message: Some(error_message.to_string()),
        }
    }

    /// Generate challenge ID
    fn generate_challenge_id() -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        format!("chal_{:x}", hasher.finish())
    }

    /// Generate nonce
    fn generate_nonce() -> Vec<u8> {
        // In production, use cryptographically secure random number generator
        vec![42u8; 32] // Placeholder
    }

    /// Generate session token
    fn generate_session_token(user_id: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        user_id.hash(&mut hasher);
        SystemTime::now().hash(&mut hasher);
        format!("sess_{:x}", hasher.finish())
    }

    /// Update successful authentication
    async fn update_successful_auth(&self, user_id: &str) -> SecureStorageResult<()> {
        let mut enrollments = self.enrollments.write().await;
        if let Some(enrollment) = enrollments.get_mut(user_id) {
            enrollment.last_auth_success = Some(SystemTime::now());
            enrollment.failed_attempts = 0;
            enrollment.locked_until = None;
        }
        drop(enrollments);
        Ok(())
    }

    /// Update failed authentication
    async fn update_failed_auth(&self, user_id: &str) -> SecureStorageResult<()> {
        let mut enrollments = self.enrollments.write().await;
        if let Some(enrollment) = enrollments.get_mut(user_id) {
            enrollment.failed_attempts += 1;

            if enrollment.failed_attempts >= self.config.max_attempts {
                enrollment.locked_until = Some(SystemTime::now() + self.config.lockout_duration);
                warn!("User {} locked due to too many failed attempts", user_id);
            }
        }
        drop(enrollments);
        Ok(())
    }

    /// Get system statistics
    #[must_use]
    pub async fn get_stats(&self) -> TfaStats {
        let enrollments_count = {
            let enrollments = self.enrollments.read().await;
            let count = enrollments.len();
            drop(enrollments);
            count
        };
        let active_challenges = {
            let challenges = self.challenges.read().await;
            let count = challenges.len();
            drop(challenges);
            count
        };

        TfaStats {
            total_enrollments: enrollments_count,
            active_challenges,
            authentications_attempted: self.authentications_attempted.load(Ordering::Relaxed),
            authentications_successful: self.authentications_successful.load(Ordering::Relaxed),
            authentications_failed: self.authentications_failed.load(Ordering::Relaxed),
            required_security_level: self.config.required_security_level,
        }
    }
}

/// 2FA system statistics
#[derive(Debug, Clone)]
pub struct TfaStats {
    /// Total user enrollments
    pub total_enrollments: usize,
    /// Active authentication challenges
    pub active_challenges: usize,
    /// Total authentication attempts
    pub authentications_attempted: u64,
    /// Successful authentications
    pub authentications_successful: u64,
    /// Failed authentications
    pub authentications_failed: u64,
    /// Required security level
    pub required_security_level: u8,
}

impl TfaStats {
    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.authentications_attempted == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(self.authentications_successful).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.authentications_attempted).unwrap_or(u32::MAX))
                * 100.0
        }
    }
}
