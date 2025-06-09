//! # Rate Limiting for 2FA
//!
//! Advanced rate limiting with exponential backoff to prevent brute-force attacks.

use super::TfaConfig;
use crate::error::SecureStorageResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Rate limiter for 2FA attempts
#[derive(Debug)]
pub struct RateLimiter {
    /// User attempt tracking
    user_attempts: RwLock<HashMap<String, UserAttemptInfo>>,
    /// Configuration
    config: RateLimitConfig,
    /// Performance counters
    requests_allowed: AtomicU64,
    requests_blocked: AtomicU64,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum attempts per time window
    pub max_attempts: u32,
    /// Time window duration
    pub time_window: Duration,
    /// Exponential backoff base multiplier
    pub backoff_multiplier: f64,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Cleanup interval for old entries
    pub cleanup_interval: Duration,
}

impl RateLimitConfig {
    /// Create production rate limit configuration
    #[must_use]
    pub const fn new_production() -> Self {
        Self {
            max_attempts: 5,                            // 5 attempts
            time_window: Duration::from_secs(300),      // per 5 minutes
            backoff_multiplier: 2.0,                    // Double each time
            max_backoff: Duration::from_secs(3600),     // Max 1 hour
            cleanup_interval: Duration::from_secs(600), // Cleanup every 10 minutes
        }
    }

    /// Create development configuration
    #[must_use]
    pub const fn new_development() -> Self {
        Self {
            max_attempts: 10,                           // More attempts for dev
            time_window: Duration::from_secs(60),       // per 1 minute
            backoff_multiplier: 1.5,                    // Gentler backoff
            max_backoff: Duration::from_secs(300),      // Max 5 minutes
            cleanup_interval: Duration::from_secs(120), // Cleanup every 2 minutes
        }
    }
}

/// User attempt tracking information
#[derive(Debug, Clone)]
struct UserAttemptInfo {
    /// Attempt timestamps within current window
    attempts: Vec<SystemTime>,
    /// Current backoff level
    backoff_level: u32,
    /// Blocked until timestamp
    blocked_until: Option<SystemTime>,
    /// Last attempt timestamp
    last_attempt: SystemTime,
}

impl UserAttemptInfo {
    /// Create new user attempt info
    fn new() -> Self {
        Self {
            attempts: Vec::with_capacity(0),
            backoff_level: 0,
            blocked_until: None,
            last_attempt: SystemTime::now(),
        }
    }

    /// Check if user is currently blocked
    fn is_blocked(&self) -> bool {
        self.blocked_until
            .is_some_and(|blocked_until| SystemTime::now() < blocked_until)
    }

    /// Clean up old attempts outside the time window
    fn cleanup_old_attempts(&mut self, window: Duration) {
        let cutoff = SystemTime::now() - window;
        self.attempts.retain(|&timestamp| timestamp > cutoff);
    }

    /// Add new attempt
    fn add_attempt(&mut self) {
        self.attempts.push(SystemTime::now());
        self.last_attempt = SystemTime::now();
    }

    /// Calculate backoff duration
    fn calculate_backoff(&self, config: &RateLimitConfig) -> Duration {
        if self.backoff_level == 0 {
            return Duration::from_secs(0);
        }

        let base_duration = config.time_window.as_secs_f64();
        let multiplier = config
            .backoff_multiplier
            .powi(i32::try_from(self.backoff_level).unwrap_or(i32::MAX));
        let backoff_secs = (base_duration * multiplier).min(config.max_backoff.as_secs_f64());

        Duration::from_secs_f64(backoff_secs)
    }

    /// Apply backoff penalty
    fn apply_backoff(&mut self, config: &RateLimitConfig) {
        self.backoff_level += 1;
        let backoff_duration = self.calculate_backoff(config);
        self.blocked_until = Some(SystemTime::now() + backoff_duration);

        debug!(
            "Applied backoff level {} for duration {:?}",
            self.backoff_level, backoff_duration
        );
    }

    /// Reset backoff on successful authentication
    fn reset_backoff(&mut self) {
        self.backoff_level = 0;
        self.blocked_until = None;
        self.attempts.clear();
    }
}

impl RateLimiter {
    /// Create new rate limiter
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(_tfa_config: TfaConfig) -> SecureStorageResult<Self> {
        let config = if cfg!(debug_assertions) {
            RateLimitConfig::new_development()
        } else {
            RateLimitConfig::new_production()
        };

        let rate_limiter = Self {
            user_attempts: RwLock::new(HashMap::new()),
            config,
            requests_allowed: AtomicU64::new(0),
            requests_blocked: AtomicU64::new(0),
        };

        // Start cleanup task
        Self::start_cleanup_task();

        debug!(
            "Initialized rate limiter with {} max attempts per {:?}",
            rate_limiter.config.max_attempts, rate_limiter.config.time_window
        );

        Ok(rate_limiter)
    }

    /// Check if request is allowed for user
    ///
    /// # Errors
    ///
    /// Returns error if rate limit check fails
    pub async fn check_rate_limit(&self, user_id: &str) -> SecureStorageResult<bool> {
        let mut user_attempts = self.user_attempts.write().await;
        let user_info = user_attempts
            .entry(user_id.to_string())
            .or_insert_with(UserAttemptInfo::new);

        // Check if user is currently blocked
        if user_info.is_blocked() {
            drop(user_attempts);
            self.requests_blocked.fetch_add(1, Ordering::Relaxed);
            warn!("Rate limit blocked request for user: {}", user_id);
            return Ok(false);
        }

        // Clean up old attempts
        user_info.cleanup_old_attempts(self.config.time_window);

        // Check if user has exceeded attempt limit
        if user_info.attempts.len() >= self.config.max_attempts as usize {
            // Apply exponential backoff
            user_info.apply_backoff(&self.config);
            let backoff_level = user_info.backoff_level;
            drop(user_attempts);
            self.requests_blocked.fetch_add(1, Ordering::Relaxed);
            warn!(
                "Rate limit exceeded for user: {} (backoff level: {})",
                user_id, backoff_level
            );
            return Ok(false);
        }

        // Record this attempt
        user_info.add_attempt();
        let attempts_count = user_info.attempts.len();
        drop(user_attempts);
        self.requests_allowed.fetch_add(1, Ordering::Relaxed);

        debug!(
            "Rate limit check passed for user: {} ({}/{} attempts)",
            user_id, attempts_count, self.config.max_attempts
        );

        Ok(true)
    }

    /// Record successful authentication (resets rate limiting)
    ///
    /// # Errors
    ///
    /// Returns error if reset fails
    pub async fn record_success(&self, user_id: &str) -> SecureStorageResult<()> {
        let mut user_attempts = self.user_attempts.write().await;

        if let Some(user_info) = user_attempts.get_mut(user_id) {
            user_info.reset_backoff();
            debug!(
                "Reset rate limiting for user after successful auth: {}",
                user_id
            );
        }
        drop(user_attempts);

        Ok(())
    }

    /// Record failed authentication
    ///
    /// # Errors
    ///
    /// Returns error if recording fails
    pub fn record_failure(&self, user_id: &str) -> SecureStorageResult<()> {
        // Failure is already recorded in check_rate_limit
        // This method can be used for additional logging or metrics
        debug!("Recorded authentication failure for user: {}", user_id);
        Ok(())
    }

    /// Get rate limit status for user
    ///
    /// # Errors
    ///
    /// Returns error if status retrieval fails
    pub async fn get_user_status(&self, user_id: &str) -> SecureStorageResult<RateLimitStatus> {
        let user_attempts = self.user_attempts.read().await;

        user_attempts.get(user_id).map_or(
            Ok(RateLimitStatus {
                is_blocked: false,
                remaining_attempts: self.config.max_attempts,
                backoff_level: 0,
                reset_time: None,
                total_attempts: 0,
            }),
            |user_info| {
                let remaining_attempts = if user_info.is_blocked() {
                    0
                } else {
                    self.config
                        .max_attempts
                        .saturating_sub(u32::try_from(user_info.attempts.len()).unwrap_or(u32::MAX))
                };

                let reset_time = if user_info.is_blocked() {
                    user_info.blocked_until
                } else if !user_info.attempts.is_empty() {
                    Some(user_info.attempts[0] + self.config.time_window)
                } else {
                    None
                };

                Ok(RateLimitStatus {
                    is_blocked: user_info.is_blocked(),
                    remaining_attempts,
                    backoff_level: user_info.backoff_level,
                    reset_time,
                    total_attempts: u32::try_from(user_info.attempts.len()).unwrap_or(u32::MAX),
                })
            },
        )
    }

    /// Start background cleanup task
    fn start_cleanup_task() {
        // In a real implementation, this would spawn a background task
        // to periodically clean up old entries
        debug!("Rate limiter cleanup task would start here");
    }

    /// Manual cleanup of old entries
    ///
    /// # Errors
    ///
    /// Returns error if cleanup operation fails
    pub async fn cleanup_old_entries(&self) -> SecureStorageResult<usize> {
        let mut user_attempts = self.user_attempts.write().await;
        let mut removed_count = 0;

        let cutoff_time = SystemTime::now() - (self.config.time_window * 2);

        user_attempts.retain(|user_id, user_info| {
            // Remove users with no recent activity
            if user_info.last_attempt < cutoff_time && !user_info.is_blocked() {
                debug!("Cleaned up old rate limit entry for user: {}", user_id);
                removed_count += 1;
                false
            } else {
                // Clean up old attempts for remaining users
                user_info.cleanup_old_attempts(self.config.time_window);
                true
            }
        });
        drop(user_attempts);

        debug!("Cleaned up {} old rate limit entries", removed_count);
        Ok(removed_count)
    }

    /// Get rate limiter statistics
    #[must_use]
    pub async fn get_stats(&self) -> RateLimitStats {
        let active_users = {
            let user_attempts = self.user_attempts.read().await;
            let count = user_attempts.len();
            drop(user_attempts);
            count
        };

        RateLimitStats {
            active_users,
            requests_allowed: self.requests_allowed.load(Ordering::Relaxed),
            requests_blocked: self.requests_blocked.load(Ordering::Relaxed),
            max_attempts: self.config.max_attempts,
            time_window_secs: self.config.time_window.as_secs(),
        }
    }
}

/// Rate limit status for a user
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    /// Whether user is currently blocked
    pub is_blocked: bool,
    /// Remaining attempts before blocking
    pub remaining_attempts: u32,
    /// Current backoff level
    pub backoff_level: u32,
    /// When the rate limit resets
    pub reset_time: Option<SystemTime>,
    /// Total attempts in current window
    pub total_attempts: u32,
}

/// Rate limiter statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Number of users with active rate limit tracking
    pub active_users: usize,
    /// Total requests allowed
    pub requests_allowed: u64,
    /// Total requests blocked
    pub requests_blocked: u64,
    /// Maximum attempts per window
    pub max_attempts: u32,
    /// Time window in seconds
    pub time_window_secs: u64,
}

impl RateLimitStats {
    /// Calculate block rate percentage
    #[must_use]
    pub fn block_rate(&self) -> f64 {
        let total_requests = self.requests_allowed + self.requests_blocked;
        if total_requests == 0 {
            0.0
        } else {
            f64::from(u32::try_from(self.requests_blocked).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total_requests).unwrap_or(u32::MAX))
                * 100.0
        }
    }
}
