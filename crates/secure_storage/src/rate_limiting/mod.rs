//! # Rate Limiting and `DDoS` Protection
//!
//! Ultra-fast rate limiting for `TallyIO` financial platform with <1ms overhead.
//! Provides protection against brute force attacks and `DDoS` attempts.

use crate::error::{SecureStorageError, SecureStorageResult};
use dashmap::DashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

// pub mod token_bucket;
// pub mod sliding_window;
// pub mod adaptive_limiter;

/// Rate limiting configuration for `DDoS` protection and brute force prevention
///
/// This configuration defines the parameters for rate limiting behavior including
/// request limits, time windows, and blacklisting thresholds.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per time window
    pub max_requests: u32,
    /// Time window duration
    pub window_duration: Duration,
    /// Burst allowance (token bucket)
    pub burst_size: u32,
    /// Refill rate (tokens per second)
    pub refill_rate: f64,
    /// Blacklist duration for violators
    pub blacklist_duration: Duration,
    /// Threshold for automatic blacklisting
    pub blacklist_threshold: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_duration: Duration::from_secs(60),
            burst_size: 10,
            refill_rate: 1.0,
            blacklist_duration: Duration::from_secs(300), // 5 minutes
            blacklist_threshold: 5,                       // 5 violations = blacklist
        }
    }
}

/// Rate limit decision returned by the rate limiter
///
/// Indicates whether a request should be allowed, denied, or if the client is blacklisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitDecision {
    /// Request allowed
    Allow {
        /// Number of remaining requests in current window
        remaining: u32,
        /// Time when the rate limit window resets
        reset_time: Instant,
    },
    /// Request denied - rate limit exceeded
    Deny {
        /// Duration to wait before retrying
        retry_after: Duration,
        /// Human-readable reason for denial
        reason: String,
    },
    /// Request denied - IP blacklisted
    Blacklisted {
        /// Time until blacklist expires
        until: Instant,
        /// Human-readable reason for blacklisting
        reason: String,
    },
}

/// Rate limiter statistics for monitoring and analysis
///
/// Provides comprehensive metrics about rate limiting performance and activity.
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Total number of requests processed
    pub total_requests: u64,
    /// Number of requests that were allowed
    pub allowed_requests: u64,
    /// Number of requests that were denied due to rate limiting
    pub denied_requests: u64,
    /// Number of currently blacklisted IP addresses
    pub blacklisted_ips: u32,
    /// Number of active rate limiting windows
    pub active_windows: u32,
}

/// Client information for rate limiting
#[derive(Debug, Clone)]
struct ClientInfo {
    request_count: u32,
    window_start: Instant,
    violation_count: u32,
    last_violation: Option<Instant>,
    blacklisted_until: Option<Instant>,
}

impl ClientInfo {
    fn new() -> Self {
        Self {
            request_count: 0,
            window_start: Instant::now(),
            violation_count: 0,
            last_violation: None,
            blacklisted_until: None,
        }
    }

    fn is_blacklisted(&self) -> bool {
        self.blacklisted_until
            .is_some_and(|until| Instant::now() < until)
    }

    fn reset_window(&mut self) {
        self.request_count = 0;
        self.window_start = Instant::now();
    }

    fn add_violation(&mut self, blacklist_duration: Duration, threshold: u32) {
        self.violation_count += 1;
        self.last_violation = Some(Instant::now());

        if self.violation_count >= threshold {
            self.blacklisted_until = Some(Instant::now() + blacklist_duration);
            warn!("IP blacklisted due to {} violations", self.violation_count);
        }
    }
}

/// High-performance rate limiter with `DDoS` protection
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: Arc<DashMap<IpAddr, ClientInfo>>,
    stats: Arc<RwLock<RateLimitStats>>,
    cleanup_interval: Duration,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    /// Create a new rate limiter with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Rate limiting configuration parameters
    ///
    /// # Returns
    ///
    /// A new `RateLimiter` instance ready for use
    #[must_use]
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(RateLimitStats {
                total_requests: 0,
                allowed_requests: 0,
                denied_requests: 0,
                blacklisted_ips: 0,
                active_windows: 0,
            })),
            cleanup_interval: Duration::from_secs(60),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Check if request should be allowed (< 1ms performance target)
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Internal` if internal state is corrupted
    pub async fn check_request(&self, client_ip: IpAddr) -> SecureStorageResult<RateLimitDecision> {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Periodic cleanup (non-blocking)
        self.maybe_cleanup().await;

        // Get or create client info
        let mut client_info = self
            .clients
            .entry(client_ip)
            .or_insert_with(ClientInfo::new);

        // Check if blacklisted
        if client_info.is_blacklisted() {
            if let Some(until) = client_info.blacklisted_until {
                // Update stats
                {
                    let mut stats = self.stats.write().await;
                    stats.denied_requests += 1;
                }

                return Ok(RateLimitDecision::Blacklisted {
                    until,
                    reason: format!(
                        "IP blacklisted due to {} violations",
                        client_info.violation_count
                    ),
                });
            }
        }

        let now = Instant::now();

        // Check if window has expired
        if now.duration_since(client_info.window_start) >= self.config.window_duration {
            client_info.reset_window();
        }

        // Check rate limit
        if client_info.request_count >= self.config.max_requests {
            // Rate limit exceeded
            client_info.add_violation(
                self.config.blacklist_duration,
                self.config.blacklist_threshold,
            );

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.denied_requests += 1;
            }

            let retry_after =
                self.config.window_duration - now.duration_since(client_info.window_start);

            return Ok(RateLimitDecision::Deny {
                retry_after,
                reason: "Rate limit exceeded".to_string(),
            });
        }

        // Allow request
        client_info.request_count += 1;
        let remaining = self.config.max_requests - client_info.request_count;
        let reset_time = client_info.window_start + self.config.window_duration;
        drop(client_info);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.allowed_requests += 1;
        }

        Ok(RateLimitDecision::Allow {
            remaining,
            reset_time,
        })
    }

    /// Get current statistics
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Internal` if stats cannot be read
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn get_stats(&self) -> SecureStorageResult<RateLimitStats> {
        let mut result = self.stats.read().await.clone();

        // Count blacklisted IPs
        result.blacklisted_ips = u32::try_from(
            self.clients
                .iter()
                .filter(|entry| entry.value().is_blacklisted())
                .count(),
        )
        .unwrap_or(u32::MAX);

        result.active_windows = u32::try_from(self.clients.len()).unwrap_or(u32::MAX);

        Ok(result)
    }

    /// Manually blacklist an IP address
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Internal` if blacklisting fails
    pub fn blacklist_ip(&self, ip: IpAddr, duration: Duration) -> SecureStorageResult<()> {
        {
            let mut client_info = self.clients.entry(ip).or_insert_with(ClientInfo::new);

            client_info.blacklisted_until = Some(Instant::now() + duration);
            client_info.violation_count += 1;
        }

        warn!("Manually blacklisted IP: {}", ip);
        Ok(())
    }

    /// Remove IP from blacklist
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::NotFound` if IP is not blacklisted
    pub fn unblacklist_ip(&self, ip: IpAddr) -> SecureStorageResult<()> {
        if let Some(mut client_info) = self.clients.get_mut(&ip) {
            if client_info.is_blacklisted() {
                client_info.blacklisted_until = None;
                client_info.violation_count = 0;
                debug!("Removed IP from blacklist: {}", ip);
                Ok(())
            } else {
                Err(SecureStorageError::NotFound {
                    resource: "blacklisted IP".to_string(),
                    identifier: ip.to_string(),
                })
            }
        } else {
            Err(SecureStorageError::NotFound {
                resource: "IP address".to_string(),
                identifier: ip.to_string(),
            })
        }
    }

    /// Get list of currently blacklisted IP addresses
    ///
    /// Returns a vector of tuples containing the IP address and the time until
    /// which it remains blacklisted.
    ///
    /// # Returns
    ///
    /// Vector of (IP address, blacklist expiry time) tuples
    #[must_use]
    pub fn get_blacklisted_ips(&self) -> Vec<(IpAddr, Instant)> {
        self.clients
            .iter()
            .filter_map(|entry| {
                let ip = *entry.key();
                let client = entry.value();
                client.blacklisted_until.and_then(|until| {
                    if Instant::now() < until {
                        Some((ip, until))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Cleanup expired entries (non-blocking)
    async fn maybe_cleanup(&self) {
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.read().await;
            last_cleanup.elapsed() >= self.cleanup_interval
        };

        if should_cleanup {
            // Spawn cleanup task to avoid blocking
            let clients = self.clients.clone();
            let last_cleanup = self.last_cleanup.clone();
            let window_duration = self.config.window_duration;

            tokio::spawn(async move {
                let now = Instant::now();
                let mut removed_count = 0_i32;

                // Remove expired entries
                clients.retain(|_ip, client| {
                    let window_expired =
                        now.duration_since(client.window_start) > window_duration * 2;
                    let blacklist_expired =
                        client.blacklisted_until.is_none_or(|until| now >= until);

                    let should_keep = !window_expired || !blacklist_expired;
                    if !should_keep {
                        removed_count += 1_i32;
                    }
                    should_keep
                });

                // Update last cleanup time
                {
                    let mut last = last_cleanup.write().await;
                    *last = now;
                }

                if removed_count > 0_i32 {
                    debug!("Cleaned up {} expired rate limit entries", removed_count);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{SecureStorageError, SecureStorageResult};
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    #[tokio::test]
    async fn test_rate_limiter_basic() -> SecureStorageResult<()> {
        let config = RateLimitConfig {
            max_requests: 5,
            window_duration: Duration::from_secs(60),
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        // First 5 requests should be allowed
        for i in 0..5 {
            let decision = limiter.check_request(ip).await?;
            match decision {
                RateLimitDecision::Allow { remaining, .. } => {
                    assert_eq!(remaining, 4 - i);
                }
                _ => {
                    return Err(SecureStorageError::InvalidInput {
                        field: "rate_limit_test".to_string(),
                        reason: "Request should be allowed".to_string(),
                    });
                }
            }
        }

        // 6th request should be denied
        let decision = limiter.check_request(ip).await?;
        match decision {
            RateLimitDecision::Deny { .. } => {
                // Expected
            }
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "rate_limit_test".to_string(),
                    reason: "Request should be denied".to_string(),
                });
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_blacklisting() -> SecureStorageResult<()> {
        let config = RateLimitConfig {
            max_requests: 1,
            blacklist_threshold: 2,
            blacklist_duration: Duration::from_secs(300),
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2));

        // Exceed rate limit twice to trigger blacklist
        for _ in 0_i32..3_i32 {
            let _ = limiter.check_request(ip).await?; // First allowed, second denied (violation 1)
            let _ = limiter.check_request(ip).await?; // Denied (violation 2, triggers blacklist)
        }

        // Next request should be blacklisted
        let decision = limiter.check_request(ip).await?;
        match decision {
            RateLimitDecision::Blacklisted { .. } => {
                // Expected
            }
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "blacklist_test".to_string(),
                    reason: "IP should be blacklisted".to_string(),
                });
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_manual_blacklist() -> SecureStorageResult<()> {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 3));

        // Manually blacklist IP
        limiter.blacklist_ip(ip, Duration::from_secs(60))?;

        // Request should be blacklisted
        let decision = limiter.check_request(ip).await?;
        match decision {
            RateLimitDecision::Blacklisted { .. } => {
                // Expected
            }
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "manual_blacklist_test".to_string(),
                    reason: "IP should be blacklisted".to_string(),
                });
            }
        }

        // Unblacklist IP
        limiter.unblacklist_ip(ip)?;

        // Request should now be allowed
        let decision = limiter.check_request(ip).await?;
        match decision {
            RateLimitDecision::Allow { .. } => {
                // Expected
            }
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "manual_blacklist_test".to_string(),
                    reason: "Request should be allowed after unblacklisting".to_string(),
                });
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_stats() -> SecureStorageResult<()> {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 4));

        // Make some requests
        for _ in 0_i32..3_i32 {
            let _ = limiter.check_request(ip).await?;
        }

        let stats = limiter.get_stats().await?;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.allowed_requests, 3);
        assert_eq!(stats.denied_requests, 0);

        Ok(())
    }
}
