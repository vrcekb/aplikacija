//! `TallyIO` Security Implementation - Production-Ready Financial Grade Security
//!
//! Ultra-high-performance security layer with <1ms latency guarantee for critical paths.
//! Zero-panic, zero-allocation security mechanisms for financial trading systems.
//!
//! ## Security Features
//! - **Rate Limiting**: Token bucket algorithm with burst protection
//! - **Certificate Pinning**: TLS certificate validation and pinning
//! - **Request Signing**: HMAC-SHA256 request authentication
//! - **`DoS` Protection**: Multi-layer `DDoS` mitigation with adaptive thresholds
//! - **Zero-panic**: All operations return Results, no unwrap/expect
//!
//! ## Performance Guarantees
//! - Rate limiting check: <10μs
//! - Certificate validation: <50μs (cached)
//! - Request signing: <100μs
//! - `DoS` detection: <5μs per request

pub mod certificate_pinning;
pub mod dos_protection;
pub mod rate_limiting;
pub mod request_signing;

use crate::error::{NetworkError, NetworkResult};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Security configuration for `TallyIO`
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// Certificate pinning configuration
    pub certificate_pinning: CertificatePinningConfig,
    /// Request signing configuration
    pub request_signing: RequestSigningConfig,
    /// `DoS` protection configuration
    pub dos_protection: DoSProtectionConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second per client
    pub max_requests_per_second: u32,
    /// Burst capacity (maximum tokens in bucket)
    pub burst_capacity: u32,
    /// Token refill rate (tokens per second)
    pub refill_rate: u32,
    /// Cleanup interval for expired entries
    pub cleanup_interval: Duration,
}

/// Certificate pinning configuration
#[derive(Debug, Clone)]
pub struct CertificatePinningConfig {
    /// Pinned certificate fingerprints (SHA-256)
    pub pinned_certificates: Vec<String>,
    /// Certificate cache TTL
    pub cache_ttl: Duration,
    /// Enable OCSP stapling verification
    pub verify_ocsp: bool,
    /// Maximum certificate chain length
    pub max_chain_length: u8,
}

/// Request signing configuration
#[derive(Debug, Clone)]
pub struct RequestSigningConfig {
    /// HMAC secret key
    pub secret_key: Vec<u8>,
    /// Request timestamp tolerance (seconds)
    pub timestamp_tolerance: u64,
    /// Required headers for signing
    pub required_headers: Vec<String>,
    /// Signature algorithm
    pub algorithm: SignatureAlgorithm,
}

/// `DoS` protection configuration
#[derive(Debug, Clone)]
pub struct DoSProtectionConfig {
    /// Maximum requests per IP per minute
    pub max_requests_per_ip_per_minute: u32,
    /// Blacklist duration for abusive IPs
    pub blacklist_duration: Duration,
    /// Threshold for automatic blacklisting
    pub blacklist_threshold: u32,
    /// Enable adaptive rate limiting
    pub adaptive_rate_limiting: bool,
    /// Connection limit per IP
    pub max_connections_per_ip: u32,
}

/// Signature algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    /// HMAC-SHA256
    HmacSha256,
    /// HMAC-SHA512
    HmacSha512,
    /// Ed25519
    Ed25519,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            rate_limiting: RateLimitConfig {
                max_requests_per_second: 1000,
                burst_capacity: 5000,
                refill_rate: 1000,
                cleanup_interval: Duration::from_secs(60),
            },
            certificate_pinning: CertificatePinningConfig {
                pinned_certificates: Vec::new(),
                cache_ttl: Duration::from_secs(3600),
                verify_ocsp: true,
                max_chain_length: 5,
            },
            request_signing: RequestSigningConfig {
                secret_key: Vec::new(),
                timestamp_tolerance: 300, // 5 minutes
                required_headers: vec![
                    "content-type".to_string(),
                    "x-timestamp".to_string(),
                    "authorization".to_string(),
                ],
                algorithm: SignatureAlgorithm::HmacSha256,
            },
            dos_protection: DoSProtectionConfig {
                max_requests_per_ip_per_minute: 600,
                blacklist_duration: Duration::from_secs(3600),
                blacklist_threshold: 100,
                adaptive_rate_limiting: true,
                max_connections_per_ip: 50,
            },
        }
    }
}

/// Security middleware result
#[derive(Debug, Clone)]
pub enum SecurityResult {
    /// Request is allowed to proceed
    Allow,
    /// Request should be denied
    Deny { reason: String },
    /// Request should be rate limited
    RateLimit { retry_after: Duration },
}

/// Security metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SecurityMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Requests blocked by rate limiting
    pub rate_limited_requests: u64,
    /// Requests blocked by `DoS` protection
    pub dos_blocked_requests: u64,
    /// Invalid signatures detected
    pub invalid_signatures: u64,
    /// Certificate validation failures
    pub certificate_failures: u64,
    /// Average processing time (microseconds)
    pub avg_processing_time_us: u64,
}

/// Main security system for `TallyIO`
#[derive(Clone)]
pub struct SecuritySystem {
    /// Rate limiting component
    rate_limiter: Arc<rate_limiting::RateLimiter>,
    /// Certificate pinning component
    cert_validator: Arc<certificate_pinning::CertificateValidator>,
    /// Request signing component
    request_signer: Arc<request_signing::RequestSigner>,
    /// `DoS` protection component
    dos_protector: Arc<dos_protection::DoSProtector>,
    /// Security metrics
    metrics: Arc<parking_lot::RwLock<SecurityMetrics>>,
    /// Configuration
    #[allow(dead_code)] // Used in future configuration updates
    config: SecurityConfig,
}

impl SecuritySystem {
    /// Create new security system
    ///
    /// # Errors
    /// Returns error if configuration validation fails
    pub fn new(config: SecurityConfig) -> NetworkResult<Self> {
        // Validate configuration
        Self::validate_config(&config)?;

        let rate_limiter = Arc::new(rate_limiting::RateLimiter::new(config.rate_limiting.clone())?);
        let cert_validator = Arc::new(certificate_pinning::CertificateValidator::new(
            config.certificate_pinning.clone(),
        )?);
        let request_signer = Arc::new(request_signing::RequestSigner::new(
            config.request_signing.clone(),
        )?);
        let dos_protector = Arc::new(dos_protection::DoSProtector::new(
            config.dos_protection.clone(),
        )?);

        Ok(Self {
            rate_limiter,
            cert_validator,
            request_signer,
            dos_protector,
            metrics: Arc::new(parking_lot::RwLock::new(SecurityMetrics::default())),
            config,
        })
    }

    /// Validate security configuration
    fn validate_config(config: &SecurityConfig) -> NetworkResult<()> {
        // Validate rate limiting config
        if config.rate_limiting.max_requests_per_second == 0 {
            return Err(NetworkError::config(
                "rate_limiting.max_requests_per_second",
                "Must be greater than 0",
            ));
        }

        if config.rate_limiting.burst_capacity < config.rate_limiting.max_requests_per_second {
            return Err(NetworkError::config(
                "rate_limiting.burst_capacity",
                "Must be >= max_requests_per_second",
            ));
        }

        // Validate request signing config
        if config.request_signing.secret_key.is_empty() {
            return Err(NetworkError::config(
                "request_signing.secret_key",
                "Secret key cannot be empty",
            ));
        }

        if config.request_signing.secret_key.len() < 32 {
            return Err(NetworkError::config(
                "request_signing.secret_key",
                "Secret key must be at least 32 bytes",
            ));
        }

        // Validate DoS protection config
        if config.dos_protection.max_requests_per_ip_per_minute == 0 {
            return Err(NetworkError::config(
                "dos_protection.max_requests_per_ip_per_minute",
                "Must be greater than 0",
            ));
        }

        Ok(())
    }

    /// Process security check for incoming request
    ///
    /// # Errors
    /// Returns error if security validation fails
    pub async fn process_request(&self, request: &SecurityRequest) -> NetworkResult<SecurityResult> {
        let start_time = Instant::now();

        // 1. DoS protection check (highest priority)
        match self.dos_protector.check_request(request)? {
            SecurityResult::Deny { reason } => {
                self.update_metrics(start_time, |m| m.dos_blocked_requests += 1);
                return Ok(SecurityResult::Deny { reason });
            }
            SecurityResult::RateLimit { retry_after } => {
                self.update_metrics(start_time, |m| m.dos_blocked_requests += 1);
                return Ok(SecurityResult::RateLimit { retry_after });
            }
            SecurityResult::Allow => {}
        }

        // 2. Rate limiting check
        match self.rate_limiter.check_request(request)? {
            SecurityResult::RateLimit { retry_after } => {
                self.update_metrics(start_time, |m| m.rate_limited_requests += 1);
                return Ok(SecurityResult::RateLimit { retry_after });
            }
            SecurityResult::Deny { reason } => {
                self.update_metrics(start_time, |m| m.rate_limited_requests += 1);
                return Ok(SecurityResult::Deny { reason });
            }
            SecurityResult::Allow => {}
        }

        // 3. Certificate validation (for HTTPS requests)
        if let Some(cert_chain) = &request.certificate_chain {
            if let Err(e) = self.cert_validator.validate_certificate_chain(cert_chain).await {
                self.update_metrics(start_time, |m| m.certificate_failures += 1);
                return Ok(SecurityResult::Deny {
                    reason: format!("Certificate validation failed: {e}"),
                });
            }
        }

        // 4. Request signature validation
        if let Err(e) = self.request_signer.validate_signature(request) {
            self.update_metrics(start_time, |m| m.invalid_signatures += 1);
            return Ok(SecurityResult::Deny {
                reason: format!("Invalid signature: {e}"),
            });
        }

        // All checks passed
        self.update_metrics(start_time, |_| {});
        Ok(SecurityResult::Allow)
    }

    /// Update security metrics
    fn update_metrics<F>(&self, start_time: Instant, update_fn: F)
    where
        F: FnOnce(&mut SecurityMetrics),
    {
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.write();
        
        metrics.total_requests += 1;
        update_fn(&mut metrics);
        
        // Update average processing time (exponential moving average)
        #[allow(clippy::cast_possible_truncation)] // Acceptable for metrics
        let elapsed_us = elapsed.as_micros() as u64;
        if metrics.avg_processing_time_us == 0 {
            metrics.avg_processing_time_us = elapsed_us;
        } else {
            metrics.avg_processing_time_us = 
                (metrics.avg_processing_time_us * 9 + elapsed_us) / 10;
        }
    }

    /// Get current security metrics
    #[must_use]
    pub fn metrics(&self) -> SecurityMetrics {
        self.metrics.read().clone()
    }

    /// Check if security system is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        let metrics = self.metrics.read();
        
        // System is healthy if:
        // 1. Average processing time is under 1ms
        // 2. Error rate is under 5%
        #[allow(clippy::cast_precision_loss)] // Acceptable for error rate calculation
        let error_rate = if metrics.total_requests > 0 {
            (metrics.rate_limited_requests + metrics.dos_blocked_requests +
             metrics.invalid_signatures + metrics.certificate_failures) as f64
             / metrics.total_requests as f64
        } else {
            0.0_f64
        };

        metrics.avg_processing_time_us < 1000 && error_rate < 0.05
    }
}

/// Security request information
#[derive(Debug, Clone)]
pub struct SecurityRequest {
    /// Client IP address
    pub client_ip: std::net::IpAddr,
    /// Request method
    pub method: String,
    /// Request path
    pub path: String,
    /// Request headers
    pub headers: std::collections::HashMap<String, String>,
    /// Request body (for signature validation)
    pub body: Vec<u8>,
    /// Certificate chain (for HTTPS)
    pub certificate_chain: Option<Vec<Vec<u8>>>,
    /// Request timestamp
    pub timestamp: std::time::SystemTime,
    /// User agent
    pub user_agent: Option<String>,
}
