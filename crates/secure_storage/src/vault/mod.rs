//! Vault implementations for secure storage

use crate::error::SecureStorageResult;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "vault")]
pub mod hashicorp_vault;
pub mod local_vault;

/// Trait for vault operations
#[async_trait]
pub trait Vault: Send + Sync {
    /// Store a value in the vault
    async fn store(&self, key: &str, value: &[u8]) -> SecureStorageResult<()>;

    /// Retrieve a value from the vault
    async fn retrieve(&self, key: &str) -> SecureStorageResult<Vec<u8>>;

    /// Delete a value from the vault
    async fn delete(&self, key: &str) -> SecureStorageResult<()>;

    /// List keys with optional prefix filter
    async fn list_keys(&self, prefix: &str) -> SecureStorageResult<Vec<String>>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> SecureStorageResult<bool>;

    /// Get metadata for a key
    async fn get_metadata(&self, key: &str) -> SecureStorageResult<VaultMetadata>;

    /// Store value with metadata
    async fn store_with_metadata(
        &self,
        key: &str,
        value: &[u8],
        metadata: VaultMetadata,
    ) -> SecureStorageResult<()>;

    /// Batch operations
    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> SecureStorageResult<()>;

    /// Retrieve multiple values by their keys in a single operation
    async fn batch_retrieve(
        &self,
        keys: Vec<String>,
    ) -> SecureStorageResult<HashMap<String, Vec<u8>>>;

    /// Delete multiple keys in a single operation
    async fn batch_delete(&self, keys: Vec<String>) -> SecureStorageResult<()>;

    /// Health check
    async fn health_check(&self) -> SecureStorageResult<VaultHealth>;

    /// Get vault statistics
    async fn get_stats(&self) -> SecureStorageResult<VaultStats>;
}

/// Comprehensive metadata associated with vault entries.
///
/// This structure contains all metadata information for entries stored in
/// the vault, including timestamps, content information, expiration settings,
/// and custom tags. Metadata is essential for vault management, lifecycle
/// policies, and operational monitoring.
///
/// # Metadata Components
///
/// ## Timestamps
/// - **Created At**: When the entry was first stored
/// - **Modified At**: When the entry was last updated
/// - **Accessed At**: When the entry was last retrieved (optional)
/// - **Expires At**: When the entry should be automatically deleted (optional)
///
/// ## Content Information
/// - **Content Type**: MIME type or format identifier
/// - **Size**: Size of the stored data in bytes
/// - **Checksum**: Integrity verification hash (optional)
/// - **Compression**: Compression algorithm used (optional)
///
/// ## Lifecycle Management
/// - **Expiration**: Automatic deletion based on time or usage
/// - **Retention Policy**: Legal or compliance-based retention requirements
/// - **Backup Status**: Whether entry is included in backups
/// - **Archival**: Long-term storage and retrieval policies
///
/// # Custom Tags
///
/// Tags provide flexible metadata extension:
/// - **Classification**: Security classification levels
/// - **Ownership**: User or department ownership information
/// - **Purpose**: Intended use or application context
/// - **Environment**: Development, staging, production indicators
///
/// # Use Cases
///
/// - **Lifecycle Management**: Automatic cleanup of expired entries
/// - **Access Control**: Tag-based permission policies
/// - **Monitoring**: Usage patterns and access analytics
/// - **Compliance**: Audit trails and retention compliance
/// - **Operations**: Backup, archival, and disaster recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultMetadata {
    /// Entry creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
    /// Last access timestamp
    pub accessed_at: Option<DateTime<Utc>>,
    /// Content type
    pub content_type: String,
    /// Content size in bytes
    pub size: usize,
    /// Custom tags
    pub tags: HashMap<String, String>,
    /// Expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,
    /// Version number
    pub version: u64,
}

impl VaultMetadata {
    /// Creates new metadata for a vault entry with current timestamp.
    ///
    /// This constructor initializes metadata with the essential information
    /// required for a new vault entry. Additional metadata can be added
    /// using the builder methods after creation.
    ///
    /// # Arguments
    ///
    /// * `content_type` - MIME type or format identifier for the stored data
    /// * `size` - Size of the data in bytes
    ///
    /// # Returns
    ///
    /// A new `VaultMetadata` instance with:
    /// - Current timestamp for creation and modification
    /// - Specified content type and size
    /// - Empty tags collection
    /// - No expiration set
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultMetadata;
    ///
    /// // Create metadata for a JSON configuration file
    /// let metadata = VaultMetadata::new("application/json".to_string(), 1024);
    ///
    /// // Create metadata for binary key material
    /// let key_metadata = VaultMetadata::new("application/octet-stream".to_string(), 32);
    /// ```
    ///
    /// # Performance
    ///
    /// This method performs minimal work and completes in microseconds.
    /// The timestamp generation uses system clock which is very fast.
    #[must_use]
    pub fn new(content_type: String, size: usize) -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            modified_at: now,
            accessed_at: None,
            content_type,
            size,
            tags: HashMap::new(),
            expires_at: None,
            version: 1,
        }
    }

    /// Determines if the vault entry has expired based on current time.
    ///
    /// This method checks whether the entry has passed its expiration time
    /// and should be considered invalid or eligible for automatic cleanup.
    /// Expired entries should not be returned to clients and may be
    /// automatically deleted by cleanup processes.
    ///
    /// # Returns
    ///
    /// `true` if the entry has an expiration time and the current time
    /// is past that expiration, `false` if the entry has not expired
    /// or has no expiration set.
    ///
    /// # Expiration Behavior
    ///
    /// - Entries with no expiration time (`expires_at` is `None`) never expire
    /// - Expiration is checked against UTC time to avoid timezone issues
    /// - Clock skew between systems should be considered in distributed environments
    /// - Expired entries should be handled gracefully by calling code
    ///
    /// # Use Cases
    ///
    /// - **Automatic Cleanup**: Background processes removing expired entries
    /// - **Access Control**: Preventing access to expired credentials
    /// - **Cache Management**: Invalidating cached data based on expiration
    /// - **Compliance**: Enforcing data retention policies
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultMetadata;
    /// use chrono::{Utc, Duration};
    ///
    /// let mut metadata = VaultMetadata::new("text/plain".to_string(), 100);
    ///
    /// // Set expiration to 1 hour from now
    /// metadata.expires_at = Some(Utc::now() + Duration::hours(1));
    /// assert!(!metadata.is_expired()); // Should not be expired yet
    ///
    /// // Set expiration to 1 hour ago
    /// metadata.expires_at = Some(Utc::now() - Duration::hours(1));
    /// assert!(metadata.is_expired()); // Should be expired
    /// ```
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .is_some_and(|expires_at| Utc::now() > expires_at)
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        self.accessed_at = Some(Utc::now());
    }

    /// Update modification time and increment version
    pub fn mark_modified(&mut self, new_size: usize) {
        self.modified_at = Utc::now();
        self.size = new_size;
        self.version += 1;
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: String, value: String) {
        self.tags.insert(key, value);
    }

    /// Remove a tag
    pub fn remove_tag(&mut self, key: &str) {
        self.tags.remove(key);
    }

    /// Checks if the metadata contains a specific tag key.
    ///
    /// This method determines whether a tag with the specified key exists
    /// in the metadata's tag collection. It performs a case-sensitive
    /// key lookup and returns a boolean result.
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to search for (case-sensitive)
    ///
    /// # Returns
    ///
    /// `true` if a tag with the specified key exists, `false` otherwise.
    ///
    /// # Performance
    ///
    /// This method uses a hash map lookup with O(1) average complexity.
    /// Performance is excellent even with large numbers of tags.
    ///
    /// # Use Cases
    ///
    /// - **Conditional Logic**: Execute different code paths based on tag presence
    /// - **Validation**: Ensure required tags are present before processing
    /// - **Filtering**: Select entries based on tag criteria
    /// - **Access Control**: Check for security classification tags
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultMetadata;
    ///
    /// let mut metadata = VaultMetadata::new("application/json".to_string(), 512);
    /// metadata.add_tag("environment".to_string(), "production".to_string());
    /// metadata.add_tag("owner".to_string(), "security-team".to_string());
    ///
    /// assert!(metadata.has_tag("environment"));
    /// assert!(metadata.has_tag("owner"));
    /// assert!(!metadata.has_tag("Environment")); // Case-sensitive
    /// assert!(!metadata.has_tag("classification"));
    /// ```
    #[must_use]
    pub fn has_tag(&self, key: &str) -> bool {
        self.tags.contains_key(key)
    }

    /// Retrieves the value of a specific tag by its key.
    ///
    /// This method performs a lookup in the metadata's tag collection
    /// and returns a reference to the tag value if found. The lookup
    /// is case-sensitive and returns `None` if the key is not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The tag key to look up (case-sensitive)
    ///
    /// # Returns
    ///
    /// `Some(&String)` containing a reference to the tag value if the key exists,
    /// `None` if the key is not found in the tag collection.
    ///
    /// # Performance
    ///
    /// This method uses a hash map lookup with O(1) average complexity.
    /// The returned reference avoids unnecessary string cloning.
    ///
    /// # Lifetime
    ///
    /// The returned reference is tied to the lifetime of the metadata object.
    /// The reference remains valid as long as the metadata is not modified.
    ///
    /// # Use Cases
    ///
    /// - **Configuration**: Retrieve configuration values stored as tags
    /// - **Classification**: Get security classification levels
    /// - **Ownership**: Determine entry ownership information
    /// - **Context**: Access application-specific metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultMetadata;
    ///
    /// let mut metadata = VaultMetadata::new("application/json".to_string(), 256);
    /// metadata.add_tag("classification".to_string(), "confidential".to_string());
    /// metadata.add_tag("department".to_string(), "engineering".to_string());
    ///
    /// // Retrieve existing tags
    /// assert_eq!(metadata.get_tag("classification"), Some(&"confidential".to_string()));
    /// assert_eq!(metadata.get_tag("department"), Some(&"engineering".to_string()));
    ///
    /// // Non-existent tag returns None
    /// assert_eq!(metadata.get_tag("nonexistent"), None);
    /// ```
    #[must_use]
    pub fn get_tag(&self, key: &str) -> Option<&String> {
        self.tags.get(key)
    }
}

/// Comprehensive health status information for vault backends.
///
/// This structure provides detailed health and operational status information
/// for vault backends, enabling monitoring, alerting, and operational decision
/// making. Health checks are essential for maintaining system reliability
/// and detecting issues before they impact operations.
///
/// # Health Monitoring
///
/// Health status includes:
/// - **Overall Status**: High-level health indicator (Healthy, Warning, Critical, Unknown)
/// - **Check Timestamp**: When the health check was performed
/// - **Response Time**: How long the health check took to complete
/// - **Detailed Information**: Specific metrics and diagnostic data
///
/// # Health Check Types
///
/// ## Connectivity Checks
/// - Network connectivity to vault backend
/// - Authentication and authorization validation
/// - SSL/TLS certificate validation
/// - Connection pool status
///
/// ## Performance Checks
/// - Response time measurements
/// - Throughput capacity assessment
/// - Resource utilization monitoring
/// - Queue depth and backlog analysis
///
/// ## Functional Checks
/// - Basic read/write operations
/// - Encryption/decryption functionality
/// - Key management operations
/// - Backup and recovery capabilities
///
/// # Monitoring Integration
///
/// Health information can be integrated with:
/// - **Monitoring Systems**: Prometheus, Grafana, `DataDog`
/// - **Alerting Platforms**: `PagerDuty`, Slack, email notifications
/// - **Load Balancers**: Health-based traffic routing
/// - **Orchestration**: Kubernetes health checks and readiness probes
///
/// # Operational Use Cases
///
/// - **Service Discovery**: Determine which vault instances are available
/// - **Load Balancing**: Route traffic to healthy instances
/// - **Alerting**: Notify operators of degraded performance
/// - **Capacity Planning**: Monitor resource utilization trends
/// - **Incident Response**: Provide diagnostic information during outages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultHealth {
    /// Overall health status
    pub status: HealthStatus,
    /// Health check timestamp
    pub checked_at: DateTime<Utc>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Additional health details
    pub details: HashMap<String, String>,
}

/// Enumeration of possible health status values for vault backends.
///
/// This enumeration provides a standardized way to represent the operational
/// health of vault backends, enabling consistent monitoring and alerting
/// across different vault types and deployment environments.
///
/// # Status Levels
///
/// ## Healthy
/// - All systems operating normally
/// - Performance within acceptable parameters
/// - No known issues or degradation
/// - Ready to handle full operational load
///
/// ## Warning
/// - Minor issues detected but service remains operational
/// - Performance may be degraded but within acceptable limits
/// - Potential issues that may require attention
/// - Service can continue but should be monitored closely
///
/// ## Critical
/// - Significant issues affecting service operation
/// - Performance severely degraded or intermittent failures
/// - Service may be unavailable or unreliable
/// - Immediate attention required to restore normal operation
///
/// ## Unknown
/// - Health status cannot be determined
/// - Health check failed or timed out
/// - Communication issues with vault backend
/// - Assume degraded service until status can be confirmed
///
/// # Operational Guidelines
///
/// - **Healthy**: Normal operations, no action required
/// - **Warning**: Monitor closely, investigate during maintenance windows
/// - **Critical**: Immediate investigation and remediation required
/// - **Unknown**: Treat as critical until status can be confirmed
///
/// # Monitoring Integration
///
/// Status values map to standard monitoring concepts:
/// - **Healthy** → Green/OK status
/// - **Warning** → Yellow/Warning status
/// - **Critical** → Red/Critical status
/// - **Unknown** → Gray/Unknown status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    /// Vault is healthy and operational
    Healthy,
    /// Vault has minor issues but is operational
    Warning,
    /// Vault has critical issues
    Critical,
    /// Vault is unreachable
    Unreachable,
}

impl HealthStatus {
    /// Determines if the health status indicates a fully healthy state.
    ///
    /// This method provides a simple boolean check for whether the vault
    /// backend is in a completely healthy state with no known issues.
    /// This is the most restrictive health check, returning `true` only
    /// for the `Healthy` status.
    ///
    /// # Returns
    ///
    /// `true` if the status is `Healthy`, `false` for all other statuses
    /// including `Warning`, `Critical`, and `Unknown`.
    ///
    /// # Use Cases
    ///
    /// - **Strict Health Checks**: Only proceed with operations if fully healthy
    /// - **Load Balancing**: Route traffic only to completely healthy instances
    /// - **Automated Decisions**: Make conservative operational choices
    /// - **SLA Compliance**: Ensure service level agreement requirements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::HealthStatus;
    ///
    /// assert!(HealthStatus::Healthy.is_healthy());
    /// assert!(!HealthStatus::Warning.is_healthy());
    /// assert!(!HealthStatus::Critical.is_healthy());
    /// assert!(!HealthStatus::Unreachable.is_healthy());
    /// ```
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime cost, performing only
    /// a simple pattern match at compile time when possible.
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Determines if the health status indicates the service is operational.
    ///
    /// This method provides a more permissive health check that considers
    /// the vault backend operational if it can handle requests, even if
    /// there are minor issues. This includes both `Healthy` and `Warning`
    /// statuses.
    ///
    /// # Returns
    ///
    /// `true` if the status is `Healthy` or `Warning`, `false` for
    /// `Critical` and `Unknown` statuses.
    ///
    /// # Operational Definition
    ///
    /// A vault is considered operational if:
    /// - It can accept and process requests
    /// - Basic functionality is available
    /// - Performance may be degraded but within acceptable limits
    /// - Service can continue with monitoring
    ///
    /// # Use Cases
    ///
    /// - **Graceful Degradation**: Continue operations with reduced performance
    /// - **Load Balancing**: Include instances with minor issues in rotation
    /// - **Maintenance Windows**: Determine if maintenance can be deferred
    /// - **Failover Decisions**: Decide whether to trigger failover procedures
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::HealthStatus;
    ///
    /// assert!(HealthStatus::Healthy.is_operational());
    /// assert!(HealthStatus::Warning.is_operational());
    /// assert!(!HealthStatus::Critical.is_operational());
    /// assert!(!HealthStatus::Unreachable.is_operational());
    /// ```
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime cost, performing only
    /// a simple pattern match at compile time when possible.
    #[must_use]
    pub const fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Warning)
    }
}

/// Comprehensive statistics and metrics for vault operations.
///
/// This structure provides detailed operational statistics for vault backends,
/// enabling performance monitoring, capacity planning, and operational insights.
/// Statistics are essential for understanding vault usage patterns and
/// optimizing system performance.
///
/// # Statistical Categories
///
/// ## Storage Metrics
/// - **Total Entries**: Number of items stored in the vault
/// - **Total Size**: Aggregate size of all stored data in bytes
/// - **Average Entry Size**: Mean size of individual entries
/// - **Storage Utilization**: Percentage of available storage used
///
/// ## Operational Metrics
/// - **Operations Count**: Total number of operations performed
/// - **Success Rate**: Percentage of successful operations
/// - **Error Rate**: Percentage of failed operations
/// - **Average Response Time**: Mean operation latency
///
/// ## Performance Metrics
/// - **Throughput**: Operations per second over time windows
/// - **Latency Percentiles**: P50, P95, P99 response times
/// - **Queue Depth**: Number of pending operations
/// - **Connection Pool Utilization**: Active vs. available connections
///
/// ## Custom Metrics
///
/// The metrics map allows for backend-specific or application-specific
/// metrics to be included:
/// - Database-specific metrics (cache hit rates, index usage)
/// - Network metrics (bandwidth utilization, packet loss)
/// - Security metrics (authentication failures, rate limiting)
/// - Business metrics (cost per operation, SLA compliance)
///
/// # Monitoring Integration
///
/// Statistics can be exported to:
/// - **Time Series Databases**: `InfluxDB`, Prometheus, `CloudWatch`
/// - **Monitoring Dashboards**: Grafana, Kibana, `DataDog`
/// - **Alerting Systems**: Based on threshold violations
/// - **Capacity Planning Tools**: For resource allocation decisions
///
/// # Performance Considerations
///
/// - Statistics collection should have minimal performance impact
/// - Use sampling for high-frequency metrics to reduce overhead
/// - Aggregate statistics periodically rather than on every operation
/// - Consider using approximate algorithms for large-scale metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStats {
    /// Total number of entries
    pub total_entries: u64,
    /// Total storage size in bytes
    pub total_size_bytes: u64,
    /// Number of operations performed
    pub operations_count: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Statistics timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl VaultStats {
    /// Creates a new statistics instance with all counters initialized to zero.
    ///
    /// This constructor initializes a fresh statistics object with default
    /// values suitable for a new vault instance or for resetting statistics.
    /// All numeric counters start at zero and the metrics map is empty.
    ///
    /// # Returns
    ///
    /// A new `VaultStats` instance with:
    /// - All counters set to zero
    /// - Empty metrics map for custom statistics
    /// - Ready for incremental updates as operations occur
    ///
    /// # Use Cases
    ///
    /// - **New Vault Instances**: Initialize statistics for new vault backends
    /// - **Statistics Reset**: Clear existing statistics for fresh monitoring periods
    /// - **Testing**: Create clean statistics objects for unit tests
    /// - **Baseline Establishment**: Start fresh measurement periods
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultStats;
    ///
    /// let stats = VaultStats::new();
    /// assert_eq!(stats.total_entries, 0);
    /// assert_eq!(stats.total_size_bytes, 0);
    /// assert_eq!(stats.operations_count, 0);
    /// assert!(stats.metrics.is_empty());
    /// ```
    ///
    /// # Performance
    ///
    /// This method performs minimal work, only initializing struct fields
    /// and creating an empty `HashMap`. Execution time is in nanoseconds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_entries: 0,
            total_size_bytes: 0,
            operations_count: 0,
            avg_response_time_ms: 0.0,
            error_rate: 0.0,
            timestamp: Utc::now(),
            metrics: HashMap::new(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Retrieves a custom metric value by name.
    ///
    /// This method looks up a custom metric in the metrics map and returns
    /// its value if found. Custom metrics allow vault backends to expose
    /// implementation-specific statistics and performance indicators.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the metric to retrieve (case-sensitive)
    ///
    /// # Returns
    ///
    /// `Some(f64)` containing the metric value if found, `None` if the
    /// metric name is not present in the metrics map.
    ///
    /// # Metric Naming Conventions
    ///
    /// Recommended metric naming patterns:
    /// - **Counters**: `operation_count`, `error_count`, `bytes_transferred`
    /// - **Gauges**: `active_connections`, `memory_usage_bytes`, `cpu_utilization`
    /// - **Histograms**: `response_time_p95`, `request_size_avg`, `latency_max`
    /// - **Rates**: `requests_per_second`, `errors_per_minute`, `throughput_mbps`
    ///
    /// # Performance
    ///
    /// This method uses a hash map lookup with O(1) average complexity.
    /// The lookup is very fast even with large numbers of metrics.
    ///
    /// # Use Cases
    ///
    /// - **Monitoring**: Export metrics to monitoring systems
    /// - **Alerting**: Check metric values against thresholds
    /// - **Debugging**: Inspect internal performance counters
    /// - **Capacity Planning**: Analyze resource utilization trends
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultStats;
    ///
    /// let mut stats = VaultStats::new();
    /// stats.add_metric("cache_hit_rate".to_string(), 0.95);
    /// stats.add_metric("average_latency_ms".to_string(), 2.5);
    ///
    /// assert_eq!(stats.get_metric("cache_hit_rate"), Some(0.95));
    /// assert_eq!(stats.get_metric("average_latency_ms"), Some(2.5));
    /// assert_eq!(stats.get_metric("nonexistent_metric"), None);
    /// ```
    #[must_use]
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }
}

impl Default for VaultStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating vault instances
pub struct VaultFactory;

impl VaultFactory {
    /// Create a local vault instance
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn create_local(database_path: &str) -> SecureStorageResult<Box<dyn Vault>> {
        let vault = local_vault::LocalVault::new(database_path).await?;
        Ok(Box::new(vault))
    }

    /// Creates a `HashiCorp` Vault backend instance with the specified configuration.
    ///
    /// This method initializes a connection to a `HashiCorp` Vault server using
    /// the provided configuration parameters. It establishes authentication,
    /// validates connectivity, and prepares the vault for secure operations.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration containing connection details, authentication,
    ///   and security settings for the `HashiCorp` Vault instance
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing the `Vault` trait, ready for use
    /// in secure storage operations.
    ///
    /// # Configuration Requirements
    ///
    /// The configuration must include:
    /// - **URL**: `HashiCorp` Vault server endpoint (`https://vault.example.com:8200`)
    /// - **Authentication**: Token, `AppRole`, or other authentication method
    /// - **Mount Path**: KV secrets engine mount point (default: `secret/`)
    /// - **TLS Settings**: Certificate validation and client certificates
    ///
    /// # Authentication Methods
    ///
    /// Supported authentication methods:
    /// - **Token**: Direct vault token authentication
    /// - **`AppRole`**: Application-based role authentication
    /// - **AWS IAM**: AWS identity-based authentication
    /// - **Kubernetes**: Service account token authentication
    /// - **LDAP**: Directory service authentication
    ///
    /// # Security Considerations
    ///
    /// - Always use HTTPS in production environments
    /// - Validate TLS certificates to prevent man-in-the-middle attacks
    /// - Use short-lived tokens and implement token renewal
    /// - Store authentication credentials securely (not in configuration files)
    /// - Enable audit logging on the `HashiCorp` Vault server
    ///
    /// # Performance
    ///
    /// Initial connection establishment may take several seconds depending on:
    /// - Network latency to vault server
    /// - Authentication method complexity
    /// - TLS handshake overhead
    /// - Vault server load and configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::vault::VaultFactory;
    /// use secure_storage::config::VaultConfig;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = VaultConfig {
    ///     // Configuration details...
    /// };
    ///
    /// let vault = VaultFactory::create_hashicorp(config).await?;
    /// // Vault is ready for secure operations
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "vault")]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Network connection to vault server fails
    /// - Authentication credentials are invalid or expired
    /// - TLS certificate validation fails
    /// - Vault server is sealed or in maintenance mode
    /// - Required secrets engine is not mounted or accessible
    /// - Configuration parameters are invalid or incomplete
    pub fn create_hashicorp(
        url: &str,
        token: &str,
        mount_path: &str,
    ) -> SecureStorageResult<Box<dyn Vault>> {
        let vault = hashicorp_vault::HashiCorpVault::new(url, token, mount_path)?;
        Ok(Box::new(vault))
    }

    /// Create vault from configuration
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn from_config(
        config: &crate::config::SecureStorageConfig,
    ) -> SecureStorageResult<Box<dyn Vault>> {
        match config.vault.vault_type {
            crate::types::VaultType::Local => {
                Self::create_local(&config.vault.connection.url).await
            }
            #[cfg(feature = "vault")]
            crate::types::VaultType::HashiCorp => {
                let token = config
                    .vault
                    .connection
                    .parameters
                    .get("token")
                    .ok_or_else(|| crate::error::SecureStorageError::Configuration {
                        field: "vault.token".to_string(),
                        reason: "HashiCorp Vault token not provided".to_string(),
                    })?;
                let default_mount = "secret".to_string();
                let mount_path = config
                    .vault
                    .connection
                    .parameters
                    .get("mount_path")
                    .unwrap_or(&default_mount);

                Self::create_hashicorp(&config.vault.connection.url, token, mount_path)
            }
            #[cfg(not(feature = "vault"))]
            crate::types::VaultType::HashiCorp => {
                Err(crate::error::SecureStorageError::Configuration {
                    field: "vault_type".to_string(),
                    reason: "HashiCorp Vault support not enabled (missing 'vault' feature)"
                        .to_string(),
                })
            }
            crate::types::VaultType::Hsm => Err(crate::error::SecureStorageError::Configuration {
                field: "vault_type".to_string(),
                reason: "HSM vault not yet implemented".to_string(),
            }),
        }
    }
}

/// Utility functions for vault operations
pub mod utils {
    use super::SecureStorageResult;

    /// Validate key format
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn validate_key(key: &str) -> SecureStorageResult<()> {
        if key.is_empty() {
            return Err(crate::error::SecureStorageError::InvalidInput {
                field: "key".to_string(),
                reason: "Key cannot be empty".to_string(),
            });
        }

        if key.len() > 1024 {
            return Err(crate::error::SecureStorageError::InvalidInput {
                field: "key".to_string(),
                reason: "Key too long (max 1024 characters)".to_string(),
            });
        }

        // Check for invalid characters
        if key.contains('\0') {
            return Err(crate::error::SecureStorageError::InvalidInput {
                field: "key".to_string(),
                reason: "Key cannot contain null characters".to_string(),
            });
        }

        Ok(())
    }

    /// Validate value size
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn validate_value_size(value: &[u8], max_size: usize) -> SecureStorageResult<()> {
        if value.len() > max_size {
            return Err(crate::error::SecureStorageError::InvalidInput {
                field: "value".to_string(),
                reason: format!(
                    "Value too large ({} bytes, max {} bytes)",
                    value.len(),
                    max_size
                ),
            });
        }
        Ok(())
    }

    /// Generate a unique key with prefix
    #[must_use]
    /// TODO: Add documentation
    pub fn generate_unique_key(prefix: &str) -> String {
        format!("{}/{}", prefix, uuid::Uuid::new_v4())
    }

    /// Parse key components
    #[must_use]
    /// TODO: Add documentation
    pub fn parse_key(key: &str) -> (Option<&str>, &str) {
        key.rfind('/').map_or((None, key), |pos| {
            let (prefix, name) = key.split_at(pos);
            (Some(prefix), &name[1..]) // Skip the '/'
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_metadata() {
        let mut metadata = VaultMetadata::new("application/json".to_string(), 1024);

        assert_eq!(metadata.content_type, "application/json");
        assert_eq!(metadata.size, 1024);
        assert_eq!(metadata.version, 1);
        assert!(!metadata.is_expired());

        // Test tagging
        metadata.add_tag("environment".to_string(), "production".to_string());
        assert!(metadata.has_tag("environment"));
        assert_eq!(
            metadata.get_tag("environment"),
            Some(&"production".to_string())
        );

        // Test modification
        metadata.mark_modified(2048);
        assert_eq!(metadata.size, 2048);
        assert_eq!(metadata.version, 2);
    }

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(HealthStatus::Healthy.is_operational());
        assert!(HealthStatus::Warning.is_operational());
        assert!(!HealthStatus::Critical.is_operational());
        assert!(!HealthStatus::Unreachable.is_operational());
    }

    #[test]
    fn test_vault_stats() {
        let mut stats = VaultStats::new();

        stats.add_metric("cpu_usage".to_string(), 0.75);
        assert_eq!(stats.get_metric("cpu_usage"), Some(0.75_f64));
        assert_eq!(stats.get_metric("memory_usage"), None);
    }

    #[test]
    fn test_key_validation() {
        assert!(utils::validate_key("valid_key").is_ok());
        assert!(utils::validate_key("").is_err());
        assert!(utils::validate_key(&"x".repeat(2000)).is_err());
        assert!(utils::validate_key("key\0with\0nulls").is_err());
    }

    #[test]
    fn test_value_size_validation() {
        let small_value = vec![0u8; 100];
        let large_value = vec![0u8; 2000];

        assert!(utils::validate_value_size(&small_value, 1024).is_ok());
        assert!(utils::validate_value_size(&large_value, 1024).is_err());
    }

    #[test]
    fn test_key_parsing() {
        assert_eq!(utils::parse_key("simple"), (None, "simple"));
        assert_eq!(utils::parse_key("prefix/key"), (Some("prefix"), "key"));
        assert_eq!(
            utils::parse_key("deep/nested/key"),
            (Some("deep/nested"), "key")
        );
    }

    #[test]
    fn test_unique_key_generation() {
        let key1 = utils::generate_unique_key("test");
        let key2 = utils::generate_unique_key("test");

        assert!(key1.starts_with("test/"));
        assert!(key2.starts_with("test/"));
        assert_ne!(key1, key2);
    }
}
