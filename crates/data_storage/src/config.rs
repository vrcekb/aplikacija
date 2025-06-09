//! `TallyIO` Data Storage Configuration (Simplified)
//!
//! Simplified configuration system for data storage without validation.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use crate::types::{CachePolicy, CompressionType};

/// Main configuration for data storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataStorageConfig {
    /// Hot storage configuration
    pub hot_storage: HotStorageConfig,

    /// Warm storage configuration
    pub warm_storage: WarmStorageConfig,

    /// Cold storage configuration
    pub cold_storage: ColdStorageConfig,

    /// Cache configuration
    pub cache: CacheConfig,

    /// Pipeline configuration
    pub pipeline: PipelineConfig,

    /// Stream processing configuration
    pub stream: StreamConfig,

    /// Indexer configuration
    pub indexer: IndexerConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,
}

/// Hot storage configuration (redb)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotStorageConfig {
    /// Database file path (None for in-memory)
    pub database_path: Option<PathBuf>,

    /// Maximum database size in bytes
    pub max_size_bytes: u64,

    /// Cache size in bytes
    pub cache_size_bytes: u64,

    /// Sync mode for durability
    pub sync_mode: SyncMode,

    /// Enable compression
    pub enable_compression: bool,

    /// Compression type
    pub compression_type: CompressionType,

    /// Use in-memory storage for ultra-low latency
    pub use_memory_storage: bool,
}

/// Warm storage configuration (PostgreSQL/TimescaleDB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmStorageConfig {
    /// Database connection URL
    pub database_url: String,

    /// Maximum number of connections in pool
    pub max_connections: u32,

    /// Minimum number of connections in pool
    pub min_connections: u32,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Query timeout
    pub query_timeout: Duration,

    /// Enable read replicas
    pub enable_read_replicas: bool,

    /// Read replica URLs
    pub read_replica_urls: Vec<String>,

    /// Enable `TimescaleDB` features
    pub enable_timescale: bool,

    /// Chunk time interval for `TimescaleDB`
    pub chunk_time_interval: Duration,

    /// Environment mode for graceful degradation
    pub environment_mode: EnvironmentMode,

    /// Enable fallback to simplified mode if database unavailable
    pub enable_fallback_mode: bool,
}

/// Environment mode for database operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentMode {
    /// Production environment - requires database connection
    Production,
    /// Development environment - allows fallback
    Development,
    /// Testing environment - graceful degradation
    Testing,
    /// CI/CD environment - simplified mode
    Ci,
}

/// Cold storage configuration (encrypted archive)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStorageConfig {
    /// Storage directory path
    pub storage_path: PathBuf,

    /// Enable encryption
    pub enable_encryption: bool,

    /// Encryption key path
    pub encryption_key_path: Option<PathBuf>,

    /// Compression type
    pub compression_type: CompressionType,

    /// Compression level (1-9)
    pub compression_level: u8,

    /// Archive rotation size in bytes
    pub rotation_size_bytes: u64,

    /// Maximum archive age before rotation
    pub max_archive_age: Duration,
}

/// Cache configuration (Redis + in-memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Redis connection URL
    pub redis_url: Option<String>,

    /// Enable Redis cache
    pub enable_redis: bool,

    /// Enable in-memory cache
    pub enable_memory_cache: bool,

    /// Memory cache size in bytes
    pub memory_cache_size_bytes: u64,

    /// Redis threshold bytes (store in Redis if larger than this)
    pub redis_threshold_bytes: Option<usize>,

    /// Default cache policy
    pub default_policy: CachePolicy,

    /// Cache TTL for different data types
    pub ttl_config: CacheTtlConfig,

    /// Enable cache compression
    pub enable_compression: bool,

    /// Cache connection timeout
    pub connection_timeout: Duration,

    /// Cache operation timeout
    pub operation_timeout: Duration,
}

/// Cache TTL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTtlConfig {
    /// TTL for opportunities (seconds)
    pub opportunities_ttl_seconds: u64,

    /// TTL for transactions (seconds)
    pub transactions_ttl_seconds: u64,

    /// TTL for blocks (seconds)
    pub blocks_ttl_seconds: u64,

    /// TTL for events (seconds)
    pub events_ttl_seconds: u64,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Batch size for processing
    pub batch_size: u32,

    /// Processing timeout
    pub processing_timeout: Duration,

    /// Number of worker threads
    pub worker_threads: u32,

    /// Buffer size for pipeline stages
    pub buffer_size: u32,

    /// Enable parallel processing
    pub enable_parallel_processing: bool,

    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Stream processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Stream buffer size
    pub buffer_size: u32,

    /// Processing interval
    pub processing_interval: Duration,

    /// Aggregation window size
    pub aggregation_window: Duration,

    /// Enable backpressure handling
    pub enable_backpressure: bool,

    /// Maximum memory usage for streams
    pub max_memory_bytes: u64,
}

/// Indexer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Enable block indexing
    pub enable_block_indexing: bool,

    /// Enable event indexing
    pub enable_event_indexing: bool,

    /// Enable transaction indexing
    pub enable_transaction_indexing: bool,

    /// Indexing batch size
    pub batch_size: u32,

    /// Indexing interval
    pub indexing_interval: Duration,

    /// Starting block number
    pub start_block: u64,

    /// Maximum blocks to process per batch
    pub max_blocks_per_batch: u32,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,

    /// Metrics collection features
    pub features: MetricsFeatures,

    /// Metrics collection interval
    pub collection_interval: Duration,

    /// Metrics retention period
    pub retention_period: Duration,
}

/// Metrics collection features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsFeatures {
    /// Enable performance metrics
    pub performance_metrics: bool,

    /// Enable error metrics
    pub error_metrics: bool,

    /// Enable cache metrics
    pub cache_metrics: bool,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,

    /// Initial retry delay
    pub initial_delay: Duration,

    /// Maximum retry delay
    pub max_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

/// Sync mode for hot storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMode {
    /// No sync (fastest, least durable)
    None,
    /// Sync on commit (balanced)
    Normal,
    /// Full sync (slowest, most durable)
    Full,
}

impl Default for HotStorageConfig {
    fn default() -> Self {
        Self {
            database_path: Some(PathBuf::from("./data/hot/tallyio.redb")),
            max_size_bytes: 1_073_741_824, // 1GB
            cache_size_bytes: 134_217_728, // 128MB
            sync_mode: SyncMode::Normal,
            enable_compression: true,
            compression_type: CompressionType::Lz4,
            use_memory_storage: false,
        }
    }
}

impl Default for WarmStorageConfig {
    fn default() -> Self {
        Self {
            database_url: "postgresql://tallyio:password@localhost:5432/tallyio".to_string(),
            max_connections: 100,
            min_connections: 10,
            connection_timeout: Duration::from_secs(5),
            query_timeout: Duration::from_secs(30),
            enable_read_replicas: false,
            read_replica_urls: Vec::with_capacity(0), // Pre-allocated empty vector for financial safety
            enable_timescale: true,
            chunk_time_interval: Duration::from_secs(86400), // 1 day
            environment_mode: EnvironmentMode::Development,
            enable_fallback_mode: true,
        }
    }
}

impl Default for ColdStorageConfig {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./data/cold"),
            enable_encryption: true,
            encryption_key_path: Some(PathBuf::from("./keys/cold_storage.key")),
            compression_type: CompressionType::Lz4,
            compression_level: 6,
            rotation_size_bytes: 1_073_741_824,              // 1GB
            max_archive_age: Duration::from_secs(2_592_000), // 30 days
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: Some("redis://localhost:6379".to_string()),
            enable_redis: true,
            enable_memory_cache: true,
            memory_cache_size_bytes: 268_435_456, // 256MB
            redis_threshold_bytes: Some(1024),    // 1KB
            default_policy: CachePolicy::Medium,
            ttl_config: CacheTtlConfig::default(),
            enable_compression: true,
            connection_timeout: Duration::from_secs(5),
            operation_timeout: Duration::from_secs(1),
        }
    }
}

impl Default for CacheTtlConfig {
    fn default() -> Self {
        Self {
            opportunities_ttl_seconds: 300, // 5 minutes
            transactions_ttl_seconds: 600,  // 10 minutes
            blocks_ttl_seconds: 3600,       // 1 hour
            events_ttl_seconds: 1800,       // 30 minutes
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            processing_timeout: Duration::from_secs(30),
            worker_threads: u32::try_from(num_cpus::get()).unwrap_or(4),
            buffer_size: 10000,
            enable_parallel_processing: true,
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            processing_interval: Duration::from_millis(100),
            aggregation_window: Duration::from_secs(60),
            enable_backpressure: true,
            max_memory_bytes: 134_217_728, // 128MB
        }
    }
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            enable_block_indexing: true,
            enable_event_indexing: true,
            enable_transaction_indexing: true,
            batch_size: 100,
            indexing_interval: Duration::from_secs(1),
            start_block: 0,
            max_blocks_per_batch: 100,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            features: MetricsFeatures {
                performance_metrics: true,
                error_metrics: true,
                cache_metrics: true,
            },
            collection_interval: Duration::from_secs(60),
            retention_period: Duration::from_secs(604_800), // 7 days
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        }
    }
}
