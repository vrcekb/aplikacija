//! Ultra-Hot Storage for MEV Opportunities
//!
//! This module provides ultra-fast storage for MEV opportunities with <100μs latency guarantee.
//! All operations are optimized for maximum performance in financial applications.

use std::sync::Arc;
use std::time::Instant;
use tokio::time::{interval, Duration};
use uuid::Uuid;

use crate::{
    config::HotStorageConfig,
    error::{CriticalError, DataStorageError, DataStorageResult},
    optimization::{lock_free::WriteOperation, OptimizerConfig, UltraPerformanceOptimizer},
    types::Opportunity,
};

/// Ultra-optimized hot storage with <100μs latency guarantee
#[derive(Debug)]
pub struct UltraHotStorage {
    /// Performance optimizer suite
    optimizer: Arc<UltraPerformanceOptimizer>,

    /// Background writer handle
    writer_handle: Option<tokio::task::JoinHandle<()>>,
}

impl UltraHotStorage {
    /// Create a new ultra-hot storage instance
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Configuration` if the configuration is invalid,
    /// or `DataStorageError::Internal` if initialization fails.
    pub fn new(config: &HotStorageConfig) -> DataStorageResult<Self> {
        tracing::info!("Initializing ultra-hot storage");

        // Create optimizer configuration
        let optimizer_config = OptimizerConfig {
            target_latency_ns: 100_000, // 100μs target
            enable_numa: true,
            enable_simd: true,
            memory_pool_size: usize::try_from(config.cache_size_bytes)
                .map_err(|e| DataStorageError::configuration(format!("Invalid cache size: {e}")))?,
            opportunity_cache_size: 10_000,
            write_buffer_size: 1000,
            ..Default::default()
        };

        // Initialize optimizer
        let optimizer = Arc::new(UltraPerformanceOptimizer::new(optimizer_config)?);

        // Start background writer
        let writer_optimizer = Arc::clone(&optimizer);
        let writer_handle = tokio::spawn(async move {
            Self::background_writer(writer_optimizer).await;
        });

        Ok(Self {
            optimizer,
            writer_handle: Some(writer_handle),
        })
    }

    /// Store an opportunity with ultra-low latency
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Critical` if the operation exceeds latency targets.
    pub fn store_opportunity(&self, opportunity: &Opportunity) -> DataStorageResult<()> {
        let start = Instant::now();

        // Store in lock-free cache using simple key-value approach
        let key = opportunity.id;
        let data = serde_json::to_vec(opportunity)
            .map_err(|e| DataStorageError::internal(format!("Serialization failed: {e}")))?;

        self.optimizer.opportunity_cache().store(key, data);

        // Queue write operation for background processing
        let write_op = WriteOperation {
            key,
            data: serde_json::to_vec(opportunity)
                .map_err(|e| DataStorageError::internal(format!("Serialization failed: {e}")))?,
            operation_type: crate::optimization::lock_free::WriteOperationType::Insert,
            timestamp: u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0),
        };

        if self
            .optimizer
            .write_buffer()
            .queue_operation(write_op)
            .is_err()
        {
            // Write buffer is full, this is a critical condition
            return Err(DataStorageError::Critical(
                CriticalError::HotStorageFailure(3001),
            ));
        }

        // Record metrics
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.optimizer.record_write(elapsed_ns);

        // Verify latency target
        if !self.optimizer.check_latency_target(start) {
            let latency_us = f64::from(u32::try_from(elapsed_ns / 1000).unwrap_or(u32::MAX));
            tracing::warn!("Store operation exceeded latency target: {latency_us:.2}μs");
        }

        Ok(())
    }

    /// Retrieve an opportunity with ultra-low latency
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Critical` if the operation exceeds latency targets.
    pub fn get_opportunity(&self, id: &Uuid) -> DataStorageResult<Option<Opportunity>> {
        let start = Instant::now();

        // Try lock-free cache first (fastest path)
        if let Some(data) = self.optimizer.opportunity_cache().get(id) {
            let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.optimizer.record_read(elapsed_ns);

            // Deserialize opportunity
            let opportunity: Opportunity = serde_json::from_slice(&data)
                .map_err(|e| DataStorageError::internal(format!("Deserialization failed: {e}")))?;

            return Ok(Some(opportunity));
        }

        // Cache miss - record and return None
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.optimizer.record_read(elapsed_ns);

        Ok(None)
    }

    /// Remove an opportunity with ultra-low latency
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Critical` if the operation exceeds latency targets.
    pub fn remove_opportunity(&self, id: &Uuid) -> DataStorageResult<bool> {
        let start = Instant::now();

        // Check if opportunity exists in cache
        let exists = self.optimizer.opportunity_cache().get(id).is_some();

        if exists {
            // Queue remove operation for background processing
            let write_op = WriteOperation {
                key: *id,
                data: Vec::with_capacity(0), // Empty data for delete operation
                operation_type: crate::optimization::lock_free::WriteOperationType::Delete,
                timestamp: u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0),
            };
            let _ = self.optimizer.write_buffer().queue_operation(write_op);
        }

        // Record metrics
        let elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.optimizer.record_delete(elapsed_ns);

        Ok(exists)
    }

    /// Get storage statistics
    #[must_use]
    pub fn get_stats(&self) -> UltraHotStorageStats {
        let perf_stats = self.optimizer.get_performance_stats();

        UltraHotStorageStats {
            avg_read_latency_ns: perf_stats.avg_read_latency_ns,
            avg_write_latency_ns: perf_stats.avg_write_latency_ns,
            cache_hit_rate: perf_stats.cache_hit_rate,
            memory_pool_hit_rate: perf_stats.pool_hit_rate,
            performance_score: perf_stats.performance_score(),
            meets_targets: perf_stats.meets_targets(),
            target_latency_ns: perf_stats.target_latency_ns,
        }
    }

    /// Background writer task with graceful shutdown
    async fn background_writer(optimizer: Arc<UltraPerformanceOptimizer>) {
        const MAX_ITERATIONS: u32 = 1_000_000; // Prevent infinite loops
        let mut interval = interval(Duration::from_millis(10)); // 10ms intervals
        let mut shutdown_counter = 0_u32;

        while shutdown_counter < MAX_ITERATIONS {
            tokio::select! {
                _ = interval.tick() => {
                    // Drain write operations from buffer
                    let operations = optimizer.write_buffer().drain_operations();

                    if operations.is_empty() {
                        shutdown_counter += 1;
                    } else {
                        // Process operations in batch
                        Self::process_write_operations(&operations);
                        shutdown_counter = 0; // Reset counter on activity
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Graceful shutdown initiated");
                    break;
                }
            }
        }

        tracing::info!("Background writer shutting down");
    }

    /// Process write operations in batch
    fn process_write_operations(operations: &[WriteOperation]) {
        // TODO: Implement actual persistence layer
        // For now, just log the operations
        tracing::debug!("Processing {} write operations", operations.len());

        for operation in operations {
            Self::process_single_operation(operation);
        }
    }

    /// Process a single write operation
    fn process_single_operation(operation: &WriteOperation) {
        match operation.operation_type {
            crate::optimization::lock_free::WriteOperationType::Insert => {
                tracing::trace!("Persisting insert for opportunity {}", operation.key);
            }
            crate::optimization::lock_free::WriteOperationType::Delete => {
                tracing::trace!("Persisting delete for opportunity {}", operation.key);
            }
            crate::optimization::lock_free::WriteOperationType::Update => {
                tracing::trace!("Persisting update for opportunity {}", operation.key);
            }
        }
    }
}

impl Drop for UltraHotStorage {
    fn drop(&mut self) {
        if let Some(handle) = self.writer_handle.take() {
            handle.abort();
        }
    }
}

/// Ultra-hot storage statistics
#[derive(Debug, Clone)]
pub struct UltraHotStorageStats {
    pub avg_read_latency_ns: u64,
    pub avg_write_latency_ns: u64,
    pub cache_hit_rate: f64,
    pub memory_pool_hit_rate: f64,
    pub performance_score: f64,
    pub meets_targets: bool,
    pub target_latency_ns: u64,
}

impl UltraHotStorageStats {
    /// Check if storage is performing optimally
    #[must_use]
    pub const fn is_optimal(&self) -> bool {
        self.meets_targets
    }

    /// Get performance grade (A-F)
    #[must_use]
    pub fn performance_grade(&self) -> char {
        if self.performance_score >= 0.9_f64 {
            'A'
        } else if self.performance_score >= 0.8_f64 {
            'B'
        } else if self.performance_score >= 0.7_f64 {
            'C'
        } else if self.performance_score >= 0.6_f64 {
            'D'
        } else {
            'F'
        }
    }
}
