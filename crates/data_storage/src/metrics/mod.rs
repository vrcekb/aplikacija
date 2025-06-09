//! Metrics collection for data storage
//!
//! Provides comprehensive metrics collection and reporting.

pub mod storage_metrics;

pub use storage_metrics::StorageMetricsCollector;

use crate::error::DataStorageResult;
use crate::types::StorageMetrics;

/// Metrics collector trait
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Record a metric
    async fn record(&self, metric: StorageMetrics) -> DataStorageResult<()>;
    
    /// Get all metrics
    async fn get_metrics(&self) -> DataStorageResult<Vec<StorageMetrics>>;
    
    /// Get metrics for a specific operation
    async fn get_operation_metrics(&self, operation: &str) -> DataStorageResult<Vec<StorageMetrics>>;
    
    /// Get aggregated metrics
    async fn get_aggregated_metrics(&self) -> DataStorageResult<AggregatedMetrics>;
    
    /// Clear old metrics
    async fn cleanup_old_metrics(&self, older_than: chrono::Duration) -> DataStorageResult<u64>;
}

/// Aggregated metrics
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Total operations
    pub total_operations: u64,
    
    /// Successful operations
    pub successful_operations: u64,
    
    /// Failed operations
    pub failed_operations: u64,
    
    /// Average latency in microseconds
    pub avg_latency_us: u64,
    
    /// 95th percentile latency in microseconds
    pub p95_latency_us: u64,
    
    /// 99th percentile latency in microseconds
    pub p99_latency_us: u64,
    
    /// Total data processed in bytes
    pub total_data_bytes: u64,
    
    /// Operations per second
    pub ops_per_second: f64,
    
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
}

impl AggregatedMetrics {
    /// Create new aggregated metrics
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            avg_latency_us: 0,
            p95_latency_us: 0,
            p99_latency_us: 0,
            total_data_bytes: 0,
            ops_per_second: 0.0,
            success_rate: 0.0,
        }
    }
    
    /// Calculate derived metrics
    pub fn calculate_derived(&mut self) {
        // Calculate success rate
        if self.total_operations > 0 {
            self.success_rate = self.successful_operations as f64 / self.total_operations as f64;
        }
        
        // Calculate ops per second (would need time window in real implementation)
        // This is a placeholder
        self.ops_per_second = 0.0;
    }
}
