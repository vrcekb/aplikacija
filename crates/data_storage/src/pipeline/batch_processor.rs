//! Batch Processing Pipeline
//!
//! High-performance batch processing for financial data with optimized throughput.
//! Handles large volumes of MEV opportunities and blockchain data efficiently.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::{
    config::PipelineConfig,
    error::DataStorageResult,
    pipeline::{PipelineMetrics, PipelineStage},
    types::{Opportunity, Transaction},
};

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size before processing
    pub min_batch_size: usize,
    /// Maximum wait time before flushing batch
    pub max_wait_time_ms: u64,
    /// Enable parallel processing within batches
    pub enable_parallel_processing: bool,
    /// Number of worker threads for parallel processing
    pub worker_threads: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            min_batch_size: 10,
            max_wait_time_ms: 100,
            enable_parallel_processing: true,
            worker_threads: num_cpus::get(),
        }
    }
}

/// Batch processor for efficient data processing
#[derive(Debug)]
pub struct BatchProcessor {
    #[allow(dead_code)] // Used for future configuration-based processing
    config: PipelineConfig,
    batch_config: BatchConfig,
    metrics: Arc<parking_lot::Mutex<PipelineMetrics>>,
    pending_opportunities: Arc<Mutex<VecDeque<Opportunity>>>,
    #[allow(dead_code)] // Used for future transaction batching
    pending_transactions: Arc<Mutex<VecDeque<Transaction>>>,
    #[allow(dead_code)] // Used for future flush timing
    last_flush: Arc<Mutex<Instant>>,
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult<T> {
    /// Successfully processed items
    pub processed: Vec<T>,
    /// Failed items with errors
    pub failed: Vec<(T, String)>,
    /// Processing statistics
    pub stats: BatchStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total items in batch
    pub total_items: usize,
    /// Successfully processed items
    pub success_count: usize,
    /// Failed items
    pub failure_count: usize,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Throughput (items per second)
    pub throughput: f64,
}

impl BatchProcessor {
    /// Create a new batch processor
    #[must_use]
    pub fn new(config: PipelineConfig, batch_config: BatchConfig) -> Self {
        Self {
            config,
            batch_config,
            metrics: Arc::new(parking_lot::Mutex::new(PipelineMetrics::new(
                "batch_processor".to_string(),
            ))),
            pending_opportunities: Arc::new(Mutex::new(VecDeque::new())),
            pending_transactions: Arc::new(Mutex::new(VecDeque::new())),
            last_flush: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Add opportunity to batch queue
    pub async fn queue_opportunity(&self, opportunity: Opportunity) -> DataStorageResult<()> {
        let mut pending = self.pending_opportunities.lock().await;
        pending.push_back(opportunity);

        // Check if we should flush
        if pending.len() >= self.batch_config.max_batch_size {
            drop(pending);
            self.flush_opportunities().await?;
        }

        Ok(())
    }

    /// Flush pending opportunities
    pub async fn flush_opportunities(&self) -> DataStorageResult<BatchResult<Opportunity>> {
        let mut pending = self.pending_opportunities.lock().await;
        if pending.is_empty() {
            return Ok(BatchResult {
                processed: Vec::new(),
                failed: Vec::new(),
                stats: BatchStats {
                    total_items: 0,
                    success_count: 0,
                    failure_count: 0,
                    processing_time_us: 0,
                    throughput: 0.0_f64,
                },
            });
        }

        let batch: Vec<Opportunity> = pending.drain(..).collect();
        drop(pending);

        self.process_opportunity_batch(batch)
    }

    /// Process batch of opportunities
    pub fn process_opportunity_batch(
        &self,
        opportunities: Vec<Opportunity>,
    ) -> DataStorageResult<BatchResult<Opportunity>> {
        let start = Instant::now();
        let total_items = opportunities.len();

        if total_items == 0 {
            return Ok(BatchResult {
                processed: Vec::new(),
                failed: Vec::new(),
                stats: BatchStats {
                    total_items: 0,
                    success_count: 0,
                    failure_count: 0,
                    processing_time_us: 0,
                    throughput: 0.0_f64,
                },
            });
        }

        let mut processed = Vec::with_capacity(total_items);
        let mut failed = Vec::new();

        // Process each opportunity
        for opportunity in opportunities {
            if Self::validate_opportunity_static(&opportunity) {
                processed.push(opportunity);
            } else {
                failed.push((opportunity, "Validation failed".to_string()));
            }
        }

        let duration = start.elapsed();
        let processing_time_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        let throughput = if duration.as_secs_f64() > 0.0_f64 {
            total_items as f64 / duration.as_secs_f64()
        } else {
            0.0_f64
        };

        // Update metrics
        self.update_metrics(processed.len(), duration);

        let stats = BatchStats {
            total_items,
            success_count: processed.len(),
            failure_count: failed.len(),
            processing_time_us,
            throughput,
        };

        tracing::info!(
            "Batch processed {} opportunities: {} success, {} failed in {:?}",
            total_items,
            processed.len(),
            failed.len(),
            duration
        );

        Ok(BatchResult {
            processed,
            failed,
            stats,
        })
    }

    /// Static opportunity validation for parallel processing
    fn validate_opportunity_static(opportunity: &Opportunity) -> bool {
        // Basic validation
        if opportunity.confidence_score < 0.0_f64 || opportunity.confidence_score > 1.0_f64 {
            return false;
        }

        if opportunity.opportunity_type.is_empty() {
            return false;
        }

        // Validate profit values
        if opportunity.profit_eth.parse::<f64>().is_err() {
            return false;
        }

        if opportunity.gas_cost.parse::<f64>().is_err() {
            return false;
        }

        if opportunity.net_profit.parse::<f64>().is_err() {
            return false;
        }

        true
    }

    /// Get pending queue sizes
    pub async fn queue_sizes(&self) -> usize {
        self.pending_opportunities.lock().await.len()
    }

    /// Update batch processing metrics
    fn update_metrics(&self, processed_count: usize, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.items_processed += processed_count as u64;

        let duration_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.total_processing_time_us += duration_us;

        if metrics.items_processed > 0 {
            metrics.avg_processing_time_us =
                metrics.total_processing_time_us / metrics.items_processed;
        }
    }

    /// Get batch processing throughput
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let metrics = self.metrics.lock();
        if metrics.total_processing_time_us > 0 {
            (metrics.items_processed as f64)
                / (metrics.total_processing_time_us as f64 / 1_000_000.0)
        } else {
            0.0_f64
        }
    }
}

#[async_trait]
impl PipelineStage<Vec<Opportunity>> for BatchProcessor {
    async fn process(&self, data: Vec<Opportunity>) -> DataStorageResult<Vec<Opportunity>> {
        let result = self.process_opportunity_batch(data)?;
        Ok(result.processed)
    }

    fn name(&self) -> &str {
        "batch_processor"
    }

    async fn metrics(&self) -> DataStorageResult<PipelineMetrics> {
        Ok(self.metrics.lock().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let config = PipelineConfig::default();
        let batch_config = BatchConfig::default();
        let processor = BatchProcessor::new(config, batch_config);
        assert_eq!(processor.name(), "batch_processor");
    }

    #[tokio::test]
    async fn test_opportunity_validation() {
        let valid_opportunity = Opportunity::new(
            "arbitrage".to_string(),
            1,
            "0.1".to_string(),
            "0.01".to_string(),
            "0.09".to_string(),
            0.8,
        );

        assert!(BatchProcessor::validate_opportunity_static(
            &valid_opportunity
        ));

        let mut invalid_opportunity = valid_opportunity.clone();
        invalid_opportunity.confidence_score = 1.5_f64; // Invalid score

        assert!(!BatchProcessor::validate_opportunity_static(
            &invalid_opportunity
        ));
    }

    #[tokio::test]
    async fn test_batch_processing() -> DataStorageResult<()> {
        let config = PipelineConfig::default();
        let batch_config = BatchConfig::default();
        let processor = BatchProcessor::new(config, batch_config);

        let opportunities = vec![
            Opportunity::new(
                "arbitrage".to_string(),
                1,
                "0.1".to_string(),
                "0.01".to_string(),
                "0.09".to_string(),
                0.8,
            ),
            Opportunity::new(
                "liquidation".to_string(),
                1,
                "0.2".to_string(),
                "0.02".to_string(),
                "0.18".to_string(),
                0.9,
            ),
        ];

        let result = processor.process_opportunity_batch(opportunities)?;
        assert_eq!(result.stats.success_count, 2);
        assert_eq!(result.stats.failure_count, 0);

        Ok(())
    }
}
