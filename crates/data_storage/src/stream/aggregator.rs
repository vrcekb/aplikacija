//! Stream Aggregator
//!
//! Real-time data aggregation for stream processing with time-based windows.
//! Optimized for financial metrics and MEV opportunity analysis.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::error::{DataStorageError, DataStorageResult};

use super::{AggregatedData, StreamData, StreamMetrics, StreamStatus};

/// Stream aggregator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorConfig {
    /// Window size in milliseconds
    pub window_size_ms: u64,
    /// Maximum number of windows to keep
    pub max_windows: usize,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 1000, // 1 second windows
            max_windows: 60,      // Keep 1 minute of data
        }
    }
}

/// Aggregation window
#[derive(Debug, Clone)]
struct AggregationWindow {
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
    transaction_count: u64,
    opportunity_count: u64,
    total_profit: f64,
    total_gas_cost: f64,
    avg_confidence: f64,
    chain_stats: HashMap<u32, ChainStats>,
}

/// Per-chain statistics
#[derive(Debug, Clone, Default)]
struct ChainStats {
    transaction_count: u64,
    opportunity_count: u64,
    total_profit: f64,
}

impl AggregationWindow {
    fn new(start_time: chrono::DateTime<chrono::Utc>, window_size_ms: u64) -> Self {
        let end_time = start_time
            + chrono::Duration::milliseconds(i64::try_from(window_size_ms).unwrap_or(1000));

        Self {
            start_time,
            end_time,
            transaction_count: 0,
            opportunity_count: 0,
            total_profit: 0.0_f64,
            total_gas_cost: 0.0_f64,
            avg_confidence: 0.0_f64,
            chain_stats: HashMap::new(),
        }
    }

    fn add_transaction(&mut self, chain_id: u32) {
        self.transaction_count += 1;

        let chain_stats = self.chain_stats.entry(chain_id).or_default();
        chain_stats.transaction_count += 1;
    }

    fn add_opportunity(&mut self, chain_id: u32, profit: f64, gas_cost: f64, confidence: f64) {
        self.opportunity_count += 1;
        self.total_profit += profit;
        self.total_gas_cost += gas_cost;

        // Update average confidence
        self.avg_confidence = (self.avg_confidence * (self.opportunity_count - 1) as f64
            + confidence)
            / self.opportunity_count as f64;

        let chain_stats = self.chain_stats.entry(chain_id).or_default();
        chain_stats.opportunity_count += 1;
        chain_stats.total_profit += profit;
    }

    fn to_aggregated_data(&self) -> AggregatedData {
        let metrics = serde_json::json!({
            "transaction_count": self.transaction_count,
            "opportunity_count": self.opportunity_count,
            "total_profit_eth": self.total_profit,
            "total_gas_cost_eth": self.total_gas_cost,
            "net_profit_eth": self.total_profit - self.total_gas_cost,
            "avg_confidence": self.avg_confidence,
            "chain_stats": self.chain_stats.iter().map(|(chain_id, stats)| {
                (chain_id.to_string(), serde_json::json!({
                    "transaction_count": stats.transaction_count,
                    "opportunity_count": stats.opportunity_count,
                    "total_profit_eth": stats.total_profit,
                }))
            }).collect::<HashMap<String, serde_json::Value>>()
        });

        AggregatedData {
            window_start: self.start_time,
            window_end: self.end_time,
            item_count: self.transaction_count + self.opportunity_count,
            metrics,
            data_type: "stream_aggregation".to_string(),
        }
    }
}

/// Stream aggregator for real-time metrics
#[derive(Debug)]
pub struct StreamAggregator {
    config: AggregatorConfig,
    windows: Arc<Mutex<VecDeque<AggregationWindow>>>,
    current_window: Arc<Mutex<Option<AggregationWindow>>>,
    metrics: Arc<parking_lot::Mutex<StreamMetrics>>,
    status: Arc<parking_lot::Mutex<StreamStatus>>,
}

impl StreamAggregator {
    /// Create a new stream aggregator
    pub fn new(config: AggregatorConfig) -> DataStorageResult<Self> {
        let metrics = StreamMetrics::new("stream_aggregator".to_string());

        let max_windows = config.max_windows;

        Ok(Self {
            config,
            windows: Arc::new(Mutex::new(VecDeque::with_capacity(max_windows))),
            current_window: Arc::new(Mutex::new(None)),
            metrics: Arc::new(parking_lot::Mutex::new(metrics)),
            status: Arc::new(parking_lot::Mutex::new(StreamStatus::Stopped)),
        })
    }

    /// Start the stream aggregator
    pub async fn start(&self) -> DataStorageResult<()> {
        {
            let mut status = self.status.lock();
            if *status == StreamStatus::Running {
                return Err(DataStorageError::stream("aggregator", "Already running"));
            }

            *status = StreamStatus::Running;
        } // Drop status lock before await

        // Initialize first window
        let now = chrono::Utc::now();
        let window = AggregationWindow::new(now, self.config.window_size_ms);
        let mut current_window = self.current_window.lock().await;
        *current_window = Some(window);

        tracing::info!(
            "Stream aggregator started with {}ms windows",
            self.config.window_size_ms
        );
        Ok(())
    }

    /// Stop the stream aggregator
    pub async fn stop(&self) -> DataStorageResult<()> {
        {
            let mut status = self.status.lock();
            *status = StreamStatus::Stopped;
        } // Drop status lock before await

        // Finalize current window
        self.finalize_current_window().await?;

        tracing::info!("Stream aggregator stopped");
        Ok(())
    }

    /// Add item to aggregation
    pub async fn add_item(&self, item: StreamData) -> DataStorageResult<()> {
        let start = Instant::now();

        // Check if we need to rotate window
        self.check_window_rotation().await?;

        // Add item to current window
        let mut current_window = self.current_window.lock().await;
        if let Some(ref mut window) = *current_window {
            match item {
                StreamData::Transaction(tx) => {
                    window.add_transaction(tx.chain_id);
                }

                StreamData::Opportunity(opp) => {
                    let profit = opp.profit_eth.parse::<f64>().unwrap_or(0.0_f64);
                    let gas_cost = opp.gas_cost.parse::<f64>().unwrap_or(0.0_f64);
                    window.add_opportunity(opp.chain_id, profit, gas_cost, opp.confidence_score);
                }

                _ => {
                    // Other types are not aggregated
                }
            }
        }

        let duration = start.elapsed();
        self.update_metrics(true, duration);

        Ok(())
    }

    /// Check if current window should be rotated
    async fn check_window_rotation(&self) -> DataStorageResult<()> {
        let now = chrono::Utc::now();
        let mut current_window = self.current_window.lock().await;

        let should_rotate = if let Some(ref window) = *current_window {
            now >= window.end_time
        } else {
            true
        };

        if should_rotate {
            // Finalize current window
            if let Some(window) = current_window.take() {
                let mut windows = self.windows.lock().await;

                // Add to completed windows
                windows.push_back(window);

                // Remove old windows if we exceed max
                while windows.len() > self.config.max_windows {
                    windows.pop_front();
                }
            }

            // Create new window
            let new_window = AggregationWindow::new(now, self.config.window_size_ms);
            *current_window = Some(new_window);

            tracing::debug!("Rotated aggregation window at {}", now);
        }

        Ok(())
    }

    /// Finalize current window
    async fn finalize_current_window(&self) -> DataStorageResult<()> {
        let mut current_window = self.current_window.lock().await;
        if let Some(window) = current_window.take() {
            let mut windows = self.windows.lock().await;
            windows.push_back(window);
        }
        Ok(())
    }

    /// Get aggregated data for the last N windows
    pub async fn get_aggregated_data(
        &self,
        window_count: usize,
    ) -> DataStorageResult<Vec<AggregatedData>> {
        let windows = self.windows.lock().await;
        let count = window_count.min(windows.len());

        let aggregated_data: Vec<AggregatedData> = windows
            .iter()
            .rev()
            .take(count)
            .map(|window| window.to_aggregated_data())
            .collect();

        Ok(aggregated_data)
    }

    /// Get current window statistics
    pub async fn current_window_stats(&self) -> Option<AggregatedData> {
        let current_window = self.current_window.lock().await;
        current_window
            .as_ref()
            .map(|window| window.to_aggregated_data())
    }

    /// Get summary statistics across all windows
    pub async fn summary_stats(&self) -> AggregationSummary {
        let windows = self.windows.lock().await;

        let mut total_transactions = 0;
        let mut total_opportunities = 0;
        let mut total_profit = 0.0_f64;
        let mut total_gas_cost = 0.0_f64;
        let mut chain_counts: HashMap<u32, u64> = HashMap::new();

        for window in windows.iter() {
            total_transactions += window.transaction_count;
            total_opportunities += window.opportunity_count;
            total_profit += window.total_profit;
            total_gas_cost += window.total_gas_cost;

            for (chain_id, stats) in &window.chain_stats {
                *chain_counts.entry(*chain_id).or_default() += stats.transaction_count;
            }
        }

        AggregationSummary {
            total_windows: windows.len(),
            total_transactions,
            total_opportunities,
            total_profit_eth: total_profit,
            total_gas_cost_eth: total_gas_cost,
            net_profit_eth: total_profit - total_gas_cost,
            chain_distribution: chain_counts,
            window_size_ms: self.config.window_size_ms,
        }
    }

    /// Update aggregator metrics
    fn update_metrics(&self, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.total_processed += 1;

        if !success {
            metrics.total_errors += 1;
        }

        // Update latency
        let latency_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.avg_latency_us = u64::midpoint(metrics.avg_latency_us, latency_us);

        // Update throughput
        let items_per_second = 1.0_f64 / duration.as_secs_f64();
        metrics.throughput = f64::midpoint(metrics.throughput, items_per_second);

        metrics.status = *self.status.lock();
        metrics.last_updated = chrono::Utc::now();
    }

    /// Get current metrics
    pub async fn metrics(&self) -> StreamMetrics {
        let mut metrics = self.metrics.lock().clone();
        let windows = self.windows.lock().await;
        metrics.buffer_size = windows.len();
        metrics
    }

    /// Health check
    /// Health check for the component
    ///
    /// # Errors
    ///
    /// Returns error if component is in error state or critical thresholds exceeded
    pub async fn health_check(&self) -> DataStorageResult<()> {
        let status = *self.status.lock();
        if status == StreamStatus::Error {
            return Err(DataStorageError::stream(
                "aggregator",
                "Aggregator in error state",
            ));
        }

        // Check if windows are being created properly
        let windows = self.windows.lock().await;
        if windows.is_empty() && *self.status.lock() == StreamStatus::Running {
            return Err(DataStorageError::stream(
                "aggregator",
                "No aggregation windows created",
            ));
        }

        Ok(())
    }
}

/// Aggregation summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSummary {
    /// Total number of windows
    pub total_windows: usize,
    /// Total transactions across all windows
    pub total_transactions: u64,
    /// Total opportunities across all windows
    pub total_opportunities: u64,
    /// Total profit in ETH
    pub total_profit_eth: f64,
    /// Total gas cost in ETH
    pub total_gas_cost_eth: f64,
    /// Net profit in ETH
    pub net_profit_eth: f64,
    /// Transaction distribution by chain
    pub chain_distribution: HashMap<u32, u64>,
    /// Window size in milliseconds
    pub window_size_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Opportunity, Transaction};

    #[tokio::test]
    async fn test_stream_aggregator_creation() -> DataStorageResult<()> {
        let config = AggregatorConfig::default();
        let _aggregator = StreamAggregator::new(config)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_aggregation_window() -> DataStorageResult<()> {
        let config = AggregatorConfig::default();
        let aggregator = StreamAggregator::new(config)?;

        aggregator.start().await?;

        // Add some transactions
        for i in 0..3 {
            let tx = Transaction::new(
                1,
                100 + i,
                format!("0x{i:x}"),
                "0xabc".to_string(),
                Some("0xdef".to_string()),
                "1.0_f64".to_string(),
                "20".to_string(),
            );

            aggregator.add_item(StreamData::Transaction(tx)).await?;
        }

        // Add some opportunities
        for _i in 0_i32..2_i32 {
            let opp = Opportunity::new(
                "arbitrage".to_string(),
                1,
                "0.1".to_string(),
                "0.01_f32".to_string(),
                "0.09".to_string(),
                0.8,
            );

            aggregator.add_item(StreamData::Opportunity(opp)).await?;
        }

        // Check current window stats
        let stats = aggregator.current_window_stats().await;
        assert!(stats.is_some());

        if let Some(stats) = stats {
            assert_eq!(stats.item_count, 5); // 3 transactions + 2 opportunities
        }

        Ok(())
    }
}
