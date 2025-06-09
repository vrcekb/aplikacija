//! Stream Processor
//!
//! High-performance stream processing for real-time MEV opportunity detection.
//! Optimized for ultra-low latency financial data processing.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    error::{DataStorageError, DataStorageResult},
    types::{Block, Event, Opportunity, Transaction},
};

use super::{StreamData, StreamHandler, StreamMetrics, StreamStatus};

/// Stream processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Processing timeout in milliseconds
    pub processing_timeout_ms: u64,
    /// Maximum latency threshold in microseconds
    pub max_latency_us: u64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            processing_timeout_ms: 1000,
            max_latency_us: 1000, // 1ms for financial data
        }
    }
}

/// Stream processor for real-time data processing
#[derive(Debug)]
pub struct StreamProcessor {
    config: ProcessorConfig,
    metrics: Arc<parking_lot::Mutex<StreamMetrics>>,
    status: Arc<parking_lot::Mutex<StreamStatus>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(config: ProcessorConfig) -> DataStorageResult<Self> {
        let metrics = StreamMetrics::new("stream_processor".to_string());

        Ok(Self {
            config,
            metrics: Arc::new(parking_lot::Mutex::new(metrics)),
            status: Arc::new(parking_lot::Mutex::new(StreamStatus::Stopped)),
        })
    }

    /// Start the stream processor
    pub fn start(&self) -> DataStorageResult<()> {
        let mut status = self.status.lock();
        if *status == StreamStatus::Running {
            return Err(DataStorageError::stream("processor", "Already running"));
        }

        *status = StreamStatus::Running;
        tracing::info!("Stream processor started");
        Ok(())
    }

    /// Stop the stream processor
    pub fn stop(&self) -> DataStorageResult<()> {
        let mut status = self.status.lock();
        *status = StreamStatus::Stopped;
        tracing::info!("Stream processor stopped");
        Ok(())
    }

    /// Process a batch of stream data
    pub fn process_batch(&self, items: Vec<StreamData>) -> DataStorageResult<Vec<StreamData>> {
        let start = Instant::now();
        let mut processed = Vec::with_capacity(items.len());

        for item in items {
            match self.process_item(item) {
                Ok(Some(processed_item)) => processed.push(processed_item),
                Ok(None) => {} // Item was filtered out
                Err(e) => {
                    tracing::warn!("Failed to process stream item: {}", e);
                    self.update_metrics(false, start.elapsed());
                }
            }
        }

        let duration = start.elapsed();
        self.update_metrics(true, duration);

        // Check latency requirement
        if duration.as_micros() > u128::from(self.config.max_latency_us) {
            tracing::warn!(
                "Stream processing exceeded latency threshold: {:?}",
                duration
            );
        }

        Ok(processed)
    }

    /// Process a single stream item
    fn process_item(&self, item: StreamData) -> DataStorageResult<Option<StreamData>> {
        match item {
            StreamData::Transaction(tx) => {
                // Process transaction for MEV opportunities
                if self.is_mev_candidate(&tx) {
                    let opportunity = self.detect_mev_opportunity(&tx)?;
                    if let Some(opp) = opportunity {
                        Ok(Some(StreamData::Opportunity(opp)))
                    } else {
                        Ok(Some(StreamData::Transaction(tx)))
                    }
                } else {
                    Ok(Some(StreamData::Transaction(tx)))
                }
            }

            StreamData::Block(block) => {
                // Process block data
                let processed_block = self.process_block(block)?;
                Ok(Some(StreamData::Block(processed_block)))
            }

            StreamData::Event(event) => {
                // Process event data
                let processed_event = self.process_event(event)?;
                Ok(Some(StreamData::Event(processed_event)))
            }

            StreamData::Opportunity(opp) => {
                // Validate and enrich opportunity
                let validated_opp = self.validate_opportunity(opp)?;
                Ok(Some(StreamData::Opportunity(validated_opp)))
            }

            StreamData::Control(_) => {
                // Control messages are not processed, just passed through
                Ok(Some(item))
            }

            StreamData::Aggregated(_) => {
                // Aggregated data is passed through
                Ok(Some(item))
            }
        }
    }

    /// Check if transaction is a MEV candidate
    fn is_mev_candidate(&self, transaction: &Transaction) -> bool {
        // Simple heuristics for MEV detection

        // Check gas price (high gas price might indicate MEV)
        if let Ok(gas_price) = transaction.gas_price.parse::<f64>() {
            if gas_price > 100.0 {
                // > 100 Gwei
                return true;
            }
        }

        // Check transaction value
        if let Ok(value) = transaction.value.parse::<f64>() {
            if value > 10.0 {
                // > 10 ETH
                return true;
            }
        }

        // Check if interacting with known DeFi protocols
        if let Some(ref to_addr) = transaction.to_address {
            // Known Uniswap V2 Router
            if to_addr == "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D" {
                return true;
            }
            // Known Uniswap V3 Router
            if to_addr == "0xE592427A0AEce92De3Edee1F18E0157C05861564" {
                return true;
            }
        }

        false
    }

    /// Detect MEV opportunity from transaction
    fn detect_mev_opportunity(
        &self,
        transaction: &Transaction,
    ) -> DataStorageResult<Option<Opportunity>> {
        // Simplified MEV detection logic
        // In production, this would involve complex analysis

        let estimated_profit = fastrand::f64() * 0.1; // Random profit 0-0.1 ETH
        let estimated_gas_cost = fastrand::f64() * 0.01_f64; // Random gas cost 0-0.01 ETH
        let net_profit = estimated_profit - estimated_gas_cost;

        if net_profit <= 0.001 {
            // Minimum 0.001 ETH profit
            return Ok(None);
        }

        let opportunity = Opportunity::new(
            "arbitrage".to_string(),
            transaction.chain_id,
            estimated_profit.to_string(),
            estimated_gas_cost.to_string(),
            net_profit.to_string(),
            0.8, // 80% confidence
        );

        Ok(Some(opportunity))
    }

    /// Process block data
    fn process_block(&self, mut block: Block) -> DataStorageResult<Block> {
        // Mark block as processed
        block.processed = true;
        block.processed_at = Some(chrono::Utc::now());
        Ok(block)
    }

    /// Process event data
    fn process_event(&self, mut event: Event) -> DataStorageResult<Event> {
        // Try to decode event data if not already decoded
        if event.decoded_data.is_none() {
            event.decoded_data = Some(serde_json::json!({
                "decoded": false,
                "reason": "ABI decoding not implemented"
            }));
        }
        Ok(event)
    }

    /// Validate and enrich opportunity
    fn validate_opportunity(&self, opportunity: Opportunity) -> DataStorageResult<Opportunity> {
        // Basic validation
        if opportunity.confidence_score < 0.0_f64 || opportunity.confidence_score > 1.0_f64 {
            return Err(DataStorageError::validation(
                "confidence_score",
                "Invalid confidence score",
            ));
        }

        // Validate profit calculations
        let profit = opportunity.profit_eth.parse::<f64>().map_err(|e| {
            DataStorageError::validation("profit_eth", format!("Invalid profit format: {e}"))
        })?;

        let gas_cost = opportunity.gas_cost.parse::<f64>().map_err(|e| {
            DataStorageError::validation("gas_cost", format!("Invalid gas cost format: {e}"))
        })?;

        let net_profit = opportunity.net_profit.parse::<f64>().map_err(|e| {
            DataStorageError::validation("net_profit", format!("Invalid net profit format: {e}"))
        })?;

        let calculated_net = profit - gas_cost;
        if (calculated_net - net_profit).abs() > 0.001 {
            return Err(DataStorageError::validation(
                "net_profit",
                "Net profit calculation mismatch",
            ));
        }

        Ok(opportunity)
    }

    /// Update processing metrics
    fn update_metrics(&self, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.total_processed += 1;

        if !success {
            metrics.total_errors += 1;
        }

        // Update latency
        let latency_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.avg_latency_us = (metrics.avg_latency_us + latency_us) / 2;

        // Update throughput
        let items_per_second = 1.0_f64 / duration.as_secs_f64();
        metrics.throughput = (metrics.throughput + items_per_second) / 2.0_f64;

        metrics.status = *self.status.lock();
        metrics.last_updated = chrono::Utc::now();
    }

    /// Get current metrics
    #[must_use]
    pub fn metrics(&self) -> StreamMetrics {
        self.metrics.lock().clone()
    }

    /// Health check
    pub fn health_check(&self) -> DataStorageResult<()> {
        let status = *self.status.lock();
        if status == StreamStatus::Error {
            return Err(DataStorageError::stream(
                "processor",
                "Processor in error state",
            ));
        }

        let metrics = self.metrics.lock();
        if metrics.avg_latency_us > self.config.max_latency_us {
            return Err(DataStorageError::stream(
                "processor",
                format!(
                    "Average latency {}μs exceeds threshold {}μs",
                    metrics.avg_latency_us, self.config.max_latency_us
                ),
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl StreamHandler<StreamData, StreamData> for StreamProcessor {
    async fn process_stream_item(&self, item: StreamData) -> DataStorageResult<Option<StreamData>> {
        self.process_item(item)
    }

    async fn handle_error(&self, error: DataStorageError) -> DataStorageResult<()> {
        tracing::error!("Stream processor error: {}", error);
        let mut status = self.status.lock();
        *status = StreamStatus::Error;
        Ok(())
    }

    async fn stream_metrics(&self) -> DataStorageResult<StreamMetrics> {
        Ok(self.metrics())
    }

    async fn health_check(&self) -> DataStorageResult<()> {
        self.health_check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_processor_creation() -> DataStorageResult<()> {
        let config = ProcessorConfig::default();
        let _processor = StreamProcessor::new(config)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_mev_candidate_detection() -> DataStorageResult<()> {
        let config = ProcessorConfig::default();
        let processor = StreamProcessor::new(config)?;

        // High gas price transaction
        let high_gas_tx = Transaction::new(
            1,
            100,
            "0x123".to_string(),
            "0xabc".to_string(),
            Some("0xdef".to_string()),
            "1.0_f64".to_string(),
            "150".to_string(), // 150 Gwei
        );

        assert!(processor.is_mev_candidate(&high_gas_tx));

        // Low gas price transaction
        let low_gas_tx = Transaction::new(
            1,
            100,
            "0x123".to_string(),
            "0xabc".to_string(),
            Some("0xdef".to_string()),
            "0.1".to_string(),
            "20".to_string(), // 20 Gwei
        );

        assert!(!processor.is_mev_candidate(&low_gas_tx));

        Ok(())
    }
}
