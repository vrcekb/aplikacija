//! Data Ingestion Pipeline
//!
//! High-performance data ingestion for real-time MEV opportunity processing.
//! Handles blockchain data streams with <1ms latency requirements.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::{
    cache::u64_to_f64_safe,
    config::PipelineConfig,
    error::{DataStorageError, DataStorageResult},
    pipeline::{PipelineMetrics, PipelineStage},
    types::{Block, Event, Transaction},
};

/// Raw blockchain data for ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RawData {
    /// Raw transaction data
    Transaction {
        hash: String,
        from: String,
        to: Option<String>,
        value: String,
        gas_price: String,
        gas_limit: u64,
        data: Vec<u8>,
        block_number: u64,
        chain_id: u32,
    },
    /// Raw block data
    Block {
        number: u64,
        hash: String,
        parent_hash: String,
        timestamp: u64,
        transactions: Vec<String>,
        chain_id: u32,
    },
    /// Raw event log data
    Event {
        address: String,
        topics: Vec<String>,
        data: Vec<u8>,
        block_number: u64,
        transaction_hash: String,
        log_index: u32,
        chain_id: u32,
    },
}

/// Data ingestion pipeline for processing raw blockchain data
#[derive(Debug)]
pub struct DataIngestion {
    #[allow(dead_code)] // Used for future configuration-based processing
    config: PipelineConfig,
    metrics: Arc<parking_lot::Mutex<PipelineMetrics>>,
    input_channel: Option<mpsc::Receiver<RawData>>,
    output_channel: Option<mpsc::Sender<ProcessedData>>,
}

/// Processed data output from ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessedData {
    /// Processed transaction
    Transaction(Transaction),
    /// Processed block
    Block(Block),
    /// Processed event
    Event(Event),
    /// Batch of processed items
    Batch(Vec<ProcessedData>),
}

impl DataIngestion {
    /// Create a new data ingestion pipeline
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(parking_lot::Mutex::new(PipelineMetrics::new(
                "ingestion".to_string(),
            ))),
            input_channel: None,
            output_channel: None,
        }
    }

    /// Set up input channel for receiving raw data
    #[must_use]
    pub fn with_input_channel(mut self, receiver: mpsc::Receiver<RawData>) -> Self {
        self.input_channel = Some(receiver);
        self
    }

    /// Set up output channel for sending processed data
    #[must_use]
    pub fn with_output_channel(mut self, sender: mpsc::Sender<ProcessedData>) -> Self {
        self.output_channel = Some(sender);
        self
    }

    /// Start the ingestion pipeline
    ///
    /// # Errors
    ///
    /// Returns error if pipeline startup fails
    #[allow(clippy::cognitive_complexity)] // Complex financial data processing logic
    pub async fn start(&mut self) -> DataStorageResult<()> {
        let mut input = self.input_channel.take().ok_or_else(|| {
            DataStorageError::pipeline("ingestion", "No input channel configured")
        })?;

        let output = self.output_channel.take().ok_or_else(|| {
            DataStorageError::pipeline("ingestion", "No output channel configured")
        })?;

        tracing::info!("Starting data ingestion pipeline");

        // Process incoming data
        while let Some(raw_data) = input.recv().await {
            let start = Instant::now();

            match self.process_raw_data(raw_data) {
                Ok(processed) => {
                    if let Err(e) = output.send(processed).await {
                        tracing::error!("Failed to send processed data: {}", e);
                        self.update_metrics(false, start.elapsed());
                        break;
                    }
                    self.update_metrics(true, start.elapsed());
                }
                Err(e) => {
                    tracing::warn!("Failed to process raw data: {}", e);
                    self.update_metrics(false, start.elapsed());
                }
            }
        }

        tracing::info!("Data ingestion pipeline stopped");
        Ok(())
    }

    /// Process raw blockchain data into structured format
    fn process_raw_data(&self, raw: RawData) -> DataStorageResult<ProcessedData> {
        match raw {
            RawData::Transaction {
                hash,
                from,
                to,
                value,
                gas_price,
                gas_limit: _,
                data: _,
                block_number,
                chain_id,
            } => {
                let transaction =
                    Transaction::new(chain_id, block_number, hash, from, to, value, gas_price);

                Ok(ProcessedData::Transaction(transaction))
            }

            RawData::Block {
                number,
                hash,
                parent_hash,
                timestamp,
                transactions,
                chain_id,
            } => {
                let block = Block {
                    number,
                    hash,
                    parent_hash,
                    timestamp: chrono::DateTime::from_timestamp(
                        i64::try_from(timestamp).unwrap_or(0),
                        0,
                    )
                    .unwrap_or_else(chrono::Utc::now),
                    chain_id,
                    transaction_count: u32::try_from(transactions.len()).map_err(|e| {
                        DataStorageError::pipeline(
                            "ingestion",
                            format!("Transaction count overflow: {e}"),
                        )
                    })?,
                    gas_used: 0,  // Would be filled from actual block data
                    gas_limit: 0, // Would be filled from actual block data
                    processed: false,
                    processed_at: None,
                };

                Ok(ProcessedData::Block(block))
            }

            RawData::Event {
                address,
                topics,
                data,
                block_number,
                transaction_hash,
                log_index,
                chain_id,
            } => {
                let event = Event {
                    id: uuid::Uuid::new_v4(),
                    block_number,
                    transaction_hash,
                    log_index,
                    contract_address: address,
                    event_signature: topics.first().cloned().unwrap_or_default(),
                    topics,
                    data: hex::encode(data),
                    chain_id,
                    created_at: chrono::Utc::now(),
                    decoded_data: None,
                };

                Ok(ProcessedData::Event(event))
            }
        }
    }

    /// Process batch of raw data items
    ///
    /// # Errors
    ///
    /// Returns error if batch processing fails
    pub fn process_batch(&self, raw_items: Vec<RawData>) -> DataStorageResult<ProcessedData> {
        let start = Instant::now();
        let mut processed_items = Vec::with_capacity(raw_items.len());
        let mut success_count = 0_i32;
        let mut error_count = 0_i32;

        for raw_item in raw_items {
            match self.process_raw_data(raw_item) {
                Ok(processed) => {
                    processed_items.push(processed);
                    success_count += 1_i32;
                }
                Err(e) => {
                    error_count += 1_i32;
                    tracing::warn!("Failed to process batch item: {}", e);
                    // Continue processing other items
                }
            }
        }

        let duration = start.elapsed();
        tracing::debug!(
            "Ingestion batch processed: {} success, {} errors in {:?}",
            success_count,
            error_count,
            duration
        );

        // Update metrics for batch
        for _ in 0_i32..success_count {
            self.update_metrics(
                true,
                duration / u32::try_from(success_count + error_count).unwrap_or(1),
            );
        }
        for _ in 0_i32..error_count {
            self.update_metrics(
                false,
                duration / u32::try_from(success_count + error_count).unwrap_or(1),
            );
        }

        Ok(ProcessedData::Batch(processed_items))
    }

    /// Update ingestion metrics
    fn update_metrics(&self, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.items_processed += 1;

        if !success {
            metrics.items_failed += 1;
        }

        let duration_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.total_processing_time_us += duration_us;
        metrics.avg_processing_time_us = metrics.total_processing_time_us / metrics.items_processed;
    }

    /// Get current ingestion rate (items per second)
    #[must_use]
    pub fn ingestion_rate(&self) -> f64 {
        let metrics = self.metrics.lock();
        if metrics.total_processing_time_us > 0 {
            u64_to_f64_safe(metrics.items_processed)
                / (u64_to_f64_safe(metrics.total_processing_time_us) / 1_000_000.0_f64)
        } else {
            0.0_f64
        }
    }

    /// Check if ingestion is meeting latency requirements
    #[must_use]
    pub fn is_meeting_latency_requirements(&self) -> bool {
        let metrics = self.metrics.lock();
        // Require <1ms average processing time for financial data
        metrics.avg_processing_time_us < 1000
    }
}

#[async_trait]
impl PipelineStage<RawData> for DataIngestion {
    async fn process(&self, data: RawData) -> DataStorageResult<RawData> {
        let start = Instant::now();

        // Validate raw data
        self.validate_raw_data(&data)?;

        // Process and update metrics
        let duration = start.elapsed();
        self.update_metrics(true, duration);

        // Return the same data (ingestion doesn't transform, just validates)
        Ok(data)
    }

    fn name(&self) -> &str {
        "ingestion"
    }

    async fn metrics(&self) -> DataStorageResult<PipelineMetrics> {
        Ok(self.metrics.lock().clone())
    }
}

impl DataIngestion {
    /// Validate raw data before processing
    fn validate_raw_data(&self, data: &RawData) -> DataStorageResult<()> {
        match data {
            RawData::Transaction {
                hash,
                from,
                chain_id,
                ..
            } => {
                if hash.is_empty() {
                    return Err(DataStorageError::validation(
                        "transaction_hash",
                        "Hash cannot be empty",
                    ));
                }
                if from.is_empty() {
                    return Err(DataStorageError::validation(
                        "transaction_from",
                        "From address cannot be empty",
                    ));
                }
                if *chain_id == 0 {
                    return Err(DataStorageError::validation(
                        "chain_id",
                        "Chain ID cannot be zero",
                    ));
                }
            }

            RawData::Block {
                number,
                hash,
                chain_id,
                ..
            } => {
                if hash.is_empty() {
                    return Err(DataStorageError::validation(
                        "block_hash",
                        "Hash cannot be empty",
                    ));
                }
                if *number == 0 {
                    return Err(DataStorageError::validation(
                        "block_number",
                        "Block number cannot be zero",
                    ));
                }
                if *chain_id == 0 {
                    return Err(DataStorageError::validation(
                        "chain_id",
                        "Chain ID cannot be zero",
                    ));
                }
            }

            RawData::Event {
                address,
                transaction_hash,
                chain_id,
                ..
            } => {
                if address.is_empty() {
                    return Err(DataStorageError::validation(
                        "event_address",
                        "Address cannot be empty",
                    ));
                }
                if transaction_hash.is_empty() {
                    return Err(DataStorageError::validation(
                        "transaction_hash",
                        "Transaction hash cannot be empty",
                    ));
                }
                if *chain_id == 0 {
                    return Err(DataStorageError::validation(
                        "chain_id",
                        "Chain ID cannot be zero",
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_ingestion_creation() {
        let config = PipelineConfig::default();
        let ingestion = DataIngestion::new(config);
        assert_eq!(ingestion.name(), "ingestion");
    }

    #[tokio::test]
    async fn test_raw_data_validation() -> DataStorageResult<()> {
        let config = PipelineConfig::default();
        let ingestion = DataIngestion::new(config);

        // Valid transaction data
        let valid_tx = RawData::Transaction {
            hash: "0x123".to_string(),
            from: "0xabc".to_string(),
            to: Some("0xdef".to_string()),
            value: "1000".to_string(),
            gas_price: "20".to_string(),
            gas_limit: 21000,
            data: vec![],
            block_number: 100,
            chain_id: 1,
        };

        assert!(ingestion.validate_raw_data(&valid_tx).is_ok());

        // Invalid transaction data (empty hash)
        let invalid_tx = RawData::Transaction {
            hash: String::new(),
            from: "0xabc".to_string(),
            to: Some("0xdef".to_string()),
            value: "1000".to_string(),
            gas_price: "20".to_string(),
            gas_limit: 21000,
            data: vec![],
            block_number: 100,
            chain_id: 1,
        };

        assert!(ingestion.validate_raw_data(&invalid_tx).is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_ingestion_metrics() {
        let config = PipelineConfig::default();
        let ingestion = DataIngestion::new(config);

        // Initially no processing
        assert!((ingestion.ingestion_rate() - 0.0_f64).abs() < f64::EPSILON);
        assert!(ingestion.is_meeting_latency_requirements()); // No data processed yet

        // Simulate some processing
        ingestion.update_metrics(true, Duration::from_micros(500));
        assert!(ingestion.is_meeting_latency_requirements()); // Under 1ms

        ingestion.update_metrics(true, Duration::from_micros(2000));
        // Average should now be 1250us, which is over 1ms requirement
        assert!(!ingestion.is_meeting_latency_requirements());
    }
}
