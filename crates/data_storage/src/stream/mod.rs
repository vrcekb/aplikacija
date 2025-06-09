//! Stream Processing Module
//!
//! Real-time stream processing for MEV opportunity detection and blockchain data analysis.
//! Optimized for ultra-low latency financial data streams.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    cache::safe_ratio,
    error::{DataStorageError, DataStorageResult},
    types::{Block, Event, Opportunity, Transaction},
};

pub mod aggregator;
pub mod buffer;
pub mod processor;

pub use aggregator::{AggregatorConfig, StreamAggregator};
pub use buffer::{BufferConfig, StreamBuffer};
pub use processor::{ProcessorConfig, StreamProcessor};

/// Stream processing trait for real-time data
#[async_trait]
pub trait StreamHandler<Input, Output>: Send + Sync {
    /// Process stream item
    ///
    /// # Errors
    ///
    /// Returns error if stream processing fails
    async fn process_stream_item(&self, item: Input) -> DataStorageResult<Option<Output>>;

    /// Handle stream error
    async fn handle_error(&self, error: DataStorageError) -> DataStorageResult<()>;

    /// Get stream metrics
    async fn stream_metrics(&self) -> DataStorageResult<StreamMetrics>;

    /// Health check for stream handler
    async fn health_check(&self) -> DataStorageResult<()>;
}

/// Stream data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamData {
    /// Real-time transaction
    Transaction(Transaction),
    /// Real-time block
    Block(Block),
    /// Real-time event
    Event(Event),
    /// Detected MEV opportunity
    Opportunity(Opportunity),
    /// Aggregated data
    Aggregated(AggregatedData),
    /// Stream control message
    Control(ControlMessage),
}

/// Aggregated stream data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedData {
    /// Aggregation window start time
    pub window_start: chrono::DateTime<chrono::Utc>,
    /// Aggregation window end time
    pub window_end: chrono::DateTime<chrono::Utc>,
    /// Number of items aggregated
    pub item_count: u64,
    /// Aggregated metrics
    pub metrics: serde_json::Value,
    /// Data type
    pub data_type: String,
}

/// Stream control messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Start stream processing
    Start,
    /// Stop stream processing
    Stop,
    /// Pause stream processing
    Pause,
    /// Resume stream processing
    Resume,
    /// Flush buffers
    Flush,
    /// Reset stream state
    Reset,
}

/// Stream processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    /// Stream name
    pub stream_name: String,
    /// Items processed per second
    pub throughput: f64,
    /// Average processing latency in microseconds
    pub avg_latency_us: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Total items processed
    pub total_processed: u64,
    /// Total errors
    pub total_errors: u64,
    /// Stream status
    pub status: StreamStatus,
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Stream processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamStatus {
    /// Stream is running
    Running,
    /// Stream is paused
    Paused,
    /// Stream is stopped
    Stopped,
    /// Stream has errors
    Error,
    /// Stream is starting up
    Starting,
    /// Stream is shutting down
    Stopping,
}

/// Stream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Stream name
    pub name: String,
    /// Buffer size for stream items
    pub buffer_size: usize,
    /// Processing timeout in milliseconds
    pub processing_timeout_ms: u64,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Maximum latency threshold in microseconds
    pub max_latency_us: u64,
    /// Enable stream metrics collection
    pub enable_metrics: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            name: "default_stream".to_string(),
            buffer_size: 10000,
            processing_timeout_ms: 1000,
            enable_backpressure: true,
            max_latency_us: 1000, // 1ms for financial data
            enable_metrics: true,
            batch_size: 100,
        }
    }
}

impl StreamMetrics {
    /// Create new stream metrics
    #[must_use]
    pub fn new(stream_name: String) -> Self {
        Self {
            stream_name,
            throughput: 0.0,
            avg_latency_us: 0,
            buffer_size: 0,
            total_processed: 0,
            total_errors: 0,
            status: StreamStatus::Stopped,
            last_updated: chrono::Utc::now(),
        }
    }

    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        safe_ratio(
            self.total_processed - self.total_errors,
            self.total_processed,
        )
    }
}
