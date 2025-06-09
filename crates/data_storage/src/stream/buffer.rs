//! Stream Buffer
//!
//! High-performance circular buffer for stream data with overflow protection.
//! Optimized for ultra-low latency financial data buffering.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::{
    cache::u64_to_f64_safe,
    error::{DataStorageError, DataStorageResult},
};

use super::{StreamData, StreamMetrics, StreamStatus};

/// Stream buffer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Buffer capacity
    pub capacity: usize,
    /// Enable overflow protection
    pub enable_overflow_protection: bool,
    /// Maximum item age in milliseconds
    pub max_item_age_ms: u64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            enable_overflow_protection: true,
            max_item_age_ms: 5000, // 5 seconds
        }
    }
}

/// Buffered stream item with timestamp
#[derive(Debug, Clone)]
struct BufferedItem {
    data: StreamData,
    timestamp: Instant,
}

/// Stream buffer for managing data flow
#[derive(Debug)]
pub struct StreamBuffer {
    config: BufferConfig,
    buffer: Arc<Mutex<VecDeque<BufferedItem>>>,
    metrics: Arc<parking_lot::Mutex<StreamMetrics>>,
    status: Arc<parking_lot::Mutex<StreamStatus>>,
}

impl StreamBuffer {
    /// Create a new stream buffer
    pub fn new(config: BufferConfig) -> DataStorageResult<Self> {
        let metrics = StreamMetrics::new("stream_buffer".to_string());

        let capacity = config.capacity;

        Ok(Self {
            config,
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            metrics: Arc::new(parking_lot::Mutex::new(metrics)),
            status: Arc::new(parking_lot::Mutex::new(StreamStatus::Stopped)),
        })
    }

    /// Start the stream buffer
    pub fn start(&self) -> DataStorageResult<()> {
        let mut status = self.status.lock();
        if *status == StreamStatus::Running {
            return Err(DataStorageError::stream("buffer", "Already running"));
        }

        *status = StreamStatus::Running;
        tracing::info!(
            "Stream buffer started with capacity {}",
            self.config.capacity
        );
        Ok(())
    }

    /// Stop the stream buffer
    pub fn stop(&self) -> DataStorageResult<()> {
        let mut status = self.status.lock();
        *status = StreamStatus::Stopped;
        tracing::info!("Stream buffer stopped");
        Ok(())
    }

    /// Push data to buffer
    pub async fn push(&self, data: StreamData) -> DataStorageResult<()> {
        let start = Instant::now();
        let mut buffer = self.buffer.lock().await;

        // Check if buffer is full
        if buffer.len() >= self.config.capacity {
            if self.config.enable_overflow_protection {
                // Remove oldest item
                buffer.pop_front();
                tracing::warn!("Buffer overflow, removed oldest item");
            } else {
                return Err(DataStorageError::stream("buffer", "Buffer overflow"));
            }
        }

        // Add new item
        let item = BufferedItem {
            data,
            timestamp: Instant::now(),
        };

        buffer.push_back(item);

        let duration = start.elapsed();
        self.update_metrics(true, duration, buffer.len());

        Ok(())
    }

    /// Pop single item from buffer
    pub async fn pop(&self) -> DataStorageResult<Option<StreamData>> {
        let start = Instant::now();
        let mut buffer = self.buffer.lock().await;

        // Clean expired items first
        self.clean_expired_items(&mut buffer);

        let result = buffer.pop_front().map(|item| item.data);

        let duration = start.elapsed();
        self.update_metrics(true, duration, buffer.len());

        Ok(result)
    }

    /// Pop batch of items from buffer
    pub async fn pop_batch(&self, count: usize) -> DataStorageResult<Vec<StreamData>> {
        let start = Instant::now();
        let mut buffer = self.buffer.lock().await;

        // Clean expired items first
        self.clean_expired_items(&mut buffer);

        let mut items = Vec::with_capacity(count.min(buffer.len()));

        for _ in 0..count {
            if let Some(item) = buffer.pop_front() {
                items.push(item.data);
            } else {
                break;
            }
        }

        let duration = start.elapsed();
        self.update_metrics(true, duration, buffer.len());

        Ok(items)
    }

    /// Get current buffer length
    pub async fn len(&self) -> usize {
        let buffer = self.buffer.lock().await;
        buffer.len()
    }

    /// Check if buffer is empty
    pub async fn is_empty(&self) -> bool {
        let buffer = self.buffer.lock().await;
        buffer.is_empty()
    }

    /// Clear all items from buffer
    pub async fn clear(&self) -> DataStorageResult<()> {
        let mut buffer = self.buffer.lock().await;
        let cleared_count = buffer.len();
        buffer.clear();

        tracing::info!("Buffer cleared, removed {} items", cleared_count);
        Ok(())
    }

    /// Clean expired items from buffer
    fn clean_expired_items(&self, buffer: &mut VecDeque<BufferedItem>) {
        let max_age = Duration::from_millis(self.config.max_item_age_ms);
        let now = Instant::now();

        let mut expired_count = 0;
        while let Some(front) = buffer.front() {
            if now.duration_since(front.timestamp) > max_age {
                buffer.pop_front();
                expired_count += 1;
            } else {
                break;
            }
        }

        if expired_count > 0 {
            tracing::debug!("Cleaned {} expired items from buffer", expired_count);
        }
    }

    /// Get buffer utilization percentage
    pub async fn utilization(&self) -> f64 {
        let buffer = self.buffer.lock().await;
        (buffer.len() as f64 / self.config.capacity as f64) * 100.0_f64
    }

    /// Update buffer metrics
    fn update_metrics(&self, success: bool, duration: Duration, buffer_size: usize) {
        let mut metrics = self.metrics.lock();
        metrics.total_processed += 1;

        if !success {
            metrics.total_errors += 1;
        }

        // Update latency
        let latency_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.avg_latency_us = u64::midpoint(metrics.avg_latency_us, latency_us);

        // Update buffer size
        metrics.buffer_size = buffer_size;

        // Update throughput
        let items_per_second = 1.0_f64 / duration.as_secs_f64();
        metrics.throughput = f64::midpoint(metrics.throughput, items_per_second);

        metrics.status = *self.status.lock();
        metrics.last_updated = chrono::Utc::now();
    }

    /// Get current metrics
    pub async fn metrics(&self) -> StreamMetrics {
        let mut metrics = self.metrics.lock().clone();
        metrics.buffer_size = self.len().await;
        metrics
    }

    /// Health check
    pub async fn health_check(&self) -> DataStorageResult<()> {
        let status = *self.status.lock();
        if status == StreamStatus::Error {
            return Err(DataStorageError::stream("buffer", "Buffer in error state"));
        }

        let utilization = self.utilization().await;
        if utilization > 90.0_f64 {
            tracing::warn!("Buffer utilization high: {:.1}%", utilization);
        }

        if utilization > 95.0_f64 {
            return Err(DataStorageError::stream(
                "buffer",
                format!("Buffer utilization critical: {utilization:.1}%"),
            ));
        }

        Ok(())
    }

    /// Get buffer statistics
    pub async fn stats(&self) -> BufferStats {
        let buffer = self.buffer.lock().await;
        let utilization = (u64_to_f64_safe(buffer.len() as u64)
            / u64_to_f64_safe(self.config.capacity as u64))
            * 100.0_f64;

        BufferStats {
            capacity: self.config.capacity,
            current_size: buffer.len(),
            utilization_percent: utilization,
            overflow_protection: self.config.enable_overflow_protection,
            max_item_age_ms: self.config.max_item_age_ms,
        }
    }
}

/// Buffer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    /// Buffer capacity
    pub capacity: usize,
    /// Current buffer size
    pub current_size: usize,
    /// Buffer utilization percentage
    pub utilization_percent: f64,
    /// Whether overflow protection is enabled
    pub overflow_protection: bool,
    /// Maximum item age in milliseconds
    pub max_item_age_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Transaction;

    #[tokio::test]
    async fn test_stream_buffer_creation() -> DataStorageResult<()> {
        let config = BufferConfig::default();
        let _buffer = StreamBuffer::new(config)?;
        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_push_pop() -> DataStorageResult<()> {
        let config = BufferConfig::default();
        let buffer = StreamBuffer::new(config)?;

        buffer.start()?;

        // Test push
        let tx = Transaction::new(
            1,
            100,
            "0x123".to_string(),
            "0xabc".to_string(),
            Some("0xdef".to_string()),
            "1.0_f64".to_string(),
            "20".to_string(),
        );

        let stream_data = StreamData::Transaction(tx);
        buffer.push(stream_data.clone()).await?;

        assert_eq!(buffer.len().await, 1);
        assert!(!buffer.is_empty().await);

        // Test pop
        let popped = buffer.pop().await?;
        assert!(popped.is_some());
        assert_eq!(buffer.len().await, 0);
        assert!(buffer.is_empty().await);

        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_batch_operations() -> DataStorageResult<()> {
        let config = BufferConfig::default();
        let buffer = StreamBuffer::new(config)?;

        buffer.start()?;

        // Push multiple items
        for i in 0..5 {
            let tx = Transaction::new(
                1,
                100 + i,
                format!("0x{i:x}"),
                "0xabc".to_string(),
                Some("0xdef".to_string()),
                "1.0_f64".to_string(),
                "20".to_string(),
            );

            let stream_data = StreamData::Transaction(tx);
            buffer.push(stream_data).await?;
        }

        assert_eq!(buffer.len().await, 5);

        // Pop batch
        let batch = buffer.pop_batch(3).await?;
        assert_eq!(batch.len(), 3);
        assert_eq!(buffer.len().await, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_buffer_overflow_protection() -> DataStorageResult<()> {
        let config = BufferConfig {
            capacity: 2, // Small capacity for testing
            enable_overflow_protection: true,
            ..BufferConfig::default()
        };

        let buffer = StreamBuffer::new(config)?;
        buffer.start()?;

        // Fill buffer to capacity
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

            let stream_data = StreamData::Transaction(tx);
            buffer.push(stream_data).await?; // Should not fail due to overflow protection
        }

        // Buffer should still be at capacity (oldest item removed)
        assert_eq!(buffer.len().await, 2);

        Ok(())
    }
}
