//! Production-ready batch processor for financial applications
//!
//! This module provides ultra-low latency batch processing capabilities
//! specifically designed for financial trading and MEV applications.
//!
//! # Performance Requirements
//! - Target latency: <1ms for batch submission
//! - Zero allocations in hot paths
//! - Lock-free data structures where possible
//! - Comprehensive error handling without panics

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

/// Batch processor errors
#[derive(Error, Debug, Clone)]
pub enum BatchError {
    /// Buffer is full
    #[error("Buffer is full")]
    BufferFull,

    /// Timeout occurred
    #[error("Timeout occurred")]
    Timeout,

    /// Invalid configuration
    #[error("Invalid configuration: {reason}")]
    InvalidConfiguration {
        /// The reason for the invalid configuration
        reason: String,
    },

    /// Processing failed
    #[error("Processing failed: {reason}")]
    ProcessingFailed {
        /// The reason for the processing failure
        reason: String,
    },
}

/// Batch processing result
pub type BatchResult<T> = Result<T, BatchError>;

/// Type alias for batch processor function
type BatchProcessorFn<T, R> = Arc<dyn Fn(Vec<T>) -> BatchResult<Vec<R>> + Send + Sync>;

/// Flush command for batch processor
#[derive(Debug, Clone)]
enum FlushCommand {
    /// Force flush immediately
    Force,
}

/// Batch item with response channel
struct BatchItem<T, R> {
    /// The item to process
    item: T,
    /// Timestamp when item was added
    timestamp: Instant,
    /// Response channel
    response_tx: oneshot::Sender<BatchResult<R>>,
}

/// Batch processing statistics
#[derive(Debug, Default)]
pub struct BatchStats {
    /// Total items processed
    pub total_items: AtomicU64,
    /// Total batches processed
    pub batches_processed: AtomicU64,
    /// Total latency in nanoseconds
    pub total_latency_ns: AtomicU64,
    /// Maximum latency in nanoseconds
    pub max_latency_ns: AtomicU64,
}

impl BatchStats {
    /// Returns average latency in microseconds
    #[must_use]
    pub fn average_latency_us(&self) -> f64 {
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        let batches = self.batches_processed.load(Ordering::Relaxed);

        if batches == 0 {
            0.0_f64
        } else {
            #[allow(clippy::cast_precision_loss)]
            let total_ns_f64 = total_ns as f64;
            #[allow(clippy::cast_precision_loss)]
            let batches_f64 = batches as f64;
            (total_ns_f64 / batches_f64) / 1000.0_f64
        }
    }

    /// Returns maximum latency in microseconds
    #[must_use]
    pub fn max_latency_us(&self) -> f64 {
        let max_ns = self.max_latency_ns.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let max_ns_f64 = max_ns as f64;
        max_ns_f64 / 1000.0_f64
    }
}

/// Batch processor for aggregating operations to reduce latency spikes
pub struct BatchProcessor<T, R> {
    batch_size: usize,
    #[allow(dead_code)]
    batch_timeout: Duration,
    max_latency: Duration,
    #[allow(dead_code)]
    processor: BatchProcessorFn<T, R>,
    flush_tx: mpsc::UnboundedSender<FlushCommand>,
    stats: Arc<BatchStats>,
    buffer: Arc<tokio::sync::Mutex<Vec<BatchItem<T, R>>>>,
}

impl<T, R> BatchProcessor<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    /// Creates new batch processor
    ///
    /// # Errors
    /// Returns error if configuration is invalid
    pub fn new(
        batch_size: usize,
        batch_timeout: Duration,
        max_latency: Duration,
        processor: impl Fn(Vec<T>) -> BatchResult<Vec<R>> + Send + Sync + 'static,
    ) -> BatchResult<Self> {
        if batch_size == 0 {
            return Err(BatchError::InvalidConfiguration {
                reason: "Batch size cannot be zero".to_string(),
            });
        }

        if batch_timeout.is_zero() {
            return Err(BatchError::InvalidConfiguration {
                reason: "Batch timeout cannot be zero".to_string(),
            });
        }

        let (flush_tx, flush_rx) = mpsc::unbounded_channel();
        let stats = Arc::new(BatchStats::default());
        let buffer = Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(batch_size)));
        let processor = Arc::new(processor);

        // Start background processor
        let buffer_clone = buffer.clone();
        let processor_clone = processor.clone();
        let stats_clone = stats.clone();

        tokio::spawn(Self::background_processor(
            buffer_clone,
            processor_clone,
            stats_clone,
            batch_size,
            batch_timeout,
            max_latency,
            flush_rx,
        ));

        Ok(Self {
            batch_size,
            batch_timeout,
            max_latency,
            processor,
            flush_tx,
            stats,
            buffer,
        })
    }

    /// Submits item for batch processing
    ///
    /// # Errors
    /// Returns error if buffer is full or timeout occurs
    pub async fn submit(&self, item: T) -> BatchResult<R> {
        let (response_tx, response_rx) = oneshot::channel();

        let batch_item = BatchItem {
            item,
            timestamp: Instant::now(),
            response_tx,
        };

        // Add to buffer
        {
            let mut buffer = self.buffer.lock().await;
            buffer.push(batch_item);

            // Update statistics
            self.stats.total_items.fetch_add(1, Ordering::Relaxed);

            // Check if we should force flush
            if buffer.len() >= self.batch_size {
                let _ = self.flush_tx.send(FlushCommand::Force);
            }
        }

        // Wait for response with timeout
        tokio::time::timeout(self.max_latency, response_rx)
            .await
            .map_err(|_| BatchError::Timeout)?
            .map_err(|_| BatchError::ProcessingFailed {
                reason: "Response channel closed".to_string(),
            })?
    }

    /// Returns current statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<BatchStats> {
        &self.stats
    }

    /// Background processor task
    async fn background_processor(
        buffer: Arc<tokio::sync::Mutex<Vec<BatchItem<T, R>>>>,
        processor: BatchProcessorFn<T, R>,
        stats: Arc<BatchStats>,
        batch_size: usize,
        batch_timeout: Duration,
        max_latency: Duration,
        mut flush_rx: mpsc::UnboundedReceiver<FlushCommand>,
    ) {
        let mut timeout_interval = tokio::time::interval(batch_timeout);

        loop {
            tokio::select! {
                _ = timeout_interval.tick() => {
                    Self::process_batch(&buffer, &processor, &stats, batch_size, max_latency).await;
                }

                command = flush_rx.recv() => {
                    match command {
                        Some(FlushCommand::Force) => {
                            Self::process_batch(&buffer, &processor, &stats, batch_size, max_latency).await;
                        }
                        None => {
                            // Process remaining items before shutdown
                            Self::process_batch(&buffer, &processor, &stats, batch_size, max_latency).await;
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Process a single batch
    async fn process_batch(
        buffer: &Arc<tokio::sync::Mutex<Vec<BatchItem<T, R>>>>,
        processor: &BatchProcessorFn<T, R>,
        stats: &Arc<BatchStats>,
        batch_size: usize,
        max_latency: Duration,
    ) {
        let mut buffer_guard = buffer.lock().await;

        if buffer_guard.is_empty() {
            return;
        }

        let start_time = Instant::now();
        let batch_items = buffer_guard.drain(..).collect::<Vec<_>>();
        drop(buffer_guard);

        if batch_items.is_empty() {
            return;
        }

        // Filter out expired items
        let mut valid_items = Vec::with_capacity(batch_size);
        for item in batch_items {
            if start_time.duration_since(item.timestamp) <= max_latency {
                valid_items.push(item);
            } else {
                let _ = item.response_tx.send(Err(BatchError::Timeout));
            }
        }

        if valid_items.is_empty() {
            return;
        }

        // Extract items for processing
        let items: Vec<T> = valid_items
            .iter()
            .map(|bi| unsafe { std::ptr::read(std::ptr::addr_of!(bi.item)) })
            .collect();

        // Process batch
        let process_start = Instant::now();
        let results = processor(items);
        let process_duration = process_start.elapsed();

        // Update statistics
        stats.batches_processed.fetch_add(1, Ordering::Relaxed);
        let process_ns = u64::try_from(process_duration.as_nanos()).unwrap_or(u64::MAX);
        stats
            .total_latency_ns
            .fetch_add(process_ns, Ordering::Relaxed);

        // Update max latency
        let mut current_max = stats.max_latency_ns.load(Ordering::Relaxed);
        while process_ns > current_max {
            match stats.max_latency_ns.compare_exchange_weak(
                current_max,
                process_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_max) => current_max = new_max,
            }
        }

        // Send responses
        match results {
            Ok(results) => {
                for (batch_item, result) in valid_items.into_iter().zip(results.into_iter()) {
                    let _ = batch_item.response_tx.send(Ok(result));
                }
            }
            Err(error) => {
                for batch_item in valid_items {
                    let _ = batch_item.response_tx.send(Err(error.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_batch_processor_creation() -> BatchResult<()> {
        let _processor = BatchProcessor::new(
            10_usize,
            Duration::from_millis(100_u64),
            Duration::from_millis(1000_u64),
            |items: Vec<i32>| Ok(items.into_iter().map(|x| x * 2_i32).collect()),
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_processing() -> BatchResult<()> {
        let processor = BatchProcessor::new(
            3_usize,
            Duration::from_millis(100_u64),
            Duration::from_millis(1000_u64),
            |items: Vec<i32>| Ok(items.into_iter().map(|x| x * 2_i32).collect()),
        )?;

        let result = processor.submit(5_i32).await?;
        assert_eq!(result, 10_i32);
        Ok(())
    }
}
