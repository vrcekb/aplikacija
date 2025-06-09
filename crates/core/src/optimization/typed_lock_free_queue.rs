//! Typed Lock-Free Queue - Production-Ready Ultra-Performance
//!
//! Type-safe lock-free queue implementation using crossbeam channels
//! for maximum performance and safety in financial applications.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError, TrySendError};
use thiserror::Error;

use super::{OptimizationError, OptimizationResult};

/// Typed lock-free queue errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TypedQueueError {
    /// Queue is full
    #[error("Queue is full (capacity: {capacity})")]
    QueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Queue is empty
    #[error("Queue is empty")]
    QueueEmpty,

    /// Queue is disconnected
    #[error("Queue is disconnected")]
    Disconnected,

    /// Invalid capacity
    #[error("Invalid capacity: {capacity}")]
    InvalidCapacity {
        /// Invalid capacity value
        capacity: usize,
    },
}

/// Result type for typed queue operations
pub type TypedQueueResult<T> = Result<T, TypedQueueError>;

/// Typed lock-free queue statistics
#[derive(Debug, Default)]
pub struct TypedQueueStats {
    /// Total items sent
    pub total_sent: AtomicU64,
    /// Total items received
    pub total_received: AtomicU64,
    /// Failed send operations
    pub failed_sends: AtomicU64,
    /// Failed receive operations
    pub failed_receives: AtomicU64,
}

impl TypedQueueStats {
    /// Get success rate for send operations
    #[must_use]
    pub fn send_success_rate(&self) -> f64 {
        let total_attempts =
            self.total_sent.load(Ordering::Relaxed) + self.failed_sends.load(Ordering::Relaxed);
        if total_attempts == 0 {
            return 1.0_f64;
        }

        let successful = self.total_sent.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            successful as f64 / total_attempts as f64
        }
    }

    /// Get throughput (items per second)
    #[must_use]
    pub fn throughput(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds <= 0.0_f64 {
            return 0.0_f64;
        }

        let total_processed =
            self.total_sent.load(Ordering::Relaxed) + self.total_received.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            total_processed as f64 / elapsed_seconds
        }
    }
}

/// Production-ready typed lock-free queue
pub struct TypedLockFreeQueue<T> {
    /// Sender handle
    sender: Sender<T>,
    /// Receiver handle
    receiver: Receiver<T>,
    /// Queue capacity (None for unbounded)
    capacity: Option<usize>,
    /// Statistics
    stats: Arc<TypedQueueStats>,
}

impl<T> TypedLockFreeQueue<T> {
    /// Create new bounded typed lock-free queue
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum queue capacity
    ///
    /// # Errors
    ///
    /// Returns error if capacity is zero
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::TypedLockFreeQueue;
    ///
    /// let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bounded(capacity: usize) -> TypedQueueResult<Self> {
        if capacity == 0 {
            return Err(TypedQueueError::InvalidCapacity { capacity });
        }

        let (sender, receiver) = bounded(capacity);

        Ok(Self {
            sender,
            receiver,
            capacity: Some(capacity),
            stats: Arc::new(TypedQueueStats::default()),
        })
    }

    /// Create new unbounded typed lock-free queue
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::TypedLockFreeQueue;
    ///
    /// let queue = TypedLockFreeQueue::<u64>::unbounded();
    /// ```
    #[must_use]
    pub fn unbounded() -> Self {
        let (sender, receiver) = unbounded();

        Self {
            sender,
            receiver,
            capacity: None,
            stats: Arc::new(TypedQueueStats::default()),
        }
    }

    /// Send item to queue (non-blocking)
    ///
    /// # Arguments
    ///
    /// * `item` - Item to send
    ///
    /// # Errors
    ///
    /// Returns error if queue is full or disconnected
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::TypedLockFreeQueue;
    ///
    /// let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;
    /// queue.try_send(42)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_send(&self, item: T) -> TypedQueueResult<()> {
        match self.sender.try_send(item) {
            Ok(()) => {
                self.stats.total_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                self.stats.failed_sends.fetch_add(1, Ordering::Relaxed);
                Err(TypedQueueError::QueueFull {
                    capacity: self.capacity.unwrap_or(0),
                })
            }
            Err(TrySendError::Disconnected(_)) => {
                self.stats.failed_sends.fetch_add(1, Ordering::Relaxed);
                Err(TypedQueueError::Disconnected)
            }
        }
    }

    /// Receive item from queue (non-blocking)
    ///
    /// # Errors
    ///
    /// Returns error if queue is empty or disconnected
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::TypedLockFreeQueue;
    ///
    /// let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;
    /// queue.try_send(42)?;
    /// let item = queue.try_recv()?;
    /// assert_eq!(item, 42);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn try_recv(&self) -> TypedQueueResult<T> {
        match self.receiver.try_recv() {
            Ok(item) => {
                self.stats.total_received.fetch_add(1, Ordering::Relaxed);
                Ok(item)
            }
            Err(TryRecvError::Empty) => {
                self.stats.failed_receives.fetch_add(1, Ordering::Relaxed);
                Err(TypedQueueError::QueueEmpty)
            }
            Err(TryRecvError::Disconnected) => {
                self.stats.failed_receives.fetch_add(1, Ordering::Relaxed);
                Err(TypedQueueError::Disconnected)
            }
        }
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Check if queue is full (for bounded queues)
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.capacity.is_some_and(|capacity| self.len() >= capacity)
    }

    /// Get current queue length (approximate)
    #[must_use]
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Get queue capacity
    #[must_use]
    pub const fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Get statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<TypedQueueStats> {
        &self.stats
    }

    /// Clone sender for multi-producer scenarios
    #[must_use]
    pub fn clone_sender(&self) -> Sender<T> {
        self.sender.clone()
    }

    /// Clone receiver for multi-consumer scenarios
    #[must_use]
    pub fn clone_receiver(&self) -> Receiver<T> {
        self.receiver.clone()
    }
}

impl<T> Clone for TypedLockFreeQueue<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
            capacity: self.capacity,
            stats: Arc::clone(&self.stats),
        }
    }
}

/// Compatibility wrapper for existing lock-free manager
pub struct TypedLockFreeManager<T> {
    /// Typed queue
    queue: TypedLockFreeQueue<T>,
}

impl<T> TypedLockFreeManager<T> {
    /// Create new typed lock-free manager
    ///
    /// # Errors
    ///
    /// Returns error if capacity is invalid
    pub fn new(capacity: usize) -> OptimizationResult<Self> {
        let queue = TypedLockFreeQueue::bounded(capacity).map_err(|e| {
            OptimizationError::LockFreeError {
                reason: format!("Failed to create typed queue: {e}"),
                operation: "new".to_string(),
            }
        })?;

        Ok(Self { queue })
    }

    /// Push item to queue
    ///
    /// # Errors
    ///
    /// Returns error if push fails
    pub fn push(&self, item: T) -> OptimizationResult<()> {
        self.queue
            .try_send(item)
            .map_err(|e| OptimizationError::LockFreeError {
                reason: format!("Failed to send: {e}"),
                operation: "push".to_string(),
            })
    }

    /// Pop item from queue
    ///
    /// # Errors
    ///
    /// Returns error if pop fails
    pub fn pop(&self) -> OptimizationResult<Option<T>> {
        match self.queue.try_recv() {
            Ok(item) => Ok(Some(item)),
            Err(TypedQueueError::QueueEmpty) => Ok(None),
            Err(e) => Err(OptimizationError::LockFreeError {
                reason: format!("Failed to receive: {e}"),
                operation: "pop".to_string(),
            }),
        }
    }

    /// Get queue length
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get capacity
    #[must_use]
    pub const fn capacity(&self) -> Option<usize> {
        self.queue.capacity()
    }

    /// Get statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<TypedQueueStats> {
        self.queue.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_typed_queue_creation() -> TypedQueueResult<()> {
        let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;
        assert_eq!(queue.capacity(), Some(1024));
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        Ok(())
    }

    #[test]
    fn test_send_recv_operations() -> TypedQueueResult<()> {
        let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;

        queue.try_send(42)?;
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let item = queue.try_recv()?;
        assert_eq!(item, 42);
        assert!(queue.is_empty());

        Ok(())
    }

    #[test]
    fn test_performance_target() -> TypedQueueResult<()> {
        let queue = TypedLockFreeQueue::<u64>::bounded(1024)?;

        let start = Instant::now();
        for i in 0..1000 {
            queue.try_send(i)?;
            assert_eq!(queue.try_recv()?, i);
        }
        let elapsed = start.elapsed();

        // Target: <200ns per operation pair
        let ns_per_op = elapsed.as_nanos() / 2000;
        println!("Typed queue performance: {ns_per_op}ns per operation");

        #[cfg(not(debug_assertions))]
        assert!(
            ns_per_op < 200,
            "Too slow: {ns_per_op}ns per op (target: <200ns)"
        );

        Ok(())
    }

    #[test]
    fn test_statistics() -> TypedQueueResult<()> {
        let queue = TypedLockFreeQueue::<u64>::bounded(4)?;

        // Send some items
        for i in 0..3 {
            queue.try_send(i)?;
        }

        // Receive some items
        for _ in 0_i32..2_i32 {
            queue.try_recv()?;
        }

        let stats = queue.stats();
        assert_eq!(
            stats.total_sent.load(std::sync::atomic::Ordering::Relaxed),
            3
        );
        assert_eq!(
            stats
                .total_received
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );

        Ok(())
    }

    #[test]
    fn test_unbounded_queue() -> TypedQueueResult<()> {
        let queue = TypedLockFreeQueue::<u64>::unbounded();
        assert_eq!(queue.capacity(), None);
        assert!(!queue.is_full()); // Unbounded queues are never full

        // Should be able to send many items
        for i in 0_u64..10000_u64 {
            queue.try_send(i)?;
        }

        assert_eq!(queue.len(), 10000);
        Ok(())
    }
}
