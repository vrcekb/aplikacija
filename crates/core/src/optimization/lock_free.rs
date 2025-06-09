//! Lock-free Data Structures
//!
//! Production-ready lock-free data structures for ultra-performance.

use std::{
    any::Any,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use crossbeam::queue::SegQueue;

use super::{OptimizationError, OptimizationResult};

/// Lock-free manager for various data structures
pub struct LockFreeManager {
    /// Lock-free queue
    queue: Arc<SegQueue<Box<dyn Any + Send + 'static>>>,

    /// Queue capacity
    capacity: usize,

    /// Statistics
    stats: Arc<LockFreeStats>,
}

/// Lock-free statistics
#[derive(Debug, Default)]
pub struct LockFreeStats {
    /// Total push operations
    pub total_pushes: AtomicU64,

    /// Total pop operations
    pub total_pops: AtomicU64,

    /// Failed operations
    pub failed_operations: AtomicU64,
}

impl LockFreeStats {
    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total_ops =
            self.total_pushes.load(Ordering::Relaxed) + self.total_pops.load(Ordering::Relaxed);
        if total_ops == 0 {
            return 1.0_f64;
        }

        let failed = self.failed_operations.load(Ordering::Relaxed);
        // Safe conversion with precision awareness for performance metrics
        #[allow(clippy::cast_precision_loss)]
        {
            1.0_f64 - (failed as f64 / total_ops as f64)
        }
    }

    /// Get push/pop ratio
    #[must_use]
    pub fn push_pop_ratio(&self) -> f64 {
        let total_pushes = self.total_pushes.load(Ordering::Relaxed);
        let total_pops = self.total_pops.load(Ordering::Relaxed);

        if total_pops == 0 {
            return f64::INFINITY;
        }

        // Safe conversion with precision awareness for performance metrics
        #[allow(clippy::cast_precision_loss)]
        {
            total_pushes as f64 / total_pops as f64
        }
    }
}

impl LockFreeManager {
    /// Create new lock-free manager
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(capacity: usize) -> OptimizationResult<Self> {
        if capacity == 0 {
            return Err(OptimizationError::LockFreeError {
                reason: "Capacity cannot be zero".to_string(),
                operation: "LockFreeQueue::new".to_string(),
            });
        }

        Ok(Self {
            queue: Arc::new(SegQueue::new()),
            capacity,
            stats: Arc::new(LockFreeStats::default()),
        })
    }

    /// Push item to lock-free queue
    ///
    /// # Errors
    ///
    /// Returns error if push fails
    pub fn push<T>(&self, item: T) -> OptimizationResult<()>
    where
        T: Send + 'static,
    {
        // Check approximate capacity (SegQueue doesn't have exact capacity)
        if self.queue.len() >= self.capacity {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
            return Err(OptimizationError::ResourceExhausted {
                resource: "LockFreeQueue capacity".to_string(),
                current_usage: Some(self.queue.len()),
                maximum_usage: Some(self.capacity),
            });
        }

        self.queue.push(Box::new(item));
        self.stats.total_pushes.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Pop item from lock-free queue
    ///
    /// # Errors
    ///
    /// Returns error if pop fails
    pub fn pop<T>(&self) -> OptimizationResult<Option<T>>
    where
        T: Send + 'static,
    {
        self.stats.total_pops.fetch_add(1, Ordering::Relaxed);

        self.queue.pop().map_or_else(
            || Ok(None),
            |boxed_item| {
                // TODO: PRODUCTION - Implement proper type-safe generic queue
                // Current implementation has type erasure limitations
                // Consider using:
                // - Separate typed queues for each type
                // - Custom trait object system
                // - crossbeam::channel for typed communication

                // Attempt unsafe downcast (not recommended for production)
                boxed_item.downcast::<T>().map_or_else(
                    |_| {
                        self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
                        Err(OptimizationError::LockFreeError {
                            reason: "Type mismatch in downcast".to_string(),
                            operation: "LockFreeQueue::pop".to_string(),
                        })
                    },
                    |item| Ok(Some(*item)),
                )
            },
        )
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
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<LockFreeStats> {
        &self.stats
    }
}

/// Lock-free counter
pub struct LockFreeCounter {
    /// Counter value
    value: AtomicU64,
}

impl LockFreeCounter {
    /// Create new counter
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Create counter with initial value
    #[must_use]
    pub const fn with_value(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Increment counter
    #[must_use]
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Decrement counter
    #[must_use]
    pub fn decrement(&self) -> u64 {
        self.value.fetch_sub(1, Ordering::Relaxed)
    }

    /// Add to counter
    #[must_use]
    pub fn add(&self, value: u64) -> u64 {
        self.value.fetch_add(value, Ordering::Relaxed)
    }

    /// Get current value
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Set value
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Compare and swap
    ///
    /// # Errors
    ///
    /// Returns `Err(actual_value)` if the current value doesn't match expected
    pub fn compare_and_swap(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.value
            .compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }
}

impl Default for LockFreeCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free flag
pub struct LockFreeFlag {
    /// Flag value
    value: std::sync::atomic::AtomicBool,
}

impl LockFreeFlag {
    /// Create new flag
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create flag with initial value
    #[must_use]
    pub const fn with_value(initial: bool) -> Self {
        Self {
            value: std::sync::atomic::AtomicBool::new(initial),
        }
    }

    /// Set flag
    pub fn set(&self) {
        self.value.store(true, Ordering::Relaxed);
    }

    /// Clear flag
    pub fn clear(&self) {
        self.value.store(false, Ordering::Relaxed);
    }

    /// Check if flag is set
    #[must_use]
    pub fn is_set(&self) -> bool {
        self.value.load(Ordering::Relaxed)
    }

    /// Test and set (returns previous value)
    #[must_use]
    pub fn test_and_set(&self) -> bool {
        self.value.swap(true, Ordering::Relaxed)
    }

    /// Compare and swap
    ///
    /// # Errors
    ///
    /// Returns `Err(actual_value)` if the current value doesn't match expected
    pub fn compare_and_swap(&self, current: bool, new: bool) -> Result<bool, bool> {
        self.value
            .compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }
}

impl Default for LockFreeFlag {
    fn default() -> Self {
        Self::new()
    }
}
