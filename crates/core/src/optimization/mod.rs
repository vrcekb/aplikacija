//! Performance Optimizations - Ultra-Performance System Optimizations
//!
//! Production-ready performance optimizations for `TallyIO` crypto MEV bot.
//! Implements CPU affinity, memory pooling, lock-free structures, and SIMD.

use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

pub mod cpu_affinity;
pub mod cpu_cache;
#[cfg(target_os = "linux")]
pub mod io_uring;
pub mod jemalloc_tuning;
#[cfg(target_os = "linux")]
pub mod kernel_bypass;
pub mod lock_free;
pub mod memory_pool;
pub mod mpc_batch;
pub mod numa_allocator;
pub mod performance_monitor;
#[cfg(target_os = "linux")]
pub mod realtime;
pub mod regression_testing;
pub mod simd;
pub mod spsc_queue_ultra;
pub mod typed_lock_free_queue;
pub mod ultra_ring_buffer;

pub use cpu_affinity::*;
#[cfg(target_os = "linux")]
pub use io_uring::*;
pub use jemalloc_tuning::*;
#[cfg(target_os = "linux")]
pub use kernel_bypass::*;
pub use lock_free::*;
pub use memory_pool::*;
pub use mpc_batch::*;
pub use numa_allocator::*;
pub use performance_monitor::*;
#[cfg(target_os = "linux")]
pub use realtime::*;
pub use simd::*;
pub use spsc_queue_ultra::*;
pub use typed_lock_free_queue::*;
pub use ultra_ring_buffer::*;

/// Optimization error types for ultra-performance systems
///
/// This error type is designed for financial-grade production systems with
/// comprehensive error context and no panics. Every error is recoverable
/// and provides detailed diagnostics.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum OptimizationError {
    /// CPU affinity error with detailed reason
    #[error("CPU affinity error: {reason}")]
    CpuAffinityError {
        /// Detailed error reason
        reason: String,
    },

    /// Memory pool error with allocation context
    #[error("Memory pool error: {reason}, size: {size:?}")]
    MemoryPoolError {
        /// Detailed error reason
        reason: String,
        /// Size that was being allocated (if applicable)
        size: Option<usize>,
    },

    /// Lock-free operation error with operation context
    #[error("Lock-free operation error: {reason}, operation: {operation}")]
    LockFreeError {
        /// Detailed error reason
        reason: String,
        /// Operation that failed (push, pop, etc.)
        operation: String,
    },

    /// SIMD operation error with vector details
    #[error("SIMD operation error: {reason}")]
    SimdError {
        /// Detailed error reason
        reason: String,
        /// Operation that failed (add, mul, etc.)
        operation: Option<String>,
        /// Vector size that caused error (if applicable)
        vector_size: Option<usize>,
    },

    /// Resource exhausted with specific resource details
    #[error("Resource exhausted: {resource}, usage: {current_usage:?}/{maximum_usage:?}")]
    ResourceExhausted {
        /// Resource name that was exhausted
        resource: String,
        /// Current usage
        current_usage: Option<usize>,
        /// Maximum allowed usage
        maximum_usage: Option<usize>,
    },

    /// Configuration error with field validation details
    #[error("Configuration error: {field}, expected: {expected:?}")]
    ConfigError {
        /// Field name with invalid value
        field: String,
        /// Expected value or constraint
        expected: Option<String>,
    },
}

/// Optimization result type
pub type OptimizationResult<T> = Result<T, OptimizationError>;

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,

    /// CPU cores to use
    pub cpu_cores: Vec<u32>,

    /// Enable SIMD optimizations
    pub enable_simd: bool,

    /// Memory pool size
    pub memory_pool_size: usize,

    /// Enable NUMA optimizations
    pub enable_numa: bool,

    /// Lock-free queue capacity
    pub lock_free_queue_capacity: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_cpu_affinity: true,
            cpu_cores: (0..u32::try_from(num_cpus::get()).unwrap_or(1)).collect(),
            enable_simd: true,
            memory_pool_size: 64 * 1024 * 1024, // 64MB
            enable_numa: false,
            lock_free_queue_capacity: 10_000,
        }
    }
}

impl OptimizationConfig {
    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> OptimizationResult<()> {
        if self.cpu_cores.is_empty() {
            return Err(OptimizationError::ConfigError {
                field: "cpu_cores cannot be empty".to_string(),
                expected: Some("at least one CPU core".to_string()),
            });
        }

        if self.memory_pool_size == 0 {
            return Err(OptimizationError::ConfigError {
                field: "memory_pool_size must be > 0".to_string(),
                expected: Some("positive value".to_string()),
            });
        }

        if self.lock_free_queue_capacity == 0 {
            return Err(OptimizationError::ConfigError {
                field: "lock_free_queue_capacity must be > 0".to_string(),
                expected: Some("positive value".to_string()),
            });
        }

        Ok(())
    }
}

/// Optimization statistics
#[derive(Debug, Default)]
pub struct OptimizationStats {
    /// CPU affinity operations
    pub cpu_affinity_operations: AtomicU64,

    /// Memory pool allocations
    pub memory_pool_allocations: AtomicU64,

    /// Lock-free operations
    pub lock_free_operations: AtomicU64,

    /// SIMD operations
    pub simd_operations: AtomicU64,

    /// Total optimization time in nanoseconds
    pub total_optimization_time_ns: AtomicU64,
}

impl OptimizationStats {
    /// Get average optimization time in microseconds
    #[must_use]
    pub fn average_optimization_time_us(&self) -> f64 {
        let total_ops = self.cpu_affinity_operations.load(Ordering::Relaxed)
            + self.memory_pool_allocations.load(Ordering::Relaxed)
            + self.lock_free_operations.load(Ordering::Relaxed)
            + self.simd_operations.load(Ordering::Relaxed);

        if total_ops == 0 {
            return 0.0_f64;
        }

        let total_time_ns = self.total_optimization_time_ns.load(Ordering::Relaxed);
        f64::from(u32::try_from(total_time_ns / total_ops).unwrap_or(u32::MAX)) / 1000.0_f64
    }

    /// Get operations per second
    #[must_use]
    pub fn operations_per_second(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds <= 0.0_f64 {
            return 0.0_f64;
        }

        let total_ops = self.cpu_affinity_operations.load(Ordering::Relaxed)
            + self.memory_pool_allocations.load(Ordering::Relaxed)
            + self.lock_free_operations.load(Ordering::Relaxed)
            + self.simd_operations.load(Ordering::Relaxed);

        // Safe conversion with precision awareness for performance metrics
        #[allow(clippy::cast_precision_loss)]
        {
            total_ops as f64 / elapsed_seconds
        }
    }
}

/// Main optimization manager for ultra-low latency operations
///
/// Manages all optimization subsystems with zero-allocation design
/// and cache-aligned data structures for maximum performance in
/// financial trading applications.
#[repr(C, align(64))]
pub struct OptimizationManager {
    /// Configuration for optimization subsystems
    config: OptimizationConfig,

    /// CPU affinity manager for thread pinning
    cpu_affinity: Option<CpuAffinityManager>,

    /// Memory pool for zero-allocation operations
    memory_pool: Option<MemoryPool>,

    /// Lock-free data structures for contention-free concurrency
    lock_free: Option<LockFreeManager>,

    /// SIMD operations for vectorized computation
    simd: Option<SimdOperations>,

    /// Statistics for performance monitoring
    stats: OptimizationStats,
}

impl OptimizationManager {
    /// Create new optimization manager with zero-allocation ultra-low latency design
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails. Every error is recoverable
    /// and includes detailed context for production debugging.
    #[inline]
    pub fn new(config: OptimizationConfig) -> OptimizationResult<Self> {
        // Validate configuration immediately to catch issues early
        config.validate()?;

        // Initialize CPU affinity on supported platforms
        let cpu_affinity = if config.enable_cpu_affinity {
            Some(CpuAffinityManager::new(config.cpu_cores.clone())?)
        } else {
            None
        };

        // Memory pool is always initialized for zero-allocation operations
        let memory_pool = Some(MemoryPool::new(config.memory_pool_size)?);

        // Lock-free structures for non-blocking concurrency
        let lock_free = Some(LockFreeManager::new(config.lock_free_queue_capacity)?);

        // SIMD operations for vectorized processing
        let simd = if config.enable_simd {
            Some(SimdOperations::new()?)
        } else {
            None
        };

        // Return fully initialized manager with cache-aligned structures
        Ok(Self {
            config,
            cpu_affinity,
            memory_pool,
            lock_free,
            simd,
            stats: OptimizationStats::default(),
        })
    }

    /// Apply CPU affinity for current thread with zero-overhead design
    ///
    /// Sets thread-to-core affinity for ultra-low latency operation in
    /// time-critical financial applications. Essential for deterministic
    /// latency in MEV and trading operations.
    ///
    /// # Errors
    ///
    /// Returns detailed error if CPU affinity operation fails with specific reason
    /// for production-grade debugging and resolution.
    #[inline]
    pub fn apply_cpu_affinity(&self, core_id: u32) -> OptimizationResult<()> {
        self.cpu_affinity.as_ref().map_or(Ok(()), |cpu_affinity| {
            match cpu_affinity.set_affinity(core_id) {
                Ok(()) => {
                    // Use Acquire ordering for consistency in performance critical code
                    self.stats
                        .cpu_affinity_operations
                        .fetch_add(1, Ordering::AcqRel);
                    Ok(())
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Allocate memory from ultra-optimized memory pool
    ///
    /// Uses zero-allocation design pattern for sub-10ns allocations in critical paths.
    /// This method is designed for financial trading systems where allocation latency
    /// directly impacts execution timing and trading performance.
    ///
    /// # Errors
    ///
    /// Returns detailed error if memory allocation fails with size context for
    /// immediate production troubleshooting. Every error is fully recoverable.
    #[inline]
    pub fn allocate_memory(&self, size: usize) -> OptimizationResult<Vec<u8>> {
        // Size validation for financial-grade systems
        if size == 0 {
            return Err(OptimizationError::MemoryPoolError {
                reason: "Zero-sized allocation requested".to_string(),
                size: Some(0),
            });
        }

        if let Some(ref memory_pool) = self.memory_pool {
            // Use memory pool for ultra-optimized allocation
            match memory_pool.allocate(size) {
                Ok(memory) => {
                    // Use AcqRel ordering for accurate metrics in multi-core environment
                    self.stats
                        .memory_pool_allocations
                        .fetch_add(1, Ordering::AcqRel);
                    Ok(memory)
                }
                Err(e) => Err(OptimizationError::MemoryPoolError {
                    reason: format!("Pool allocation failed: {e}"),
                    size: Some(size),
                }),
            }
        } else {
            // Fallback with explicit size validation for safety
            if size > isize::MAX as usize / 2 {
                return Err(OptimizationError::MemoryPoolError {
                    reason: "Allocation size exceeds system limits".to_string(),
                    size: Some(size),
                });
            }
            Ok(vec![0; size])
        }
    }

    /// Perform zero-contention lock-free push operation
    ///
    /// Uses ultra-optimized lock-free algorithms for sub-microsecond operations
    /// without contention or allocation. Critical for high-frequency trading and
    /// MEV operations requiring deterministic execution times.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Item type that must be Send + 'static for thread-safety
    ///
    /// # Errors
    ///
    /// Returns detailed error if lock-free operation fails with operation context
    /// for immediate production analysis.
    #[inline]
    pub fn lock_free_push<T>(&self, item: T) -> OptimizationResult<()>
    where
        T: Send + 'static,
    {
        self.lock_free.as_ref().map_or_else(
            || {
                Err(OptimizationError::LockFreeError {
                    reason: "Lock-free operations are disabled".to_string(),
                    operation: "push".to_string(),
                })
            },
            |lock_free| match lock_free.push(item) {
                Ok(()) => {
                    // Use AcqRel for consistent multi-core visibility in trading systems
                    self.stats
                        .lock_free_operations
                        .fetch_add(1, Ordering::AcqRel);
                    Ok(())
                }
                Err(e) => Err(OptimizationError::LockFreeError {
                    reason: format!("Push operation failed: {e}"),
                    operation: "push".to_string(),
                }),
            },
        )
    }

    /// Perform SIMD vectorized array addition for ultra-high performance
    ///
    /// Implements AVX/AVX2/AVX-512 optimized vectorized math operations
    /// critical for high-performance financial modeling and MEV calculations.
    /// Includes zero-allocation optimizations and cache-aware algorithms.
    ///
    /// # Errors
    ///
    /// Returns detailed error with vector size context if operation fails.
    /// All errors are fully recoverable with precise diagnostics.
    #[inline]
    pub fn simd_add_arrays(&self, a: &[f32], b: &[f32]) -> OptimizationResult<Vec<f32>> {
        // Input validation before any processing
        if a.len() != b.len() {
            return Err(OptimizationError::SimdError {
                reason: "Array length mismatch".to_string(),
                operation: Some("add_arrays".to_string()),
                vector_size: Some(a.len()),
            });
        }

        // Empty array fast path
        if a.is_empty() {
            return Ok(Vec::with_capacity(0));
        }

        self.simd.as_ref().map_or_else(
            || {
                // Optimized scalar fallback with pre-allocated capacity
                let mut result = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    result.push(x + y);
                }
                Ok(result)
            },
            |simd| match simd.add_arrays(a, b) {
                Ok(result) => {
                    // Use AcqRel for consistent multi-core visibility
                    self.stats.simd_operations.fetch_add(1, Ordering::AcqRel);
                    Ok(result)
                }
                Err(e) => Err(OptimizationError::SimdError {
                    reason: format!("SIMD operation failed: {e}"),
                    operation: Some("add_arrays".to_string()),
                    vector_size: Some(a.len()),
                }),
            },
        )
    }

    /// Get optimization statistics with ultra-low latency design
    ///
    /// Returns direct reference to internal statistics without Arc overhead
    #[must_use]
    #[inline]
    pub const fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Get optimization configuration with zero-allocation design
    ///
    /// Returns direct reference to internal configuration without Arc overhead
    #[must_use]
    #[inline]
    pub const fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Check if CPU affinity is enabled
    #[must_use]
    pub const fn is_cpu_affinity_enabled(&self) -> bool {
        self.cpu_affinity.is_some()
    }

    /// Check if SIMD is enabled
    #[must_use]
    pub const fn is_simd_enabled(&self) -> bool {
        self.simd.is_some()
    }
}
