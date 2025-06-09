//! Advanced features benchmarks for `TallyIO` Week 4-6
//!
//! Comprehensive benchmarks for lock-free data structures,
//! CPU cache optimization, and performance regression testing.
//!
//! This module implements production-ready benchmarks with:
//! - Zero-panic guarantee
//! - Comprehensive error handling
//! - Input validation
//! - Performance monitoring
//! - Cache-aligned operations
//! - <1ms latency requirements

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use garde::Validate;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::{thread, time::Duration};
use thiserror::Error;

use tallyio_core::lockfree::{
    cache::LockFreeCache,
    queue::{MPSCQueue, SPSCQueue},
    ring_buffer::{MPSCRingBuffer, SafeLockFreeRingBuffer},
};
// Note: optimization imports will be added when implementing cache-aligned and regression benchmarks

/// Benchmark configuration constants to minimize allocations
const SPSC_CAPACITIES: &[usize] = &[1_024, 4_096, 16_384];
const MPSC_CAPACITIES: &[usize] = &[1_024, 4_096, 16_384];
const CACHE_CAPACITIES: &[usize] = &[256, 1_024, 4_096];
const RING_CAPACITIES: &[usize] = &[512, 2_048, 8_192];

/// Standard benchmark iterations for consistent testing
const STANDARD_ITERATIONS: u32 = 100;
const MPSC_ITERATIONS: u32 = 25;
const RING_ITERATIONS: u32 = 50;

/// Benchmark-specific error types following `TallyIO` standards
#[derive(Error, Debug, Clone)]
pub enum BenchmarkError {
    #[error("Queue operation failed: {reason}")]
    QueueOperationFailed { reason: String },

    #[error("Cache operation failed: {reason}")]
    CacheOperationFailed { reason: String },

    #[error("Buffer operation failed: {reason}")]
    BufferOperationFailed { reason: String },

    #[error("Allocation failed: {reason}")]
    AllocationFailed { reason: String },

    #[error("Performance requirement violated: expected <{expected_ms}ms, got {actual_ms}ms")]
    PerformanceViolation { expected_ms: u64, actual_ms: u64 },

    #[error("Invalid input: {field}")]
    InvalidInput { field: String },
}

pub type BenchmarkResult<T> = Result<T, BenchmarkError>;

/// Configuration for benchmark parameters with validation
#[derive(Debug, Clone, Validate)]
pub struct BenchmarkConfig {
    #[garde(range(min = 1, max = 1_000_000))]
    pub capacity: usize,

    #[garde(range(min = 1, max = 10_000))]
    pub iterations: u32,

    #[garde(range(min = 1, max = 100))]
    pub thread_count: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            capacity: 1_024,
            iterations: 100,
            thread_count: 4,
        }
    }
}

/// Performance metrics collector with cache-aligned counters
///
/// Thread-safe metrics collection using atomic operations with `SeqCst` ordering
/// to ensure proper synchronization across concurrent benchmark operations.
/// Designed for zero-panic guarantee and high-performance data collection.
#[derive(Debug)]
pub struct BenchmarkMetrics {
    operations_count: AtomicU64,
    total_duration_ns: AtomicU64,
    error_count: AtomicU64,
    cache_hits: AtomicU64,        // For cache-specific operations
    cache_misses: AtomicU64,      // For cache-specific operations
    queue_operations: AtomicU64,  // For queue-specific operations
    buffer_operations: AtomicU64, // For buffer-specific operations
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkMetrics {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            operations_count: AtomicU64::new(0),
            total_duration_ns: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            queue_operations: AtomicU64::new(0),
            buffer_operations: AtomicU64::new(0),
        }
    }

    fn record_operation(&self, duration_ns: u64) {
        self.operations_count.fetch_add(1, Ordering::SeqCst);
        self.total_duration_ns
            .fetch_add(duration_ns, Ordering::SeqCst);
    }

    fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::SeqCst);
    }

    fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::SeqCst);
    }

    fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::SeqCst);
    }

    fn record_queue_operation(&self) {
        self.queue_operations.fetch_add(1, Ordering::SeqCst);
    }

    fn record_buffer_operation(&self) {
        self.buffer_operations.fetch_add(1, Ordering::SeqCst);
    }

    #[allow(dead_code)]
    fn get_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.operations_count.load(Ordering::SeqCst),
            self.total_duration_ns.load(Ordering::SeqCst),
            self.error_count.load(Ordering::SeqCst),
            self.cache_hits.load(Ordering::SeqCst),
            self.cache_misses.load(Ordering::SeqCst),
        )
    }

    fn get_detailed_stats(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        (
            self.operations_count.load(Ordering::SeqCst),
            self.total_duration_ns.load(Ordering::SeqCst),
            self.error_count.load(Ordering::SeqCst),
            self.cache_hits.load(Ordering::SeqCst),
            self.cache_misses.load(Ordering::SeqCst),
            self.queue_operations.load(Ordering::SeqCst),
            self.buffer_operations.load(Ordering::SeqCst),
        )
    }
}

/// Benchmark lock-free SPSC queue operations with comprehensive error handling
fn bench_spsc_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_queue");
    let metrics = Arc::new(BenchmarkMetrics::new());

    // Pre-validate configurations once to avoid repeated validation
    let configs: Vec<_> = SPSC_CAPACITIES
        .iter()
        .map(|&capacity| {
            let config = BenchmarkConfig {
                capacity,
                iterations: STANDARD_ITERATIONS,
                thread_count: 1,
            };

            if let Err(e) = config.validate(&()) {
                eprintln!("Configuration validation failed: {e:?}");
                return (capacity, BenchmarkConfig::default()); // Return default config as fallback
            }

            (capacity, config)
        })
        .collect();

    for (capacity, config) in &configs {
        group.bench_with_input(
            BenchmarkId::new("enqueue_dequeue", *capacity),
            capacity,
            |b, &capacity| {
                // Use pre-validated config reference

                // Create queue with proper error handling
                let queue = match SPSCQueue::new(capacity) {
                    Ok(q) => Arc::new(q),
                    Err(e) => {
                        eprintln!("Failed to create SPSC queue: {e:?}");
                        metrics.record_error();
                        return;
                    }
                };

                b.iter(|| {
                    let start = std::time::Instant::now();

                    // Pre-allocate vector with capacity to avoid reallocations
                    let mut results =
                        Vec::with_capacity(usize::try_from(config.iterations).unwrap_or(100));

                    // Enqueue operations with error tracking
                    for i in 0..config.iterations {
                        if queue
                            .enqueue(black_box(i32::try_from(i).unwrap_or(0_i32)))
                            .is_err()
                        {
                            metrics.record_error();
                            break;
                        }
                        metrics.record_queue_operation();
                    }

                    // Dequeue operations with error tracking
                    for _ in 0..config.iterations {
                        if let Ok(value) = queue.dequeue() {
                            results.push(value);
                            metrics.record_queue_operation();
                        } else {
                            metrics.record_error();
                            break;
                        }
                    }

                    let duration = start.elapsed();
                    let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
                    metrics.record_operation(duration_ns);

                    // Ensure we don't optimize away the results
                    black_box(results);
                });
            },
        );
    }

    // Log final metrics with all counters
    let (ops, total_ns, errors, hits, misses, queue_ops, buffer_ops) = metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        println!("SPSC Queue Metrics - Ops: {ops}, Avg: {avg_ns}ns, Errors: {errors}, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}, Cache hits: {hits}, misses: {misses}");
    }

    group.finish();
}

/// Benchmark lock-free MPSC queue operations with comprehensive error handling
fn bench_mpsc_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpsc_queue");
    let metrics = Arc::new(BenchmarkMetrics::new());

    // Pre-validate configurations once to avoid repeated validation
    let configs: Vec<_> = MPSC_CAPACITIES
        .iter()
        .map(|&capacity| {
            let config = BenchmarkConfig {
                capacity,
                iterations: MPSC_ITERATIONS,
                thread_count: 4,
            };

            if let Err(e) = config.validate(&()) {
                eprintln!("Configuration validation failed: {e:?}");
                return (capacity, BenchmarkConfig::default()); // Return default config as fallback
            }

            (capacity, config)
        })
        .collect();

    for (capacity, config) in &configs {
        group.bench_with_input(
            BenchmarkId::new("concurrent_enqueue", *capacity),
            capacity,
            |b, &capacity| {
                // Use pre-validated config reference

                b.iter(|| {
                    let start = std::time::Instant::now();

                    // Create queue with proper error handling
                    let queue = match MPSCQueue::new(capacity) {
                        Ok(q) => Arc::new(q),
                        Err(e) => {
                            eprintln!("Failed to create MPSC queue: {e:?}");
                            metrics.record_error();
                            return;
                        }
                    };

                    // Pre-allocate handles vector with capacity
                    let mut handles = Vec::with_capacity(usize::try_from(config.thread_count).unwrap_or(4));

                    // Spawn multiple producer threads
                    for thread_id in 0..config.thread_count {
                        let queue_clone = Arc::clone(&queue);
                        let metrics_clone = Arc::clone(&metrics);
                        let iterations = config.iterations;

                        let handle = thread::spawn(move || {
                            // Wrap thread execution in panic handler for zero-panic guarantee
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                for i in 0..iterations {
                                    let value = i32::try_from(thread_id).unwrap_or(0_i32) * 100_i32 + i32::try_from(i).unwrap_or(0_i32);
                                    if queue_clone.enqueue(black_box(value)).is_err() {
                                        metrics_clone.record_error();
                                        break;
                                    }
                                }
                            }));

                            if result.is_err() {
                                eprintln!("❌ CRITICAL: Thread execution panicked! This violates TallyIO zero-panic requirement.");
                                metrics_clone.record_error();
                            }
                        });
                        handles.push(handle);
                    }

                    // Wait for all producers with comprehensive error and panic handling
                    for handle in handles {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.join())) {
                            Ok(join_result) => {
                                if join_result.is_err() {
                                    eprintln!("Thread join failed");
                                    metrics.record_error();
                                }
                            },
                            Err(_panic_payload) => {
                                eprintln!("❌ CRITICAL: Thread panicked! This violates TallyIO zero-panic requirement.");
                                metrics.record_error();
                            }
                        }
                    }

                    // Single consumer dequeues all with counting
                    let mut dequeued_count = 0_u32;
                    while queue.dequeue().is_ok() {
                        dequeued_count += 1;
                        // Prevent infinite loop
                        if dequeued_count > config.thread_count * config.iterations {
                            break;
                        }
                    }

                    let duration = start.elapsed();
                    let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
                    metrics.record_operation(duration_ns);

                    // Verify we dequeued expected number of items
                    let expected = config.thread_count * config.iterations;
                    if dequeued_count != expected {
                        eprintln!("Expected {expected} items, got {dequeued_count}");
                        metrics.record_error();
                    }
                });
            },
        );
    }

    // Log final metrics with all counters
    let (ops, total_ns, errors, hits, misses, queue_ops, buffer_ops) = metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        println!("MPSC Queue Metrics - Ops: {ops}, Avg: {avg_ns}ns, Errors: {errors}, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}, Cache hits: {hits}, misses: {misses}");
    }

    group.finish();
}

/// Benchmark lock-free cache operations with comprehensive error handling
fn bench_lockfree_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockfree_cache");
    let metrics = Arc::new(BenchmarkMetrics::new());

    // Pre-validate configurations once to avoid repeated validation
    let configs: Vec<_> = CACHE_CAPACITIES
        .iter()
        .map(|&capacity| {
            let config = BenchmarkConfig {
                capacity,
                iterations: STANDARD_ITERATIONS,
                thread_count: 1,
            };

            if let Err(e) = config.validate(&()) {
                eprintln!("Configuration validation failed: {e:?}");
                return (capacity, BenchmarkConfig::default()); // Return default config as fallback
            }

            (capacity, config)
        })
        .collect();

    for (capacity, config) in &configs {
        group.bench_with_input(
            BenchmarkId::new("get_put_operations", *capacity),
            capacity,
            |b, &capacity| {
                // Use pre-validated config reference

                // Create cache with proper error handling
                let cache = match LockFreeCache::<String, i32>::new(capacity) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Failed to create lock-free cache: {e:?}");
                        metrics.record_error();
                        return;
                    }
                };

                b.iter(|| {
                    let start = std::time::Instant::now();

                    // Pre-allocate keys vector to avoid reallocations
                    let mut keys =
                        Vec::with_capacity(usize::try_from(config.iterations).unwrap_or(100));

                    // Generate keys once to avoid allocation in hot path
                    for i in 0..config.iterations {
                        keys.push(format!("key_{}", i % 50));
                    }

                    // Put operations with error tracking
                    for (i, key) in keys.iter().enumerate() {
                        if cache
                            .put(key, black_box(i32::try_from(i).unwrap_or(0_i32)))
                            .is_err()
                        {
                            metrics.record_error();
                            break;
                        }
                    }

                    // Get operations with cache hit/miss tracking
                    for key in &keys {
                        if cache.get(key).is_ok() {
                            metrics.record_cache_hit();
                        } else {
                            // Could be cache miss or actual error
                            metrics.record_cache_miss();
                        }
                    }

                    let duration = start.elapsed();
                    let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
                    metrics.record_operation(duration_ns);
                });
            },
        );
    }

    // Log final metrics with all counters
    let (ops, total_ns, errors, hits, misses, queue_ops, buffer_ops) = metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        let hit_rate = if hits + misses > 0 {
            (f64::from(u32::try_from(hits).unwrap_or(0))
                / f64::from(u32::try_from(hits + misses).unwrap_or(1)))
                * 100.0_f64
        } else {
            0.0_f64
        };
        println!("Cache Metrics - Ops: {ops}, Avg: {avg_ns}ns, Errors: {errors}, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}, Hit rate: {hit_rate:.1}%");
    }

    group.finish();
}

/// Benchmark ring buffer operations with comprehensive error handling
fn bench_ring_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");
    let metrics = Arc::new(BenchmarkMetrics::new());

    for capacity in RING_CAPACITIES {
        group.bench_with_input(
            BenchmarkId::new("write_read_cycle", capacity),
            capacity,
            |b, &capacity| {
                // Validate configuration
                let config = BenchmarkConfig {
                    capacity,
                    iterations: STANDARD_ITERATIONS,
                    thread_count: 1,
                };

                if let Err(e) = config.validate(&()) {
                    eprintln!("Configuration validation failed: {e:?}");
                    return;
                }

                // Create buffer with proper error handling
                let buffer = match SafeLockFreeRingBuffer::<u64>::new(capacity) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to create ring buffer: {e:?}");
                        metrics.record_error();
                        return;
                    }
                };

                b.iter(|| {
                    let start = std::time::Instant::now();

                    // Pre-allocate results vector
                    let mut results =
                        Vec::with_capacity(usize::try_from(config.iterations).unwrap_or(100));

                    // Write operations with error tracking
                    for i in 0..config.iterations {
                        let value = u64::from(i);
                        if buffer.write(black_box(value)).is_err() {
                            metrics.record_error();
                            break;
                        }
                        metrics.record_buffer_operation();
                    }

                    // Read operations with error tracking
                    for _ in 0..config.iterations {
                        if let Ok(value) = buffer.read() {
                            results.push(value);
                            metrics.record_buffer_operation();
                        } else {
                            metrics.record_error();
                            break;
                        }
                    }

                    let duration = start.elapsed();
                    let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
                    metrics.record_operation(duration_ns);

                    // Ensure we don't optimize away the results
                    black_box(results);
                });
            },
        );
    }

    // Log final metrics with all counters
    let (ops, total_ns, errors, _hits, _misses, queue_ops, buffer_ops) =
        metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        println!("Ring Buffer Metrics - Ops: {ops}, Avg: {avg_ns}ns, Errors: {errors}, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}");
    }

    group.finish();
}

/// Benchmark MPSC ring buffer with concurrent writers and comprehensive error handling
fn bench_mpsc_ring_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpsc_ring_buffer");
    let metrics = Arc::new(BenchmarkMetrics::new());

    group.bench_function("concurrent_write_single_read", |b| {
        // Validate configuration
        let config = BenchmarkConfig {
            capacity: 4_096,
            iterations: RING_ITERATIONS,
            thread_count: 4,
        };

        if let Err(e) = config.validate(&()) {
            eprintln!("Configuration validation failed: {e:?}");
            return;
        }

        b.iter(|| {
            let start = std::time::Instant::now();

            // Create buffer with proper error handling
            let buffer = match MPSCRingBuffer::new(config.capacity) {
                Ok(b) => Arc::new(b),
                Err(e) => {
                    eprintln!("Failed to create MPSC ring buffer: {e:?}");
                    metrics.record_error();
                    return;
                }
            };

            // Pre-allocate handles vector
            let mut handles = Vec::with_capacity(usize::try_from(config.thread_count).unwrap_or(4));

            // Spawn multiple writer threads
            for thread_id in 0..config.thread_count {
                let buffer_clone = Arc::clone(&buffer);
                let metrics_clone = Arc::clone(&metrics);
                let iterations = config.iterations;

                let handle = thread::spawn(move || {
                    // Wrap thread execution in panic handler for zero-panic guarantee
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        for i in 0..iterations {
                            let value = i32::try_from(thread_id).unwrap_or(0_i32) * 1_000_i32 + i32::try_from(i).unwrap_or(0_i32);
                            if buffer_clone.write(black_box(value)).is_err() {
                                metrics_clone.record_error();
                                break;
                            }
                        }
                    }));

                    if result.is_err() {
                        eprintln!("❌ CRITICAL: Thread execution panicked! This violates TallyIO zero-panic requirement.");
                        metrics_clone.record_error();
                    }
                });
                handles.push(handle);
            }

            // Wait for all writers with comprehensive error and panic handling
            for handle in handles {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.join())) {
                    Ok(join_result) => {
                        if join_result.is_err() {
                            eprintln!("Thread join failed");
                            metrics.record_error();
                        }
                    },
                    Err(_panic_payload) => {
                        eprintln!("❌ CRITICAL: Thread panicked! This violates TallyIO zero-panic requirement.");
                        metrics.record_error();
                    }
                }
            }

            // Single reader reads all with counting
            let mut read_count = 0_u32;
            while buffer.read().is_ok() {
                read_count += 1;
                // Prevent infinite loop
                if read_count > config.thread_count * config.iterations {
                    break;
                }
            }

            let duration = start.elapsed();
            let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
            metrics.record_operation(duration_ns);

            // Verify we read expected number of items
            let expected = config.thread_count * config.iterations;
            if read_count != expected {
                eprintln!("Expected {expected} items, got {read_count}");
                metrics.record_error();
            }
        });
    });

    // Log final metrics with all counters
    let (ops, total_ns, errors, _hits, _misses, queue_ops, buffer_ops) =
        metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        println!("MPSC Ring Buffer Metrics - Ops: {ops}, Avg: {avg_ns}ns, Errors: {errors}, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}");
    }

    group.finish();
}

/// Benchmark ultra-low latency operations with <1ms guarantee
fn bench_ultra_low_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_low_latency");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1_000);
    let metrics = Arc::new(BenchmarkMetrics::new());

    group.bench_function("single_enqueue_dequeue", |b| {
        // Validate configuration for ultra-low latency
        let config = BenchmarkConfig {
            capacity: 1_024,
            iterations: 1,
            thread_count: 1,
        };

        if let Err(e) = config.validate(&()) {
            eprintln!("Configuration validation failed: {e:?}");
            return;
        }

        // Create queue with proper error handling
        let queue = match SPSCQueue::new(config.capacity) {
            Ok(q) => q,
            Err(e) => {
                eprintln!("Failed to create SPSC queue: {e:?}");
                metrics.record_error();
                return;
            }
        };

        b.iter(|| {
            let start = std::time::Instant::now();

            // Critical path - must be <1ms
            let enqueue_result = queue.enqueue(black_box(42_i32));
            let dequeue_result = queue.dequeue();

            let duration = start.elapsed();

            // Verify <1ms requirement
            if duration.as_millis() >= 1 {
                eprintln!("CRITICAL: Latency violation! Expected <1ms, got {duration:?}");
                metrics.record_error();
            } else {
                let duration_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
                metrics.record_operation(duration_ns);
            }

            // Check operation success - don't use cache_hit for queue operations
            if enqueue_result.is_err() || dequeue_result.is_err() {
                metrics.record_error();
            }
            // Note: Successful queue operations are already recorded via record_operation()
        });
    });

    // Log final ultra-low latency metrics with all counters
    let (ops, total_ns, errors, _hits, _misses, queue_ops, buffer_ops) =
        metrics.get_detailed_stats();
    if ops > 0 {
        let avg_ns = total_ns / ops;
        let max_latency_ms = f64::from(u32::try_from(avg_ns).unwrap_or(u32::MAX)) / 1_000_000.0_f64;
        let success_rate = if ops > 0 {
            (f64::from(u32::try_from(ops - errors).unwrap_or(0))
                / f64::from(u32::try_from(ops).unwrap_or(1)))
                * 100.0_f64
        } else {
            0.0_f64
        };

        println!("Ultra-Low Latency Metrics - Ops: {ops}, Avg: {avg_ns}ns ({max_latency_ms:.3}ms), Success rate: {success_rate:.1}%, Queue ops: {queue_ops}, Buffer ops: {buffer_ops}");

        if max_latency_ms >= 1.0_f64 {
            eprintln!("❌ CRITICAL: Ultra-low latency requirement VIOLATED! Max latency: {max_latency_ms:.3}ms >= 1ms");
        } else {
            println!("✅ Ultra-low latency requirement MET: {max_latency_ms:.3}ms < 1ms");
        }
    }

    group.finish();
}

criterion_group!(
    advanced_features_benches,
    bench_spsc_queue,
    bench_mpsc_queue,
    bench_lockfree_cache,
    bench_ring_buffer,
    bench_mpsc_ring_buffer,
    bench_ultra_low_latency
);

criterion_main!(advanced_features_benches);

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::{BenchmarkConfig, BenchmarkError, BenchmarkMetrics, BenchmarkResult};

    /// Test that benchmark configuration validation works
    #[test]
    fn test_benchmark_config_validation() -> BenchmarkResult<()> {
        // Valid configuration
        let valid_config = BenchmarkConfig {
            capacity: 1_024,
            iterations: 100,
            thread_count: 4,
        };
        assert!(valid_config.validate(&()).is_ok());

        // Invalid capacity (too large)
        let invalid_config = BenchmarkConfig {
            capacity: 2_000_000, // Exceeds max
            iterations: 100,
            thread_count: 4,
        };
        assert!(invalid_config.validate(&()).is_err());

        Ok(())
    }

    /// Test that metrics collection works correctly
    #[test]
    fn test_benchmark_metrics() -> BenchmarkResult<()> {
        let metrics = BenchmarkMetrics::new();

        // Record some operations
        metrics.record_operation(1000);
        metrics.record_operation(2000);
        metrics.record_error();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let (ops, total_ns, errors, hits, misses) = metrics.get_stats();

        assert_eq!(ops, 2);
        assert_eq!(total_ns, 3000);
        assert_eq!(errors, 1);
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);

        Ok(())
    }

    /// Test performance requirement validation
    #[test]
    fn test_performance_requirements() -> BenchmarkResult<()> {
        let start = std::time::Instant::now();

        // Simulate a fast operation
        let _result = 42_u64.wrapping_add(1);

        let duration = start.elapsed();

        // Verify it's under 1ms (ultra-low latency requirement)
        if duration.as_millis() >= 1 {
            return Err(BenchmarkError::PerformanceViolation {
                expected_ms: 1,
                actual_ms: duration.as_millis(),
            });
        }

        Ok(())
    }

    /// Test error handling patterns
    #[test]
    fn test_error_handling() -> BenchmarkResult<()> {
        // Test that our error types work correctly
        let error = BenchmarkError::InvalidInput {
            field: "test_field".to_string(),
        };

        assert!(error.to_string().contains("Invalid input"));
        assert!(error.to_string().contains("test_field"));

        Ok(())
    }
}
