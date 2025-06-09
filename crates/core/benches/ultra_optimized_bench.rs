//! Ultra-optimized benchmarks for <1ms latency validation
//!
//! This benchmark suite validates the performance targets for ultra-optimized
//! components designed for MEV trading and financial applications.

#![allow(clippy::unwrap_used)] // Benchmarks can use unwrap for simplicity

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tallyio_core::lockfree::ultra::{UltraMemoryPool, UltraSPSCQueue};

/// Test task for benchmarking
#[allow(dead_code)]
struct BenchTask {
    id: u64,
    counter: Arc<AtomicU64>,
}

#[allow(dead_code)]
impl BenchTask {
    const fn new(id: u64, counter: Arc<AtomicU64>) -> Self {
        Self { id, counter }
    }

    fn execute(&self) {
        self.counter.fetch_add(1_u64, Ordering::Relaxed);
    }

    fn priority(&self) -> u8 {
        u8::try_from(self.id % 10_u64).unwrap_or(0_u8)
    }

    const fn estimated_duration_us() -> u64 {
        50_u64 // 50 microseconds estimated
    }
}

/// Benchmark SPSC queue performance
fn bench_ultra_spsc_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_spsc_queue");

    for capacity in &[256_usize, 1024_usize, 4096_usize] {
        group.bench_with_input(
            BenchmarkId::new("enqueue_dequeue", capacity),
            capacity,
            |b, &capacity| {
                let queue = UltraSPSCQueue::<u64>::new(capacity).unwrap();

                b.iter(|| {
                    // Target: <100ns per operation
                    let value = black_box(42_u64);
                    queue.try_enqueue(value).unwrap();
                    black_box(queue.try_dequeue().unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory pool performance
fn bench_ultra_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_memory_pool");

    for size in &[64_usize, 256_usize, 1024_usize, 4096_usize] {
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            size,
            |b, &size| {
                let pool = UltraMemoryPool::new();

                b.iter(|| {
                    // Target: <50ns per operation
                    let ptr = pool.allocate(black_box(size)).unwrap();
                    unsafe {
                        pool.deallocate(ptr, size).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark queue throughput under load
fn bench_queue_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_throughput");

    group.bench_function("high_throughput", |b| {
        let queue = UltraSPSCQueue::<u64>::new(8192).unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            // Fill queue
            for _ in 0_i32..1_000_i32 {
                if queue.try_enqueue(counter).is_ok() {
                    counter += 1_u64;
                }
            }

            // Drain queue
            let mut drained = 0_i32;
            while queue.try_dequeue().is_some() {
                drained += 1_i32;
            }

            black_box(drained);
        });
    });

    group.finish();
}

/// Benchmark memory pool under different allocation patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    group.bench_function("mixed_sizes", |b| {
        let pool = UltraMemoryPool::new();
        let sizes = [64, 128, 256, 512, 1024];
        let mut ptrs = Vec::with_capacity(100);

        b.iter(|| {
            // Allocate mixed sizes
            for &size in &sizes {
                for _ in 0_i32..20_i32 {
                    if let Ok(ptr) = pool.allocate(size) {
                        ptrs.push((ptr, size));
                    }
                }
            }

            // Deallocate all
            while let Some((ptr, size)) = ptrs.pop() {
                unsafe {
                    let _ = pool.deallocate(ptr, size);
                }
            }
        });
    });

    group.finish();
}

/// Benchmark queue latency under different load factors
fn bench_latency_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_under_load");

    for load_factor in &[0.1_f64, 0.5_f64, 0.8_f64, 0.95_f64] {
        group.bench_with_input(
            BenchmarkId::new("load_factor", format!("{load_factor:.1}")),
            load_factor,
            |b, &load_factor| {
                let queue = UltraSPSCQueue::<u64>::new(1024_usize).unwrap();

                // Pre-fill queue to desired load factor
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let target_items = (1024.0_f64 * load_factor) as usize;
                for i in 0_usize..target_items {
                    let _ = queue.try_enqueue(u64::try_from(i).unwrap_or(0_u64));
                }

                b.iter(|| {
                    // Single operation under load
                    if queue.try_enqueue(black_box(999_u64)).is_err() {
                        // Queue full, dequeue one item
                        black_box(queue.try_dequeue());
                        let _ = queue.try_enqueue(black_box(999_u64));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache-aligned operations
fn bench_cache_aligned(c: &mut Criterion) {
    use std::sync::atomic::AtomicUsize;
    use tallyio_core::lockfree::ultra::CacheAligned;

    let mut group = c.benchmark_group("cache_aligned");

    group.bench_function("atomic_operations", |b| {
        let counter = CacheAligned::new(AtomicUsize::new(0));

        b.iter(|| {
            // Target: <10ns per operation
            counter.fetch_add(black_box(1_usize), Ordering::Relaxed);
            black_box(counter.load(Ordering::Relaxed));
        });
    });

    group.finish();
}

/// End-to-end latency benchmark
fn bench_end_to_end_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_latency");

    group.bench_function("complete_workflow", |b| {
        let queue = UltraSPSCQueue::<u64>::new(1024_usize).unwrap();
        let pool = UltraMemoryPool::new();

        b.iter(|| {
            // Simulate complete MEV workflow
            // 1. Allocate memory for transaction data
            let ptr = pool.allocate(black_box(256_usize)).unwrap();

            // 2. Queue transaction for processing
            let tx_id = black_box(12345_u64);
            queue.try_enqueue(tx_id).unwrap();

            // 3. Process transaction
            let processed_tx = queue.try_dequeue().unwrap();
            black_box(processed_tx);

            // 4. Deallocate memory
            unsafe {
                pool.deallocate(ptr, 256_usize).unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    ultra_optimized_benches,
    bench_ultra_spsc_queue,
    bench_ultra_memory_pool,
    bench_queue_throughput,
    bench_memory_patterns,
    bench_latency_under_load,
    bench_cache_aligned,
    bench_end_to_end_latency
);

criterion_main!(ultra_optimized_benches);
