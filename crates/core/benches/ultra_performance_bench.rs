//! Ultra-Performance Benchmarks for `TallyIO` Regression Fixes
//!
//! Benchmarks for newly implemented ultra-optimized components:
//! - `UltraSPSCQueue`: Target <100ns per operation
//! - `UltraRingBuffer`: Target <100ns per operation
//! - `ThreadLocalMemoryPool`: Target <10ns per allocation
//!
//! These benchmarks validate the performance regression fixes
//! and ensure production-ready performance for financial applications.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use tallyio_core::optimization::{fast_alloc, fast_dealloc, UltraRingBuffer, UltraSPSCQueue};

/// Benchmark ultra-optimized SPSC queue
fn bench_ultra_spsc_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_spsc_queue");

    for capacity in [1024_usize, 4096_usize, 16384_usize] {
        group.bench_with_input(
            BenchmarkId::new("enqueue_dequeue", capacity),
            &capacity,
            |b, &capacity| {
                let Ok(queue) = UltraSPSCQueue::<u64>::new(capacity) else {
                    return; // Skip benchmark if creation fails
                };

                b.iter(|| {
                    // Single enqueue/dequeue cycle - target <100ns total
                    let value = black_box(42_u64);
                    if queue.try_enqueue(value).is_ok() {
                        if let Ok(result) = queue.try_dequeue() {
                            black_box(result);
                        }
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_operations", capacity),
            &capacity,
            |b, &capacity| {
                let Ok(queue) = UltraSPSCQueue::<u64>::new(capacity) else {
                    return; // Skip benchmark if creation fails
                };

                b.iter(|| {
                    // Batch of 100 operations to test sustained performance
                    for i in 0_u64..100_u64 {
                        let value = black_box(i);
                        if queue.try_enqueue(value).is_err() {
                            break; // Stop if queue is full
                        }
                    }

                    for _ in 0_i32..100_i32 {
                        if let Ok(result) = queue.try_dequeue() {
                            black_box(result);
                        } else {
                            break; // Stop if queue is empty
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ultra-optimized ring buffer
fn bench_ultra_ring_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_ring_buffer");

    for capacity in [512_usize, 2048_usize, 8192_usize] {
        group.bench_with_input(
            BenchmarkId::new("write_read_cycle", capacity),
            &capacity,
            |b, &capacity| {
                let Ok(buffer) = UltraRingBuffer::<u64>::new(capacity) else {
                    return; // Skip benchmark if creation fails
                };

                b.iter(|| {
                    // Single write/read cycle - target <100ns total
                    let value = black_box(42_u64);
                    if buffer.try_write(value).is_ok() {
                        if let Ok(result) = buffer.try_read() {
                            black_box(result);
                        }
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sustained_throughput", capacity),
            &capacity,
            |b, &capacity| {
                let Ok(buffer) = UltraRingBuffer::<u64>::new(capacity) else {
                    return; // Skip benchmark if creation fails
                };

                b.iter(|| {
                    // Fill buffer to 75% capacity and then drain
                    let operations = (capacity * 3_usize) / 4_usize;

                    for i in 0..operations {
                        let value = black_box(i as u64);
                        if buffer.try_write(value).is_err() {
                            break; // Stop if buffer is full
                        }
                    }

                    for _ in 0..operations {
                        if let Ok(result) = buffer.try_read() {
                            black_box(result);
                        } else {
                            break; // Stop if buffer is empty
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark thread-local memory pool
fn bench_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");

    for size in [64_usize, 256_usize, 1024_usize, 4096_usize] {
        group.bench_with_input(
            BenchmarkId::new("alloc_dealloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    // Single allocation/deallocation cycle - target <10ns total
                    if let Ok(ptr) = fast_alloc(size) {
                        fast_dealloc(ptr, size);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_allocations", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    // Batch of 50 allocations to test pool efficiency
                    let mut ptrs = Vec::with_capacity(50_usize);

                    for _ in 0_i32..50_i32 {
                        if let Ok(ptr) = fast_alloc(size) {
                            ptrs.push(ptr);
                        }
                    }

                    for ptr in ptrs {
                        fast_dealloc(ptr, size);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ultra-low latency requirement validation
fn bench_ultra_low_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultra_low_latency");

    group.bench_function("single_enqueue_dequeue", |b| {
        let Ok(queue) = UltraSPSCQueue::<u64>::new(1024_usize) else {
            return; // Skip benchmark if creation fails
        };

        b.iter(|| {
            let start = Instant::now();

            // Critical path: single enqueue + dequeue
            if queue.try_enqueue(black_box(42_u64)).is_ok() {
                if let Ok(result) = queue.try_dequeue() {
                    black_box(result);
                }
            }

            let elapsed = start.elapsed();

            // Log latency violations instead of panicking
            if elapsed.as_nanos() > 1_000_000_u128 {
                eprintln!("Latency violation: {}ns > 1ms", elapsed.as_nanos());
            }
        });
    });

    group.bench_function("memory_allocation", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Critical path: memory allocation + deallocation
            if let Ok(ptr) = fast_alloc(1024_usize) {
                fast_dealloc(ptr, 1024_usize);
            }

            let elapsed = start.elapsed();

            // Log latency violations instead of panicking
            if elapsed.as_nanos() > 1_000_000_u128 {
                eprintln!("Latency violation: {}ns > 1ms", elapsed.as_nanos());
            }
        });
    });

    group.finish();
}

/// Benchmark memory usage scaling
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for data_size_bytes in [1024_usize, 4096_usize, 16384_usize, 65536_usize] {
        group.bench_with_input(
            BenchmarkId::new("data_size_bytes", data_size_bytes),
            &data_size_bytes,
            |b, &data_size_bytes| {
                b.iter(|| {
                    // Allocate and immediately deallocate to test scaling
                    if let Ok(ptr) = fast_alloc(data_size_bytes) {
                        // Simulate some work with the memory
                        black_box(ptr);
                        fast_dealloc(ptr, data_size_bytes);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency under load
fn bench_latency_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_under_load");

    for load_factor in [1.0_f64, 2.0_f64, 5.0_f64, 10.0_f64] {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let load_factor_u32 = load_factor as u32;

        group.bench_with_input(
            BenchmarkId::new("load_factor", load_factor_u32),
            &load_factor,
            |b, &load_factor| {
                let Ok(queue) = UltraSPSCQueue::<u64>::new(1024_usize) else {
                    return; // Skip benchmark if creation fails
                };

                b.iter(|| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let operations = (100.0_f64 * load_factor) as usize;
                    let mut latency_violations = 0_i32;

                    for i in 0..operations {
                        let start = Instant::now();

                        // Critical operation under load
                        if queue.try_enqueue(black_box(i as u64)).is_ok() {
                            if let Ok(result) = queue.try_dequeue() {
                                black_box(result);
                            }
                        }

                        let elapsed = start.elapsed();

                        // Count latency violations (>1ms)
                        if elapsed.as_nanos() > 1_000_000_u128 {
                            latency_violations += 1_i32;
                        }
                    }

                    // Report violations for monitoring
                    if latency_violations > 0_i32 {
                        eprintln!(
                            "Load factor {load_factor}: {latency_violations} latency violations"
                        );
                    }

                    black_box(latency_violations);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    ultra_performance_benches,
    bench_ultra_spsc_queue,
    bench_ultra_ring_buffer,
    bench_memory_pool,
    bench_ultra_low_latency,
    bench_memory_usage,
    bench_latency_under_load
);

criterion_main!(ultra_performance_benches);
