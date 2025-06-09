//! Integration tests for ultra-optimized components
//!
//! These tests validate the integration and performance of ultra-optimized
//! components designed for <1ms latency requirements.

#![allow(clippy::unwrap_used)] // Tests can use unwrap for simplicity

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tallyio_core::lockfree::ultra::{CacheAligned, UltraMemoryPool, UltraSPSCQueue};

/// Test task implementation
#[allow(dead_code)]
struct TestTask {
    id: u64,
    counter: Arc<AtomicU64>,
}

#[allow(dead_code)]
impl TestTask {
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
        50_u64
    }
}

#[test]
fn test_ultra_spsc_queue_basic_operations() {
    let queue = UltraSPSCQueue::<u64>::new(1024).unwrap();

    // Test basic enqueue/dequeue
    assert!(queue.try_enqueue(42).is_ok());
    assert!(queue.try_enqueue(43).is_ok());

    assert_eq!(queue.try_dequeue(), Some(42));
    assert_eq!(queue.try_dequeue(), Some(43));
    assert_eq!(queue.try_dequeue(), None);
}

#[test]
fn test_ultra_spsc_queue_capacity_limits() {
    let queue = UltraSPSCQueue::<u64>::new(2).unwrap();

    // Fill to capacity
    assert!(queue.try_enqueue(1).is_ok());
    assert!(queue.try_enqueue(2).is_ok());

    // Should fail when full
    assert!(queue.try_enqueue(3).is_err());

    // Dequeue one and try again
    assert_eq!(queue.try_dequeue(), Some(1));
    assert!(queue.try_enqueue(3).is_ok());
}

#[test]
fn test_ultra_memory_pool_basic_operations() {
    let pool = UltraMemoryPool::new();

    // Test allocation and deallocation
    let ptr = pool.allocate(64).unwrap();
    unsafe {
        pool.deallocate(ptr, 64).unwrap();
    }

    let stats = pool.stats();
    assert_eq!(stats.total_allocations, 1);
    assert_eq!(stats.total_deallocations, 1);
    assert_eq!(stats.current_usage(), 0);
}

#[test]
fn test_ultra_memory_pool_different_sizes() {
    let pool = UltraMemoryPool::new();
    let sizes = [64, 256, 1024, 4096];
    let mut ptrs = Vec::new();

    // Allocate different sizes
    for &size in &sizes {
        let ptr = pool.allocate(size).unwrap();
        ptrs.push((ptr, size));
    }

    // Deallocate all
    for (ptr, size) in ptrs {
        unsafe {
            pool.deallocate(ptr, size).unwrap();
        }
    }

    let stats = pool.stats();
    assert_eq!(stats.total_allocations, sizes.len());
    assert_eq!(stats.total_deallocations, sizes.len());
    assert_eq!(stats.current_usage(), 0);
}

#[test]
fn test_cache_aligned_operations() {
    let counter = CacheAligned::new(AtomicU64::new(0));

    // Test atomic operations
    counter.fetch_add(1, Ordering::Relaxed);
    counter.fetch_add(2, Ordering::Relaxed);

    assert_eq!(counter.load(Ordering::Relaxed), 3);

    // Test alignment
    let ptr = std::ptr::from_ref::<AtomicU64>(counter.get()) as usize;
    assert_eq!(
        ptr % 64_usize,
        0_usize,
        "CacheAligned should be 64-byte aligned"
    );
}

#[test]
fn test_ultra_spsc_queue_performance_target() {
    let queue = UltraSPSCQueue::<u64>::new(16384).unwrap(); // Povečaj kapaciteto
    let iterations = 8_000; // Zmanjšaj iteracije da se izognemo polnemu queue-u

    // Warm up
    for i in 0..100 {
        let _ = queue.try_enqueue(i);
        let _ = queue.try_dequeue();
    }

    // Measure enqueue performance
    let start = Instant::now();
    let mut enqueued = 0_i32;
    for i in 0..iterations {
        if queue.try_enqueue(i).is_ok() {
            enqueued += 1_i32;
        } else {
            // Queue is full, stop
            break;
        }
    }
    let enqueue_time = start.elapsed();

    // Measure dequeue performance
    let start = Instant::now();
    let mut dequeued = 0_i32;
    for _ in 0_i32..enqueued {
        if queue.try_dequeue().is_some() {
            dequeued += 1_i32;
        } else {
            // Queue is empty, stop
            break;
        }
    }
    let dequeue_time = start.elapsed();

    let avg_enqueue_ns = if enqueued > 0_i32 {
        let enqueued_u128 = u128::try_from(enqueued).unwrap_or(1_u128);
        enqueue_time.as_nanos() / enqueued_u128
    } else {
        0_u128
    };
    let avg_dequeue_ns = if dequeued > 0_i32 {
        let dequeued_u128 = u128::try_from(dequeued).unwrap_or(1_u128);
        dequeue_time.as_nanos() / dequeued_u128
    } else {
        0_u128
    };

    println!("Average enqueue time: {avg_enqueue_ns}ns");
    println!("Average dequeue time: {avg_dequeue_ns}ns");

    // Performance targets: <1000ns per operation (relaxed for integration test)
    assert!(
        avg_enqueue_ns < 1000,
        "Enqueue too slow: {avg_enqueue_ns}ns"
    );
    assert!(
        avg_dequeue_ns < 1000,
        "Dequeue too slow: {avg_dequeue_ns}ns"
    );
}

#[test]
fn test_ultra_memory_pool_performance_target() {
    let pool = UltraMemoryPool::new();
    let iterations = 1000;

    // Warm up
    let mut ptrs = Vec::with_capacity(100);
    for _ in 0_i32..100_i32 {
        ptrs.push(pool.allocate(64_usize).unwrap());
    }
    for ptr in ptrs {
        unsafe {
            pool.deallocate(ptr, 64_usize).unwrap();
        }
    }

    // Measure allocation performance
    let start = Instant::now();
    let mut ptrs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        ptrs.push(pool.allocate(64_usize).unwrap());
    }
    let alloc_time = start.elapsed();

    // Measure deallocation performance
    let start = Instant::now();
    for ptr in ptrs {
        unsafe {
            pool.deallocate(ptr, 64_usize).unwrap();
        }
    }
    let dealloc_time = start.elapsed();

    let avg_alloc_ns = alloc_time.as_nanos() / iterations as u128;
    let avg_dealloc_ns = dealloc_time.as_nanos() / iterations as u128;

    println!("Average allocation time: {avg_alloc_ns}ns");
    println!("Average deallocation time: {avg_dealloc_ns}ns");

    // Performance targets: <1000ns per operation (relaxed for system allocation)
    assert!(avg_alloc_ns < 1000, "Allocation too slow: {avg_alloc_ns}ns");
    assert!(
        avg_dealloc_ns < 1000,
        "Deallocation too slow: {avg_dealloc_ns}ns"
    );
}

#[test]
fn test_end_to_end_workflow() {
    let queue = UltraSPSCQueue::<u64>::new(1024).unwrap();
    let pool = UltraMemoryPool::new();
    let counter = Arc::new(AtomicU64::new(0));

    // Simulate MEV workflow
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        // 1. Allocate memory for transaction data
        let ptr = pool.allocate(256).unwrap();

        // 2. Queue transaction for processing
        queue.try_enqueue(i).unwrap();

        // 3. Process transaction
        let tx_id = queue.try_dequeue().unwrap();
        counter.fetch_add(tx_id, Ordering::Relaxed);

        // 4. Deallocate memory
        unsafe {
            pool.deallocate(ptr, 256).unwrap();
        }
    }

    let total_time = start.elapsed();
    let avg_time_us = total_time.as_micros() / u128::from(iterations);

    println!("Average end-to-end time: {avg_time_us}μs");

    // Target: <100μs per complete workflow
    assert!(avg_time_us < 100, "End-to-end too slow: {avg_time_us}μs");

    // Verify all transactions processed
    let expected_sum = (0..iterations).sum::<u64>();
    assert_eq!(counter.load(Ordering::Relaxed), expected_sum);

    // Verify memory pool stats
    let stats = pool.stats();
    assert_eq!(
        stats.total_allocations,
        usize::try_from(iterations).unwrap_or(0_usize)
    );
    assert_eq!(
        stats.total_deallocations,
        usize::try_from(iterations).unwrap_or(0_usize)
    );
    assert_eq!(stats.current_usage(), 0);
}

#[test]
fn test_concurrent_safety() {
    use std::thread;

    let queue = Arc::new(UltraSPSCQueue::<u64>::new(8192).unwrap());
    let pool = Arc::new(UltraMemoryPool::new());
    let counter = Arc::new(AtomicU64::new(0));

    let producer_queue = queue.clone();
    let producer_counter = counter.clone();

    // Producer thread
    let producer = thread::spawn(move || {
        for i in 0..1000 {
            while producer_queue.try_enqueue(i).is_err() {
                thread::yield_now();
            }
            producer_counter.fetch_add(1, Ordering::Relaxed);
        }
    });

    let consumer_counter = counter.clone();

    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut consumed = 0_u32;
        while consumed < 1000_u32 {
            if queue.try_dequeue().is_some() {
                consumed += 1_u32;
                consumer_counter.fetch_add(1_u64, Ordering::Relaxed);
            } else {
                thread::yield_now();
            }
        }
    });

    // Memory allocator thread
    let allocator_pool = pool.clone();
    let allocator = thread::spawn(move || {
        for _ in 0_i32..500_i32 {
            let ptr = allocator_pool.allocate(128_usize).unwrap();
            unsafe {
                allocator_pool.deallocate(ptr, 128_usize).unwrap();
            }
        }
    });

    // Wait for all threads
    producer.join().unwrap();
    consumer.join().unwrap();
    allocator.join().unwrap();

    // Verify results
    assert_eq!(counter.load(Ordering::Relaxed), 2000); // 1000 produced + 1000 consumed

    let stats = pool.stats();
    assert_eq!(stats.total_allocations, 500);
    assert_eq!(stats.total_deallocations, 500);
    assert_eq!(stats.current_usage(), 0);
}
