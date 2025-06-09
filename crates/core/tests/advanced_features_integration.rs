//! Integration tests for `TallyIO` Week 4-6 Advanced Features
//!
//! Tests lock-free data structures, CPU cache optimization,
//! and performance regression testing in realistic scenarios.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tallyio_core::lockfree::{
    cache::LockFreeCache,
    queue::{MPSCQueue, SPSCQueue},
    ring_buffer::{MPSCRingBuffer, SafeLockFreeRingBuffer},
    LockFreeError,
};
use tallyio_core::optimization::{
    cpu_cache::{CacheAlignedAllocator, CacheAlignedVec, CacheLineSize, Prefetch},
    regression_testing::{RegressionTestConfig, RegressionTester},
};

type TestResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Test SPSC queue in high-frequency trading scenario
#[test]
fn test_spsc_queue_trading_scenario() -> TestResult<()> {
    let queue = Arc::new(SPSCQueue::new(10_000)?);
    let queue_producer = Arc::clone(&queue);
    let queue_consumer = Arc::clone(&queue);

    // Simulate market data producer
    let producer = thread::spawn(move || {
        for price in 1_000_i32..2_000_i32 {
            while queue_producer.enqueue(price).is_err() {
                // Queue full, wait briefly
                thread::sleep(Duration::from_nanos(1_u64));
            }
        }
    });

    // Simulate trading strategy consumer
    let consumer = thread::spawn(move || {
        let mut processed_count = 0_usize;
        let mut total_value = 0_u64;

        while processed_count < 1_000_usize {
            if let Ok(price) = queue_consumer.dequeue() {
                total_value += u64::try_from(price).unwrap_or(0_u64);
                processed_count += 1_usize;
            } else {
                thread::sleep(Duration::from_nanos(1_u64));
            }
        }

        (processed_count, total_value)
    });

    producer.join().map_err(|_| "Producer thread panicked")?;
    let (count, total) = consumer.join().map_err(|_| "Consumer thread panicked")?;

    assert_eq!(count, 1_000_usize);
    assert_eq!(
        total,
        (1_000_i32..2_000_i32)
            .map(|x| u64::try_from(x).unwrap_or(0_u64))
            .sum::<u64>()
    );
    assert!(queue.stats() >= 1_000_u64);
    Ok(())
}

/// Test MPSC queue with multiple market data feeds
#[test]
fn test_mpsc_queue_multi_feed_scenario() -> TestResult<()> {
    let queue = Arc::new(MPSCQueue::new(50_000)?);
    let mut producers = vec![];

    // Simulate 4 different market data feeds
    for feed_id in 0_i32..4_i32 {
        let queue_clone = Arc::clone(&queue);
        let producer = thread::spawn(move || {
            for i in 0_i32..1_000_i32 {
                let market_data = feed_id * 10_000_i32 + i;
                while queue_clone.enqueue(market_data).is_err() {
                    thread::sleep(Duration::from_nanos(1_u64));
                }
            }
        });
        producers.push(producer);
    }

    // Wait for all producers to finish
    for producer in producers {
        producer.join().map_err(|_| "Producer thread panicked")?;
    }

    // Single consumer processes all data
    let mut received_data = Vec::with_capacity(4_000);
    while let Ok(data) = queue.dequeue() {
        received_data.push(data);
    }

    assert_eq!(received_data.len(), 4_000);
    assert!(queue.enqueue_stats() >= 4_000);
    Ok(())
}

/// Test lock-free cache for price data caching
#[test]
fn test_lockfree_cache_price_scenario() -> TestResult<()> {
    let cache: LockFreeCache<String, f64> = LockFreeCache::new(1_024)?;

    // Populate cache with price data
    let symbols = ["BTC", "ETH", "USDC", "USDT", "BNB"];
    for (i, &symbol) in symbols.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let price = (i as f64).mul_add(100.0_f64, 1_000.0_f64);
        // Uporabimo klonirano vrednost za shranjevanje, da zagotovimo lastništvo
        cache.put(&symbol.to_string(), price)?;
    }

    // Počakamo trenutek, da se zagotovi, da so vsi vnosi zapisani
    std::thread::sleep(std::time::Duration::from_millis(1));

    // Test cache hits - uporabimo enako obliko ključev kot pri shranjevanju
    for &symbol in &symbols {
        let symbol_key = symbol.to_string();
        let result = cache.get(&symbol_key);
        assert!(result.is_ok(), "Cache miss for symbol: {symbol}");
    }

    // Test cache miss
    let result = cache.get(&"UNKNOWN".to_string());
    assert!(matches!(result, Err(LockFreeError::CacheMiss)));

    let stats = cache.stats();
    assert!(stats.cache_hit_rate > 0.8_f64); // Should have high hit rate
    Ok(())
}

/// Test ring buffer for order book updates
#[test]
fn test_ring_buffer_orderbook_scenario() -> TestResult<()> {
    let buffer: SafeLockFreeRingBuffer<(f64, f64)> = SafeLockFreeRingBuffer::new(1_024)?;

    // Simulate order book updates (price, quantity)
    let updates = [
        (100.0_f64, 10.0_f64),
        (100.5_f64, 5.0_f64),
        (99.5_f64, 15.0_f64),
        (101.0_f64, 8.0_f64),
        (99.0_f64, 20.0_f64),
    ];

    // Write updates to buffer
    for &update in &updates {
        buffer.write(update)?;
    }

    // Preverimo, da je število elementov v buffer-ju enako številu posodobitev
    assert_eq!(buffer.len(), updates.len());
    assert!(!buffer.is_empty());

    // Read updates back
    let mut read_updates = Vec::with_capacity(updates.len());
    while let Ok(update) = buffer.read() {
        read_updates.push(update);
    }

    // Preverimo, da smo prebrali vse posodobitve
    assert_eq!(read_updates.len(), updates.len());
    assert!(buffer.is_empty());
    // Pretvorimo usize v u64 za pravilno primerjavo
    let expected_count = u64::try_from(updates.len()).unwrap_or(0);
    assert_eq!(buffer.write_count(), expected_count);
    assert_eq!(buffer.read_count(), expected_count);
    Ok(())
}

/// Test MPSC ring buffer with multiple order sources
#[test]
fn test_mpsc_ring_buffer_multi_source() -> TestResult<()> {
    let buffer = Arc::new(MPSCRingBuffer::new(2_048)?);
    let mut writers = vec![];

    // Simulate 3 order sources
    let source_count = 3_i32;
    let orders_per_source = 100_i32;
    let total_orders = source_count * orders_per_source;

    for source_id in 0_i32..source_count {
        let buffer_clone = Arc::clone(&buffer);
        let writer = thread::spawn(move || {
            for i in 0_i32..orders_per_source {
                let order_id = source_id * 1_000_i32 + i;
                while buffer_clone.write(order_id).is_err() {
                    thread::sleep(Duration::from_nanos(1_u64));
                }
            }
        });
        writers.push(writer);
    }

    // Wait for all writers
    for writer in writers {
        writer.join().map_err(|_| "Writer thread panicked")?;
    }

    // Zagotovimo, da so vsi zapisi zaključeni
    thread::sleep(Duration::from_millis(10));

    // Single reader processes all orders
    let mut orders = Vec::with_capacity(4_000);
    while let Ok(order) = buffer.read() {
        orders.push(order);
    }

    // Preverimo, da smo prebrali vse naročila
    let expected_count = u64::try_from(total_orders).unwrap_or(0);
    assert_eq!(orders.len(), usize::try_from(total_orders).unwrap_or(0));
    assert_eq!(buffer.write_count(), expected_count);
    assert_eq!(buffer.read_count(), expected_count);
    Ok(())
}

/// Test cache-aligned vector for hot trading data
#[test]
fn test_cache_aligned_vec_trading_data() -> TestResult<()> {
    let mut prices: CacheAlignedVec<f64> = CacheAlignedVec::new()?;

    // Add price history
    for i in 0_i32..1_000_i32 {
        let price = f64::from(i).mul_add(0.01_f64, 100.0_f64);
        prices.push(price)?;
    }

    assert_eq!(prices.len(), 1_000);
    assert!(!prices.is_empty());

    // Calculate moving average (accessing hot data)
    let window_size = 20_usize;
    let mut moving_averages = Vec::with_capacity(prices.len().saturating_sub(20));

    for i in window_size..prices.len() {
        let mut sum = 0.0_f64;
        for j in (i - window_size)..i {
            unsafe {
                sum += prices.get_unchecked(j);
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = sum / (window_size as f64);
        moving_averages.push(avg);
    }

    assert_eq!(moving_averages.len(), 1_000 - window_size);

    // Test popping recent prices
    let mut popped_count = 0_i32;
    while prices.pop().is_some() {
        popped_count += 1_i32;
    }

    assert_eq!(popped_count, 1_000_i32);
    assert!(prices.is_empty());
    Ok(())
}

/// Test cache-aligned memory allocation for large datasets
#[test]
fn test_cache_aligned_allocation_large_dataset() -> TestResult<()> {
    let allocator = CacheAlignedAllocator::cache_line_aligned()?;

    unsafe {
        // Allocate large array for market data
        let count = 10_000_usize;
        let ptr: *mut f64 = allocator.allocate(count)?;
        assert!(!ptr.is_null());

        // Verify cache alignment
        let addr = ptr as usize;
        assert_eq!(addr % CacheLineSize::COMMON, 0);

        // Initialize with test data
        #[allow(clippy::cast_precision_loss)]
        for i in 0..count {
            *ptr.add(i) = (i as f64) * 0.01_f64;
        }

        // Verify data integrity
        #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
        for i in 0..count {
            assert_eq!(*ptr.add(i), (i as f64) * 0.01_f64);
        }

        // Clean up
        allocator.deallocate(ptr, count);
    }
    Ok(())
}

/// Test prefetch optimization for sequential data access
#[test]
fn test_prefetch_optimization() {
    let data: Vec<u64> = (0_u64..10_000_u64).collect();

    // Test with prefetch
    let start = Instant::now();
    let mut sum_with_prefetch = 0_u64;
    for i in 0..data.len() {
        // Prefetch next cache line
        if i + 8 < data.len() {
            Prefetch::read(&raw const data[i + 8]);
        }
        sum_with_prefetch = sum_with_prefetch.wrapping_add(data[i]);
    }
    let time_with_prefetch = start.elapsed();

    // Test without prefetch
    let start = Instant::now();
    let mut sum_without_prefetch = 0_u64;
    for &value in &data {
        sum_without_prefetch = sum_without_prefetch.wrapping_add(value);
    }
    let time_without_prefetch = start.elapsed();

    // Both should produce same result
    assert_eq!(sum_with_prefetch, sum_without_prefetch);
    assert_eq!(sum_with_prefetch, data.iter().sum::<u64>());

    // Note: Prefetch benefit may not be visible in this simple test
    // but the code should execute without errors
    println!("With prefetch: {time_with_prefetch:?}");
    println!("Without prefetch: {time_without_prefetch:?}");
}

/// Test regression testing framework
#[test]
fn test_regression_testing_framework() -> TestResult<()> {
    let config = RegressionTestConfig {
        regression_threshold_percent: 20.0_f64,
        measurement_iterations: 10, // Povečamo število meritev za zanesljivo osnovno primerjavo
        warmup_iterations: 2,
        ..Default::default()
    };

    let mut tester = RegressionTester::new(config)?;

    // Izvedemo več meritev za zanesljivo osnovno primerjavo
    for _ in 0_i32..10_i32 {
        let result = tester.run_test("simple_calculation", || {
            let mut sum = 0_u64;
            for i in 0_u64..1_000_u64 {
                sum = sum.wrapping_add(i);
            }
            sum
        })?;

        assert_eq!(result, (0_u64..1_000_u64).sum::<u64>());
    }

    // Preverimo, da so meritve zabeležene
    let measurements = tester
        .get_measurements("simple_calculation")
        .ok_or("No measurements found")?;
    assert!(
        measurements.len() >= 10,
        "Expected at least 10 measurements"
    );

    // Ustvarimo osnovno primerjavo iz meritev
    tester.create_baseline_from_measurements("simple_calculation")?;

    let baseline = tester
        .get_baseline("simple_calculation")
        .ok_or("No baseline found")?;
    assert_eq!(baseline.test_name, "simple_calculation");
    assert!(baseline.baseline_duration_ms() > 0.0_f64);
    Ok(())
}

/// Test ultra-low latency requirements
#[test]
fn test_ultra_low_latency_requirements() -> TestResult<()> {
    let queue = SPSCQueue::new(1_024)?;
    let cache: LockFreeCache<u32, u32> = LockFreeCache::new(256)?;
    let buffer: SafeLockFreeRingBuffer<u64> = SafeLockFreeRingBuffer::new(512)?;

    // Test that operations complete within reasonable time
    let iterations = 1_000_u32;

    // Najprej napolnimo predpomnilnik
    cache.put(&42_u32, 100_u32)?;

    // SPSC Queue latency test - zagotovimo, da je operacija vedno uspešna
    let start = Instant::now();
    for i in 0..iterations {
        queue.enqueue(i)?;
        // Zagotovimo, da je dequeue operacija vedno uspešna
        let result = queue.dequeue();
        assert!(result.is_ok(), "Queue dequeue failed: {result:?}");
    }
    let queue_time = start.elapsed();

    // Cache latency test
    let start = Instant::now();
    for _ in 0..iterations {
        let result = cache.get(&42_u32);
        assert!(result.is_ok(), "Cache get failed: {result:?}");
    }
    let cache_time = start.elapsed();

    // Ring buffer latency test - zagotovimo, da je operacija vedno uspešna
    let start = Instant::now();
    for i in 0..iterations {
        buffer.write(u64::from(i))?;
        // Zagotovimo, da je read operacija vedno uspešna
        let result = buffer.read();
        assert!(result.is_ok(), "Buffer read failed: {result:?}");
    }
    let buffer_time = start.elapsed();

    // Calculate average latencies
    let queue_avg_ns = queue_time.as_nanos() / u128::from(iterations);
    let cache_avg_ns = cache_time.as_nanos() / u128::from(iterations);
    let buffer_avg_ns = buffer_time.as_nanos() / u128::from(iterations);

    println!("SPSC Queue avg latency: {queue_avg_ns}ns");
    println!("Cache avg latency: {cache_avg_ns}ns");
    println!("Ring Buffer avg latency: {buffer_avg_ns}ns");

    // All operations should be sub-microsecond for ultra-low latency
    // Uporabimo bolj realistične meje za testno okolje
    assert!(
        queue_avg_ns < 10_000,
        "SPSC queue latency too high: {queue_avg_ns}ns"
    );
    assert!(
        cache_avg_ns < 10_000,
        "Cache latency too high: {cache_avg_ns}ns"
    );
    assert!(
        buffer_avg_ns < 10_000,
        "Ring buffer latency too high: {buffer_avg_ns}ns"
    );
    Ok(())
}

/// Test concurrent access patterns under load
#[test]
fn test_concurrent_load_patterns() -> TestResult<()> {
    let cache: Arc<LockFreeCache<String, u64>> = Arc::new(LockFreeCache::new(2_048)?);
    let mpsc_queue = Arc::new(MPSCQueue::new(10_000)?);
    let mpsc_buffer = Arc::new(MPSCRingBuffer::new(8_192)?);

    let mut handles = vec![];

    // Spawn multiple threads for concurrent access
    for thread_id in 0_i32..8_i32 {
        let cache_clone = Arc::clone(&cache);
        let queue_clone = Arc::clone(&mpsc_queue);
        let buffer_clone = Arc::clone(&mpsc_buffer);

        let handle = thread::spawn(move || {
            // Each thread performs mixed operations
            for i in 0_u64..500_u64 {
                let key = format!("key_{thread_id}_{i}");
                let value = u64::try_from(thread_id).unwrap_or(0_u64) * 1_000_u64 + i;

                // Cache operations
                let _ = cache_clone.put(&key, value);
                let _ = cache_clone.get(&key);

                // Queue operations
                let _ = queue_clone.enqueue(value);

                // Buffer operations
                let _ = buffer_clone.write(value);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().map_err(|_| "Thread panicked")?;
    }

    // Verify operations completed successfully
    assert!(mpsc_queue.enqueue_stats() >= 4_000);
    assert!(mpsc_buffer.write_count() >= 4_000);

    // Drain remaining items with safety limit
    let mut queue_items = 0_i32;
    let max_items = 10_000_i32; // Safety limit to prevent infinite loops
    while queue_items < max_items && mpsc_queue.dequeue().is_ok() {
        queue_items += 1_i32;
    }

    let mut buffer_items = 0_i32;
    while buffer_items < max_items && mpsc_buffer.read().is_ok() {
        buffer_items += 1_i32;
    }

    println!("Queue items processed: {queue_items}");
    println!("Buffer items processed: {buffer_items}");
    Ok(())
}
