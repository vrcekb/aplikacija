//! Mempool Performance Benchmarks
//!
//! Production-ready benchmarks for `TallyIO` mempool monitoring performance.
//! Validates transaction processing and filtering performance under high load.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use tallyio_core::types::{Address, BlockNumber, Gas, Price, TxHash};

/// Mock transaction for benchmarking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MockTransaction {
    hash: TxHash,
    from: Address,
    to: Option<Address>,
    value: Price,
    gas_limit: Gas,
    gas_price: Price,
    data: Vec<u8>,
    block_number: Option<BlockNumber>,
}

impl MockTransaction {
    fn new() -> Self {
        Self {
            hash: TxHash::new([1; 32]),
            from: Address::new([1; 20]),
            to: Some(Address::new([2; 20])),
            value: Price::from_ether(1),
            gas_limit: Gas::new(21_000),
            gas_price: Price::from_gwei(20),
            data: Vec::with_capacity(0),
            block_number: Some(BlockNumber::new(1000)),
        }
    }

    fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    const fn with_value(mut self, value: Price) -> Self {
        self.value = value;
        self
    }

    const fn with_gas_price(mut self, gas_price: Price) -> Self {
        self.gas_price = gas_price;
        self
    }
}

/// Mock mempool for benchmarking
struct MockMempool {
    transactions: Vec<MockTransaction>,
    capacity: usize,
}

impl MockMempool {
    fn new(capacity: usize) -> Self {
        Self {
            transactions: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn add_transaction(&mut self, tx: MockTransaction) -> bool {
        if self.transactions.len() < self.capacity {
            self.transactions.push(tx);
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    fn get_transactions(&self) -> &[MockTransaction] {
        &self.transactions
    }

    fn filter_by_value(&self, min_value: Price) -> Vec<&MockTransaction> {
        self.transactions
            .iter()
            .filter(|tx| tx.value.wei() >= min_value.wei())
            .collect()
    }

    fn filter_by_gas_price(&self, min_gas_price: Price) -> Vec<&MockTransaction> {
        self.transactions
            .iter()
            .filter(|tx| tx.gas_price.wei() >= min_gas_price.wei())
            .collect()
    }

    fn filter_mev_opportunities(&self) -> Vec<&MockTransaction> {
        self.transactions
            .iter()
            .filter(|tx| {
                // Simple MEV detection: high value transactions with contract calls
                tx.value.wei() > Price::from_ether(10).wei() && !tx.data.is_empty()
            })
            .collect()
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.transactions.clear();
    }

    const fn len(&self) -> usize {
        self.transactions.len()
    }
}

/// Benchmark transaction addition
fn bench_transaction_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_addition");

    for capacity in &[1_000_usize, 10_000_usize, 100_000_usize] {
        group.bench_with_input(
            BenchmarkId::new("capacity", capacity),
            capacity,
            |b, &capacity| {
                b.iter(|| {
                    let mut mempool = MockMempool::new(capacity);
                    for i in 0_i32..1_000_i32 {
                        #[allow(clippy::unwrap_used)]
                        let tx = MockTransaction::new()
                            .with_value(Price::from_ether(u64::try_from(i % 100_i32).unwrap_or(0)));
                        black_box(mempool.add_transaction(tx));
                    }
                    black_box(mempool.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark transaction filtering by value
fn bench_value_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_filtering");

    for tx_count in &[1_000_usize, 10_000_usize, 100_000_usize] {
        group.bench_with_input(
            BenchmarkId::new("transaction_count", tx_count),
            tx_count,
            |b, &tx_count| {
                let mut mempool = MockMempool::new(tx_count);

                // Fill mempool with transactions
                for i in 0_i32..i32::try_from(tx_count).unwrap_or(1_000_i32) {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::unwrap_used
                    )]
                    let tx = MockTransaction::new()
                        .with_value(Price::from_ether(u64::try_from(i % 100_i32).unwrap_or(0)));
                    mempool.add_transaction(tx);
                }

                let min_value = Price::from_ether(50);

                b.iter(|| {
                    let filtered = mempool.filter_by_value(min_value);
                    black_box(filtered.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark transaction filtering by gas price
fn bench_gas_price_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("gas_price_filtering");

    for tx_count in &[1_000_usize, 10_000_usize, 100_000_usize] {
        group.bench_with_input(
            BenchmarkId::new("transaction_count", tx_count),
            tx_count,
            |b, &tx_count| {
                let mut mempool = MockMempool::new(tx_count);

                // Fill mempool with transactions
                for i in 0_i32..i32::try_from(tx_count).unwrap_or(1_000_i32) {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::unwrap_used
                    )]
                    let tx = MockTransaction::new()
                        .with_gas_price(Price::from_gwei(u64::try_from(i % 100_i32).unwrap_or(0)));
                    mempool.add_transaction(tx);
                }

                let min_gas_price = Price::from_gwei(50);

                b.iter(|| {
                    let filtered = mempool.filter_by_gas_price(min_gas_price);
                    black_box(filtered.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark MEV opportunity detection
fn bench_mev_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("mev_detection");

    for tx_count in &[1_000_usize, 10_000_usize, 100_000_usize] {
        group.bench_with_input(
            BenchmarkId::new("transaction_count", tx_count),
            tx_count,
            |b, &tx_count| {
                let mut mempool = MockMempool::new(tx_count);

                // Fill mempool with transactions (10% are potential MEV opportunities)
                for i in 0_i32..i32::try_from(tx_count).unwrap_or(1_000_i32) {
                    #[allow(clippy::unwrap_used)]
                    let tx = if i % 10_i32 == 0_i32 {
                        MockTransaction::new()
                            .with_value(Price::from_ether(20))
                            .with_data(vec![1, 2, 3, 4]) // Contract call data
                    } else {
                        MockTransaction::new().with_value(Price::from_ether(1))
                    };
                    mempool.add_transaction(tx);
                }

                b.iter(|| {
                    let opportunities = mempool.filter_mev_opportunities();
                    black_box(opportunities.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent mempool operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");

    for thread_count in &[1_i32, 2_i32, 4_i32, 8_i32] {
        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let mempool =
                        std::sync::Arc::new(std::sync::Mutex::new(MockMempool::new(10_000)));
                    let mut handles =
                        Vec::with_capacity(usize::try_from(thread_count).unwrap_or(4));

                    for _ in 0_i32..thread_count {
                        let mempool_clone = std::sync::Arc::clone(&mempool);
                        let handle = std::thread::spawn(move || {
                            for i in 0_i32..1_000_i32 {
                                #[allow(
                                    clippy::cast_possible_truncation,
                                    clippy::cast_sign_loss,
                                    clippy::unwrap_used
                                )]
                                let tx = MockTransaction::new().with_value(Price::from_ether(
                                    u64::try_from(i % 100_i32).unwrap_or(0),
                                ));
                                #[allow(clippy::unwrap_used)]
                                let mut pool = mempool_clone.lock().unwrap();
                                black_box(pool.add_transaction(tx));
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        #[allow(clippy::unwrap_used)]
                        {
                            handle.join().unwrap();
                        }
                    }

                    #[allow(clippy::unwrap_used)]
                    let pool = mempool.lock().unwrap();
                    black_box(pool.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for data_size in &[0_usize, 64_usize, 256_usize, 1_024_usize, 4_096_usize] {
        group.bench_with_input(
            BenchmarkId::new("data_size_bytes", data_size),
            data_size,
            |b, &data_size| {
                b.iter(|| {
                    let mut mempool = MockMempool::new(1_000);

                    for i in 0_i32..1_000_i32 {
                        let data = vec![0u8; data_size];
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            clippy::unwrap_used
                        )]
                        let tx = MockTransaction::new()
                            .with_value(Price::from_ether(u64::try_from(i % 100_i32).unwrap_or(0)))
                            .with_data(data);
                        black_box(mempool.add_transaction(tx));
                    }

                    black_box(mempool.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency under high load
fn bench_latency_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_under_load");
    group.measurement_time(Duration::from_secs(10));

    for load_factor in &[1.0_f64, 2.0_f64, 5.0_f64, 10.0_f64] {
        group.bench_with_input(
            BenchmarkId::new("load_factor", format!("{load_factor:.1}")),
            load_factor,
            |b, &load_factor| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let tx_count = (1_000.0_f64 * load_factor) as usize;
                let mut mempool = MockMempool::new(tx_count * 2);

                // Pre-fill mempool
                for i in 0_i32..i32::try_from(tx_count).unwrap_or(1_000_i32) {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::unwrap_used
                    )]
                    let tx = MockTransaction::new()
                        .with_value(Price::from_ether(u64::try_from(i % 100_i32).unwrap_or(0)));
                    mempool.add_transaction(tx);
                }

                b.iter(|| {
                    let start = std::time::Instant::now();

                    // Perform operations
                    let tx = MockTransaction::new().with_value(Price::from_ether(50));
                    black_box(mempool.add_transaction(tx));

                    let filtered = mempool.filter_by_value(Price::from_ether(25));
                    black_box(filtered.len());

                    let opportunities = mempool.filter_mev_opportunities();
                    black_box(opportunities.len());

                    let elapsed = start.elapsed();

                    // Log latency for analysis (don't panic in benchmarks)
                    if elapsed >= Duration::from_millis(1) {
                        eprintln!("Warning: Latency requirement violated: {elapsed:?}");
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_transaction_addition,
    bench_value_filtering,
    bench_gas_price_filtering,
    bench_mev_detection,
    bench_concurrent_operations,
    bench_memory_usage,
    bench_latency_under_load
);

criterion_main!(benches);
