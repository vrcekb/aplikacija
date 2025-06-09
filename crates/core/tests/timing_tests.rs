//! Timing and performance tests for `TallyIO` core functionality.
//! These tests validate latency requirements and performance characteristics.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tallyio_core::prelude::*;
use tallyio_core::state::{
    GlobalState, LocalState, MarketState, MevOpportunity, MevType, PositionState, StateError,
    StateResult,
};

/// Test sub-millisecond latency for market operations
#[test]
fn test_market_operation_latency() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(100);

    // Test market update latency
    let start = Instant::now();
    let market = MarketState::new(token_a, token_b, price);
    state.update_market(market)?;
    let update_latency = start.elapsed();

    assert!(
        update_latency < Duration::from_millis(1),
        "Market update should be < 1ms, got: {update_latency:?}"
    );

    // Test market read latency
    let start = Instant::now();
    let _retrieved = state.get_market(token_a, token_b)?;
    let read_latency = start.elapsed();

    assert!(
        read_latency < Duration::from_millis(1),
        "Market read should be < 1ms, got: {read_latency:?}"
    );
    Ok(())
}

/// Test position operation latency
#[test]
fn test_position_operation_latency() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    // Test position update latency
    let start = Instant::now();
    let position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(100),
        Price::from_ether(50),
        1.5_f64,
    );
    state.update_position(position)?;
    let update_latency = start.elapsed();

    assert!(
        update_latency < Duration::from_millis(1),
        "Position update should be < 1ms, got: {update_latency:?}"
    );

    // Test position read latency
    let start = Instant::now();
    let _retrieved = state.get_position(owner, protocol)?;
    let read_latency = start.elapsed();

    assert!(
        read_latency < Duration::from_millis(1),
        "Position read should be < 1ms, got: {read_latency:?}"
    );
    Ok(())
}

/// Test MEV opportunity detection latency
#[test]
fn test_mev_detection_latency() {
    let tx_hash = TxHash::new([1; 32]);

    // Test MEV opportunity creation latency
    let start = Instant::now();
    let opportunity =
        MevOpportunity::new(MevType::Arbitrage, tx_hash, Price::from_ether(10), 100_000);
    let creation_latency = start.elapsed();

    assert!(
        creation_latency < Duration::from_millis(1),
        "MEV opportunity creation should be < 1ms, got: {creation_latency:?}"
    );

    // Test profitability check latency
    let start = Instant::now();
    let gas_price = Price::from_gwei(20);
    let _is_profitable = opportunity.is_profitable(gas_price);
    let check_latency = start.elapsed();

    assert!(
        check_latency < Duration::from_millis(1),
        "MEV profitability check should be < 1ms, got: {check_latency:?}"
    );
}

/// Test concurrent operation latency
#[test]
fn test_concurrent_operation_latency() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = Arc::new(LocalState::new(Arc::clone(&global_state)));

    let num_threads = 10;
    let operations_per_thread = 100;

    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let state_clone: Arc<LocalState> = Arc::clone(&state);
            std::thread::spawn(move || -> StateResult<()> {
                for j in 0..operations_per_thread {
                    #[allow(clippy::cast_possible_truncation)]
                    let token_a = Address::new([i as u8; 20]);
                    #[allow(clippy::cast_possible_truncation)]
                    let token_b = Address::new([j as u8; 20]);
                    let price = Price::from_ether(u64::from(i * 100 + j) + 1);

                    let market = MarketState::new(token_a, token_b, price);
                    state_clone.update_market(market)?;
                }
                Ok(())
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().map_err(|_| StateError::Corruption {
            reason: "Thread panicked".to_string(),
        })?;
        result?;
    }

    let total_time = start.elapsed();
    let total_operations = num_threads * operations_per_thread;
    let avg_latency = total_time / total_operations;

    assert!(
        avg_latency < Duration::from_millis(1),
        "Average concurrent operation latency should be < 1ms, got: {avg_latency:?}"
    );
    Ok(())
}

/// Test memory allocation performance
#[test]
fn test_memory_allocation_performance() {
    let start = Instant::now();

    // Create many objects to test allocation performance
    let mut markets = Vec::with_capacity(1_000);
    for i in 0_i32..1_000_i32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let token_a = Address::new([i as u8; 20]);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let token_b = Address::new([(i + 1_i32) as u8; 20]);
        #[allow(clippy::cast_sign_loss)]
        let price = Price::from_ether(i as u64 + 1);

        let market = MarketState::new(token_a, token_b, price);
        markets.push(market);
    }

    let allocation_time = start.elapsed();
    let avg_allocation_time = allocation_time / 1_000;

    // Verify we actually created the markets
    assert_eq!(markets.len(), 1_000, "Should have created 1000 markets");

    assert!(
        avg_allocation_time < Duration::from_micros(10),
        "Average allocation time should be < 10μs, got: {avg_allocation_time:?}"
    );
}
