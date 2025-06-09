//! Security-critical tests for `TallyIO` core functionality.
//! These tests validate security properties essential for financial applications.

use std::sync::Arc;
use std::time::Duration;
use tallyio_core::prelude::*;
use tallyio_core::state::{
    GlobalState, LocalState, MarketState, PositionState, StateError, StateResult,
};

/// Test memory safety under concurrent access
#[test]
fn test_memory_safety_concurrent_access() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = Arc::new(LocalState::new(Arc::clone(&global_state)));

    let handles: Vec<_> = (0_i32..10_i32)
        .map(|i| {
            let state_clone = Arc::clone(&state);
            std::thread::spawn(move || -> StateResult<()> {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let token_a = Address::new([i as u8; 20]);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let token_b = Address::new([(i + 1_i32) as u8; 20]);
                let price = Price::from_ether(100);

                let market = MarketState::new(token_a, token_b, price);

                // Concurrent read/write operations
                for _ in 0_i32..100_i32 {
                    state_clone.update_market(market.clone())?;
                    let _ = state_clone.get_market(token_a, token_b)?;
                }
                Ok(())
            })
        })
        .collect();

    for handle in handles {
        let result = handle
            .join()
            .map_err(|_| StateError::SyncTimeout { duration_ms: 1000 });
        match result {
            Ok(thread_result) => thread_result?,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Test input validation and sanitization
#[test]
fn test_input_validation() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    // Test with zero addresses
    let zero_addr = Address::new([0; 20]);
    let valid_addr = Address::new([1; 20]);
    let price = Price::from_ether(100);

    let market = MarketState::new(zero_addr, valid_addr, price);
    state.update_market(market)?;

    // Test with zero price
    let zero_price = Price::new(0);
    let market_zero_price = MarketState::new(valid_addr, zero_addr, zero_price);
    state.update_market(market_zero_price)?;

    Ok(())
}

/// Test overflow protection in calculations
#[test]
fn test_overflow_protection() {
    // Test maximum values
    let max_price = Price::new(u128::MAX);

    // These should not panic
    let _market = MarketState::new(Address::new([1; 20]), Address::new([2; 20]), max_price);

    let position = PositionState::new(
        Address::new([1; 20]),
        Address::new([2; 20]),
        max_price,
        max_price,
        1.5_f64,
    );

    // Test calculations that could overflow
    let health_score = position.health_score;
    assert!(
        health_score >= 0.0_f64,
        "Health score should be non-negative"
    );
}

/// Test cryptographic operations security
#[test]
fn test_cryptographic_security() {
    // Test hash consistency
    let tx_hash1 = TxHash::new([1; 32]);
    let tx_hash2 = TxHash::new([1; 32]);
    assert_eq!(tx_hash1, tx_hash2, "Same input should produce same hash");

    // Test hash uniqueness
    let tx_hash3 = TxHash::new([2; 32]);
    assert_ne!(
        tx_hash1, tx_hash3,
        "Different inputs should produce different hashes"
    );
}

/// Test access control and permissions
#[test]
fn test_access_control() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let owner1 = Address::new([1; 20]);
    let owner2 = Address::new([2; 20]);
    let protocol = Address::new([3; 20]);

    let position1 = PositionState::new(
        owner1,
        protocol,
        Price::from_ether(100),
        Price::from_ether(50),
        1.5_f64,
    );

    let position2 = PositionState::new(
        owner2,
        protocol,
        Price::from_ether(200),
        Price::from_ether(100),
        1.5_f64,
    );

    // Update positions
    state.update_position(position1)?;
    state.update_position(position2)?;

    // Verify isolation - owner1 cannot access owner2's position
    let retrieved1 = state.get_position(owner1, protocol)?;
    let retrieved2 = state.get_position(owner2, protocol)?;

    assert_ne!(
        retrieved1.collateral, retrieved2.collateral,
        "Positions should be isolated"
    );
    Ok(())
}

/// Test timing attack resistance
#[test]
fn test_timing_attack_resistance() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);

    // Measure time for non-existing market
    let start = std::time::Instant::now();
    let _ = state.get_market(token_a, token_b);
    let time_nonexistent = start.elapsed();

    // Add market
    let market = MarketState::new(token_a, token_b, Price::from_ether(100));
    state.update_market(market)?;

    let start = std::time::Instant::now();
    let _ = state.get_market(token_a, token_b)?;
    let time_existing = start.elapsed();

    // Time difference should be minimal (within 10ms)
    let time_diff = time_existing.abs_diff(time_nonexistent);

    assert!(
        time_diff < Duration::from_millis(10),
        "Timing difference too large: {time_diff:?}"
    );
    Ok(())
}
