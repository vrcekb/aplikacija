//! State consistency tests for `TallyIO` core functionality.
//! These tests validate state management and consistency properties.

use std::sync::Arc;
use tallyio_core::prelude::*;
use tallyio_core::state::{
    GlobalState, LocalState, MarketState, PositionState, StateError, StateResult,
};

/// Test state consistency across multiple operations
#[test]
fn test_state_consistency_multiple_operations() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let initial_price = Price::from_ether(100);

    // Initial state
    let market = MarketState::new(token_a, token_b, initial_price);
    state.update_market(market)?;

    // Verify initial state
    let retrieved = state.get_market(token_a, token_b)?;
    assert_eq!(retrieved.price, initial_price, "Initial price should match");

    // Update price multiple times
    for i in 1..=10 {
        let new_price = Price::from_ether(100 + i);
        let updated_market = MarketState::new(token_a, token_b, new_price);
        state.update_market(updated_market)?;

        let current = state.get_market(token_a, token_b)?;
        assert_eq!(
            current.price, new_price,
            "Price should be updated consistently"
        );
    }
    Ok(())
}

/// Test state isolation between different markets
#[test]
fn test_state_isolation() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    // Create multiple markets
    let markets = [
        (
            Address::new([1; 20]),
            Address::new([2; 20]),
            Price::from_ether(100),
        ),
        (
            Address::new([3; 20]),
            Address::new([4; 20]),
            Price::from_ether(200),
        ),
        (
            Address::new([5; 20]),
            Address::new([6; 20]),
            Price::from_ether(300),
        ),
    ];

    // Update all markets
    for (token_a, token_b, price) in &markets {
        let market = MarketState::new(*token_a, *token_b, *price);
        state.update_market(market)?;
    }

    // Verify each market maintains its state independently
    for (token_a, token_b, expected_price) in &markets {
        let retrieved = state.get_market(*token_a, *token_b)?;
        assert_eq!(
            retrieved.price, *expected_price,
            "Market price should be isolated"
        );
    }

    // Modify one market and verify others are unaffected
    let new_price = Price::from_ether(999);
    let updated_market = MarketState::new(markets[0].0, markets[0].1, new_price);
    state.update_market(updated_market)?;

    // Check first market is updated
    let updated = state.get_market(markets[0].0, markets[0].1)?;
    assert_eq!(updated.price, new_price, "First market should be updated");

    // Check other markets are unchanged
    for (token_a, token_b, expected_price) in &markets[1..] {
        let retrieved = state.get_market(*token_a, *token_b)?;
        assert_eq!(
            retrieved.price, *expected_price,
            "Other markets should be unchanged"
        );
    }
    Ok(())
}

/// Test position state consistency
#[test]
fn test_position_state_consistency() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = LocalState::new(Arc::clone(&global_state));

    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    // Create initial position
    let initial_position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(100),
        Price::from_ether(50),
        1.5_f64,
    );

    state.update_position(initial_position.clone())?;

    // Verify initial position
    let retrieved = state.get_position(owner, protocol)?;
    assert_eq!(retrieved.collateral, initial_position.collateral);
    assert_eq!(retrieved.debt, initial_position.debt);

    // Update position multiple times
    for i in 1..=10 {
        let new_collateral = Price::from_ether(100 + i);
        let new_debt = Price::from_ether(50 + i / 2);

        let updated_position =
            PositionState::new(owner, protocol, new_collateral, new_debt, 1.5_f64);
        state.update_position(updated_position.clone())?;

        let current = state.get_position(owner, protocol)?;
        assert_eq!(
            current.collateral, new_collateral,
            "Collateral should be updated"
        );
        assert_eq!(current.debt, new_debt, "Debt should be updated");
    }
    Ok(())
}

/// Test concurrent state modifications
#[test]
fn test_concurrent_state_modifications() -> StateResult<()> {
    let global_state = Arc::new(GlobalState::new());
    let state = Arc::new(LocalState::new(Arc::clone(&global_state)));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);

    // Spawn multiple threads modifying the same market
    let handles: Vec<_> = (0_i32..10_i32)
        .map(|i| {
            let state_clone = Arc::clone(&state);
            std::thread::spawn(move || -> StateResult<()> {
                for j in 0_i32..100_i32 {
                    #[allow(clippy::cast_sign_loss)]
                    let price = Price::from_ether((i * 100 + j) as u64);
                    let market = MarketState::new(token_a, token_b, price);
                    state_clone.update_market(market)?;
                }
                Ok(())
            })
        })
        .collect();

    for handle in handles {
        match handle.join() {
            Ok(thread_result) => thread_result?,
            Err(_) => return Err(StateError::SyncTimeout { duration_ms: 1000 }),
        }
    }

    // Verify state is still consistent
    let final_market = state.get_market(token_a, token_b)?;
    assert_eq!(final_market.token_a, token_a);
    assert_eq!(final_market.token_b, token_b);
    assert!(final_market.price.wei() > 0, "Price should be positive");
    Ok(())
}
