//! State Management Integration Tests
//!
//! Production-ready integration tests for `TallyIO` state management.
//! Validates global and local state synchronization under various scenarios.

use std::sync::Arc;
use std::time::Duration;

use tallyio_core::state::{
    GlobalState, LocalState, MarketState, MevOpportunity, MevType, PositionState, StateError,
};
use tallyio_core::types::{Address, BlockNumber, Price, TxHash};

#[test]
fn test_global_state_creation() {
    let global_state = GlobalState::new();
    assert_eq!(global_state.market_count(), 0);
    assert_eq!(global_state.position_count(), 0);
    assert_eq!(global_state.mev_opportunity_count(), 0);
}

#[test]
fn test_market_state_operations() {
    let global_state = GlobalState::new();

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(1);

    let market = MarketState::new(token_a, token_b, price);

    // Update market state
    assert!(global_state.update_market(market).is_ok());
    assert_eq!(global_state.market_count(), 1);

    // Retrieve market state
    let retrieved = global_state.get_market(token_a, token_b);
    assert!(retrieved.is_ok());
    #[allow(clippy::unwrap_used)]
    let retrieved_market = retrieved.unwrap();
    assert_eq!(retrieved_market.token_a, token_a);
    assert_eq!(retrieved_market.token_b, token_b);
    assert_eq!(retrieved_market.price.wei(), price.wei());
}

#[test]
fn test_market_state_not_found() {
    let global_state = GlobalState::new();

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);

    // Try to get non-existent market
    let result = global_state.get_market(token_a, token_b);
    assert!(matches!(result, Err(StateError::NotFound { .. })));
}

#[test]
fn test_position_state_operations() {
    let global_state = GlobalState::new();

    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);
    let collateral = Price::from_ether(100);
    let debt = Price::from_ether(50);
    let liquidation_threshold = 1.5_f64;

    let position = PositionState::new(owner, protocol, collateral, debt, liquidation_threshold);

    // Update position state
    assert!(global_state.update_position(position).is_ok());
    assert_eq!(global_state.position_count(), 1);

    // Retrieve position state
    let retrieved = global_state.get_position(owner, protocol);
    assert!(retrieved.is_ok());
    #[allow(clippy::unwrap_used)]
    let retrieved_position = retrieved.unwrap();
    assert_eq!(retrieved_position.owner, owner);
    assert_eq!(retrieved_position.protocol, protocol);
    assert!(!retrieved_position.is_liquidatable());
}

#[test]
fn test_liquidatable_position() {
    let global_state = GlobalState::new();

    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);
    let collateral = Price::from_ether(100);
    let debt = Price::from_ether(80); // High debt ratio
    let liquidation_threshold = 1.5_f64;

    let position = PositionState::new(owner, protocol, collateral, debt, liquidation_threshold);

    // Position should be liquidatable
    assert!(position.is_liquidatable());

    #[allow(clippy::unwrap_used)]
    {
        global_state.update_position(position).unwrap();
    }

    // Get liquidatable positions
    let liquidatable = global_state.get_liquidatable_positions();
    assert_eq!(liquidatable.len(), 1);
    assert_eq!(liquidatable[0].owner, owner);
}

#[test]
fn test_mev_opportunity_operations() {
    let global_state = GlobalState::new();

    let tx_hash = TxHash::new([1; 32]);
    let estimated_profit = Price::from_ether(5);
    let gas_required = 200_000;

    let opportunity =
        MevOpportunity::new(MevType::Arbitrage, tx_hash, estimated_profit, gas_required);

    // Add MEV opportunity
    assert!(global_state.add_mev_opportunity(opportunity).is_ok());
    assert_eq!(global_state.mev_opportunity_count(), 1);

    // Retrieve MEV opportunity
    let retrieved = global_state.get_mev_opportunity(tx_hash);
    assert!(retrieved.is_ok());
    #[allow(clippy::unwrap_used)]
    let retrieved_opportunity = retrieved.unwrap();
    assert_eq!(retrieved_opportunity.target_tx, tx_hash);
    assert!(retrieved_opportunity.is_valid());

    // Remove MEV opportunity
    assert!(global_state.remove_mev_opportunity(tx_hash).is_ok());
    assert_eq!(global_state.mev_opportunity_count(), 0);
}

#[test]
fn test_mev_opportunity_expiration() {
    let global_state = GlobalState::new();

    let tx_hash = TxHash::new([1; 32]);
    let estimated_profit = Price::from_ether(5);
    let gas_required = 200_000;

    let mut opportunity =
        MevOpportunity::new(MevType::Arbitrage, tx_hash, estimated_profit, gas_required);

    // Set expiration to past - use a simple approach
    let now = std::time::Instant::now();
    #[allow(clippy::unwrap_used)]
    let past = now.checked_sub(Duration::from_secs(1)).unwrap();
    opportunity.expires_at = past;

    // Test time comparison directly
    let current_time = std::time::Instant::now();
    assert!(
        current_time > past,
        "Current time should be after past time"
    );

    // Test add_mev_opportunity
    #[allow(clippy::unwrap_used)]
    {
        global_state.add_mev_opportunity(opportunity).unwrap();
    }

    // Test get_mev_opportunity - this should return NotFound for expired opportunity
    let result = global_state.get_mev_opportunity(tx_hash);
    assert!(matches!(result, Err(StateError::NotFound { .. })));

    // Test passes if we get here without hanging
}

#[test]
fn test_local_state_caching() {
    let global_state = Arc::new(GlobalState::new());
    let local_state = LocalState::new(Arc::clone(&global_state));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(1);

    let market = MarketState::new(token_a, token_b, price);

    // Update through local state (write-through)
    assert!(local_state.update_market(market).is_ok());

    // First read should be cache miss
    let retrieved1 = local_state.get_market(token_a, token_b);
    assert!(retrieved1.is_ok());

    // Second read should be cache hit
    let retrieved2 = local_state.get_market(token_a, token_b);
    assert!(retrieved2.is_ok());

    let (hit_ratio, _) = local_state.cache_stats();
    assert!(hit_ratio > 0.0_f64);
}

#[test]
fn test_local_state_cache_invalidation() {
    let global_state = Arc::new(GlobalState::new());
    let local_state = LocalState::new(Arc::clone(&global_state));

    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(1);

    let market = MarketState::new(token_a, token_b, price);

    // Update and read to populate cache
    #[allow(clippy::unwrap_used)]
    {
        local_state.update_market(market).unwrap();
        local_state.get_market(token_a, token_b).unwrap();
    }

    // Invalidate cache
    local_state.invalidate_cache();

    // Next read should be cache miss
    let result = local_state.get_market(token_a, token_b);
    assert!(result.is_ok());
}

#[test]
fn test_concurrent_state_access() {
    let global_state = Arc::new(GlobalState::new());
    let mut handles = Vec::new();

    // Spawn multiple threads accessing state concurrently
    for i in 0_i32..4_i32 {
        let global_state_clone = Arc::clone(&global_state);

        let handle = std::thread::spawn(move || {
            for j in 0_i32..100_i32 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let token_a = Address::new([i as u8; 20]);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let token_b = Address::new([j as u8; 20]);
                #[allow(clippy::unwrap_used)]
                let price = Price::from_ether(u64::try_from(j).unwrap_or(0));

                let market = MarketState::new(token_a, token_b, price);

                // Update market state
                let result = global_state_clone.update_market(market);
                assert!(result.is_ok(), "Market update failed: {result:?}");

                // Read market state
                let retrieved = global_state_clone.get_market(token_a, token_b);
                assert!(retrieved.is_ok(), "Market retrieval failed: {retrieved:?}");
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        #[allow(clippy::unwrap_used)]
        {
            handle.join().unwrap();
        }
    }

    // Verify final state
    assert!(global_state.market_count() > 0);
}

#[test]
fn test_state_version_tracking() {
    let global_state = GlobalState::new();

    let initial_version = global_state.version();

    // Update block number should increment version
    global_state.update_block(BlockNumber::new(100));
    let new_version = global_state.version();
    assert!(new_version.is_newer_than(initial_version));

    // Add market should increment version
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let market = MarketState::new(token_a, token_b, Price::from_ether(1));

    #[allow(clippy::unwrap_used)]
    {
        global_state.update_market(market).unwrap();
    }
    let latest_version = global_state.version();
    assert!(latest_version.is_newer_than(new_version));
}

#[test]
fn test_state_statistics() {
    let global_state = GlobalState::new();
    let stats = global_state.stats();

    // Initial stats should be zero
    assert_eq!(stats.reads.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(stats.writes.load(std::sync::atomic::Ordering::Relaxed), 0);

    // Perform some operations
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let market = MarketState::new(token_a, token_b, Price::from_ether(1));

    #[allow(clippy::unwrap_used)]
    {
        global_state.update_market(market).unwrap(); // Write
        global_state.get_market(token_a, token_b).unwrap(); // Read
    }

    // Stats should be updated
    assert!(stats.reads.load(std::sync::atomic::Ordering::Relaxed) > 0);
    assert!(stats.writes.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

#[test]
fn test_market_state_liquidity_check() {
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(1);

    let mut market = MarketState::new(token_a, token_b, price);
    market.liquidity = Price::from_ether(1000);

    // Check liquidity
    assert!(market.has_sufficient_liquidity(Price::from_ether(500)));
    assert!(!market.has_sufficient_liquidity(Price::from_ether(2000)));
}

#[test]
fn test_position_state_price_update() {
    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);
    let collateral = Price::from_ether(100);
    let debt = Price::from_ether(50);
    let liquidation_threshold = 1.5_f64;

    let mut position = PositionState::new(owner, protocol, collateral, debt, liquidation_threshold);

    // Initial state should be healthy
    assert!(!position.is_liquidatable());

    // Update with lower collateral price (simulating price drop)
    let collateral_price = Price::from_ether(1); // 1 ETH per collateral token
    let debt_price = Price::from_ether(2); // 2 ETH per debt token

    position.update_prices(collateral_price, debt_price);

    // Position should now be liquidatable
    assert!(position.is_liquidatable());
}

#[test]
fn test_mev_opportunity_profitability() {
    let tx_hash = TxHash::new([1; 32]);
    let estimated_profit = Price::from_ether(1); // 1 ETH profit
    let gas_required = 200_000;

    let opportunity =
        MevOpportunity::new(MevType::Arbitrage, tx_hash, estimated_profit, gas_required);

    // Check profitability with different gas prices
    // Low gas price: 10 gwei * 200k gas = 2M gwei = 0.002 ETH < 1 ETH profit
    let low_gas_price = Price::from_gwei(10);
    assert!(opportunity.is_profitable(low_gas_price));

    // Medium gas price: 100 gwei * 200k gas = 20M gwei = 0.02 ETH < 1 ETH profit
    let medium_gas_price = Price::from_gwei(100);
    assert!(opportunity.is_profitable(medium_gas_price));

    // Very high gas price: 10,000 gwei * 200k gas = 2B gwei = 2 ETH > 1 ETH profit
    let very_high_gas_price = Price::from_gwei(10_000);
    assert!(!opportunity.is_profitable(very_high_gas_price));
}
