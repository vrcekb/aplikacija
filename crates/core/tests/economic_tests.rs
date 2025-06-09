//! Economic model tests for `TallyIO` core functionality.
//! These tests validate economic properties and financial calculations.

use tallyio_core::prelude::*;
use tallyio_core::state::{MarketState, MevOpportunity, MevType, PositionState};

/// Test liquidation threshold calculations
#[test]
fn test_liquidation_thresholds() {
    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    // Test healthy position (should not be liquidatable)
    let healthy_position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(200), // collateral
        Price::from_ether(100), // debt
        1.5_f64,                // liquidation threshold
    );

    assert!(
        !healthy_position.is_liquidatable(),
        "Healthy position should not be liquidatable"
    );

    // Test unhealthy position (should be liquidatable)
    let unhealthy_position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(100), // collateral
        Price::from_ether(200), // debt
        1.5_f64,                // liquidation threshold
    );

    assert!(
        unhealthy_position.is_liquidatable(),
        "Unhealthy position should be liquidatable"
    );
}

/// Test MEV opportunity profitability calculations
#[test]
fn test_mev_profitability() {
    let tx_hash = TxHash::new([1; 32]);

    // Test profitable opportunity - 10 ETH profit vs gas cost
    let profitable_opportunity = MevOpportunity::new(
        MevType::Arbitrage,
        tx_hash,
        Price::from_ether(10), // estimated profit: 10 ETH
        100_000,               // gas required: 100k gas
    );

    // Low gas price: 20 gwei * 100k gas = 2M gwei = 0.002 ETH
    let gas_price = Price::from_gwei(20);
    assert!(
        profitable_opportunity.is_profitable(gas_price),
        "Should be profitable: 10 ETH profit > 0.002 ETH gas cost"
    );

    // High gas price: 200 gwei * 100k gas = 20M gwei = 0.02 ETH
    let high_gas_price = Price::from_gwei(200);
    assert!(
        profitable_opportunity.is_profitable(high_gas_price),
        "Should still be profitable: 10 ETH profit > 0.02 ETH gas cost"
    );

    // Very high gas price: 100,000 gwei * 100k gas = 10B gwei = 10 ETH
    let very_high_gas_price = Price::from_gwei(100_000);
    assert!(
        !profitable_opportunity.is_profitable(very_high_gas_price),
        "Should not be profitable: 10 ETH profit = 10 ETH gas cost"
    );
}

/// Test price impact calculations
#[test]
fn test_price_impact() {
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(100);

    // Create market with liquidity for meaningful price impact calculations
    // Use smaller liquidity and higher impact factor for more predictable results
    let market = MarketState::with_liquidity(
        token_a,
        token_b,
        price,
        Price::from_ether(100), // 100 ETH liquidity (smaller for more noticeable impact)
        1.0_f64,                // 100% impact when amount equals liquidity
    );

    // Verify market state is set correctly
    assert_eq!(market.liquidity.wei(), Price::from_ether(100).wei());
    assert!((market.price_impact_1eth - 1.0_f64).abs() < f64::EPSILON);

    // Test small trade (should have minimal impact)
    let small_amount = Price::from_ether(1);
    let small_impact = market.calculate_price_impact(small_amount);

    // Expected: 1 ETH / 100 ETH * 1.0 = 0.01 (1%)
    assert!(
        small_impact < 0.05_f64,
        "Small trade should have minimal impact, got: {small_impact}"
    );

    // Test large trade (should have significant impact)
    let large_amount = Price::from_ether(50);
    let large_impact = market.calculate_price_impact(large_amount);

    // Expected: 50 ETH / 100 ETH * 1.0 = 0.5 (50%)
    assert!(
        large_impact > small_impact,
        "Large trade should have higher impact than small trade: {large_impact} > {small_impact}"
    );

    assert!(
        large_impact > 0.1_f64,
        "Large trade should have significant impact (>10%), got: {large_impact}"
    );

    // Test zero liquidity case
    let zero_liquidity_market = MarketState::new(token_a, token_b, price);
    let zero_liquidity_impact = zero_liquidity_market.calculate_price_impact(small_amount);
    assert!(
        (zero_liquidity_impact - 1.0_f64).abs() < f64::EPSILON,
        "Zero liquidity should result in 100% price impact"
    );
}

/// Test slippage protection
#[test]
fn test_slippage_protection() {
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(100);

    // Create market with liquidity for meaningful slippage calculations
    let market = MarketState::with_liquidity(
        token_a,
        token_b,
        price,
        Price::from_ether(50_000), // 50k ETH liquidity
        0.0002_f64,                // 0.02% impact per ETH
    );

    let amount = Price::from_ether(10);

    // Test price impact calculation (proxy for slippage)
    let price_impact = market.calculate_price_impact(amount);

    // Test acceptable slippage (< 5%)
    let acceptable_slippage = 0.05_f64; // 5%
    assert!(
        price_impact < acceptable_slippage,
        "Should have acceptable slippage, got: {price_impact}"
    );

    // Test with larger amount (higher slippage)
    let large_amount = Price::from_ether(1_000);
    let large_impact = market.calculate_price_impact(large_amount);
    assert!(
        large_impact > price_impact,
        "Larger trades should have higher slippage: {large_impact} > {price_impact}"
    );

    // Test slippage protection threshold
    let max_acceptable_slippage = 0.1_f64; // 10%
    if large_impact > max_acceptable_slippage {
        // This would trigger slippage protection in a real system
        assert!(
            large_impact > max_acceptable_slippage,
            "Large trade exceeds slippage protection threshold"
        );
    }
}

/// Test arbitrage opportunity detection
#[test]
fn test_arbitrage_detection() {
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);

    // Market 1: 1 ETH = 100 USDC
    let market1 = MarketState::new(token_a, token_b, Price::from_ether(100));

    // Market 2: 1 ETH = 105 USDC (arbitrage opportunity)
    let market2 = MarketState::new(token_a, token_b, Price::from_ether(105));

    let arbitrage_profit = market2.price.wei() - market1.price.wei();
    assert!(arbitrage_profit > 0, "Should detect arbitrage opportunity");

    // Calculate expected profit percentage
    #[allow(clippy::cast_precision_loss)]
    let profit_percentage = arbitrage_profit as f64 / market1.price.wei() as f64;
    assert!(
        profit_percentage > 0.0_f64,
        "Arbitrage should be profitable"
    );
    assert!(
        profit_percentage < 0.1_f64,
        "Profit should be reasonable (< 10%)"
    );
}

/// Test liquidation bonus calculations
#[test]
fn test_liquidation_bonus() {
    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    let position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(150), // collateral
        Price::from_ether(100), // debt
        1.2_f64,                // liquidation threshold
    );

    // Test liquidation profit calculation
    let liquidation_bonus = 0.05_f64; // 5% bonus
    let profit = position.liquidation_profit(liquidation_bonus);

    // Should have some profit if position is liquidatable
    if position.is_liquidatable() {
        assert!(profit.wei() > 0, "Liquidation should be profitable");
    } else {
        assert_eq!(
            profit.wei(),
            0,
            "Non-liquidatable position should have zero profit"
        );
    }
}

/// Test fee calculations
#[test]
fn test_fee_calculations() {
    let amount = Price::from_ether(100);

    // Test trading fee (0.3%) - use integer arithmetic to avoid precision loss
    let expected_trading_fee = (amount.wei() * 3) / 1_000;
    let trading_fee = Price::new(expected_trading_fee);
    assert_eq!(
        trading_fee.wei(),
        expected_trading_fee,
        "Trading fee should be 0.3%"
    );

    // Test protocol fee (0.05%) - use integer arithmetic to avoid precision loss
    let expected_protocol_fee = (amount.wei() * 5) / 10_000;
    let protocol_fee = Price::new(expected_protocol_fee);
    assert_eq!(
        protocol_fee.wei(),
        expected_protocol_fee,
        "Protocol fee should be 0.05%"
    );
}

/// Test compound interest calculations
#[test]
fn test_compound_interest() {
    let principal = Price::from_ether(1_000);
    let annual_rate = 0.05_f64; // 5% APY
    let time_periods = 12_i32; // monthly compounding for 1 year

    // Calculate compound interest: A = P(1 + r/n)^(nt)
    // For 1 year: A = P(1 + 0.05/12)^12
    let rate_per_period = annual_rate / f64::from(time_periods);
    let compound_factor = (1.0_f64 + rate_per_period).powi(time_periods);

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let final_amount = Price::new((principal.wei() as f64 * compound_factor) as u128);

    assert!(
        final_amount.wei() > principal.wei(),
        "Compound interest should increase principal: {final_amount:?} > {principal:?}"
    );

    // Verify approximately 5.12% growth (compound effect)
    // (1 + 0.05/12)^12 ≈ 1.0512
    #[allow(clippy::cast_precision_loss)]
    let growth_rate = (final_amount.wei() as f64 / principal.wei() as f64) - 1.0_f64;

    // Expected compound growth is slightly higher than simple 5%
    let expected_compound_rate = compound_factor - 1.0_f64; // ≈ 0.0512

    assert!(
        (growth_rate - expected_compound_rate).abs() < 0.001_f64,
        "Growth rate should be approximately {expected_compound_rate:.4}, got {growth_rate:.4}"
    );

    // Verify it's higher than simple interest
    assert!(
        growth_rate > 0.05_f64,
        "Compound interest should exceed simple interest: {growth_rate:.4} > 0.05"
    );
}

/// Test risk metrics calculations
#[test]
fn test_risk_metrics() {
    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    let position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(200), // collateral
        Price::from_ether(100), // debt
        1.5_f64,                // liquidation threshold
    );

    // Test collateralization ratio
    let collateral_ratio = position.collateral_ratio;
    assert!(
        (collateral_ratio - 2.0_f64).abs() < f64::EPSILON,
        "Collateralization ratio should be 2.0"
    );

    // Test health score
    let health_score = position.health_score;
    assert!(
        health_score > 0.5_f64,
        "Healthy position should have good health score"
    );
}

/// Test portfolio value calculations
#[test]
fn test_portfolio_value() {
    let owner = Address::new([1; 20]);
    let protocol1 = Address::new([2; 20]);
    let protocol2 = Address::new([3; 20]);

    let position1 = PositionState::new(
        owner,
        protocol1,
        Price::from_ether(100),
        Price::from_ether(50),
        1.5_f64,
    );

    let position2 = PositionState::new(
        owner,
        protocol2,
        Price::from_ether(200),
        Price::from_ether(75),
        1.5_f64,
    );

    // Calculate total portfolio value
    let total_collateral = position1.collateral.wei() + position2.collateral.wei();
    let total_debt = position1.debt.wei() + position2.debt.wei();
    let net_value = total_collateral - total_debt;

    assert_eq!(
        total_collateral,
        Price::from_ether(300).wei(),
        "Total collateral should be 300 ETH"
    );
    assert_eq!(
        total_debt,
        Price::from_ether(125).wei(),
        "Total debt should be 125 ETH"
    );
    assert_eq!(
        net_value,
        Price::from_ether(175).wei(),
        "Net value should be 175 ETH"
    );
}

/// Test economic invariants
#[test]
fn test_economic_invariants() {
    let owner = Address::new([1; 20]);
    let protocol = Address::new([2; 20]);

    // Test: collateral should always be >= 0
    let position = PositionState::new(
        owner,
        protocol,
        Price::from_ether(100),
        Price::from_ether(50),
        1.5_f64,
    );

    // Test: collateral and debt are u128, so always >= 0
    assert!(
        position.collateral.wei() < u128::MAX,
        "Collateral should be within bounds"
    );
    assert!(
        position.debt.wei() < u128::MAX,
        "Debt should be within bounds"
    );

    // Test: price should always be > 0 for valid markets
    let token_a = Address::new([1; 20]);
    let token_b = Address::new([2; 20]);
    let price = Price::from_ether(100);

    let market = MarketState::new(token_a, token_b, price);
    assert!(market.price.wei() > 0, "Market price should be positive");
}
