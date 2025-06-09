//! MEV Opportunity Types
//!
//! Production-ready types for MEV opportunity detection and execution.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{Address, Price, TxHash};

/// MEV opportunity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpportunityType {
    /// Arbitrage opportunity between DEXes
    Arbitrage,
    /// Sandwich attack opportunity
    Sandwich,
    /// Liquidation opportunity
    Liquidation,
    /// Front-running opportunity
    Frontrun,
    /// Back-running opportunity
    Backrun,
}

/// MEV opportunity
#[derive(Debug, Clone)]
pub struct Opportunity {
    /// Opportunity type
    pub opportunity_type: OpportunityType,

    /// Target transaction
    pub target_tx: TxHash,

    /// Estimated profit in wei
    pub estimated_profit: Price,

    /// Required gas for execution
    pub gas_required: u64,

    /// Opportunity expiration time
    pub expires_at: Instant,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Competition level (0.0 = no competition, 1.0 = high competition)
    pub competition: f64,

    /// Related addresses
    pub addresses: Vec<Address>,
}

impl Opportunity {
    /// Create new opportunity
    #[must_use]
    pub fn new(
        opportunity_type: OpportunityType,
        target_tx: TxHash,
        estimated_profit: Price,
        gas_required: u64,
    ) -> Self {
        Self {
            opportunity_type,
            target_tx,
            estimated_profit,
            gas_required,
            expires_at: Instant::now() + Duration::from_secs(12), // One block
            confidence: 0.5_f64,
            competition: 0.5_f64,
            addresses: Vec::with_capacity(2), // Typically 2 addresses for most MEV ops
        }
    }

    /// Check if opportunity is still valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }

    /// Calculate expected value considering competition
    #[must_use]
    pub fn expected_value(&self) -> Price {
        let success_probability = self.confidence * (1.0_f64 - self.competition);

        // Use safe arithmetic to prevent precision loss and overflow
        let profit_wei = self.estimated_profit.wei();

        // For very large values, use integer arithmetic to avoid precision loss
        if profit_wei > (1_u128 << 53) {
            // Use integer arithmetic for large values
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let probability_scaled = (success_probability * 1_000_000.0_f64) as u64;
            let expected_wei = profit_wei
                .saturating_mul(u128::from(probability_scaled))
                .saturating_div(1_000_000);
            Price::new(expected_wei)
        } else {
            // Safe to use floating point for smaller values
            #[allow(clippy::cast_precision_loss)]
            let profit_f64 = profit_wei as f64;
            let expected_f64 = profit_f64 * success_probability;

            // Clamp to valid u128 range
            let expected_wei = if expected_f64 < 0.0_f64 {
                0_u128
            } else if expected_f64 > 1e30_f64 {
                // Conservative upper bound to avoid overflow
                u128::MAX
            } else {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    expected_f64 as u128
                }
            };

            Price::new(expected_wei)
        }
    }

    /// Check if opportunity is profitable after gas costs
    #[must_use]
    pub fn is_profitable(&self, gas_price: Price) -> bool {
        let gas_cost = Price::new(u128::from(self.gas_required) * gas_price.wei());
        self.estimated_profit.wei() > gas_cost.wei()
    }
}
