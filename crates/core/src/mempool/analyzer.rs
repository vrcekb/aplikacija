//! Transaction Analyzer - Ultra-Performance MEV Detection
//!
//! Production-ready transaction analysis for MEV opportunity detection.

use std::sync::Arc;
use std::time::Instant;

use crate::types::{AnalysisResult, Price, Transaction};

use super::{MempoolConfig, MempoolResult};

/// Transaction analyzer for MEV detection
pub struct TransactionAnalyzer {
    /// Configuration
    config: Arc<MempoolConfig>,
}

impl TransactionAnalyzer {
    /// Create new transaction analyzer
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub const fn new(config: Arc<MempoolConfig>) -> MempoolResult<Self> {
        Ok(Self { config })
    }

    /// Analyze transaction for MEV opportunities
    ///
    /// # Errors
    ///
    /// Returns error if analysis fails
    pub fn analyze(&self, transaction: &Transaction) -> MempoolResult<AnalysisResult> {
        let start_time = Instant::now();

        if !self.config.enable_mev_detection {
            return Ok(AnalysisResult::no_opportunities(start_time.elapsed()));
        }

        // Quick checks for MEV potential
        let has_opportunities = self.detect_mev_potential(transaction);

        if has_opportunities {
            let estimated_profit = Self::estimate_profit(transaction);
            let confidence = Self::calculate_confidence(transaction);

            Ok(AnalysisResult::with_opportunities(
                1,
                estimated_profit,
                start_time.elapsed(),
                confidence,
            ))
        } else {
            Ok(AnalysisResult::no_opportunities(start_time.elapsed()))
        }
    }

    /// Detect MEV potential in transaction
    fn detect_mev_potential(&self, transaction: &Transaction) -> bool {
        // Check if transaction value meets minimum threshold
        if transaction.value.wei() < self.config.min_mev_value.wei() {
            return false;
        }

        // Check for contract interactions (potential DEX trades)
        if transaction.is_contract_call() {
            return true;
        }

        // Check for high gas price (potential front-running target)
        if let Some(gas_price) = transaction.gas_price {
            if gas_price.wei() > Price::from_gwei(50).wei() {
                return true;
            }
        }

        // Check for EIP-1559 transactions with high priority fee
        if let Some(priority_fee) = transaction.max_priority_fee_per_gas {
            if priority_fee.wei() > Price::from_gwei(10).wei() {
                return true;
            }
        }

        false
    }

    /// Estimate potential profit from MEV opportunity
    fn estimate_profit(transaction: &Transaction) -> Price {
        // Simple profit estimation based on transaction value
        let base_profit = transaction.value.wei() / 1000; // 0.1% of transaction value

        // Adjust based on gas price (higher gas = more urgent = more profit potential)
        let gas_multiplier = transaction.gas_price.map_or(1.0_f64, |gas_price| {
            let gwei = gas_price.wei() / 1_000_000_000;
            #[allow(clippy::cast_precision_loss)]
            let gwei_f64 = gwei as f64;
            1.0_f64 + (gwei_f64 / 100.0_f64) // +1% per gwei above 0
        });

        #[allow(clippy::cast_precision_loss)]
        let base_profit_f64 = base_profit as f64;

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let estimated_profit = (base_profit_f64 * gas_multiplier) as u128;

        Price::new(estimated_profit)
    }

    /// Calculate confidence score for MEV opportunity
    fn calculate_confidence(transaction: &Transaction) -> f64 {
        let mut confidence = 0.0_f64;

        // Base confidence for contract calls
        if transaction.is_contract_call() {
            confidence += 0.3_f64;
        }

        // Confidence based on transaction value
        #[allow(clippy::cast_precision_loss)]
        let value_eth = transaction.value.wei() as f64 / 1e18_f64;
        if value_eth > 10.0_f64 {
            confidence += 0.3_f64;
        } else if value_eth > 1.0_f64 {
            confidence += 0.2_f64;
        } else if value_eth > 0.1_f64 {
            confidence += 0.1_f64;
        }

        // Confidence based on gas price
        if let Some(gas_price) = transaction.gas_price {
            let gwei = gas_price.wei() / 1_000_000_000;
            if gwei > 100 {
                confidence += 0.3_f64;
            } else if gwei > 50 {
                confidence += 0.2_f64;
            } else if gwei > 20 {
                confidence += 0.1_f64;
            }
        }

        // Confidence based on data size (complex transactions)
        if transaction.data.len() > 1000 {
            confidence += 0.1_f64;
        }

        confidence.min(1.0_f64)
    }

    /// Analyze batch of transactions
    ///
    /// # Errors
    ///
    /// Returns error if batch analysis fails
    pub fn analyze_batch(
        &self,
        transactions: &[Transaction],
    ) -> MempoolResult<Vec<AnalysisResult>> {
        let mut results = Vec::with_capacity(transactions.len());

        for transaction in transactions {
            let result = self.analyze(transaction)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get analyzer statistics
    #[must_use]
    pub fn get_stats(&self) -> AnalyzerStats {
        AnalyzerStats {
            mev_detection_enabled: self.config.enable_mev_detection,
            min_mev_value: self.config.min_mev_value,
        }
    }
}

/// Analyzer statistics
#[derive(Debug, Clone)]
pub struct AnalyzerStats {
    /// Whether MEV detection is enabled
    pub mev_detection_enabled: bool,

    /// Minimum MEV value threshold
    pub min_mev_value: Price,
}
