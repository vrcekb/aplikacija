//! Result Types
//!
//! Production-ready result types for `TallyIO` operations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::Price;

/// Execution result for strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution was successful
    pub success: bool,

    /// Gas used during execution
    pub gas_used: u64,

    /// Actual execution time
    pub execution_time: Duration,

    /// Result data
    pub data: Vec<u8>,

    /// Error message if execution failed
    pub error: Option<String>,

    /// Profit made (if any)
    pub profit: Option<Price>,

    /// Transaction hashes of executed transactions
    pub transaction_hashes: Vec<String>,
}

impl ExecutionResult {
    /// Create successful result
    #[must_use]
    pub fn success(gas_used: u64, execution_time: Duration, data: Vec<u8>) -> Self {
        Self {
            success: true,
            gas_used,
            execution_time,
            data,
            error: None,
            profit: None,
            transaction_hashes: Vec::with_capacity(1), // Usually 1 transaction
        }
    }

    /// Create failed result
    #[must_use]
    pub fn failure(error: String, gas_used: u64, execution_time: Duration) -> Self {
        Self {
            success: false,
            gas_used,
            execution_time,
            data: Vec::with_capacity(0),
            error: Some(error),
            profit: None,
            transaction_hashes: Vec::with_capacity(0),
        }
    }

    /// Add profit information
    #[must_use]
    pub const fn with_profit(mut self, profit: Price) -> Self {
        self.profit = Some(profit);
        self
    }

    /// Add transaction hash
    #[must_use]
    pub fn with_transaction(mut self, tx_hash: String) -> Self {
        self.transaction_hashes.push(tx_hash);
        self
    }
}

/// Analysis result for opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Whether analysis found opportunities
    pub has_opportunities: bool,

    /// Number of opportunities found
    pub opportunity_count: usize,

    /// Total estimated profit
    pub total_estimated_profit: Price,

    /// Analysis execution time
    pub analysis_time: Duration,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Additional metadata
    pub metadata: Vec<(String, String)>,
}

impl AnalysisResult {
    /// Create result with no opportunities
    #[must_use]
    pub fn no_opportunities(analysis_time: Duration) -> Self {
        Self {
            has_opportunities: false,
            opportunity_count: 0,
            total_estimated_profit: Price::new(0),
            analysis_time,
            confidence: 0.0_f64,
            metadata: Vec::with_capacity(4), // Typical metadata size
        }
    }

    /// Create result with opportunities
    #[must_use]
    pub fn with_opportunities(
        opportunity_count: usize,
        total_estimated_profit: Price,
        analysis_time: Duration,
        confidence: f64,
    ) -> Self {
        Self {
            has_opportunities: opportunity_count > 0,
            opportunity_count,
            total_estimated_profit,
            analysis_time,
            confidence,
            metadata: Vec::with_capacity(4), // Typical metadata size
        }
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.push((key, value));
        self
    }
}
