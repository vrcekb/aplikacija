//! Transaction Filter - Ultra-Performance Transaction Filtering
//!
//! Production-ready transaction filtering for mempool optimization.

use std::collections::HashSet;
use std::sync::Arc;

use crate::types::{Address, Price, Transaction};

use super::{MempoolConfig, MempoolResult};

/// Transaction filter for mempool optimization
pub struct TransactionFilter {
    /// Configuration
    config: Arc<MempoolConfig>,

    /// Blacklisted addresses
    blacklisted_addresses: HashSet<Address>,

    /// Minimum gas price
    min_gas_price: Price,

    /// Minimum transaction value
    min_value: Price,
}

impl TransactionFilter {
    /// Create new transaction filter
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(config: Arc<MempoolConfig>) -> MempoolResult<Self> {
        Ok(Self {
            config,
            blacklisted_addresses: HashSet::new(),
            min_gas_price: Price::from_gwei(1), // 1 gwei minimum
            min_value: Price::new(0),           // No minimum value by default
        })
    }

    /// Check if transaction should be included in mempool
    ///
    /// # Errors
    ///
    /// Returns error if filtering fails
    pub fn should_include(&self, transaction: &Transaction) -> MempoolResult<bool> {
        // Check blacklisted addresses
        if self.is_blacklisted(&transaction.from) {
            return Ok(false);
        }

        if let Some(to) = transaction.to {
            if self.is_blacklisted(&to) {
                return Ok(false);
            }
        }

        // Check minimum gas price
        if !self.meets_gas_price_requirement(transaction) {
            return Ok(false);
        }

        // Check minimum value
        if transaction.value.wei() < self.min_value.wei() {
            return Ok(false);
        }

        // Check for spam transactions (very small gas limit)
        if transaction.gas_limit.amount() < self.config.min_gas_limit {
            return Ok(false);
        }

        // Check for suspiciously high gas limit
        if transaction.gas_limit.amount() > self.config.max_gas_limit {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check if address is blacklisted
    fn is_blacklisted(&self, address: &Address) -> bool {
        self.blacklisted_addresses.contains(address)
    }

    /// Check if transaction meets gas price requirements
    const fn meets_gas_price_requirement(&self, transaction: &Transaction) -> bool {
        match transaction.gas_price {
            Some(gas_price) => gas_price.wei() >= self.min_gas_price.wei(),
            None => {
                // For EIP-1559 transactions, check max fee per gas
                if let Some(max_fee) = transaction.max_fee_per_gas {
                    max_fee.wei() >= self.min_gas_price.wei()
                } else {
                    false
                }
            }
        }
    }

    /// Add address to blacklist
    pub fn blacklist_address(&mut self, address: Address) {
        self.blacklisted_addresses.insert(address);
    }

    /// Remove address from blacklist
    pub fn unblacklist_address(&mut self, address: &Address) {
        self.blacklisted_addresses.remove(address);
    }

    /// Set minimum gas price
    pub const fn set_min_gas_price(&mut self, min_gas_price: Price) {
        self.min_gas_price = min_gas_price;
    }

    /// Set minimum transaction value
    pub const fn set_min_value(&mut self, min_value: Price) {
        self.min_value = min_value;
    }

    /// Get filter statistics
    #[must_use]
    pub fn get_stats(&self) -> FilterStats {
        FilterStats {
            blacklisted_count: self.blacklisted_addresses.len(),
            min_gas_price: self.min_gas_price,
            min_value: self.min_value,
        }
    }

    /// Filter transactions by value
    #[must_use]
    pub fn filter_by_value<'a>(
        &self,
        transactions: &'a [Transaction],
        min_value: Price,
    ) -> Vec<&'a Transaction> {
        transactions
            .iter()
            .filter(|tx| tx.value.wei() >= min_value.wei())
            .collect()
    }

    /// Filter transactions by gas price
    #[must_use]
    pub fn filter_by_gas_price<'a>(
        &self,
        transactions: &'a [Transaction],
        min_gas_price: Price,
    ) -> Vec<&'a Transaction> {
        transactions
            .iter()
            .filter(|tx| {
                tx.gas_price.map_or_else(
                    || {
                        tx.max_fee_per_gas
                            .is_some_and(|max_fee| max_fee.wei() >= min_gas_price.wei())
                    },
                    |gas_price| gas_price.wei() >= min_gas_price.wei(),
                )
            })
            .collect()
    }

    /// Filter contract calls
    #[must_use]
    pub fn filter_contract_calls<'a>(
        &self,
        transactions: &'a [Transaction],
    ) -> Vec<&'a Transaction> {
        transactions
            .iter()
            .filter(|tx| tx.is_contract_call())
            .collect()
    }

    /// Filter high-value transactions
    #[must_use]
    pub fn filter_high_value<'a>(
        &self,
        transactions: &'a [Transaction],
        threshold: Price,
    ) -> Vec<&'a Transaction> {
        transactions
            .iter()
            .filter(|tx| tx.value.wei() >= threshold.wei())
            .collect()
    }

    /// Filter transactions by address
    #[must_use]
    pub fn filter_by_address<'a>(
        &self,
        transactions: &'a [Transaction],
        address: &Address,
    ) -> Vec<&'a Transaction> {
        transactions
            .iter()
            .filter(|tx| tx.from == *address || (tx.to == Some(*address)))
            .collect()
    }
}

/// Filter statistics
#[derive(Debug, Clone)]
pub struct FilterStats {
    /// Number of blacklisted addresses
    pub blacklisted_count: usize,

    /// Minimum gas price requirement
    pub min_gas_price: Price,

    /// Minimum transaction value requirement
    pub min_value: Price,
}
