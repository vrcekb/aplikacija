//! Data Validation Pipeline
//!
//! Comprehensive validation for financial data integrity and security.
//! Ensures all data meets strict quality standards before storage.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    config::PipelineConfig,
    error::{DataStorageError, DataStorageResult},
    pipeline::{PipelineMetrics, PipelineStage},
    types::{Opportunity, Transaction},
};

/// Validation rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    /// Transaction validation rules
    pub transaction_rules: TransactionValidationRules,
    /// Opportunity validation rules
    pub opportunity_rules: OpportunityValidationRules,
    /// Block validation rules
    pub block_rules: BlockValidationRules,
    /// Event validation rules
    pub event_rules: EventValidationRules,
    /// General validation settings
    pub general_rules: GeneralValidationRules,
}

/// Transaction validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionValidationRules {
    /// Maximum gas price in Gwei
    pub max_gas_price_gwei: f64,
    /// Minimum gas limit
    pub min_gas_limit: u64,
    /// Maximum gas limit
    pub max_gas_limit: u64,
    /// Valid address format regex
    pub address_format_regex: String,
    /// Maximum transaction value in ETH
    pub max_value_eth: f64,
    /// Required fields
    pub required_fields: Vec<String>,
}

/// Opportunity validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityValidationRules {
    /// Minimum confidence score
    pub min_confidence_score: f64,
    /// Maximum confidence score
    pub max_confidence_score: f64,
    /// Minimum profit in ETH
    pub min_profit_eth: f64,
    /// Maximum profit in ETH (sanity check)
    pub max_profit_eth: f64,
    /// Valid opportunity types
    pub valid_opportunity_types: Vec<String>,
    /// Maximum gas cost in ETH
    pub max_gas_cost_eth: f64,
}

/// Block validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockValidationRules {
    /// Maximum block size
    pub max_block_size: u64,
    /// Maximum transaction count per block
    pub max_transaction_count: u32,
    /// Valid hash format regex
    pub hash_format_regex: String,
    /// Maximum timestamp deviation from current time (seconds)
    pub max_timestamp_deviation_sec: i64,
}

/// Event validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventValidationRules {
    /// Maximum topics count
    pub max_topics_count: usize,
    /// Maximum data size in bytes
    pub max_data_size: usize,
    /// Valid address format regex
    pub address_format_regex: String,
    /// Required fields
    pub required_fields: Vec<String>,
}

/// General validation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralValidationRules {
    /// Enable strict validation mode
    pub strict_mode: bool,
    /// Enable data sanitization
    pub enable_sanitization: bool,
    /// Maximum processing time per item (microseconds)
    pub max_processing_time_us: u64,
    /// Enable duplicate detection
    pub enable_duplicate_detection: bool,
}

/// Data validation pipeline
#[derive(Debug)]
pub struct DataValidation {
    #[allow(dead_code)] // Used for future configuration-based validation
    config: PipelineConfig,
    rules: ValidationRules,
    metrics: Arc<parking_lot::Mutex<PipelineMetrics>>,
    seen_hashes: Arc<parking_lot::Mutex<HashSet<String>>>,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Sanitized data (if sanitization was applied)
    pub sanitized_data: Option<serde_json::Value>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Field that failed validation
    pub field: String,
    /// Error message
    pub message: String,
    /// Severity level
    pub severity: ValidationSeverity,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Field that triggered warning
    pub field: String,
    /// Warning message
    pub message: String,
}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical error - data must be rejected
    Critical,
    /// High severity - data should be rejected
    High,
    /// Medium severity - data may be accepted with warnings
    Medium,
    /// Low severity - informational only
    Low,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            transaction_rules: TransactionValidationRules {
                max_gas_price_gwei: 1000.0,
                min_gas_limit: 21000,
                max_gas_limit: 30_000_000,
                address_format_regex: r"^0x[a-fA-F0-9]{40}$".to_string(),
                max_value_eth: 1000.0,
                required_fields: vec![
                    "hash".to_string(),
                    "from_address".to_string(),
                    "chain_id".to_string(),
                ],
            },
            opportunity_rules: OpportunityValidationRules {
                min_confidence_score: 0.0,
                max_confidence_score: 1.0,
                min_profit_eth: 0.001,
                max_profit_eth: 100.0,
                valid_opportunity_types: vec![
                    "arbitrage".to_string(),
                    "liquidation".to_string(),
                    "sandwich".to_string(),
                    "frontrun".to_string(),
                ],
                max_gas_cost_eth: 1.0,
            },
            block_rules: BlockValidationRules {
                max_block_size: 30_000_000,
                max_transaction_count: 1000,
                hash_format_regex: r"^0x[a-fA-F0-9]{64}$".to_string(),
                max_timestamp_deviation_sec: 3600, // 1 hour
            },
            event_rules: EventValidationRules {
                max_topics_count: 4,
                max_data_size: 1024 * 1024, // 1MB
                address_format_regex: r"^0x[a-fA-F0-9]{40}$".to_string(),
                required_fields: vec![
                    "contract_address".to_string(),
                    "transaction_hash".to_string(),
                    "chain_id".to_string(),
                ],
            },
            general_rules: GeneralValidationRules {
                strict_mode: true,
                enable_sanitization: true,
                max_processing_time_us: 1000, // 1ms for financial data
                enable_duplicate_detection: true,
            },
        }
    }
}

impl DataValidation {
    /// Create a new data validation pipeline
    #[must_use]
    pub fn new(config: PipelineConfig, rules: ValidationRules) -> Self {
        Self {
            config,
            rules,
            metrics: Arc::new(parking_lot::Mutex::new(PipelineMetrics::new(
                "validation".to_string(),
            ))),
            seen_hashes: Arc::new(parking_lot::Mutex::new(HashSet::new())),
        }
    }

    /// Validate transaction data
    pub fn validate_transaction(
        &self,
        transaction: &Transaction,
    ) -> DataStorageResult<ValidationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check required fields
        if transaction.tx_hash.is_empty() {
            errors.push(ValidationError {
                code: "MISSING_HASH".to_string(),
                field: "tx_hash".to_string(),
                message: "Transaction hash is required".to_string(),
                severity: ValidationSeverity::Critical,
            });
        }

        if transaction.from_address.is_empty() {
            errors.push(ValidationError {
                code: "MISSING_FROM".to_string(),
                field: "from_address".to_string(),
                message: "From address is required".to_string(),
                severity: ValidationSeverity::Critical,
            });
        }

        // Validate address format
        if !self.is_valid_address(&transaction.from_address) {
            errors.push(ValidationError {
                code: "INVALID_ADDRESS_FORMAT".to_string(),
                field: "from_address".to_string(),
                message: "Invalid address format".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        if let Some(ref to_addr) = transaction.to_address {
            if !self.is_valid_address(to_addr) {
                errors.push(ValidationError {
                    code: "INVALID_ADDRESS_FORMAT".to_string(),
                    field: "to_address".to_string(),
                    message: "Invalid address format".to_string(),
                    severity: ValidationSeverity::High,
                });
            }
        }

        // Validate gas price
        if let Ok(gas_price_gwei) = transaction.gas_price.parse::<f64>() {
            if gas_price_gwei > self.rules.transaction_rules.max_gas_price_gwei {
                warnings.push(ValidationWarning {
                    code: "HIGH_GAS_PRICE".to_string(),
                    field: "gas_price".to_string(),
                    message: format!("Gas price {gas_price_gwei} Gwei exceeds recommended maximum"),
                });
            }
        } else {
            errors.push(ValidationError {
                code: "INVALID_GAS_PRICE".to_string(),
                field: "gas_price".to_string(),
                message: "Invalid gas price format".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        // Validate transaction value
        if let Ok(value_eth) = transaction.value.parse::<f64>() {
            if value_eth > self.rules.transaction_rules.max_value_eth {
                warnings.push(ValidationWarning {
                    code: "HIGH_VALUE".to_string(),
                    field: "value".to_string(),
                    message: format!("Transaction value {value_eth} ETH is unusually high"),
                });
            }
        } else {
            errors.push(ValidationError {
                code: "INVALID_VALUE".to_string(),
                field: "value".to_string(),
                message: "Invalid value format".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        // Check for duplicates
        if self.rules.general_rules.enable_duplicate_detection {
            let mut seen = self.seen_hashes.lock();
            if seen.contains(&transaction.tx_hash) {
                errors.push(ValidationError {
                    code: "DUPLICATE_TRANSACTION".to_string(),
                    field: "tx_hash".to_string(),
                    message: "Duplicate transaction hash detected".to_string(),
                    severity: ValidationSeverity::Critical,
                });
            } else {
                seen.insert(transaction.tx_hash.clone());
            }
        }

        let duration = start.elapsed();
        self.update_metrics(errors.is_empty(), duration);

        // Check processing time
        if duration.as_micros() > u128::from(self.rules.general_rules.max_processing_time_us) {
            warnings.push(ValidationWarning {
                code: "SLOW_VALIDATION".to_string(),
                field: "processing_time".to_string(),
                message: format!("Validation took {}μs, exceeds target", duration.as_micros()),
            });
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty()
                || (!self.rules.general_rules.strict_mode
                    && errors.iter().all(|e| e.severity == ValidationSeverity::Low)),
            errors,
            warnings,
            sanitized_data: None,
        })
    }

    /// Validate opportunity data
    pub fn validate_opportunity(
        &self,
        opportunity: &Opportunity,
    ) -> DataStorageResult<ValidationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate confidence score
        if opportunity.confidence_score < self.rules.opportunity_rules.min_confidence_score
            || opportunity.confidence_score > self.rules.opportunity_rules.max_confidence_score
        {
            errors.push(ValidationError {
                code: "INVALID_CONFIDENCE".to_string(),
                field: "confidence_score".to_string(),
                message: format!(
                    "Confidence score {} is out of valid range [0.0, 1.0]",
                    opportunity.confidence_score
                ),
                severity: ValidationSeverity::High,
            });
        }

        // Validate opportunity type
        if !self
            .rules
            .opportunity_rules
            .valid_opportunity_types
            .contains(&opportunity.opportunity_type)
        {
            errors.push(ValidationError {
                code: "INVALID_OPPORTUNITY_TYPE".to_string(),
                field: "opportunity_type".to_string(),
                message: format!("Unknown opportunity type: {}", opportunity.opportunity_type),
                severity: ValidationSeverity::Medium,
            });
        }

        // Validate profit values
        if let Ok(profit_eth) = opportunity.profit_eth.parse::<f64>() {
            if profit_eth < self.rules.opportunity_rules.min_profit_eth {
                warnings.push(ValidationWarning {
                    code: "LOW_PROFIT".to_string(),
                    field: "profit_eth".to_string(),
                    message: format!("Profit {profit_eth} ETH is below minimum threshold"),
                });
            }

            if profit_eth > self.rules.opportunity_rules.max_profit_eth {
                warnings.push(ValidationWarning {
                    code: "HIGH_PROFIT".to_string(),
                    field: "profit_eth".to_string(),
                    message: format!("Profit {profit_eth} ETH seems unusually high"),
                });
            }
        } else {
            errors.push(ValidationError {
                code: "INVALID_PROFIT_FORMAT".to_string(),
                field: "profit_eth".to_string(),
                message: "Invalid profit format".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        // Validate gas cost
        if let Ok(gas_cost_eth) = opportunity.gas_cost.parse::<f64>() {
            if gas_cost_eth > self.rules.opportunity_rules.max_gas_cost_eth {
                warnings.push(ValidationWarning {
                    code: "HIGH_GAS_COST".to_string(),
                    field: "gas_cost".to_string(),
                    message: format!("Gas cost {gas_cost_eth} ETH is very high"),
                });
            }
        } else {
            errors.push(ValidationError {
                code: "INVALID_GAS_COST_FORMAT".to_string(),
                field: "gas_cost".to_string(),
                message: "Invalid gas cost format".to_string(),
                severity: ValidationSeverity::High,
            });
        }

        // Validate net profit calculation
        if let (Ok(profit), Ok(gas_cost), Ok(net_profit)) = (
            opportunity.profit_eth.parse::<f64>(),
            opportunity.gas_cost.parse::<f64>(),
            opportunity.net_profit.parse::<f64>(),
        ) {
            let calculated_net = profit - gas_cost;
            let diff = (calculated_net - net_profit).abs();
            if diff > 0.001 {
                // 0.001 ETH tolerance
                errors.push(ValidationError {
                    code: "INCORRECT_NET_PROFIT".to_string(),
                    field: "net_profit".to_string(),
                    message: format!(
                        "Net profit calculation error: expected {calculated_net}, got {net_profit}"
                    ),
                    severity: ValidationSeverity::High,
                });
            }
        }

        let duration = start.elapsed();
        self.update_metrics(errors.is_empty(), duration);

        Ok(ValidationResult {
            is_valid: errors.is_empty()
                || (!self.rules.general_rules.strict_mode
                    && errors.iter().all(|e| e.severity == ValidationSeverity::Low)),
            errors,
            warnings,
            sanitized_data: None,
        })
    }

    /// Validate address format using regex
    fn is_valid_address(&self, address: &str) -> bool {
        if let Ok(regex) = regex::Regex::new(&self.rules.transaction_rules.address_format_regex) {
            regex.is_match(address)
        } else {
            // Fallback validation
            address.starts_with("0x") && address.len() == 42
        }
    }

    /// Update validation metrics
    fn update_metrics(&self, success: bool, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.items_processed += 1;

        if !success {
            metrics.items_failed += 1;
        }

        let duration_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        metrics.total_processing_time_us += duration_us;
        metrics.avg_processing_time_us = metrics.total_processing_time_us / metrics.items_processed;
    }

    /// Get validation statistics
    #[must_use]
    pub fn validation_stats(&self) -> ValidationStats {
        let metrics = self.metrics.lock();
        let seen_count = self.seen_hashes.lock().len();

        ValidationStats {
            total_validated: metrics.items_processed,
            total_failed: metrics.items_failed,
            success_rate: if metrics.items_processed > 0 {
                (metrics.items_processed - metrics.items_failed) as f64
                    / metrics.items_processed as f64
            } else {
                0.0
            },
            avg_validation_time_us: metrics.avg_processing_time_us,
            unique_items_seen: seen_count,
        }
    }
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    /// Total items validated
    pub total_validated: u64,
    /// Total items that failed validation
    pub total_failed: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average validation time in microseconds
    pub avg_validation_time_us: u64,
    /// Number of unique items seen (for duplicate detection)
    pub unique_items_seen: usize,
}

#[async_trait]
impl PipelineStage<Transaction> for DataValidation {
    async fn process(&self, data: Transaction) -> DataStorageResult<Transaction> {
        let result = self.validate_transaction(&data)?;

        if !result.is_valid {
            let error_messages: Vec<String> = result
                .errors
                .iter()
                .map(|e| format!("{}: {}", e.field, e.message))
                .collect();
            return Err(DataStorageError::validation(
                "transaction",
                error_messages.join("; "),
            ));
        }

        Ok(data)
    }

    fn name(&self) -> &str {
        "validation"
    }

    async fn metrics(&self) -> DataStorageResult<PipelineMetrics> {
        Ok(self.metrics.lock().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_creation() {
        let config = PipelineConfig::default();
        let rules = ValidationRules::default();
        let validation = DataValidation::new(config, rules);
        assert_eq!(validation.name(), "validation");
    }

    #[tokio::test]
    async fn test_address_validation() {
        let config = PipelineConfig::default();
        let rules = ValidationRules::default();
        let validation = DataValidation::new(config, rules);

        assert!(validation.is_valid_address("0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"));
        assert!(!validation.is_valid_address("invalid_address"));
        assert!(!validation.is_valid_address("0x123")); // Too short
    }

    #[tokio::test]
    async fn test_transaction_validation() -> DataStorageResult<()> {
        let config = PipelineConfig::default();
        let rules = ValidationRules::default();
        let validation = DataValidation::new(config, rules);

        let valid_tx = Transaction::new(
            1,
            100,
            "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
                .to_string(),
            "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            Some("0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string()),
            "1.0".to_string(),
            "20".to_string(),
        );

        let result = validation.validate_transaction(&valid_tx)?;
        assert!(result.is_valid);

        Ok(())
    }
}
