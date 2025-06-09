//! Data Transformation Pipeline
//!
//! Transforms raw blockchain data into MEV opportunities and enriched data structures.
//! Optimized for ultra-low latency financial data processing.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    config::PipelineConfig,
    error::{DataStorageError, DataStorageResult},
    pipeline::{PipelineMetrics, PipelineStage},
    types::{Block, Event, Opportunity, Transaction},
};

/// Transformation rules for different data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRules {
    /// MEV opportunity detection rules
    pub mev_rules: MevDetectionRules,
    /// Data enrichment rules
    pub enrichment_rules: EnrichmentRules,
    /// Filtering rules
    pub filter_rules: FilterRules,
}

/// MEV opportunity detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevDetectionRules {
    /// Minimum profit threshold in ETH
    pub min_profit_eth: f64,
    /// Maximum gas cost threshold in ETH
    pub max_gas_cost_eth: f64,
    /// Supported protocols for MEV detection
    pub supported_protocols: Vec<String>,
    /// DEX addresses for arbitrage detection
    pub dex_addresses: HashMap<String, String>,
    /// Lending protocol addresses
    pub lending_addresses: HashMap<String, String>,
}

/// Data enrichment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentRules {
    /// Add token metadata
    pub add_token_metadata: bool,
    /// Add protocol information
    pub add_protocol_info: bool,
    /// Add historical price data
    pub add_price_data: bool,
    /// Calculate profit estimates
    pub calculate_profits: bool,
}

/// Data filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRules {
    /// Minimum transaction value in ETH
    pub min_transaction_value: f64,
    /// Blacklisted addresses
    pub blacklisted_addresses: Vec<String>,
    /// Whitelisted protocols
    pub whitelisted_protocols: Vec<String>,
    /// Filter by chain IDs
    pub allowed_chain_ids: Vec<u32>,
}

/// Data transformation pipeline
#[derive(Debug)]
pub struct DataTransformation {
    #[allow(dead_code)] // Used for future configuration-based processing
    config: PipelineConfig,
    rules: TransformationRules,
    metrics: Arc<parking_lot::Mutex<PipelineMetrics>>,
    opportunity_cache: Arc<parking_lot::Mutex<HashMap<String, Opportunity>>>,
}

/// Transformed data output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformedData {
    /// MEV opportunity detected
    Opportunity(Opportunity),
    /// Enriched transaction
    EnrichedTransaction(EnrichedTransaction),
    /// Enriched block
    EnrichedBlock(EnrichedBlock),
    /// Enriched event
    EnrichedEvent(EnrichedEvent),
    /// Filtered out (no transformation needed)
    Filtered,
}

/// Enriched transaction with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedTransaction {
    /// Base transaction
    pub transaction: Transaction,
    /// Token metadata
    pub token_metadata: Option<TokenMetadata>,
    /// Protocol information
    pub protocol_info: Option<ProtocolInfo>,
    /// Estimated gas cost in ETH
    pub gas_cost_eth: Option<f64>,
    /// Transaction value in USD
    pub value_usd: Option<f64>,
}

/// Enriched block with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedBlock {
    /// Base block
    pub block: Block,
    /// MEV opportunities in this block
    pub mev_opportunities: Vec<Opportunity>,
    /// Total MEV value in ETH
    pub total_mev_value: f64,
    /// Block utilization percentage
    pub utilization_percent: f64,
}

/// Enriched event with decoded data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedEvent {
    /// Base event
    pub event: Event,
    /// Decoded event data
    pub decoded_data: Option<serde_json::Value>,
    /// Event type classification
    pub event_type: EventType,
    /// Related MEV opportunity ID
    pub related_opportunity_id: Option<uuid::Uuid>,
}

/// Token metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetadata {
    /// Token symbol
    pub symbol: String,
    /// Token name
    pub name: String,
    /// Token decimals
    pub decimals: u8,
    /// Current price in USD
    pub price_usd: Option<f64>,
}

/// Protocol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInfo {
    /// Protocol name
    pub name: String,
    /// Protocol type (DEX, Lending, etc.)
    pub protocol_type: ProtocolType,
    /// Protocol version
    pub version: Option<String>,
    /// Total value locked
    pub tvl_usd: Option<f64>,
}

/// Protocol type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolType {
    /// Decentralized exchange
    Dex,
    /// Lending protocol
    Lending,
    /// Yield farming
    YieldFarming,
    /// Derivatives
    Derivatives,
    /// Bridge
    Bridge,
    /// Other/Unknown
    Other,
}

/// Event type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Token swap
    Swap,
    /// Liquidity provision
    LiquidityAdd,
    /// Liquidity removal
    LiquidityRemove,
    /// Lending/borrowing
    Lending,
    /// Liquidation
    Liquidation,
    /// Transfer
    Transfer,
    /// Other/Unknown
    Other,
}

impl Default for TransformationRules {
    fn default() -> Self {
        Self {
            mev_rules: MevDetectionRules {
                min_profit_eth: 0.01,
                max_gas_cost_eth: 0.1,
                supported_protocols: vec![
                    "Uniswap".to_string(),
                    "SushiSwap".to_string(),
                    "Aave".to_string(),
                    "Compound".to_string(),
                ],
                dex_addresses: HashMap::new(),
                lending_addresses: HashMap::new(),
            },
            enrichment_rules: EnrichmentRules {
                add_token_metadata: true,
                add_protocol_info: true,
                add_price_data: true,
                calculate_profits: true,
            },
            filter_rules: FilterRules {
                min_transaction_value: 0.001,
                blacklisted_addresses: Vec::new(),
                whitelisted_protocols: Vec::new(),
                allowed_chain_ids: vec![1, 137, 42161, 10], // Ethereum, Polygon, Arbitrum, Optimism
            },
        }
    }
}

impl DataTransformation {
    /// Create a new data transformation pipeline
    #[must_use]
    pub fn new(config: PipelineConfig, rules: TransformationRules) -> Self {
        Self {
            config,
            rules,
            metrics: Arc::new(parking_lot::Mutex::new(PipelineMetrics::new(
                "transformation".to_string(),
            ))),
            opportunity_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }
    }

    /// Transform transaction data
    fn transform_transaction(
        &self,
        transaction: Transaction,
    ) -> DataStorageResult<TransformedData> {
        let start = Instant::now();

        // Apply filtering rules
        if !self.should_process_transaction(&transaction)? {
            return Ok(TransformedData::Filtered);
        }

        // Check for MEV opportunities
        if let Some(opportunity) = self.detect_mev_opportunity(&transaction)? {
            // Cache the opportunity
            let mut cache = self.opportunity_cache.lock();
            cache.insert(opportunity.id.to_string(), opportunity.clone());

            self.update_metrics(true, start.elapsed());
            return Ok(TransformedData::Opportunity(opportunity));
        }

        // Enrich transaction data
        let enriched = self.enrich_transaction(transaction)?;

        self.update_metrics(true, start.elapsed());
        Ok(TransformedData::EnrichedTransaction(enriched))
    }

    /// Transform block data
    #[allow(dead_code)] // Part of transformation API
    fn transform_block(&self, block: Block) -> DataStorageResult<TransformedData> {
        let start = Instant::now();

        // Get MEV opportunities for this block
        let cache = self.opportunity_cache.lock();
        let mev_opportunities: Vec<Opportunity> = cache
            .values()
            .filter(|opp| {
                // Filter opportunities that might be in this block
                opp.chain_id == block.chain_id
            })
            .cloned()
            .collect();

        let total_mev_value = mev_opportunities
            .iter()
            .map(|opp| opp.profit_eth.parse::<f64>().unwrap_or(0.0))
            .sum();

        let utilization_percent = if block.gas_limit > 0 {
            (block.gas_used as f64 / block.gas_limit as f64) * 100.0
        } else {
            0.0
        };

        let enriched_block = EnrichedBlock {
            block,
            mev_opportunities,
            total_mev_value,
            utilization_percent,
        };

        self.update_metrics(true, start.elapsed());
        Ok(TransformedData::EnrichedBlock(enriched_block))
    }

    /// Transform event data
    #[allow(dead_code)] // Part of transformation API
    fn transform_event(&self, event: Event) -> DataStorageResult<TransformedData> {
        let start = Instant::now();

        // Classify event type
        let event_type = self.classify_event_type(&event);

        // Try to decode event data
        let decoded_data = self.decode_event_data(&event)?;

        // Check if event is related to any cached MEV opportunity
        let cache = self.opportunity_cache.lock();
        let related_opportunity_id = cache
            .values()
            .find(|opp| {
                // Simple heuristic: same chain and recent
                opp.chain_id == event.chain_id
            })
            .map(|opp| opp.id);

        let enriched_event = EnrichedEvent {
            event,
            decoded_data,
            event_type,
            related_opportunity_id,
        };

        self.update_metrics(true, start.elapsed());
        Ok(TransformedData::EnrichedEvent(enriched_event))
    }

    /// Check if transaction should be processed based on filter rules
    fn should_process_transaction(&self, transaction: &Transaction) -> DataStorageResult<bool> {
        // Check chain ID
        if !self
            .rules
            .filter_rules
            .allowed_chain_ids
            .contains(&transaction.chain_id)
        {
            return Ok(false);
        }

        // Check blacklisted addresses
        if self
            .rules
            .filter_rules
            .blacklisted_addresses
            .contains(&transaction.from_address)
        {
            return Ok(false);
        }

        if let Some(ref to) = transaction.to_address {
            if self.rules.filter_rules.blacklisted_addresses.contains(to) {
                return Ok(false);
            }
        }

        // Check minimum transaction value
        let value_eth = transaction.value.parse::<f64>().map_err(|e| {
            DataStorageError::validation("transaction_value", format!("Invalid value format: {e}"))
        })?;

        if value_eth < self.rules.filter_rules.min_transaction_value {
            return Ok(false);
        }

        Ok(true)
    }

    /// Detect MEV opportunities in transaction
    fn detect_mev_opportunity(
        &self,
        transaction: &Transaction,
    ) -> DataStorageResult<Option<Opportunity>> {
        // Simplified MEV detection logic
        // In production, this would involve complex analysis of transaction data

        // Check if transaction interacts with known DEX
        let is_dex_interaction = self
            .rules
            .mev_rules
            .dex_addresses
            .values()
            .any(|addr| transaction.to_address.as_ref() == Some(addr));

        if !is_dex_interaction {
            return Ok(None);
        }

        // Simulate MEV opportunity detection
        let estimated_profit = fastrand::f64() * 0.1; // Random profit 0-0.1 ETH
        let estimated_gas_cost = fastrand::f64() * 0.01; // Random gas cost 0-0.01 ETH

        if estimated_profit < self.rules.mev_rules.min_profit_eth {
            return Ok(None);
        }

        if estimated_gas_cost > self.rules.mev_rules.max_gas_cost_eth {
            return Ok(None);
        }

        let net_profit = estimated_profit - estimated_gas_cost;
        if net_profit <= 0.0 {
            return Ok(None);
        }

        let opportunity = Opportunity::new(
            "arbitrage".to_string(),
            transaction.chain_id,
            estimated_profit.to_string(),
            estimated_gas_cost.to_string(),
            net_profit.to_string(),
            0.8, // 80% confidence
        );

        Ok(Some(opportunity))
    }

    /// Enrich transaction with additional metadata
    fn enrich_transaction(
        &self,
        transaction: Transaction,
    ) -> DataStorageResult<EnrichedTransaction> {
        // Simplified enrichment logic
        let token_metadata = if self.rules.enrichment_rules.add_token_metadata {
            Some(TokenMetadata {
                symbol: "ETH".to_string(),
                name: "Ethereum".to_string(),
                decimals: 18,
                price_usd: Some(2000.0), // Mock price
            })
        } else {
            None
        };

        let protocol_info = if self.rules.enrichment_rules.add_protocol_info {
            Some(ProtocolInfo {
                name: "Unknown".to_string(),
                protocol_type: ProtocolType::Other,
                version: None,
                tvl_usd: None,
            })
        } else {
            None
        };

        let gas_cost_eth = if self.rules.enrichment_rules.calculate_profits {
            let gas_price_gwei = transaction.gas_price.parse::<f64>().unwrap_or(0.0);
            let gas_used = transaction.gas_used.unwrap_or(21000);
            Some((gas_price_gwei * gas_used as f64) / 1e9) // Convert to ETH
        } else {
            None
        };

        let value_usd = if self.rules.enrichment_rules.add_price_data {
            let value_eth = transaction.value.parse::<f64>().unwrap_or(0.0);
            Some(value_eth * 2000.0) // Mock ETH price
        } else {
            None
        };

        Ok(EnrichedTransaction {
            transaction,
            token_metadata,
            protocol_info,
            gas_cost_eth,
            value_usd,
        })
    }

    /// Classify event type based on signature
    #[allow(dead_code)] // Part of transformation API
    fn classify_event_type(&self, event: &Event) -> EventType {
        match event.event_signature.as_str() {
            sig if sig.contains("Swap") => EventType::Swap,
            sig if sig.contains("Transfer") => EventType::Transfer,
            sig if sig.contains("Mint") || sig.contains("Burn") => EventType::LiquidityAdd,
            sig if sig.contains("Liquidat") => EventType::Liquidation,
            _ => EventType::Other,
        }
    }

    /// Decode event data (simplified)
    #[allow(dead_code)] // Part of transformation API
    fn decode_event_data(&self, _event: &Event) -> DataStorageResult<Option<serde_json::Value>> {
        // In production, this would use ABI decoding
        Ok(Some(serde_json::json!({
            "decoded": false,
            "reason": "ABI decoding not implemented"
        })))
    }

    /// Update transformation metrics
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
}

#[async_trait]
impl PipelineStage<Transaction> for DataTransformation {
    async fn process(&self, data: Transaction) -> DataStorageResult<Transaction> {
        let _result = self.transform_transaction(data.clone())?;
        Ok(data) // Return original data for pipeline compatibility
    }

    fn name(&self) -> &str {
        "transformation"
    }

    async fn metrics(&self) -> DataStorageResult<PipelineMetrics> {
        Ok(self.metrics.lock().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transformation_creation() {
        let config = PipelineConfig::default();
        let rules = TransformationRules::default();
        let transformation = DataTransformation::new(config, rules);
        assert_eq!(transformation.name(), "transformation");
    }

    #[tokio::test]
    async fn test_transaction_filtering() -> DataStorageResult<()> {
        let config = PipelineConfig::default();
        let rules = TransformationRules::default();
        let transformation = DataTransformation::new(config, rules);

        let valid_tx = Transaction::new(
            1, // Ethereum mainnet
            100,
            "0x123".to_string(),
            "0xabc".to_string(),
            Some("0xdef".to_string()),
            "1.0".to_string(), // 1 ETH, above minimum
            "20".to_string(),
        );

        assert!(transformation.should_process_transaction(&valid_tx)?);

        let invalid_tx = Transaction::new(
            999, // Unsupported chain
            100,
            "0x123".to_string(),
            "0xabc".to_string(),
            Some("0xdef".to_string()),
            "1.0".to_string(),
            "20".to_string(),
        );

        assert!(!transformation.should_process_transaction(&invalid_tx)?);

        Ok(())
    }
}
