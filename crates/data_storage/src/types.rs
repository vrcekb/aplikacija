//! `TallyIO` Data Storage Types (Simplified)
//!
//! Core data types for the storage layer without validation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Transaction data for MEV opportunities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Transaction {
    /// Unique transaction identifier
    pub id: Uuid,

    /// Blockchain chain ID
    pub chain_id: u32,

    /// Block number
    pub block_number: u64,

    /// Transaction hash (hex string with 0x prefix)
    pub tx_hash: String,

    /// From address (hex string with 0x prefix)
    pub from_address: String,

    /// To address (optional, hex string with 0x prefix)
    pub to_address: Option<String>,

    /// Transaction value in wei (as string to avoid precision loss)
    pub value: String,

    /// Gas price in wei (as string to avoid precision loss)
    pub gas_price: String,

    /// Gas used
    pub gas_used: Option<u64>,

    /// Transaction status (0 = failed, 1 = success)
    pub status: u8,

    /// Transaction timestamp
    pub created_at: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// MEV opportunity data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Opportunity {
    /// Unique opportunity identifier
    pub id: Uuid,

    /// Type of opportunity (arbitrage, liquidation, sandwich, etc.)
    pub opportunity_type: String,

    /// Blockchain chain ID
    pub chain_id: u32,

    /// Protocol name (Uniswap, Aave, etc.)
    pub protocol: Option<String>,

    /// Estimated profit in ETH (as string to avoid precision loss)
    pub profit_eth: String,

    /// Estimated gas cost in ETH (as string to avoid precision loss)
    pub gas_cost: String,

    /// Net profit (profit - `gas_cost`) in ETH
    pub net_profit: String,

    /// Confidence score (0.0 - 1.0)
    pub confidence_score: f64,

    /// Whether opportunity was executed
    pub executed: bool,

    /// Execution transaction hash (if executed)
    pub execution_tx_hash: Option<String>,

    /// Opportunity discovery timestamp
    pub created_at: DateTime<Utc>,

    /// Opportunity expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,

    /// Additional opportunity data
    pub data: serde_json::Value,
}

/// Block data for indexing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Block {
    /// Block number
    pub number: u64,

    /// Block hash
    pub hash: String,

    /// Parent block hash
    pub parent_hash: String,

    /// Block timestamp
    pub timestamp: DateTime<Utc>,

    /// Chain ID
    pub chain_id: u32,

    /// Number of transactions in block
    pub transaction_count: u32,

    /// Gas used in block
    pub gas_used: u64,

    /// Gas limit for block
    pub gas_limit: u64,

    /// Block processing status
    pub processed: bool,

    /// Processing timestamp
    pub processed_at: Option<DateTime<Utc>>,
}

/// Event data for indexing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Event {
    /// Event ID
    pub id: Uuid,

    /// Block number
    pub block_number: u64,

    /// Transaction hash
    pub transaction_hash: String,

    /// Log index within transaction
    pub log_index: u32,

    /// Contract address that emitted the event
    pub contract_address: String,

    /// Event signature (topic0)
    pub event_signature: String,

    /// Event topics
    pub topics: Vec<String>,

    /// Event data (hex encoded)
    pub data: String,

    /// Chain ID
    pub chain_id: u32,

    /// Event timestamp
    pub created_at: DateTime<Utc>,

    /// Decoded event data (if available)
    pub decoded_data: Option<serde_json::Value>,
}

/// Query filter for opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityFilter {
    /// Filter by opportunity type
    pub opportunity_type: Option<String>,

    /// Filter by chain ID
    pub chain_id: Option<u32>,

    /// Filter by protocol
    pub protocol: Option<String>,

    /// Minimum profit threshold
    pub min_profit: Option<String>,

    /// Maximum profit threshold  
    pub max_profit: Option<String>,

    /// Minimum confidence score
    pub min_confidence: Option<f64>,

    /// Filter by execution status
    pub executed: Option<bool>,

    /// Start time for filtering
    pub start_time: Option<DateTime<Utc>>,

    /// End time for filtering
    pub end_time: Option<DateTime<Utc>>,

    /// Limit number of results
    pub limit: Option<u32>,

    /// Offset for pagination
    pub offset: Option<u32>,
}

/// Storage tier for data placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    /// Hot storage - <1ms access, limited capacity
    Hot,
    /// Warm storage - <10ms access, high capacity
    Warm,
    /// Cold storage - <100ms access, unlimited capacity
    Cold,
}

/// Cache policy for data retention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Never cache
    None,
    /// Cache for short duration (1 minute)
    Short,
    /// Cache for medium duration (10 minutes)
    Medium,
    /// Cache for long duration (1 hour)
    Long,
    /// Cache permanently until evicted
    Permanent,
}

/// Data compression type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// LZ4 compression (fast)
    Lz4,
    /// Gzip compression (balanced)
    Gzip,
    /// Zstd compression (high ratio)
    Zstd,
}

/// Storage operation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Operation type
    pub operation: String,

    /// Storage tier used
    pub tier: StorageTier,

    /// Operation duration in microseconds
    pub duration_us: u64,

    /// Data size in bytes
    pub data_size: u64,

    /// Whether operation was successful
    pub success: bool,

    /// Error code if operation failed
    pub error_code: Option<u16>,

    /// Timestamp of operation
    pub timestamp: DateTime<Utc>,
}

impl Default for OpportunityFilter {
    fn default() -> Self {
        Self {
            opportunity_type: None,
            chain_id: None,
            protocol: None,
            min_profit: None,
            max_profit: None,
            min_confidence: None,
            executed: None,
            start_time: None,
            end_time: None,
            limit: Some(100),
            offset: Some(0),
        }
    }
}

impl Transaction {
    /// Create a new transaction
    #[must_use]
    pub fn new(
        chain_id: u32,
        block_number: u64,
        tx_hash: String,
        from_address: String,
        to_address: Option<String>,
        value: String,
        gas_price: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            chain_id,
            block_number,
            tx_hash,
            from_address,
            to_address,
            value,
            gas_price,
            gas_used: None,
            status: 1,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

impl Opportunity {
    /// Create a new opportunity
    #[must_use]
    pub fn new(
        opportunity_type: String,
        chain_id: u32,
        profit_eth: String,
        gas_cost: String,
        net_profit: String,
        confidence_score: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            opportunity_type,
            chain_id,
            protocol: None,
            profit_eth,
            gas_cost,
            net_profit,
            confidence_score,
            executed: false,
            execution_tx_hash: None,
            created_at: Utc::now(),
            expires_at: None,
            data: serde_json::Value::Null,
        }
    }
}
