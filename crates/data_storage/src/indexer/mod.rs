//! Blockchain indexing implementations
//!
//! Provides indexing for blocks, transactions, and events.

pub mod block_indexer;
pub mod event_indexer;
pub mod transaction_indexer;

pub use block_indexer::BlockIndexer;
pub use event_indexer::EventIndexer;
pub use transaction_indexer::TransactionIndexer;

use crate::error::DataStorageResult;

/// Indexer trait for different blockchain data types
#[async_trait::async_trait]
pub trait Indexer<T>: Send + Sync {
    /// Index data
    async fn index(&self, data: T) -> DataStorageResult<()>;
    
    /// Get indexer name
    fn name(&self) -> &str;
    
    /// Get indexer metrics
    async fn metrics(&self) -> DataStorageResult<IndexerMetrics>;
    
    /// Get current indexing status
    async fn status(&self) -> DataStorageResult<IndexerStatus>;
}

/// Indexer metrics
#[derive(Debug, Clone)]
pub struct IndexerMetrics {
    /// Indexer name
    pub indexer_name: String,
    
    /// Number of items indexed
    pub items_indexed: u64,
    
    /// Number of items failed
    pub items_failed: u64,
    
    /// Current block height
    pub current_block: u64,
    
    /// Indexing rate (items per second)
    pub indexing_rate: f64,
}

/// Indexer status
#[derive(Debug, Clone)]
pub struct IndexerStatus {
    /// Whether indexer is running
    pub is_running: bool,
    
    /// Current block being processed
    pub current_block: u64,
    
    /// Latest block available
    pub latest_block: u64,
    
    /// Blocks behind latest
    pub blocks_behind: u64,
    
    /// Estimated time to catch up
    pub estimated_catchup_time: std::time::Duration,
}

impl IndexerMetrics {
    /// Create new indexer metrics
    pub fn new(indexer_name: String) -> Self {
        Self {
            indexer_name,
            items_indexed: 0,
            items_failed: 0,
            current_block: 0,
            indexing_rate: 0.0,
        }
    }
}
