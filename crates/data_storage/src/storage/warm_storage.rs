//! Warm Storage Implementation using PostgreSQL/TimescaleDB
//!
//! Storage tier optimized for analytical queries and historical data with <10ms latency.
//! Uses `PostgreSQL` with `TimescaleDB` extension for time-series data optimization.

#[cfg(feature = "warm-storage")]
use chrono::DateTime;
use chrono::Utc;
#[cfg(feature = "warm-storage")]
use deadpool_postgres::{Config, Pool, Runtime};
#[cfg(feature = "warm-storage")]
use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;
#[cfg(feature = "warm-storage")]
use tokio_postgres::NoTls;
#[cfg(feature = "warm-storage")]
use uuid::Uuid;

use crate::{
    config::WarmStorageConfig,
    error::DataStorageResult,
    types::{
        Block, Event, Opportunity, OpportunityFilter, StorageMetrics, StorageTier, Transaction,
    },
};

#[cfg(feature = "warm-storage")]
use crate::{config::EnvironmentMode, error::DataStorageError};

/// Warm storage implementation using `PostgreSQL`
///
/// This storage provides analytical capabilities and historical data access
/// with <10ms latency for complex queries.
#[derive(Debug)]
#[allow(dead_code)] // Fields will be used in full implementation
pub struct WarmStorage {
    #[cfg(feature = "warm-storage")]
    /// `PostgreSQL` connection pool
    pool: Arc<Pool>,

    #[cfg(feature = "warm-storage")]
    /// Read replica pools (if configured)
    read_pools: Vec<Arc<Pool>>,

    /// Configuration
    config: WarmStorageConfig,

    /// Operation metrics
    metrics: Arc<parking_lot::Mutex<Vec<StorageMetrics>>>,

    /// Fallback mode indicator
    is_fallback_mode: bool,
}

impl WarmStorage {
    /// Create a new warm storage instance
    ///
    /// # Arguments
    ///
    /// * `config` - Warm storage configuration
    ///
    /// # Errors
    ///
    /// Returns error if database connection fails
    pub fn new(config: &WarmStorageConfig) -> DataStorageResult<Self> {
        #[cfg(feature = "warm-storage")]
        {
            // Environment-aware connection handling
            match config.environment_mode {
                EnvironmentMode::Production => {
                    // Production requires database connection
                    Self::create_production_storage(config)
                }
                EnvironmentMode::Development | EnvironmentMode::Testing | EnvironmentMode::Ci => {
                    // Non-production environments allow graceful degradation
                    Self::create_with_fallback(config)
                }
            }
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation without PostgreSQL
            Ok(Self {
                config: config.clone(),
                metrics: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(1000))),
                is_fallback_mode: true,
            })
        }
    }

    #[cfg(feature = "warm-storage")]
    /// Create production storage - requires database connection
    fn create_production_storage(config: &WarmStorageConfig) -> DataStorageResult<Self> {
        // Create PostgreSQL connection pool using deadpool-postgres
        let mut pg_config = Config::new();
        pg_config.url = Some(config.database_url.clone());
        pg_config.pool = Some(deadpool_postgres::PoolConfig::new(
            config.max_connections as usize,
        ));

        let pool = pg_config
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| {
                tracing::error!("PRODUCTION: PostgreSQL connection required but failed: {e}");
                DataStorageError::database(
                    "create_pool",
                    format!("Production PostgreSQL connection failed: {e}"),
                )
            })?;

        // For now, skip read replicas in simplified implementation
        let read_pools = Vec::with_capacity(0);

        let storage = Self {
            pool: Arc::new(pool),
            read_pools,
            config: config.clone(),
            metrics: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(1000))),
            is_fallback_mode: false,
        };

        // Skip migrations for now in simplified implementation
        // storage.run_migrations().await?;

        Ok(storage)
    }

    #[cfg(feature = "warm-storage")]
    /// Create storage with fallback for non-production environments
    fn create_with_fallback(config: &WarmStorageConfig) -> DataStorageResult<Self> {
        if config.enable_fallback_mode {
            // Try to create PostgreSQL connection, but fallback if it fails
            let mut pg_config = Config::new();
            pg_config.url = Some(config.database_url.clone());
            pg_config.pool = Some(deadpool_postgres::PoolConfig::new(
                config.max_connections as usize,
            ));

            match pg_config.create_pool(Some(Runtime::Tokio1), NoTls) {
                Ok(pool) => {
                    tracing::info!(
                        "PostgreSQL connection successful in {} mode",
                        match config.environment_mode {
                            EnvironmentMode::Development => "development",
                            EnvironmentMode::Testing => "testing",
                            EnvironmentMode::Ci => "CI",
                            EnvironmentMode::Production => "production", // Should not reach here
                        }
                    );

                    let read_pools = Vec::with_capacity(0);
                    Ok(Self {
                        pool: Arc::new(pool),
                        read_pools,
                        config: config.clone(),
                        metrics: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(1000))),
                        is_fallback_mode: false,
                    })
                }
                Err(e) => {
                    tracing::warn!(
                        "PostgreSQL not available in {} mode, using simplified storage: {e}",
                        match config.environment_mode {
                            EnvironmentMode::Development => "development",
                            EnvironmentMode::Testing => "testing",
                            EnvironmentMode::Ci => "CI",
                            EnvironmentMode::Production => "production", // Should not reach here
                        }
                    );

                    // Fallback to simplified mode - create dummy pool for structure compatibility
                    let dummy_config = Config::new();
                    let dummy_pool = dummy_config
                        .create_pool(Some(Runtime::Tokio1), NoTls)
                        .map_err(|e| DataStorageError::database("dummy_pool", e.to_string()))?;

                    Ok(Self {
                        pool: Arc::new(dummy_pool),
                        read_pools: Vec::with_capacity(0),
                        config: config.clone(),
                        metrics: Arc::new(parking_lot::Mutex::new(Vec::with_capacity(1000))),
                        is_fallback_mode: true,
                    })
                }
            }
        } else {
            // Fallback disabled, require connection
            Self::create_production_storage(config)
        }
    }

    /// Run database migrations (placeholder)
    #[allow(dead_code)] // Will be used in full implementation
    fn run_migrations() {
        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL migrations
            tracing::warn!("PostgreSQL migrations not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            tracing::info!("Warm storage migrations skipped (simplified mode)");
        }
    }

    /// Create database indexes (placeholder)
    #[allow(dead_code)] // Will be used in full implementation
    fn create_indexes() {
        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL index creation
            tracing::warn!("PostgreSQL index creation not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            tracing::info!("Warm storage index creation skipped (simplified mode)");
        }
    }

    /// Setup `TimescaleDB` (placeholder)
    #[allow(dead_code)] // Will be used in full implementation
    fn setup_timescale() {
        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement TimescaleDB setup
            tracing::warn!("TimescaleDB setup not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            tracing::info!("TimescaleDB setup skipped (simplified mode)");
        }
    }

    /// Store a transaction
    /// Store transaction in warm storage
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_transaction(
        &self,
        #[cfg_attr(feature = "warm-storage", allow(unused_variables))] transaction: &Transaction,
    ) -> DataStorageResult<()> {
        let start = Instant::now();

        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL transaction storage
            tracing::warn!("PostgreSQL transaction storage not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - just log for now
            let transaction_id = &transaction.id;
            tracing::info!("Storing transaction {} (simplified mode)", transaction_id);
        }

        let duration = start.elapsed();
        self.record_metric("store_transaction", duration, 0, true);

        Ok(())
    }

    /// Store an opportunity
    /// Store opportunity in warm storage
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    #[cfg_attr(not(feature = "warm-storage"), allow(clippy::unused_async))]
    pub async fn store_opportunity(&self, opportunity: &Opportunity) -> DataStorageResult<()> {
        let start = Instant::now();

        #[cfg(feature = "warm-storage")]
        {
            let client = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get connection: {e}"))
            })?;

            let stmt = client
                .prepare_cached(
                    "INSERT INTO opportunities (
                    id, opportunity_type, chain_id, protocol, profit_eth, gas_cost, net_profit,
                    confidence_score, executed, execution_tx_hash, created_at, expires_at, data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
                )
                .await
                .map_err(|e| DataStorageError::database("prepare_statement", e.to_string()))?;

            client
                .execute(
                    &stmt,
                    &[
                        &opportunity.id.to_string(),
                        &opportunity.opportunity_type,
                        &i32::try_from(opportunity.chain_id).map_err(|e| {
                            DataStorageError::database("chain_id_conversion", e.to_string())
                        })?,
                        &opportunity.protocol,
                        &opportunity.profit_eth,
                        &opportunity.gas_cost,
                        &opportunity.net_profit,
                        &opportunity.confidence_score,
                        &opportunity.executed,
                        &opportunity.execution_tx_hash,
                        &opportunity.created_at.timestamp(),
                        &opportunity.expires_at.map(|dt| dt.timestamp()),
                        &serde_json::to_string(&opportunity.data)
                            .map_err(|e| DataStorageError::serialization(e.to_string()))?,
                    ],
                )
                .await
                .map_err(|e| DataStorageError::database("insert_opportunity", e.to_string()))?;

            tracing::debug!("Stored opportunity {} in PostgreSQL", opportunity.id);
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - just log for now
            tracing::info!("Storing opportunity {} (simplified mode)", opportunity.id);
        }

        let duration = start.elapsed();
        self.record_metric("store_opportunity", duration, 0, true);

        Ok(())
    }

    /// Query opportunities with filters
    /// Query opportunities from warm storage
    ///
    /// # Errors
    ///
    /// Returns error if query operation fails
    #[cfg_attr(not(feature = "warm-storage"), allow(clippy::unused_async))]
    pub async fn query_opportunities(
        &self,
        #[cfg_attr(not(feature = "warm-storage"), allow(unused_variables))]
        filter: &OpportunityFilter,
    ) -> DataStorageResult<Vec<Opportunity>> {
        let start = Instant::now();

        #[cfg(feature = "warm-storage")]
        {
            let client = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get connection: {e}"))
            })?;

            // Build dynamic query based on filter - simplified approach
            let mut query = "SELECT id, opportunity_type, chain_id, protocol, profit_eth, gas_cost, net_profit, confidence_score, executed, execution_tx_hash, created_at, expires_at, data FROM opportunities WHERE 1=1".to_string();

            // For now, use a simple approach without dynamic parameters
            // In production, you'd want proper parameter binding
            if let Some(ref opp_type) = filter.opportunity_type {
                let escaped_type: String = opp_type.replace('\'', "''");
                write!(&mut query, " AND opportunity_type = '{escaped_type}'")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            if let Some(chain_id) = filter.chain_id {
                write!(&mut query, " AND chain_id = {chain_id}")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            if let Some(ref protocol) = filter.protocol {
                let escaped_protocol: String = protocol.replace('\'', "''");
                write!(&mut query, " AND protocol = '{escaped_protocol}'")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            if let Some(executed) = filter.executed {
                write!(&mut query, " AND executed = {executed}")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            // Add ordering and limits
            query.push_str(" ORDER BY created_at DESC");

            if let Some(limit) = filter.limit {
                write!(&mut query, " LIMIT {limit}")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            if let Some(offset) = filter.offset {
                write!(&mut query, " OFFSET {offset}")
                    .map_err(|e| DataStorageError::database("query_build", e.to_string()))?;
            }

            let rows = client
                .query(&query, &[])
                .await
                .map_err(|e| DataStorageError::database("execute_query", e.to_string()))?;

            let mut opportunities = Vec::with_capacity(rows.len());

            for row in rows {
                let id_str: String = row.get(0);
                let id = Uuid::parse_str(&id_str)
                    .map_err(|e| DataStorageError::database("parse_uuid", e.to_string()))?;

                let created_timestamp: i64 = row.get(10);
                let created_at =
                    DateTime::from_timestamp(created_timestamp, 0).ok_or_else(|| {
                        DataStorageError::database(
                            "parse_timestamp",
                            "Invalid timestamp".to_string(),
                        )
                    })?;

                let expires_timestamp: Option<i64> = row.get(11);
                let expires_at = expires_timestamp.and_then(|ts| DateTime::from_timestamp(ts, 0));

                let data_str: String = row.get(12);
                let data = serde_json::from_str(&data_str)
                    .map_err(|e| DataStorageError::serialization(e.to_string()))?;

                let opportunity = Opportunity {
                    id,
                    opportunity_type: row.get(1),
                    chain_id: u32::try_from(row.get::<_, i32>(2)).map_err(|e| {
                        DataStorageError::database("chain_id_conversion", e.to_string())
                    })?,
                    protocol: row.get(3),
                    profit_eth: row.get(4),
                    gas_cost: row.get(5),
                    net_profit: row.get(6),
                    confidence_score: row.get(7),
                    executed: row.get(8),
                    execution_tx_hash: row.get(9),
                    created_at,
                    expires_at,
                    data,
                };

                opportunities.push(opportunity);
            }

            let duration = start.elapsed();
            self.record_metric("query_opportunities", duration, opportunities.len(), true);

            Ok(opportunities)
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - return empty results for now
            let _ = filter; // Suppress unused variable warning
            tracing::info!("Querying opportunities with filter (simplified mode)");
            let opportunities = Vec::new();

            let duration = start.elapsed();
            self.record_metric("query_opportunities", duration, opportunities.len(), true);

            Ok(opportunities)
        }
    }

    /// Store a block
    /// Store block in warm storage
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_block(
        &self,
        #[cfg_attr(feature = "warm-storage", allow(unused_variables))] block: &Block,
    ) -> DataStorageResult<()> {
        let start = Instant::now();

        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL block storage
            tracing::warn!("PostgreSQL block storage not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - just log for now
            let block_number = block.number;
            tracing::info!("Storing block {} (simplified mode)", block_number);
        }

        let duration = start.elapsed();
        self.record_metric("store_block", duration, 0, true);

        Ok(())
    }

    /// Store an event
    /// Store event in warm storage
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_event(
        &self,
        #[cfg_attr(feature = "warm-storage", allow(unused_variables))] event: &Event,
    ) -> DataStorageResult<()> {
        let start = Instant::now();

        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL event storage
            tracing::warn!("PostgreSQL event storage not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - just log for now
            let event_id = &event.id;
            tracing::info!("Storing event {} (simplified mode)", event_id);
        }

        let duration = start.elapsed();
        self.record_metric("store_event", duration, 0, true);

        Ok(())
    }

    /// Get storage metrics
    /// Get storage metrics
    ///
    /// # Errors
    ///
    /// Returns error if metrics collection fails
    pub fn get_metrics(&self) -> DataStorageResult<Vec<StorageMetrics>> {
        let metrics = self.metrics.lock().clone();
        Ok(metrics)
    }

    /// Perform health check
    /// Perform health check on warm storage
    ///
    /// # Errors
    ///
    /// Returns error if health check fails
    pub fn health_check(&self) -> DataStorageResult<()> {
        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL health check
            tracing::warn!("PostgreSQL health check not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            // Simplified implementation - always healthy
            tracing::info!("Warm storage health check (simplified mode): OK");
        }

        Ok(())
    }

    /// Shutdown storage gracefully
    /// Shutdown warm storage gracefully
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub fn shutdown(&self) -> DataStorageResult<()> {
        #[cfg(feature = "warm-storage")]
        {
            // TODO: Implement PostgreSQL shutdown
            tracing::info!("PostgreSQL shutdown not yet implemented");
        }

        #[cfg(not(feature = "warm-storage"))]
        {
            tracing::info!("Warm storage shutdown completed (simplified mode)");
        }

        Ok(())
    }

    #[cfg(feature = "warm-storage")]
    /// Get read pool (round-robin selection)
    #[allow(dead_code)] // Will be used in full implementation
    fn get_read_pool(&self) -> &Arc<Pool> {
        if self.read_pools.is_empty() {
            &self.pool
        } else {
            // Simple round-robin selection
            let index = fastrand::usize(0..self.read_pools.len());
            &self.read_pools[index]
        }
    }

    /// Record operation metrics
    fn record_metric(
        &self,
        operation: &str,
        duration: std::time::Duration,
        data_size: usize,
        success: bool,
    ) {
        let metric = StorageMetrics {
            operation: operation.to_string(),
            tier: StorageTier::Warm,
            duration_us: u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
            data_size: u64::try_from(data_size).unwrap_or(0),
            success,
            error_code: if success { None } else { Some(1100) },
            timestamp: Utc::now(),
        };

        let mut metrics = self.metrics.lock();
        metrics.push(metric);

        // Keep only last 1000 metrics
        if metrics.len() > 1000 {
            metrics.drain(0..500);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_warm_storage_creation() -> Result<(), Box<dyn std::error::Error>> {
        // This test would require a real PostgreSQL instance
        // In practice, you'd use testcontainers or similar for integration tests
        let config = WarmStorageConfig::default();
        let _storage = WarmStorage::new(&config)?;
        Ok(())
    }
}
