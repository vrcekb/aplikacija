//! Integration tests for `TallyIO` Data Storage
//!
//! Comprehensive integration tests covering all storage tiers and operations.

#![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap for simplicity
#![allow(clippy::expect_used)] // Tests are allowed to use expect for simplicity
#![allow(clippy::panic)] // Tests are allowed to panic
#![allow(clippy::unreadable_literal)] // Test data can have unreadable literals
#![allow(clippy::default_numeric_fallback)] // Test data can have default numeric types
#![allow(clippy::uninlined_format_args)] // Test formatting can be verbose
#![allow(clippy::unused_async)] // Test helper functions can be async

use std::time::Instant;
use uuid::Uuid;

use tallyio_data_storage::{
    config::EnvironmentMode,
    error::{DataStorageError, DataStorageResult},
    types::{Block, Event, Opportunity, OpportunityFilter, Transaction},
    DataStorage, DataStorageConfig,
};

/// Test basic data storage operations
#[tokio::test]
async fn test_basic_storage_operations() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Test opportunity storage
    let opportunity = create_test_opportunity();
    storage.store_opportunity_fast(&opportunity)?;

    // Test opportunity retrieval
    let retrieved = storage.get_opportunity_fast(&opportunity.id)?;
    assert!(retrieved.is_some());
    match retrieved {
        Some(opp) => assert_eq!(opp.id, opportunity.id),
        None => panic!("Failed to retrieve opportunity - this should not happen in tests"),
    }

    // Test transaction storage
    let transaction = create_test_transaction();
    storage.store_transaction(&transaction)?;

    // Test block storage
    let block = create_test_block();
    storage.store_block(&block)?;

    // Test event storage
    let event = create_test_event();
    storage.store_event(&event)?;

    Ok(())
}

/// Test query operations
#[tokio::test]
async fn test_query_operations() -> DataStorageResult<()> {
    let config = create_test_config().await;

    // Try to create storage, but skip test if database is not available
    let storage = match DataStorage::new(config).await {
        Ok(storage) => storage,
        Err(DataStorageError::Database {
            operation: _,
            reason,
        }) if reason.contains("refused") || reason.contains("connection") => {
            println!("Database not available, skipping test");
            return Ok(());
        }
        Err(e) => return Err(e),
    };

    // Store multiple opportunities
    for i in 0..10 {
        let mut opportunity = create_test_opportunity();
        opportunity.opportunity_type = format!("test_type_{}", i % 3);
        opportunity.chain_id = (i % 2) + 1;
        storage.store_opportunity_fast(&opportunity)?;
    }

    // Test basic query
    let filter = OpportunityFilter {
        limit: Some(5),
        offset: Some(0),
        ..Default::default()
    };
    let results = storage.query_opportunities(&filter).await?;
    assert!(results.len() <= 5);

    // Test filtered query
    let filter = OpportunityFilter {
        opportunity_type: Some("test_type_0".to_string()),
        limit: Some(10),
        offset: Some(0),
        ..Default::default()
    };
    let results = storage.query_opportunities(&filter).await?;
    assert!(results.iter().all(|o| o.opportunity_type == "test_type_0"));

    Ok(())
}

/// Test performance requirements
#[tokio::test]
async fn test_performance_requirements() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    let opportunity = create_test_opportunity();

    // Test hot path latency (<1ms)
    let start = Instant::now();
    storage.store_opportunity_fast(&opportunity)?;
    let store_duration = start.elapsed();

    let start = Instant::now();
    let _retrieved = storage.get_opportunity_fast(&opportunity.id)?;
    let get_duration = start.elapsed();

    // Assert <1ms requirement for hot path (1000 microseconds)
    assert!(
        store_duration.as_micros() < 1000,
        "Store too slow: {store_duration:?}"
    );
    assert!(
        get_duration.as_micros() < 1000,
        "Get too slow: {get_duration:?}"
    );

    Ok(())
}

/// Test concurrent operations
#[tokio::test]
async fn test_concurrent_operations() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Create multiple concurrent tasks
    let mut handles = Vec::new();

    for i in 0..100 {
        let storage_clone = storage.clone();
        let handle = tokio::spawn(async move {
            let mut opportunity = create_test_opportunity();
            opportunity.opportunity_type = format!("concurrent_test_{}", i);

            storage_clone.store_opportunity_fast(&opportunity)?;
            let retrieved = storage_clone.get_opportunity_fast(&opportunity.id)?;

            assert!(retrieved.is_some());
            Ok::<(), tallyio_data_storage::error::DataStorageError>(())
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap()?;
    }

    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Test getting non-existent opportunity
    let non_existent_id = Uuid::new_v4();
    let result = storage.get_opportunity_fast(&non_existent_id)?;
    assert!(result.is_none());

    Ok(())
}

/// Test health check
#[tokio::test]
async fn test_health_check() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Health check should pass
    storage.health_check().await?;

    Ok(())
}

/// Test metrics collection
#[tokio::test]
async fn test_metrics_collection() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Perform some operations
    let opportunity = create_test_opportunity();
    storage.store_opportunity_fast(&opportunity)?;
    storage.get_opportunity_fast(&opportunity.id)?;

    // Get metrics
    let metrics = storage.get_metrics()?;
    assert!(!metrics.is_empty());

    Ok(())
}

/// Test graceful shutdown
#[tokio::test]
async fn test_graceful_shutdown() -> DataStorageResult<()> {
    let config = create_test_config().await;
    let storage = DataStorage::new(config).await?;

    // Perform some operations
    let opportunity = create_test_opportunity();
    storage.store_opportunity_fast(&opportunity)?;

    // Shutdown should complete without errors
    storage.shutdown()?;

    Ok(())
}

// Helper functions

async fn create_test_config() -> DataStorageConfig {
    let mut config = DataStorageConfig::default();

    // Use temporary directories for testing with unique names
    let temp_dir = tempfile::tempdir().unwrap();

    // Create unique database name with timestamp to avoid conflicts
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    // Use in-memory storage for ultra-low latency testing
    config.hot_storage.use_memory_storage = true;
    config.hot_storage.database_path = None;
    config.cold_storage.storage_path = temp_dir.path().join(format!("cold_{timestamp}"));
    config.cold_storage.enable_encryption = false; // Disable for testing

    // Use in-memory cache for testing
    config.cache.redis_url = None;
    config.cache.enable_memory_cache = true;

    // PRODUCTION-GRADE: Configure warm storage for testing environment
    config.warm_storage.environment_mode = EnvironmentMode::Testing;
    config.warm_storage.enable_fallback_mode = true;
    config.warm_storage.database_url = "postgresql://test:test@localhost:5432/test_db".to_string();
    config.warm_storage.max_connections = 1;

    config
}

fn create_test_opportunity() -> Opportunity {
    Opportunity::new(
        "arbitrage".to_string(),
        1,
        "1.5".to_string(),
        "0.1".to_string(),
        "1.4".to_string(),
        0.95,
    )
}

fn create_test_transaction() -> Transaction {
    Transaction::new(
        1,
        12345,
        "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
        "0x1234567890abcdef1234567890abcdef12345678".to_string(),
        Some("0xabcdef1234567890abcdef1234567890abcdef12".to_string()),
        "1000000000000000000".to_string(),
        "20000000000".to_string(),
    )
}

fn create_test_block() -> Block {
    use chrono::Utc;

    Block {
        number: 12345,
        hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string(),
        parent_hash: "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            .to_string(),
        timestamp: Utc::now(),
        chain_id: 1,
        transaction_count: 100,
        gas_used: 8000000,
        gas_limit: 15000000,
        processed: false,
        processed_at: None,
    }
}

fn create_test_event() -> Event {
    use chrono::Utc;

    Event {
        id: Uuid::new_v4(),
        block_number: 12345,
        transaction_hash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            .to_string(),
        log_index: 0,
        contract_address: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
        event_signature: "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            .to_string(),
        topics: vec![
            "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef".to_string(),
            "0x0000000000000000000000001234567890abcdef1234567890abcdef12345678".to_string(),
        ],
        data: "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000".to_string(),
        chain_id: 1,
        created_at: Utc::now(),
        decoded_data: None,
    }
}
