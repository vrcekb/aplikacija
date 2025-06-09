//! Vault implementation tests

#![allow(clippy::unwrap_used)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::uninlined_format_args)]

use secure_storage::error::SecureStorageResult;
use secure_storage::vault::{local_vault::LocalVault, Vault, VaultMetadata};
use tempfile::tempdir;

#[tokio::test]
async fn test_local_vault_basic_operations() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Test store and retrieve
    let key = "test_key";
    let value = b"test_value_data";

    vault.store(key, value).await?;
    let retrieved = vault.retrieve(key).await?;
    assert_eq!(value, retrieved.as_slice());

    // Test exists
    assert!(vault.exists(key).await?);
    assert!(!vault.exists("non_existent").await?);

    // Test delete
    vault.delete(key).await?;
    assert!(!vault.exists(key).await?);

    Ok(())
}

#[tokio::test]
async fn test_local_vault_list_keys() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Store multiple keys
    vault.store("prefix/key1", b"value1").await?;
    vault.store("prefix/key2", b"value2").await?;
    vault.store("other/key3", b"value3").await?;

    // List keys with prefix
    let keys = vault.list_keys("prefix/").await?;
    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&"prefix/key1".to_string()));
    assert!(keys.contains(&"prefix/key2".to_string()));

    // List all keys
    let all_keys = vault.list_keys("").await?;
    assert_eq!(all_keys.len(), 3);

    Ok(())
}

#[tokio::test]
async fn test_local_vault_metadata() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    let key = "metadata_test";
    let value = b"test_data_with_metadata";

    // Create metadata
    let mut metadata = VaultMetadata::new("application/json".to_string(), value.len());
    metadata.add_tag("environment".to_string(), "test".to_string());
    metadata.add_tag("version".to_string(), "1.0".to_string());

    // Store with metadata
    vault
        .store_with_metadata(key, value, metadata.clone())
        .await?;

    // Retrieve metadata
    let retrieved_metadata = vault.get_metadata(key).await?;
    assert_eq!(retrieved_metadata.content_type, "application/json");
    assert_eq!(retrieved_metadata.size, value.len());
    assert_eq!(
        retrieved_metadata.get_tag("environment"),
        Some(&"test".to_string())
    );
    assert_eq!(
        retrieved_metadata.get_tag("version"),
        Some(&"1.0".to_string())
    );

    Ok(())
}

#[tokio::test]
async fn test_local_vault_batch_operations() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Prepare batch data
    let items = vec![
        ("batch1".to_string(), b"value1".to_vec()),
        ("batch2".to_string(), b"value2".to_vec()),
        ("batch3".to_string(), b"value3".to_vec()),
    ];

    // Batch store
    vault.batch_store(items.clone()).await?;

    // Batch retrieve
    let keys: Vec<String> = items.iter().map(|(k, _)| k.clone()).collect();
    let retrieved = vault.batch_retrieve(keys.clone()).await?;

    assert_eq!(retrieved.len(), 3);
    assert_eq!(retrieved.get("batch1"), Some(&b"value1".to_vec()));
    assert_eq!(retrieved.get("batch2"), Some(&b"value2".to_vec()));
    assert_eq!(retrieved.get("batch3"), Some(&b"value3".to_vec()));

    // Batch delete
    vault.batch_delete(keys).await?;

    // Verify deletion
    assert!(!vault.exists("batch1").await?);
    assert!(!vault.exists("batch2").await?);
    assert!(!vault.exists("batch3").await?);

    Ok(())
}

#[tokio::test]
async fn test_local_vault_health_check() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    let health = vault.health_check().await?;
    assert!(health.status.is_healthy());
    assert!(health.response_time_ms < 1000); // Should be very fast for local vault

    Ok(())
}

#[tokio::test]
async fn test_local_vault_stats() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Store some data
    vault.store("stats_test1", b"data1").await?;
    vault.store("stats_test2", b"data2_longer").await?;

    let stats = vault.get_stats().await?;
    assert_eq!(stats.total_entries, 2);
    assert_eq!(stats.total_size_bytes, 5 + 12); // "data1" + "data2_longer"
    assert!(stats.operations_count > 0);

    Ok(())
}

#[tokio::test]
async fn test_local_vault_performance() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    let key = "performance_test";
    let value = vec![0u8; 1024]; // 1KB of data

    // Measure store performance
    let start = std::time::Instant::now();
    vault.store(key, &value).await?;
    let store_time = start.elapsed();

    // Measure retrieve performance
    let start = std::time::Instant::now();
    let _retrieved = vault.retrieve(key).await?;
    let retrieve_time = start.elapsed();

    // Performance requirements: <20ms for vault operations
    assert!(
        store_time.as_millis() < 20,
        "Store took {}ms",
        store_time.as_millis()
    );
    assert!(
        retrieve_time.as_millis() < 20,
        "Retrieve took {}ms",
        retrieve_time.as_millis()
    );

    Ok(())
}

#[tokio::test]
async fn test_local_vault_concurrent_access() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = std::sync::Arc::new(LocalVault::new(db_path.to_str().unwrap()).await?);

    // Spawn multiple concurrent operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let vault_clone = vault.clone();
        let handle = tokio::spawn(async move {
            let key = format!("concurrent_key_{}", i);
            let value = format!("value_{}", i).into_bytes();

            vault_clone.store(&key, &value).await?;
            let retrieved = vault_clone.retrieve(&key).await?;
            assert_eq!(value, retrieved);

            Ok::<(), secure_storage::error::SecureStorageError>(())
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap()?;
    }

    // Verify all keys exist
    let keys = vault.list_keys("concurrent_key_").await?;
    assert_eq!(keys.len(), 10);

    Ok(())
}

#[tokio::test]
async fn test_local_vault_error_handling() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Test retrieving non-existent key
    let result = vault.retrieve("non_existent_key").await;
    assert!(result.is_err());

    // Test deleting non-existent key
    let result = vault.delete("non_existent_key").await;
    assert!(result.is_err());

    // Test invalid key names
    let result = vault.store("", b"value").await;
    assert!(result.is_err());

    let result = vault.store(&"x".repeat(2000), b"value").await;
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_local_vault_cleanup_expired() -> SecureStorageResult<()> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vault.db");

    let vault = LocalVault::new(db_path.to_str().unwrap()).await?;

    // Store data with expiration
    let mut metadata = VaultMetadata::new("text/plain".to_string(), 10);
    metadata.expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(1)); // Already expired

    vault
        .store_with_metadata("expired_key", b"expired_data", metadata)
        .await?;

    // Store non-expired data
    vault.store("normal_key", b"normal_data").await?;

    // Cleanup expired entries
    let deleted_count = vault.cleanup_expired().await?;
    assert_eq!(deleted_count, 1);

    // Verify expired key is gone but normal key remains
    assert!(!vault.exists("expired_key").await?);
    assert!(vault.exists("normal_key").await?);

    Ok(())
}
