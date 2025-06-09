//! # Secure Enclave Integration Tests
//!
//! Comprehensive tests for Secure Enclave integration with HSM and MPC systems

use secure_storage::secure_enclave::{
    integration::{
        HsmEnclaveIntegration, HsmIntegrationConfig, MpcEnclaveIntegration, MpcIntegrationConfig,
    },
    EnclaveConfig, EnclavePlatform, IntegrationMode, SecureEnclaveSystem,
};
use secure_storage::types::KeyId;
use std::sync::Arc;

/// Test basic enclave system initialization
#[tokio::test]
async fn test_enclave_system_initialization() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    let stats = enclave_system.get_stats();
    assert_eq!(stats.platform, EnclavePlatform::Simulation);
    assert_eq!(stats.operations_total, 0);

    Ok(())
}

/// Test enclave system with different integration modes
#[tokio::test]
async fn test_integration_modes() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;

    // Test standalone mode
    let standalone =
        SecureEnclaveSystem::new_with_integration(config.clone(), IntegrationMode::Standalone)
            .await?;
    let capabilities = standalone.get_integration_capabilities();
    assert!(!capabilities.has_hsm());
    assert!(!capabilities.has_mpc());

    // Test HSM integration mode
    let hsm_mode =
        SecureEnclaveSystem::new_with_integration(config.clone(), IntegrationMode::EnclaveHsm)
            .await?;
    let capabilities = hsm_mode.get_integration_capabilities();
    assert!(capabilities.has_hsm());
    assert!(!capabilities.has_mpc());

    // Test MPC integration mode
    let mpc_mode =
        SecureEnclaveSystem::new_with_integration(config.clone(), IntegrationMode::EnclaveMpc)
            .await?;
    let capabilities = mpc_mode.get_integration_capabilities();
    assert!(!capabilities.has_hsm());
    assert!(capabilities.has_mpc());

    // Test full integration mode
    let full_mode =
        SecureEnclaveSystem::new_with_integration(config, IntegrationMode::FullIntegration).await?;
    let capabilities = full_mode.get_integration_capabilities();
    assert!(capabilities.has_hsm());
    assert!(capabilities.has_mpc());

    Ok(())
}

/// Test HSM-Enclave integration
#[tokio::test]
async fn test_hsm_enclave_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(config, IntegrationMode::EnclaveHsm).await?,
    );

    let hsm_config = HsmIntegrationConfig::default();
    let hsm_integration = HsmEnclaveIntegration::new(enclave_system, hsm_config)?;

    // Test key generation
    let key_id = KeyId::new("test_key".to_string());
    let key_data = hsm_integration.generate_key(&key_id, 2048).await?;
    assert_eq!(key_data.len(), 256); // 2048 bits = 256 bytes

    // Test signing
    let message = b"test message";
    let signature = hsm_integration.sign_data(&key_id, message).await?;
    assert_eq!(signature.len(), 64); // Expected signature length

    // Test statistics
    let stats = hsm_integration.get_integration_stats();
    assert!(!stats.hsm_available); // HSM not actually initialized
    assert!(stats.enclave_available);
    assert!(stats.fallback_enabled);

    Ok(())
}

/// Test MPC-Enclave integration
#[tokio::test]
async fn test_mpc_enclave_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(config, IntegrationMode::EnclaveMpc).await?,
    );

    let mpc_config = MpcIntegrationConfig::default();
    let mut mpc_integration = MpcEnclaveIntegration::new(enclave_system, mpc_config)?;

    // Initialize MPC system
    mpc_integration.initialize_mpc()?;

    // Test threshold signature
    let key_id = KeyId::new("threshold_key".to_string());
    let message = b"threshold message";
    let value = 2_000_000; // Above threshold

    let signature = mpc_integration
        .threshold_sign(&key_id, message, value)
        .await?;
    assert_eq!(signature.len(), 64);

    // Test statistics
    let stats = mpc_integration.get_mpc_stats();
    assert!(stats.mpc_available);
    assert!(stats.enclave_available);
    assert_eq!(stats.threshold, 3);
    assert_eq!(stats.total_parties, 5);

    Ok(())
}

/// Test circuit breaker functionality
#[tokio::test]
async fn test_circuit_breaker() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Test normal operation
    let result = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::KeyGeneration,
            "test_op_1".to_string(),
            || Ok(vec![0x0042_u8; 32]),
        )
        .await;

    assert!(result.is_ok());
    if let Ok(enclave_result) = result {
        assert_eq!(enclave_result.result, vec![0x0042_u8; 32]);
        assert!(enclave_result.execution_time_ns > 0);
    }

    Ok(())
}

/// Test critical path performance
#[tokio::test]
async fn test_critical_path_performance() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Test critical operation with latency requirement
    let start = std::time::Instant::now();
    let result = enclave_system.execute_critical_operation(
        secure_storage::secure_enclave::EnclaveOperation::Hashing,
        "critical_hash",
        || {
            // Simulate fast hash operation
            Ok(vec![0x00FF_u8; 32])
        },
    );
    let elapsed = start.elapsed();

    // Check if result is ok, if not print the error
    if let Err(ref e) = result {
        println!("Critical operation failed: {e:?}");
    }
    assert!(result.is_ok());

    // Be more lenient with timing in tests due to system overhead
    assert!(
        elapsed.as_millis() <= 10,
        "Critical operation took {}ms, expected ≤10ms",
        elapsed.as_millis()
    );

    Ok(())
}

/// Test batch operations
#[tokio::test]
async fn test_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Execute operations individually since batch operations need same closure type
    let key_result1 = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::KeyGeneration,
            "batch_key_1".to_string(),
            || Ok(vec![0x0001_u8; 32]),
        )
        .await?;

    let key_result2 = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::KeyGeneration,
            "batch_key_2".to_string(),
            || Ok(vec![0x0002_u8; 32]),
        )
        .await?;

    let hash_result3 = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::Hashing,
            "batch_hash_1".to_string(),
            || Ok(vec![0x0003_u8; 32]),
        )
        .await?;

    let operation_results = vec![key_result1, key_result2, hash_result3];
    assert_eq!(operation_results.len(), 3);

    // Verify results
    assert_eq!(operation_results[0].result, vec![0x0001_u8; 32]);
    assert_eq!(operation_results[1].result, vec![0x0002_u8; 32]);
    assert_eq!(operation_results[2].result, vec![0x0003_u8; 32]);

    // Check that all operations completed successfully
    for result in &operation_results {
        assert!(result.execution_time_ns > 0);
    }

    Ok(())
}

/// Test sealed storage integration
#[tokio::test]
async fn test_sealed_storage() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Test storing sealed data
    let test_data = b"sensitive test data";
    enclave_system
        .store_sealed_data("test_sealed_key", test_data)
        .await?;

    // Test retrieving sealed data
    let retrieved_data = enclave_system
        .retrieve_sealed_data("test_sealed_key")
        .await?;
    assert_eq!(retrieved_data, test_data);

    // Verify sealed storage is available
    assert!(enclave_system.has_sealed_storage());

    Ok(())
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Test operation that fails
    let result: Result<secure_storage::secure_enclave::EnclaveResult<Vec<u8>>, _> = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::KeyGeneration,
            "failing_op".to_string(),
            || {
                Err(secure_storage::error::SecureStorageError::InvalidInput {
                    field: "test".to_string(),
                    reason: "Intentional test failure".to_string(),
                })
            },
        )
        .await;

    assert!(result.is_err());

    // Verify that system continues to work after failure
    let success_result = enclave_system
        .execute_secure_operation(
            secure_storage::secure_enclave::EnclaveOperation::Hashing,
            "recovery_op".to_string(),
            || Ok(vec![0x00AA_u8; 32]),
        )
        .await;

    assert!(success_result.is_ok());

    Ok(())
}

/// Performance benchmark test
#[tokio::test]
async fn test_performance_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    const NUM_OPERATIONS: usize = 100;

    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;
    let start = std::time::Instant::now();

    for i in 0..NUM_OPERATIONS {
        let _ = enclave_system
            .execute_secure_operation(
                secure_storage::secure_enclave::EnclaveOperation::Hashing,
                format!("perf_test_{i}"),
                move || Ok(vec![u8::try_from(i).unwrap_or(0_u8); 32]),
            )
            .await?;
    }

    let elapsed = start.elapsed();
    let avg_time_per_op = elapsed / u32::try_from(NUM_OPERATIONS).unwrap_or(1_u32);

    println!("Average time per operation: {avg_time_per_op:?}");

    // Verify performance is reasonable (be lenient for test environment)
    assert!(
        avg_time_per_op.as_millis() < 50,
        "Average operation time {}ms is too slow",
        avg_time_per_op.as_millis()
    );

    // Check final statistics
    let stats = enclave_system.get_stats();
    assert_eq!(stats.operations_total, NUM_OPERATIONS as u64);
    assert_eq!(stats.operations_successful, NUM_OPERATIONS as u64);
    assert_eq!(stats.operations_failed, 0);

    Ok(())
}
