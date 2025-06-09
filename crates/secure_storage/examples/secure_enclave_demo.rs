//! # Secure Enclave Integration Demo
//!
//! Comprehensive demonstration of `TallyIO`'s Secure Enclave integration
//! with HSM and MPC systems for financial applications.

use secure_storage::secure_enclave::{
    integration::{
        HsmEnclaveIntegration, HsmIntegrationConfig, MpcEnclaveIntegration, MpcIntegrationConfig,
    },
    EnclaveConfig, EnclaveOperation, IntegrationMode, SecureEnclaveSystem,
};
use secure_storage::types::KeyId;
use std::sync::Arc;
use std::time::{Duration, Instant};

const ITERATIONS: usize = 100;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 TallyIO Secure Enclave Integration Demo");
    println!("==========================================\n");

    // Demo 1: Basic Enclave Operations
    demo_basic_enclave_operations().await?;

    // Demo 2: HSM Integration
    demo_hsm_integration().await?;

    // Demo 3: MPC Integration
    demo_mpc_integration().await?;

    // Demo 4: Performance Benchmarks
    demo_performance_benchmarks().await?;

    // Demo 5: Error Handling and Recovery
    demo_error_handling().await?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demo 1: Basic Enclave Operations
async fn demo_basic_enclave_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Demo 1: Basic Enclave Operations");
    println!("-----------------------------------");

    // Initialize enclave system
    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    println!("✅ Enclave system initialized");

    // Key generation
    let key_result = enclave_system
        .execute_secure_operation(
            EnclaveOperation::KeyGeneration,
            "demo_key_001".to_string(),
            || {
                // Simulate key generation
                let key = vec![0x0042_u8; 32]; // 256-bit key
                Ok(key)
            },
        )
        .await?;

    println!(
        "🔑 Key generated in {}μs",
        key_result.execution_time_ns / 1000
    );
    println!("   Key length: {} bytes", key_result.result.len());

    // Digital signature
    let signature_result = enclave_system
        .execute_secure_operation(
            EnclaveOperation::DigitalSignature,
            "demo_signature_001".to_string(),
            || {
                // Simulate digital signature
                let signature = vec![0x00AB_u8; 64]; // 512-bit signature
                Ok(signature)
            },
        )
        .await?;

    println!(
        "✍️  Message signed in {}μs",
        signature_result.execution_time_ns / 1000
    );
    println!(
        "   Signature length: {} bytes",
        signature_result.result.len()
    );

    // Hash computation
    let hash_result = enclave_system
        .execute_secure_operation(
            EnclaveOperation::Hashing,
            "demo_hash_001".to_string(),
            || {
                // Simulate hash computation
                let hash = vec![0x00CD_u8; 32]; // SHA-256 hash
                Ok(hash)
            },
        )
        .await?;

    println!(
        "🔢 Hash computed in {}μs",
        hash_result.execution_time_ns / 1000
    );

    // Display statistics
    let stats = enclave_system.get_stats();
    println!("\n📊 Enclave Statistics:");
    println!("   Platform: {:?}", stats.platform);
    println!("   Total operations: {}", stats.operations_total);
    println!("   Successful operations: {}", stats.operations_successful);
    println!(
        "   Average execution time: {}μs",
        stats.average_execution_time_ns / 1000
    );

    println!("\n");
    Ok(())
}

/// Demo 2: HSM Integration
async fn demo_hsm_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Demo 2: HSM Integration");
    println!("--------------------------");

    // Create enclave system with HSM integration
    let config = EnclaveConfig::new_development()?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(config, IntegrationMode::EnclaveHsm).await?,
    );

    println!("✅ Enclave system with HSM integration initialized");

    // Configure HSM integration
    let hsm_config = HsmIntegrationConfig {
        hsm_key_generation: true,
        hsm_signing: true,
        enclave_fallback: true,
        hsm_timeout: Duration::from_millis(100),
        hsm_key_size_threshold: 2048,
    };

    let hsm_integration = HsmEnclaveIntegration::new(enclave_system, hsm_config)?;

    // Generate keys of different sizes
    let small_key_id = KeyId::new("small_key_1024".to_string());
    let large_key_id = KeyId::new("large_key_4096".to_string());

    // Small key (will use enclave)
    let start = Instant::now();
    let small_key = hsm_integration.generate_key(&small_key_id, 1024).await?;
    let small_key_time = start.elapsed();

    println!("🔑 Small key (1024-bit) generated in {small_key_time:?}");
    println!("   Key length: {} bytes", small_key.len());

    // Large key (will attempt HSM, fallback to enclave)
    let start = Instant::now();
    let large_key = hsm_integration.generate_key(&large_key_id, 4096).await?;
    let large_key_time = start.elapsed();

    println!("🔑 Large key (4096-bit) generated in {large_key_time:?}");
    println!("   Key length: {} bytes", large_key.len());

    // Sign data with both keys
    let message = b"HSM integration test message";

    let small_signature = hsm_integration.sign_data(&small_key_id, message).await?;
    let large_signature = hsm_integration.sign_data(&large_key_id, message).await?;

    println!("✍️  Small key signature: {} bytes", small_signature.len());
    println!("✍️  Large key signature: {} bytes", large_signature.len());

    // Display integration statistics
    let stats = hsm_integration.get_integration_stats();
    println!("\n📊 HSM Integration Statistics:");
    println!("   HSM available: {}", stats.hsm_available);
    println!("   Enclave available: {}", stats.enclave_available);
    println!("   Fallback enabled: {}", stats.fallback_enabled);

    println!("\n");
    Ok(())
}

/// Demo 3: MPC Integration
async fn demo_mpc_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤝 Demo 3: MPC Integration");
    println!("--------------------------");

    // Create enclave system with MPC integration
    let config = EnclaveConfig::new_development()?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(config, IntegrationMode::EnclaveMpc).await?,
    );

    println!("✅ Enclave system with MPC integration initialized");

    // Configure MPC integration
    let mpc_config = MpcIntegrationConfig {
        threshold: 3,
        total_parties: 5,
        mpc_threshold_signatures: true,
        mpc_key_generation: true,
        mpc_timeout: Duration::from_millis(500),
        mpc_value_threshold: 1_000_000,
    };

    let mut mpc_integration = MpcEnclaveIntegration::new(enclave_system, mpc_config)?;

    // Initialize MPC system
    mpc_integration.initialize_mpc()?;
    println!("✅ MPC system initialized");

    // Test threshold signatures with different values
    let key_id = KeyId::new("multisig_key_001".to_string());
    let message = b"Multi-party computation test";

    // Low value transaction (will use enclave)
    let start = Instant::now();
    let low_value_sig = mpc_integration
        .threshold_sign(&key_id, message, 500_000)
        .await?;
    let low_value_time = start.elapsed();

    println!(
        "✍️  Low value signature ({} units) in {:?}",
        500_000_u64, low_value_time
    );
    println!("   Signature length: {} bytes", low_value_sig.len());

    // High value transaction (will use MPC)
    let start = Instant::now();
    let high_value_sig = mpc_integration
        .threshold_sign(&key_id, message, 5_000_000_u64)
        .await?;
    let high_value_time = start.elapsed();

    println!(
        "✍️  High value signature ({} units) in {:?}",
        5_000_000_u64, high_value_time
    );
    println!("   Signature length: {} bytes", high_value_sig.len());

    // Display MPC statistics
    let stats = mpc_integration.get_mpc_stats();
    println!("\n📊 MPC Integration Statistics:");
    println!("   MPC available: {}", stats.mpc_available);
    println!("   Enclave available: {}", stats.enclave_available);
    println!("   Threshold: {}/{}", stats.threshold, stats.total_parties);

    println!("\n");
    Ok(())
}

/// Demo 4: Performance Benchmarks
async fn demo_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Demo 4: Performance Benchmarks");
    println!("---------------------------------");

    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Benchmark different operations
    let operations = vec![
        ("Key Generation", EnclaveOperation::KeyGeneration),
        ("Digital Signature", EnclaveOperation::DigitalSignature),
        ("Hash Computation", EnclaveOperation::Hashing),
        ("Encryption", EnclaveOperation::Cryptography),
        ("Random Generation", EnclaveOperation::RandomGeneration),
    ];

    for (name, operation) in operations {
        let start = Instant::now();

        for i in 0..ITERATIONS {
            let _ = enclave_system
                .execute_secure_operation(
                    operation.clone(),
                    format!("bench_{name}_{i}"),
                    move || Ok(vec![u8::try_from(i).unwrap_or(0_u8); 32]),
                )
                .await?;
        }

        let total_time = start.elapsed();
        let avg_time = total_time / u32::try_from(ITERATIONS).unwrap_or(1_u32);

        println!("📈 {name}: {ITERATIONS} iterations in {total_time:?}");
        println!("   Average: {avg_time:?} per operation");
        println!(
            "   Throughput: {:.0} ops/sec",
            1.0_f64 / avg_time.as_secs_f64()
        );
    }

    // Critical path performance test
    println!("\n🎯 Critical Path Performance Test:");
    let start = Instant::now();
    let _result = enclave_system.execute_critical_operation(
        EnclaveOperation::Hashing,
        "critical_test",
        || Ok(vec![0x00FF_u8; 32]),
    )?;
    let elapsed = start.elapsed();

    println!("   Critical operation completed in {elapsed:?}");
    println!(
        "   Target: <1ms, Achieved: {:.2}ms",
        elapsed.as_secs_f64() * 1_000.0_f64
    );

    if elapsed.as_millis() <= 1 {
        println!("   ✅ Performance target met!");
    } else {
        println!("   ⚠️  Performance target missed (test environment)");
    }

    println!("\n");
    Ok(())
}

/// Demo 5: Error Handling and Recovery
async fn demo_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  Demo 5: Error Handling and Recovery");
    println!("---------------------------------------");

    let config = EnclaveConfig::new_development()?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;

    // Test operation that fails
    println!("🔥 Testing error handling...");
    let result: Result<secure_storage::secure_enclave::EnclaveResult<Vec<u8>>, _> = enclave_system
        .execute_secure_operation(
            EnclaveOperation::KeyGeneration,
            "failing_operation".to_string(),
            || {
                Err(secure_storage::error::SecureStorageError::InvalidInput {
                    field: "test".to_string(),
                    reason: "Intentional test failure".to_string(),
                })
            },
        )
        .await;

    match result {
        Ok(_) => println!("   ❌ Expected failure but got success"),
        Err(e) => println!("   ✅ Error handled correctly: {e}"),
    }

    // Test recovery after failure
    println!("🔄 Testing recovery after failure...");
    let recovery_result = enclave_system
        .execute_secure_operation(
            EnclaveOperation::Hashing,
            "recovery_operation".to_string(),
            || Ok(vec![0x00AA_u8; 32]),
        )
        .await?;

    println!(
        "   ✅ Recovery successful in {}μs",
        recovery_result.execution_time_ns / 1000
    );

    // Test circuit breaker behavior
    println!("⚡ Testing circuit breaker...");

    // Simulate multiple failures to trigger circuit breaker
    for i in 0_i32..3_i32 {
        let _: Result<secure_storage::secure_enclave::EnclaveResult<Vec<u8>>, _> = enclave_system
            .execute_secure_operation(
                EnclaveOperation::KeyGeneration,
                format!("circuit_test_{i}"),
                || {
                    Err(secure_storage::error::SecureStorageError::InvalidInput {
                        field: "circuit_test".to_string(),
                        reason: "Circuit breaker test".to_string(),
                    })
                },
            )
            .await;
    }

    // Test that system still works
    let _final_result = enclave_system
        .execute_secure_operation(EnclaveOperation::Hashing, "final_test".to_string(), || {
            Ok(vec![0x00BB_u8; 32])
        })
        .await?;

    println!("   ✅ System operational after circuit breaker test");

    // Final statistics
    let stats = enclave_system.get_stats();
    println!("\n📊 Final Statistics:");
    println!("   Total operations: {}", stats.operations_total);
    println!("   Successful operations: {}", stats.operations_successful);
    println!("   Failed operations: {}", stats.operations_failed);
    let success_rate = if stats.operations_total > 0 {
        f64::from(u32::try_from(stats.operations_successful).unwrap_or(0_u32))
            / f64::from(u32::try_from(stats.operations_total).unwrap_or(1_u32))
            * 100.0_f64
    } else {
        0.0_f64
    };
    println!("   Success rate: {success_rate:.1}%");

    println!("\n");
    Ok(())
}
