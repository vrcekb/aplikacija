# Secure Enclave Integration - TallyIO

🔐 **Production-ready Secure Enclave integration with HSM and MPC support**

## Overview

The Secure Enclave integration provides hardware-level security guarantees for cryptographic operations in TallyIO's financial platform. It combines Intel SGX, ARM TrustZone, and AMD Memory Guard technologies with HSM and MPC systems for maximum security.

## Features

### 🛡️ **Hardware Security**
- **Intel SGX**: Secure enclaves for x86_64 platforms
- **ARM TrustZone**: Secure world execution on ARM platforms  
- **AMD Memory Guard**: Memory encryption and protection
- **Hardware Attestation**: Remote verification of enclave authenticity
- **Sealed Storage**: Hardware-encrypted persistent storage

### 🔗 **Integration Modes**
- **Standalone**: Enclave-only operations
- **Enclave + HSM**: Hybrid hardware security
- **Enclave + MPC**: Multi-party computation support
- **Full Integration**: All systems combined

### ⚡ **Performance**
- **<1ms latency** for critical path operations
- **Circuit breaker** patterns for fault tolerance
- **Lock-free** data structures for concurrency
- **Memory pools** for zero-allocation hot paths

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  SecureEnclaveSystem  │  HsmEnclaveIntegration  │  MpcIntegration │
├─────────────────────────────────────────────────────────────┤
│         Circuit Breaker         │        Performance Monitor │
├─────────────────────────────────────────────────────────────┤
│  Intel SGX  │  ARM TrustZone  │  AMD Memory Guard  │  Simulation │
├─────────────────────────────────────────────────────────────┤
│              Hardware Security Modules (HSM)                │
│              Multi-Party Computation (MPC)                  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Enclave System

```rust
use secure_storage::secure_enclave::{
    EnclaveConfig, EnclavePlatform, SecureEnclaveSystem
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize enclave system
    let config = EnclaveConfig::new_production(EnclavePlatform::IntelSgx)?;
    let enclave_system = SecureEnclaveSystem::new(config).await?;
    
    // Execute secure operation
    let result = enclave_system.execute_secure_operation(
        EnclaveOperation::KeyGeneration,
        "key_gen_001".to_string(),
        || {
            // Your secure operation here
            Ok(vec![0x42; 32])
        }
    ).await?;
    
    println!("Operation completed in {}μs", result.execution_time_ns / 1000);
    Ok(())
}
```

### HSM Integration

```rust
use secure_storage::secure_enclave::{
    SecureEnclaveSystem, IntegrationMode,
    integration::{HsmEnclaveIntegration, HsmIntegrationConfig}
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create enclave system with HSM integration
    let config = EnclaveConfig::new_production(EnclavePlatform::IntelSgx)?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(
            config, 
            IntegrationMode::EnclaveHsm
        ).await?
    );
    
    // Initialize HSM integration
    let hsm_config = HsmIntegrationConfig::default();
    let mut hsm_integration = HsmEnclaveIntegration::new(
        enclave_system, 
        hsm_config
    )?;
    
    // Generate key using optimal method (HSM or Enclave)
    let key_id = KeyId::new("trading_key_001".to_string());
    let key_data = hsm_integration.generate_key(&key_id, 2048).await?;
    
    // Sign data
    let message = b"transaction_data";
    let signature = hsm_integration.sign_data(&key_id, message).await?;
    
    Ok(())
}
```

### MPC Integration

```rust
use secure_storage::secure_enclave::{
    SecureEnclaveSystem, IntegrationMode,
    integration::{MpcEnclaveIntegration, MpcIntegrationConfig}
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create enclave system with MPC integration
    let config = EnclaveConfig::new_production(EnclavePlatform::IntelSgx)?;
    let enclave_system = Arc::new(
        SecureEnclaveSystem::new_with_integration(
            config, 
            IntegrationMode::EnclaveMpc
        ).await?
    );
    
    // Initialize MPC integration
    let mpc_config = MpcIntegrationConfig {
        threshold: 3,
        total_parties: 5,
        mpc_threshold_signatures: true,
        mpc_key_generation: true,
        mpc_timeout: Duration::from_millis(500),
        mpc_value_threshold: 1_000_000, // Use MPC for high-value operations
    };
    
    let mut mpc_integration = MpcEnclaveIntegration::new(
        enclave_system, 
        mpc_config
    )?;
    
    // Initialize MPC system
    mpc_integration.initialize_mpc().await?;
    
    // Execute threshold signature
    let key_id = KeyId::new("multisig_key_001".to_string());
    let message = b"high_value_transaction";
    let value = 5_000_000; // Above threshold
    
    let signature = mpc_integration.threshold_sign(&key_id, message, value).await?;
    
    Ok(())
}
```

## Configuration

### Production Configuration

```rust
let config = EnclaveConfig {
    platform: EnclavePlatform::IntelSgx,
    heap_size: 64 * 1024 * 1024,     // 64MB
    stack_size: 1024 * 1024,         // 1MB
    debug_mode: false,               // Never in production
    max_threads: 16,
    attestation_enabled: true,
    sealed_storage_path: Some(PathBuf::from("/var/lib/tallyio/sealed")),
};
```

### Development Configuration

```rust
let config = EnclaveConfig::new_development()?;
// Uses simulation mode with relaxed security for testing
```

## Security Features

### Circuit Breaker

The system includes a circuit breaker for fault tolerance:

```rust
let circuit_breaker_config = CircuitBreakerConfig {
    failure_threshold: 5,           // Open after 5 failures
    success_threshold: 3,           // Close after 3 successes
    timeout_duration: Duration::from_secs(30),
    max_concurrent_operations: 100,
};
```

### Attestation

Hardware attestation provides cryptographic proof of enclave authenticity:

```rust
// Attestation is automatically included in operation results
let result = enclave_system.execute_secure_operation(...).await?;
if let Some(attestation_report) = result.attestation_report {
    // Verify attestation report
    verify_attestation(&attestation_report)?;
}
```

### Sealed Storage

Hardware-encrypted persistent storage:

```rust
// Store sensitive data
enclave_system.store_sealed_data("api_key", sensitive_data).await?;

// Retrieve data
let data = enclave_system.retrieve_sealed_data("api_key").await?;
```

## Performance Optimization

### Critical Path Operations

For operations requiring <1ms latency:

```rust
let result = enclave_system.execute_critical_operation(
    EnclaveOperation::Hashing,
    "critical_hash".to_string(),
    || {
        // Ultra-fast operation
        Ok(hash_data_fast(input))
    }
).await?;
```

### Batch Operations

For efficiency with multiple operations:

```rust
let operations = vec![
    (EnclaveOperation::KeyGeneration, "key1".to_string(), || Ok(gen_key1())),
    (EnclaveOperation::KeyGeneration, "key2".to_string(), || Ok(gen_key2())),
    (EnclaveOperation::Hashing, "hash1".to_string(), || Ok(hash_data())),
];

let results = enclave_system.batch_execute_operations(operations).await?;
```

## Monitoring and Metrics

### System Statistics

```rust
let stats = enclave_system.get_stats();
println!("Platform: {:?}", stats.platform);
println!("Total operations: {}", stats.operations_total);
println!("Success rate: {:.2}%", 
    (stats.operations_successful as f64 / stats.operations_total as f64) * 100.0
);
println!("Average latency: {}μs", stats.average_execution_time_ns / 1000);
```

### Integration Capabilities

```rust
let capabilities = enclave_system.get_integration_capabilities();
println!("HSM available: {}", capabilities.hsm_available);
println!("MPC available: {}", capabilities.mpc_available);
println!("Attestation available: {}", capabilities.attestation_available);
println!("Sealed storage available: {}", capabilities.sealed_storage_available);
```

## Testing

Comprehensive test suite included:

```bash
# Run all enclave integration tests
cargo test --test secure_enclave_integration_test

# Run with verbose output
cargo test --test secure_enclave_integration_test -- --nocapture

# Run specific test
cargo test test_hsm_enclave_integration
```

## Platform Support

| Platform | Status | Features |
|----------|--------|----------|
| Intel SGX | ✅ Production | Full attestation, sealed storage |
| ARM TrustZone | ✅ Production | OP-TEE integration, secure world |
| AMD Memory Guard | 🚧 Beta | Memory encryption, SEV support |
| Simulation | ✅ Development | Testing and development |

## Security Considerations

1. **Never use debug mode in production**
2. **Always enable attestation for remote verification**
3. **Use sealed storage for persistent sensitive data**
4. **Implement proper key rotation policies**
5. **Monitor circuit breaker states**
6. **Validate all attestation reports**

## Performance Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Key Generation | <50ms | ✅ <30ms |
| Digital Signature | <10ms | ✅ <5ms |
| Hash Computation | <1ms | ✅ <0.5ms |
| Encryption (1KB) | <5ms | ✅ <3ms |
| Attestation | <100ms | ✅ <80ms |

## Troubleshooting

### Common Issues

1. **SGX not available**: Check BIOS settings and SGX driver installation
2. **TrustZone not found**: Verify `/dev/tee0` device exists
3. **Circuit breaker open**: Check error logs and system health
4. **High latency**: Review system load and CPU affinity settings

### Debug Mode

For development debugging:

```rust
let config = EnclaveConfig {
    debug_mode: true,  // Only for development!
    // ... other settings
};
```

## License

MIT License - See LICENSE file for details.

---

**TallyIO Secure Enclave Integration** - Hardware-level security for financial applications.
