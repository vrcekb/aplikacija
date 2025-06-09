//! # MPC Integration Tests
//!
//! Comprehensive tests for multi-party computation functionality
//! including threshold signatures, distributed key generation, and verification.

use secure_storage::error::SecureStorageResult;
use secure_storage::mpc::{
    threshold::ThresholdSignatureSystem,
    verification::{Commitment, CommitmentType, MpcVerificationSystem, ProofType, ZkProof},
    MpcSystem, PartyId, ThresholdConfig,
};
use secure_storage::types::KeyId;
use std::time::Duration;

/// Test basic MPC system initialization
#[tokio::test]
async fn test_mpc_system_initialization() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let stats = mpc_system.get_stats().await;

    assert_eq!(stats.party_id, party_id);
    assert_eq!(stats.total_operations, 0);
    assert_eq!(stats.active_shares, 0);

    Ok(())
}

/// Test threshold configuration validation
#[tokio::test]
async fn test_threshold_config_validation() {
    // Valid configuration
    let valid_config = ThresholdConfig::new(5, 3, Duration::from_secs(60));
    assert!(valid_config.is_ok());

    // Invalid: threshold > total_parties
    let invalid_config1 = ThresholdConfig::new(3, 5, Duration::from_secs(60));
    assert!(invalid_config1.is_err());

    // Invalid: threshold = 0
    let invalid_config2 = ThresholdConfig::new(5, 0, Duration::from_secs(60));
    assert!(invalid_config2.is_err());
}

/// Test distributed key generation
#[tokio::test]
async fn test_distributed_key_generation() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let key_id = KeyId::new("test_key_1".to_string());

    // Generate distributed key
    let share = mpc_system.generate_distributed_key(key_id.clone()).await?;

    // Verify share properties
    assert_eq!(share.party_id, party_id);
    assert!(!share.share_value().is_empty());
    assert!(!share.verification_data.is_empty());
    assert!(share.verify()?);

    // Check system stats
    let stats = mpc_system.get_stats().await;
    assert_eq!(stats.active_shares, 1);
    assert_eq!(stats.total_operations, 1);

    Ok(())
}

/// Test threshold signature creation and verification
#[tokio::test]
async fn test_threshold_signature() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let key_id = KeyId::new("test_key_2".to_string());

    // Generate key share
    let _share = mpc_system.generate_distributed_key(key_id.clone()).await?;

    // Create threshold signature
    let message = b"test message for signing";
    let participants = vec![PartyId::new(1), PartyId::new(2)];

    let signature = mpc_system
        .threshold_sign(&key_id, message, &participants)
        .await?;

    // Verify signature properties
    assert!(!signature.is_empty());
    assert_eq!(signature.len(), 48); // BLS signature size

    Ok(())
}

/// Test threshold signature system directly
#[tokio::test]
async fn test_threshold_signature_system() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let threshold_system = ThresholdSignatureSystem::new(party_id, config)?;

    // Create a mock secret share
    let share_value = vec![42u8; 32]; // 32-byte secret
    let verification_data = vec![1u8; 48]; // Mock verification data
    let share = secure_storage::mpc::SecretShare::new(party_id, share_value, verification_data, 1);

    // Create partial signature
    let message = b"test message";
    let operation_id = "test_op_1";

    let partial_sig = threshold_system
        .create_partial_signature(&share, message, operation_id)
        .await?;

    // Verify partial signature properties
    assert_eq!(partial_sig.party_id, party_id);
    assert!(!partial_sig.signature.is_empty());
    assert!(!partial_sig.proof.is_empty());

    Ok(())
}

/// Test MPC verification system
#[tokio::test]
async fn test_mpc_verification_system() -> SecureStorageResult<()> {
    let mut verification_system = MpcVerificationSystem::new();

    let operation_id = "test_verification_1".to_string();
    let parties = vec![PartyId::new(1), PartyId::new(2), PartyId::new(3)];

    // Create verification context
    verification_system.create_context(operation_id.clone(), parties.clone())?;

    // Add commitments and proofs
    for party_id in parties {
        let commitment = Commitment::new(
            vec![vec![1u8; 48]; 3], // Mock commitment values
            CommitmentType::Pedersen,
            vec![2u8; 48], // Mock generator
        );

        let proof = ZkProof::new(
            ProofType::ShareGeneration,
            vec![3u8; 64], // Mock proof data
            vec![4u8; 32], // Mock public params
            vec![5u8; 48], // Mock verification key
        );

        verification_system.add_commitment(&operation_id, party_id, commitment)?;
        verification_system.add_proof(&operation_id, party_id, proof)?;
    }

    // Verify all data
    let challenge = b"verification_challenge";
    let result = verification_system.verify_context(&operation_id, challenge)?;

    assert!(result, "Verification should succeed with valid data");

    // Check stats
    let stats = verification_system.get_stats();
    assert_eq!(stats.verifications_performed, 1);
    assert_eq!(stats.verifications_successful, 1);
    assert_eq!(stats.verifications_failed, 0);

    Ok(())
}

/// Test key refresh functionality
#[tokio::test]
async fn test_key_refresh() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let mut config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;
    config.proactive_security = true;
    config.refresh_interval = Duration::from_millis(100); // Short interval for testing

    let mpc_system = MpcSystem::new(party_id, config)?;

    // Generate some keys
    let key_id1 = KeyId::new("refresh_test_1".to_string());
    let key_id2 = KeyId::new("refresh_test_2".to_string());

    let _share1 = mpc_system.generate_distributed_key(key_id1).await?;
    let _share2 = mpc_system.generate_distributed_key(key_id2).await?;

    // Wait for refresh interval
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Check if refresh is needed
    assert!(mpc_system.needs_refresh().await);

    // Perform key refresh
    let refreshed_count = mpc_system.refresh_keys().await?;
    assert_eq!(refreshed_count, 2);

    // Verify refresh was performed
    assert!(!mpc_system.needs_refresh().await);

    Ok(())
}

/// Test share validation
#[tokio::test]
async fn test_share_validation() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let key_id = KeyId::new("validation_test".to_string());

    // Generate key share
    let _share = mpc_system.generate_distributed_key(key_id).await?;

    // Validate all shares
    let validation_result = mpc_system.validate_all_shares().await?;
    assert!(validation_result, "All shares should be valid");

    Ok(())
}

/// Test error handling for insufficient participants
#[tokio::test]
async fn test_insufficient_participants_error() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let key_id = KeyId::new("insufficient_test".to_string());

    // Generate key share
    let _share = mpc_system.generate_distributed_key(key_id.clone()).await?;

    // Try to sign with insufficient participants
    let message = b"test message";
    let insufficient_participants = vec![PartyId::new(1)]; // Only 1 participant, need 2

    let result = mpc_system
        .threshold_sign(&key_id, message, &insufficient_participants)
        .await;
    assert!(
        result.is_err(),
        "Should fail with insufficient participants"
    );

    Ok(())
}

/// Performance test for MPC operations
#[tokio::test]
async fn test_mpc_performance() -> SecureStorageResult<()> {
    let party_id = PartyId::new(1);
    let config = ThresholdConfig::new(3, 2, Duration::from_secs(30))?;

    let mpc_system = MpcSystem::new(party_id, config)?;
    let key_id = KeyId::new("performance_test".to_string());

    // Measure key generation time
    let start = std::time::Instant::now();
    let _share = mpc_system.generate_distributed_key(key_id.clone()).await?;
    let key_gen_time = start.elapsed();

    // Should be reasonably fast (under 200ms for simulation with crypto operations)
    assert!(
        key_gen_time < Duration::from_millis(200),
        "Key generation took too long: {key_gen_time:?}"
    );

    // Measure signing time
    let message = b"performance test message";
    let participants = vec![PartyId::new(1), PartyId::new(2)];

    let start = std::time::Instant::now();
    let _signature = mpc_system
        .threshold_sign(&key_id, message, &participants)
        .await?;
    let signing_time = start.elapsed();

    // Should be reasonably fast (under 100ms for simulation with crypto operations)
    assert!(
        signing_time < Duration::from_millis(100),
        "Threshold signing took too long: {signing_time:?}"
    );

    Ok(())
}
