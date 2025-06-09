//! Encryption implementation tests

#![allow(clippy::unwrap_used)]
#![allow(clippy::panic)]
#![allow(clippy::default_numeric_fallback)]
#![allow(clippy::uninlined_format_args)]

use secure_storage::encryption::{
    aes_gcm::AesGcmEncryption,
    key_derivation::{
        derive_key_argon2id, derive_key_argon2id_with_config, derive_key_with_salt, Argon2Config,
    },
    utils, Encryption, EncryptionFactory,
};
use secure_storage::error::{SecureStorageError, SecureStorageResult};
use secure_storage::types::{EncryptionAlgorithm, KeyId};

#[tokio::test]
async fn test_aes_gcm_encryption_roundtrip() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;

    // Generate a key
    let key_material = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let key_id = key_material.metadata.id.clone();

    // Test data
    let plaintext = b"This is a test message for AES-GCM encryption and decryption";

    // Encrypt
    let encrypted = encryption
        .encrypt(plaintext, &key_id, EncryptionAlgorithm::Aes256Gcm)
        .await?;

    // Verify encrypted data structure
    assert_eq!(encrypted.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert_eq!(encrypted.key_id, key_id);
    assert_eq!(encrypted.nonce.len(), 12); // AES-GCM nonce size
    assert!(encrypted.tag.is_some());
    if let Some(ref tag) = encrypted.tag {
        assert_eq!(tag.len(), 16); // AES-GCM tag size
    } else {
        panic!("Expected tag to be present for AES-GCM encryption");
    }
    assert_ne!(encrypted.ciphertext, plaintext); // Should be different

    // Decrypt
    let decrypted = encryption.decrypt(&encrypted).await?;

    assert_eq!(plaintext, decrypted.as_slice());
    Ok(())
}

#[tokio::test]
async fn test_aes_gcm_key_generation() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;

    // Generate multiple keys
    let key1 = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let key2 = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;

    // Keys should be different
    assert_ne!(key1.key, key2.key);
    assert_ne!(key1.metadata.id, key2.metadata.id);

    // Keys should have correct properties
    assert_eq!(key1.key.len(), 32); // 256 bits
    assert_eq!(key1.metadata.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert_eq!(key1.metadata.key_size, 256);

    Ok(())
}

#[tokio::test]
async fn test_aes_gcm_key_derivation() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;

    let password = b"strong_password_for_testing_123";
    let salt = utils::generate_salt(32);

    // Derive key
    let key_material = encryption
        .derive_key(password, &salt, EncryptionAlgorithm::Aes256Gcm)
        .await?;

    assert_eq!(key_material.key.len(), 32);
    assert_eq!(
        key_material.metadata.algorithm,
        EncryptionAlgorithm::Aes256Gcm
    );

    // Derive the same key again with same password and salt
    let key_material2 = encryption
        .derive_key(password, &salt, EncryptionAlgorithm::Aes256Gcm)
        .await?;

    // Keys should be the same (deterministic derivation)
    assert_eq!(key_material.key, key_material2.key);

    Ok(())
}

#[tokio::test]
async fn test_encryption_factory() -> SecureStorageResult<()> {
    // Test default encryption
    let encryption = EncryptionFactory::create_default()?;
    let algorithms = encryption.supported_algorithms();
    assert!(!algorithms.is_empty());
    assert!(algorithms.contains(&EncryptionAlgorithm::Aes256Gcm));

    // Test specific algorithm
    let aes_encryption = EncryptionFactory::create(EncryptionAlgorithm::Aes256Gcm)?;
    let aes_algorithms = aes_encryption.supported_algorithms();
    assert!(aes_algorithms.contains(&EncryptionAlgorithm::Aes256Gcm));

    Ok(())
}

#[tokio::test]
async fn test_encryption_performance() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;

    // Generate a key
    let key_material = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let key_id = key_material.metadata.id.clone();

    // Test with 1KB of data
    let plaintext = vec![0u8; 1024];

    // Measure encryption time
    let start = std::time::Instant::now();
    let encrypted = encryption
        .encrypt(&plaintext, &key_id, EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let encrypt_time = start.elapsed();

    // Measure decryption time
    let start = std::time::Instant::now();
    let _decrypted = encryption.decrypt(&encrypted).await?;
    let decrypt_time = start.elapsed();

    // Performance requirements: <5ms for 1KB
    assert!(
        encrypt_time.as_millis() < 5,
        "Encryption took {}ms",
        encrypt_time.as_millis()
    );
    assert!(
        decrypt_time.as_millis() < 5,
        "Decryption took {}ms",
        decrypt_time.as_millis()
    );

    Ok(())
}

#[tokio::test]
async fn test_encryption_with_different_data_sizes() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;
    let key_material = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let key_id = key_material.metadata.id.clone();

    // Test different data sizes
    let sizes = vec![0, 1, 16, 64, 256, 1024, 4096];

    for size in sizes {
        let plaintext = vec![0u8; size];

        let encrypted = encryption
            .encrypt(&plaintext, &key_id, EncryptionAlgorithm::Aes256Gcm)
            .await?;

        let decrypted = encryption.decrypt(&encrypted).await?;
        assert_eq!(plaintext, decrypted);
    }

    Ok(())
}

#[test]
fn test_argon2_key_derivation() -> SecureStorageResult<()> {
    let password = b"test_password_for_argon2";
    let salt = utils::generate_salt(32);

    // Test basic key derivation
    let key = derive_key_argon2id(password, &salt, 32)?;
    assert_eq!(key.len(), 32);

    // Test deterministic derivation
    let key2 = derive_key_argon2id(password, &salt, 32)?;
    assert_eq!(key, key2);

    // Test different salt produces different key
    let salt2 = utils::generate_salt(32);
    let key3 = derive_key_argon2id(password, &salt2, 32)?;
    assert_ne!(key, key3);

    Ok(())
}

#[test]
fn test_argon2_config_validation() -> SecureStorageResult<()> {
    // Test valid configurations
    let config = Argon2Config::default();
    config.validate()?;

    let fast_config = Argon2Config::fast();
    fast_config.validate()?;

    let secure_config = Argon2Config::secure();
    secure_config.validate()?;

    // Test invalid configurations
    let invalid_config = Argon2Config {
        memory_cost: 512, // Too low
        ..Argon2Config::default()
    };
    assert!(invalid_config.validate().is_err());

    let invalid_config2 = Argon2Config {
        memory_cost: 1024,
        time_cost: 0, // Invalid
        ..Argon2Config::default()
    };
    assert!(invalid_config2.validate().is_err());

    Ok(())
}

#[test]
fn test_key_derivation_with_salt() -> SecureStorageResult<()> {
    let password = b"test_password_with_auto_salt";
    let config = Argon2Config::fast(); // Use fast config for testing

    // Derive key with automatic salt generation
    let derived1 = derive_key_with_salt(password, &config)?;
    let derived2 = derive_key_with_salt(password, &config)?;

    // Keys should be different due to different salts
    assert_ne!(derived1.key, derived2.key);
    assert_ne!(derived1.salt, derived2.salt);

    // But same password + same salt should produce same key
    let derived3 =
        derive_key_argon2id_with_config(password, &derived1.salt, config.output_length, &config)?;
    assert_eq!(derived1.key, derived3);

    Ok(())
}

#[test]
fn test_utility_functions() -> SecureStorageResult<()> {
    // Test nonce generation
    let nonce1 = utils::generate_nonce(EncryptionAlgorithm::Aes256Gcm);
    let nonce2 = utils::generate_nonce(EncryptionAlgorithm::Aes256Gcm);

    assert_eq!(nonce1.len(), 12); // AES-GCM nonce size
    assert_eq!(nonce2.len(), 12);
    assert_ne!(nonce1, nonce2); // Should be different

    // Test salt generation
    let salt1 = utils::generate_salt(32);
    let salt2 = utils::generate_salt(32);

    assert_eq!(salt1.len(), 32);
    assert_eq!(salt2.len(), 32);
    assert_ne!(salt1, salt2); // Should be different

    // Test constant time comparison
    let a = b"hello";
    let b = b"hello";
    let c = b"world";

    assert!(utils::constant_time_eq(a, b));
    assert!(!utils::constant_time_eq(a, c));
    assert!(!utils::constant_time_eq(a, b"hell")); // Different lengths

    // Test secure random generation
    let bytes1 = utils::secure_random_bytes(32)?;
    let bytes2 = utils::secure_random_bytes(32)?;

    assert_eq!(bytes1.len(), 32);
    assert_eq!(bytes2.len(), 32);
    assert_ne!(bytes1, bytes2); // Should be different

    Ok(())
}

#[tokio::test]
async fn test_encryption_error_handling() -> SecureStorageResult<()> {
    let encryption = AesGcmEncryption::new()?;

    // Test with non-existent key
    let non_existent_key = KeyId::new("non_existent");
    let plaintext = b"test data";

    let result = encryption
        .encrypt(plaintext, &non_existent_key, EncryptionAlgorithm::Aes256Gcm)
        .await;

    assert!(result.is_err());

    // Test with unsupported algorithm
    let result = encryption
        .generate_key(EncryptionAlgorithm::ChaCha20Poly1305)
        .await;
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_concurrent_encryption() -> SecureStorageResult<()> {
    let encryption = std::sync::Arc::new(AesGcmEncryption::new()?);

    // Generate a shared key
    let key_material = encryption
        .generate_key(EncryptionAlgorithm::Aes256Gcm)
        .await?;
    let key_id = key_material.metadata.id.clone();

    // Spawn multiple concurrent encryption operations
    let mut handles = Vec::new();

    for i in 0..10 {
        let encryption_clone = encryption.clone();
        let key_id_clone = key_id.clone();

        let handle = tokio::spawn(async move {
            let plaintext = format!("test message {}", i).into_bytes();

            let encrypted = encryption_clone
                .encrypt(&plaintext, &key_id_clone, EncryptionAlgorithm::Aes256Gcm)
                .await?;

            let decrypted = encryption_clone.decrypt(&encrypted).await?;
            assert_eq!(plaintext, decrypted);

            Ok::<(), secure_storage::error::SecureStorageError>(())
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        handle.await.map_err(|e| SecureStorageError::Encryption {
            reason: format!("Task join error: {e}"),
        })??;
    }

    Ok(())
}
