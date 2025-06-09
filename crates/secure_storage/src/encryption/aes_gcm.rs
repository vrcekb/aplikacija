//! AES-256-GCM encryption implementation

use crate::encryption::{utils, Encryption};
use crate::error::{EncryptionError, SecureStorageResult};
use crate::types::{EncryptedData, EncryptionAlgorithm, KeyId, KeyMaterial, KeyMetadata};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use async_trait::async_trait;
use dashmap::DashMap;
use std::sync::Arc;
use zeroize::Zeroize;

/// AES-256-GCM encryption implementation
pub struct AesGcmEncryption {
    /// Key storage
    keys: Arc<DashMap<KeyId, SecureKey>>,
}

/// Secure key wrapper that zeroizes on drop
struct SecureKey {
    key: Key<Aes256Gcm>,
}

impl Drop for SecureKey {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

impl AesGcmEncryption {
    /// Create a new AES-GCM encryption instance
    ///
    /// # Errors
    ///
    /// Currently this function cannot fail, but returns a Result for future extensibility
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn new() -> SecureStorageResult<Self> {
        Ok(Self {
            keys: Arc::new(DashMap::new()),
        })
    }

    /// Add a key to the key store
    ///
    /// # Errors
    ///
    /// Returns `EncryptionError::InvalidKey` if the key material is not 32 bytes (256 bits)
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn add_key(&self, key_material: &KeyMaterial) -> SecureStorageResult<()> {
        if key_material.key.len() != 32 {
            return Err(EncryptionError::InvalidKey {
                reason: format!("Expected 32 bytes, got {}", key_material.key.len()),
            }
            .into());
        }

        let key = Key::<Aes256Gcm>::from_slice(&key_material.key);
        let secure_key = SecureKey { key: *key };

        self.keys
            .insert(key_material.metadata.id.clone(), secure_key);
        Ok(())
    }

    /// Get a key from the key store
    fn get_key(&self, key_id: &KeyId) -> SecureStorageResult<Key<Aes256Gcm>> {
        let entry = self
            .keys
            .get(key_id)
            .ok_or_else(|| EncryptionError::InvalidKey {
                reason: format!("Key not found: {key_id}"),
            })?;

        Ok(entry.key)
    }

    /// Perform encryption with timing measurement
    #[inline]
    fn encrypt_internal(
        cipher: &Aes256Gcm,
        nonce: &[u8],
        plaintext: &[u8],
        aad: Option<&[u8]>,
    ) -> SecureStorageResult<(Vec<u8>, Option<Vec<u8>>)> {
        let start = std::time::Instant::now();

        let nonce = Nonce::from_slice(nonce);

        let ciphertext = aad
            .map_or_else(
                || cipher.encrypt(nonce, plaintext),
                |aad| {
                    cipher.encrypt(
                        nonce,
                        aes_gcm::aead::Payload {
                            msg: plaintext,
                            aad,
                        },
                    )
                },
            )
            .map_err(|e| EncryptionError::CipherOperation {
                reason: format!("AES-GCM encryption failed: {e}"),
            })?;

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 5 {
            tracing::warn!(
                "AES-GCM encryption took {}ms (target: <5ms)",
                elapsed.as_millis()
            );
        }

        // For AES-GCM, the authentication tag is included in the ciphertext
        // We need to separate it for our EncryptedData structure
        if ciphertext.len() < 16 {
            return Err(EncryptionError::CipherOperation {
                reason: "Ciphertext too short to contain authentication tag".to_string(),
            }
            .into());
        }

        let (data, tag) = ciphertext.split_at(ciphertext.len() - 16);
        Ok((data.to_vec(), Some(tag.to_vec())))
    }

    /// Perform decryption with timing measurement
    #[inline]
    fn decrypt_internal(
        cipher: &Aes256Gcm,
        nonce: &[u8],
        ciphertext: &[u8],
        tag: Option<&[u8]>,
        aad: Option<&[u8]>,
    ) -> SecureStorageResult<Vec<u8>> {
        let start = std::time::Instant::now();

        let nonce = Nonce::from_slice(nonce);

        // Reconstruct the full ciphertext with tag for AES-GCM
        let full_ciphertext = tag.map_or_else(
            || ciphertext.to_vec(),
            |tag| {
                let mut full = Vec::with_capacity(ciphertext.len() + tag.len());
                full.extend_from_slice(ciphertext);
                full.extend_from_slice(tag);
                full
            },
        );

        let plaintext = aad
            .map_or_else(
                || cipher.decrypt(nonce, full_ciphertext.as_slice()),
                |aad| {
                    cipher.decrypt(
                        nonce,
                        aes_gcm::aead::Payload {
                            msg: &full_ciphertext,
                            aad,
                        },
                    )
                },
            )
            .map_err(|_e| EncryptionError::AuthenticationFailed)?;

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 5 {
            tracing::warn!(
                "AES-GCM decryption took {}ms (target: <5ms)",
                elapsed.as_millis()
            );
        }

        Ok(plaintext)
    }
}

#[async_trait]
impl Encryption for AesGcmEncryption {
    async fn encrypt(
        &self,
        data: &[u8],
        key_id: &KeyId,
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<EncryptedData> {
        if algorithm != EncryptionAlgorithm::Aes256Gcm {
            return Err(EncryptionError::UnsupportedAlgorithm {
                algorithm: algorithm.to_string(),
            }
            .into());
        }

        let key = self.get_key(key_id)?;
        let cipher = Aes256Gcm::new(&key);

        // Generate a random nonce
        let nonce = utils::generate_nonce(algorithm);

        // Perform encryption
        let (ciphertext, tag) = Self::encrypt_internal(
            &cipher, &nonce, data, None, // No AAD for now
        )?;

        Ok(
            EncryptedData::new(algorithm, key_id.clone(), nonce, ciphertext)
                .with_tag(tag.unwrap_or_else(Vec::new)),
        )
    }

    async fn decrypt(&self, encrypted_data: &EncryptedData) -> SecureStorageResult<Vec<u8>> {
        if encrypted_data.algorithm != EncryptionAlgorithm::Aes256Gcm {
            return Err(EncryptionError::UnsupportedAlgorithm {
                algorithm: encrypted_data.algorithm.to_string(),
            }
            .into());
        }

        let key = self.get_key(&encrypted_data.key_id)?;
        let cipher = Aes256Gcm::new(&key);

        Self::decrypt_internal(
            &cipher,
            &encrypted_data.nonce,
            &encrypted_data.ciphertext,
            encrypted_data.tag.as_deref(),
            encrypted_data.aad.as_deref(),
        )
    }

    async fn generate_key(
        &self,
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<KeyMaterial> {
        if algorithm != EncryptionAlgorithm::Aes256Gcm {
            return Err(EncryptionError::UnsupportedAlgorithm {
                algorithm: algorithm.to_string(),
            }
            .into());
        }

        // Generate a random 256-bit key
        let key_bytes = utils::secure_random_bytes(32)?;

        let key_id = KeyId::generate();
        let metadata = KeyMetadata::new(
            key_id, algorithm, 256, // 256 bits
        );

        let key_material = KeyMaterial::new(key_bytes, metadata);

        // Store the key
        self.add_key(&key_material)?;

        Ok(key_material)
    }

    async fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<KeyMaterial> {
        if algorithm != EncryptionAlgorithm::Aes256Gcm {
            return Err(EncryptionError::UnsupportedAlgorithm {
                algorithm: algorithm.to_string(),
            }
            .into());
        }

        // Use Argon2id for key derivation
        let key_bytes = super::key_derivation::derive_key_argon2id(
            password, salt, 32, // 256 bits
        )?;

        let key_id = KeyId::generate();
        let metadata = KeyMetadata::new(key_id, algorithm, 256);

        let key_material = KeyMaterial::new(key_bytes, metadata);

        // Store the key
        self.add_key(&key_material)?;

        Ok(key_material)
    }

    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm> {
        vec![EncryptionAlgorithm::Aes256Gcm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::KeyId;
    use crate::SecureStorageError;

    #[tokio::test]
    async fn test_aes_gcm_roundtrip() -> SecureStorageResult<()> {
        let encryption = AesGcmEncryption::new()?;

        // Generate a key
        let key_material = encryption
            .generate_key(EncryptionAlgorithm::Aes256Gcm)
            .await?;
        let key_id = key_material.metadata.id.clone();

        // Test data
        let plaintext = b"This is a test message for AES-GCM encryption";

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
            return Err(SecureStorageError::Encryption {
                reason: "Expected tag to be present for AES-GCM".to_string(),
            });
        }

        // Decrypt
        let decrypted = encryption.decrypt(&encrypted).await?;

        assert_eq!(plaintext, decrypted.as_slice());
        Ok(())
    }

    #[tokio::test]
    async fn test_key_derivation() -> SecureStorageResult<()> {
        let encryption = AesGcmEncryption::new()?;

        let password = b"strong_password_123";
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
        assert_eq!(key_material.metadata.key_size, 256);

        // Derive the same key again with same password and salt
        let key_material2 = encryption
            .derive_key(password, &salt, EncryptionAlgorithm::Aes256Gcm)
            .await?;

        // Keys should be the same (deterministic derivation)
        assert_eq!(key_material.key, key_material2.key);

        Ok(())
    }

    #[tokio::test]
    async fn test_performance_requirements() -> SecureStorageResult<()> {
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
    async fn test_invalid_key_handling() -> SecureStorageResult<()> {
        let encryption = AesGcmEncryption::new()?;

        let non_existent_key = KeyId::new("non_existent");
        let plaintext = b"test data";

        // Should fail with non-existent key
        let result = encryption
            .encrypt(plaintext, &non_existent_key, EncryptionAlgorithm::Aes256Gcm)
            .await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_unsupported_algorithm() -> SecureStorageResult<()> {
        let encryption = AesGcmEncryption::new()?;

        let result = encryption
            .generate_key(EncryptionAlgorithm::ChaCha20Poly1305)
            .await;
        assert!(result.is_err());

        Ok(())
    }
}
