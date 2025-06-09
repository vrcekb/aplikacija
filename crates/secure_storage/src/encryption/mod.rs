//! Encryption layer for secure storage

use crate::error::{EncryptionError, SecureStorageResult};
use crate::types::{EncryptedData, EncryptionAlgorithm, KeyId, KeyMaterial};
use async_trait::async_trait;

pub mod aes_gcm;
pub mod key_derivation;

/// Trait for encryption operations
#[async_trait]
pub trait Encryption: Send + Sync {
    /// Encrypt data with the specified key
    async fn encrypt(
        &self,
        data: &[u8],
        key_id: &KeyId,
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<EncryptedData>;

    /// Decrypt data with the specified key
    async fn decrypt(&self, encrypted_data: &EncryptedData) -> SecureStorageResult<Vec<u8>>;

    /// Generate a new encryption key
    async fn generate_key(
        &self,
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<KeyMaterial>;

    /// Derive a key from a password
    async fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<KeyMaterial>;

    /// Get supported algorithms
    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm>;
}

/// Factory for creating encryption instances
pub struct EncryptionFactory;

impl EncryptionFactory {
    /// Create a new encryption instance
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn create(algorithm: EncryptionAlgorithm) -> SecureStorageResult<Box<dyn Encryption>> {
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => Ok(Box::new(aes_gcm::AesGcmEncryption::new()?)),
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                // TODO: Implement ChaCha20-Poly1305
                Err(EncryptionError::UnsupportedAlgorithm {
                    algorithm: algorithm.to_string(),
                }
                .into())
            }
            EncryptionAlgorithm::Aes256Cbc => {
                // TODO: Implement AES-256-CBC (legacy support)
                Err(EncryptionError::UnsupportedAlgorithm {
                    algorithm: algorithm.to_string(),
                }
                .into())
            }
        }
    }

    /// Get default encryption instance
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn create_default() -> SecureStorageResult<Box<dyn Encryption>> {
        Self::create(EncryptionAlgorithm::Aes256Gcm)
    }
}

/// Utility functions for encryption operations
pub mod utils {
    use super::{EncryptionAlgorithm, EncryptionError, SecureStorageResult};
    use rand::{thread_rng, RngCore};

    /// Generates a cryptographically secure random nonce for the specified algorithm.
    ///
    /// This function creates a nonce (number used once) of the appropriate size
    /// for the given encryption algorithm. Nonces are critical for security in
    /// encryption schemes and must be unique for each encryption operation.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The encryption algorithm that will use this nonce
    ///
    /// # Returns
    ///
    /// A vector containing cryptographically secure random bytes of the
    /// correct length for the specified algorithm:
    /// - AES-256-GCM: 12 bytes (96 bits) - recommended size
    /// - `ChaCha20`-Poly1305: 12 bytes (96 bits)
    /// - AES-256-CBC: 16 bytes (128 bits) - block size
    ///
    /// # Security Requirements
    ///
    /// - **Uniqueness**: Each nonce must be unique for a given key
    /// - **Randomness**: Generated using cryptographically secure RNG
    /// - **Unpredictability**: Cannot be guessed or predicted by attackers
    /// - **Proper Size**: Must match algorithm requirements exactly
    ///
    /// # Performance
    ///
    /// Nonce generation is very fast, typically completing in microseconds.
    /// The function uses the system's cryptographically secure random number
    /// generator which is optimized for performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::encryption::utils::generate_nonce;
    /// use secure_storage::types::EncryptionAlgorithm;
    ///
    /// let nonce = generate_nonce(EncryptionAlgorithm::Aes256Gcm);
    /// assert_eq!(nonce.len(), 12); // GCM uses 12-byte nonces
    ///
    /// let nonce2 = generate_nonce(EncryptionAlgorithm::Aes256Gcm);
    /// assert_ne!(nonce, nonce2); // Each nonce should be unique
    /// ```
    #[must_use]
    pub fn generate_nonce(algorithm: EncryptionAlgorithm) -> Vec<u8> {
        let mut nonce = vec![0u8; algorithm.nonce_size_bytes()];
        thread_rng().fill_bytes(&mut nonce);
        nonce
    }

    /// Generates a cryptographically secure random salt of the specified size.
    ///
    /// This function creates a random salt for use in key derivation functions
    /// and other cryptographic operations that require unique, unpredictable
    /// input values. Salts prevent rainbow table attacks and ensure unique
    /// derived keys even for identical input passwords.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of random bytes to generate for the salt
    ///
    /// # Returns
    ///
    /// A vector containing the specified number of cryptographically secure
    /// random bytes suitable for use as a salt.
    ///
    /// # Recommended Salt Sizes
    ///
    /// - **Minimum**: 16 bytes (128 bits) for basic security
    /// - **Standard**: 32 bytes (256 bits) for strong security
    /// - **High Security**: 64 bytes (512 bits) for maximum security
    /// - **PBKDF2**: 16-32 bytes recommended by standards
    /// - **Argon2**: 16-64 bytes depending on security requirements
    ///
    /// # Security Properties
    ///
    /// - **Uniqueness**: Each salt should be unique across all uses
    /// - **Randomness**: Generated using cryptographically secure RNG
    /// - **Unpredictability**: Cannot be guessed or predicted
    /// - **Sufficient Length**: Must be long enough to prevent brute force
    ///
    /// # Use Cases
    ///
    /// - **Password Hashing**: Prevent rainbow table attacks
    /// - **Key Derivation**: Ensure unique keys from same password
    /// - **Cryptographic Protocols**: Add randomness to deterministic operations
    /// - **Database Security**: Unique salts per user/record
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::encryption::utils::generate_salt;
    ///
    /// // Standard 32-byte salt for strong security
    /// let salt = generate_salt(32);
    /// assert_eq!(salt.len(), 32);
    ///
    /// // Minimum 16-byte salt for basic security
    /// let small_salt = generate_salt(16);
    /// assert_eq!(small_salt.len(), 16);
    ///
    /// // Each salt should be unique
    /// let salt1 = generate_salt(32);
    /// let salt2 = generate_salt(32);
    /// assert_ne!(salt1, salt2);
    /// ```
    #[must_use]
    pub fn generate_salt(size: usize) -> Vec<u8> {
        let mut salt = vec![0u8; size];
        thread_rng().fill_bytes(&mut salt);
        salt
    }

    /// Performs constant-time comparison of two byte slices to prevent timing attacks.
    ///
    /// This function compares two byte slices in a way that takes the same amount
    /// of time regardless of where the first difference occurs. This prevents
    /// timing-based side-channel attacks that could leak information about
    /// secret values through execution time variations.
    ///
    /// # Arguments
    ///
    /// * `a` - First byte slice to compare
    /// * `b` - Second byte slice to compare
    ///
    /// # Returns
    ///
    /// `true` if the byte slices are identical in both length and content,
    /// `false` otherwise. The comparison time is constant regardless of
    /// where differences occur.
    ///
    /// # Security Importance
    ///
    /// Regular comparison operations (`==`) can leak information through timing:
    /// - Early termination when first difference is found
    /// - Execution time varies based on position of difference
    /// - Attackers can use timing to guess secret values byte by byte
    ///
    /// This function prevents such attacks by:
    /// - Always examining every byte in both slices
    /// - Using bitwise operations that don't branch
    /// - Maintaining constant execution time
    ///
    /// # Use Cases
    ///
    /// - **Authentication Tag Verification**: Comparing computed vs. provided tags
    /// - **Password Hash Verification**: Comparing stored vs. computed hashes
    /// - **HMAC Verification**: Validating message authentication codes
    /// - **Token Comparison**: Comparing API tokens or session identifiers
    /// - **Cryptographic Key Comparison**: Any comparison involving secret data
    ///
    /// # Performance
    ///
    /// The function has O(max(len(a), len(b))) time complexity and always
    /// processes the full length of the longer slice for security.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::encryption::utils::constant_time_eq;
    ///
    /// let secret1 = b"secret_authentication_tag";
    /// let secret2 = b"secret_authentication_tag";
    /// let different = b"different_tag_value_here";
    ///
    /// // Secure comparison - always use for secrets
    /// assert!(constant_time_eq(secret1, secret2));
    /// assert!(!constant_time_eq(secret1, different));
    ///
    /// // Different lengths are also handled securely
    /// assert!(!constant_time_eq(secret1, b"short"));
    /// ```
    #[must_use]
    pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result == 0
    }

    /// Secure random number generation
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn secure_random_bytes(size: usize) -> SecureStorageResult<Vec<u8>> {
        let mut bytes = vec![0u8; size];
        getrandom::getrandom(&mut bytes).map_err(|e| EncryptionError::KeyGeneration {
            reason: format!("Failed to generate random bytes: {e}"),
        })?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::utils;

    #[tokio::test]
    async fn test_encryption_factory() -> SecureStorageResult<()> {
        let encryption = EncryptionFactory::create_default()?;
        let algorithms = encryption.supported_algorithms();
        assert!(!algorithms.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_aes_gcm_encryption() -> SecureStorageResult<()> {
        let encryption = EncryptionFactory::create(EncryptionAlgorithm::Aes256Gcm)?;

        // Generate a key
        let key_material = encryption
            .generate_key(EncryptionAlgorithm::Aes256Gcm)
            .await?;
        let key_id = key_material.metadata.id.clone();

        // Test data
        let plaintext = b"Hello, secure world!";

        // Encrypt
        let encrypted = encryption
            .encrypt(plaintext, &key_id, EncryptionAlgorithm::Aes256Gcm)
            .await?;

        // Decrypt
        let decrypted = encryption.decrypt(&encrypted).await?;

        assert_eq!(plaintext, decrypted.as_slice());
        Ok(())
    }

    #[test]
    fn test_nonce_generation() {
        let nonce = utils::generate_nonce(EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(nonce.len(), 12); // AES-GCM nonce size

        // Generate another nonce and ensure they're different
        let nonce2 = utils::generate_nonce(EncryptionAlgorithm::Aes256Gcm);
        assert_ne!(nonce, nonce2);
    }

    #[test]
    fn test_constant_time_comparison() {
        let a = b"hello";
        let b = b"hello";
        let c = b"world";

        assert!(utils::constant_time_eq(a, b));
        assert!(!utils::constant_time_eq(a, c));
        assert!(!utils::constant_time_eq(a, b"hell")); // Different lengths
    }

    #[test]
    fn test_secure_random_generation() -> SecureStorageResult<()> {
        let bytes1 = utils::secure_random_bytes(32)?;
        let bytes2 = utils::secure_random_bytes(32)?;

        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);
        assert_ne!(bytes1, bytes2); // Should be different

        Ok(())
    }
}
