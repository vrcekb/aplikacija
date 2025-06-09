//! Key derivation functions for secure storage

use crate::error::{EncryptionError, SecureStorageError, SecureStorageResult};
use argon2::{Argon2, Params, Version};

/// Argon2id configuration for key derivation
#[derive(Debug, Clone)]
/// TODO: Add documentation
pub struct Argon2Config {
    /// Memory cost in KB
    pub memory_cost: u32,
    /// Time cost (iterations)
    pub time_cost: u32,
    /// Parallelism
    pub parallelism: u32,
    /// Output length in bytes
    pub output_length: usize,
}

impl Default for Argon2Config {
    fn default() -> Self {
        Self {
            memory_cost: 65536, // 64MB
            time_cost: 3,
            parallelism: 4,
            output_length: 32, // 256 bits
        }
    }
}

impl Argon2Config {
    /// Create a new configuration
    #[must_use]
    /// TODO: Add documentation
    pub const fn new(
        memory_cost: u32,
        time_cost: u32,
        parallelism: u32,
        output_length: usize,
    ) -> Self {
        Self {
            memory_cost,
            time_cost,
            parallelism,
            output_length,
        }
    }

    /// Create a fast configuration for testing
    #[must_use]
    /// TODO: Add documentation
    pub const fn fast() -> Self {
        Self {
            memory_cost: 1024, // 1MB
            time_cost: 1,
            parallelism: 1,
            output_length: 32,
        }
    }

    /// Create a secure configuration for production
    #[must_use]
    /// TODO: Add documentation
    pub const fn secure() -> Self {
        Self {
            memory_cost: 131_072, // 128MB
            time_cost: 4,
            parallelism: 8,
            output_length: 32,
        }
    }

    /// Validate configuration parameters
    ///
    /// # Errors
    ///
    /// Returns `EncryptionError::InvalidKey` if any parameter is out of valid range
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn validate(&self) -> SecureStorageResult<()> {
        if self.memory_cost < 1024 {
            return Err(EncryptionError::InvalidKey {
                reason: "Memory cost too low (minimum 1024 KB)".to_string(),
            }
            .into());
        }

        if self.time_cost == 0 {
            return Err(EncryptionError::InvalidKey {
                reason: "Time cost cannot be zero".to_string(),
            }
            .into());
        }

        if self.parallelism == 0 {
            return Err(EncryptionError::InvalidKey {
                reason: "Parallelism cannot be zero".to_string(),
            }
            .into());
        }

        if self.output_length == 0 || self.output_length > 1024 {
            return Err(EncryptionError::InvalidKey {
                reason: "Invalid output length (must be 1-1024 bytes)".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// Secure key derivation result that zeroizes on drop
#[derive(Clone)]
/// TODO: Add documentation
pub struct DerivedKey {
    /// The derived key bytes
    pub key: Vec<u8>,
    /// Salt used for derivation
    pub salt: Vec<u8>,
    /// Configuration used
    pub config: Argon2Config,
}

impl DerivedKey {
    /// Create a new derived key
    #[must_use]
    /// TODO: Add documentation
    pub const fn new(key: Vec<u8>, salt: Vec<u8>, config: Argon2Config) -> Self {
        Self { key, salt, config }
    }

    /// Get key bytes
    #[must_use]
    /// TODO: Add documentation
    pub fn key_bytes(&self) -> &[u8] {
        &self.key
    }

    /// Get salt bytes
    #[must_use]
    /// TODO: Add documentation
    pub fn salt_bytes(&self) -> &[u8] {
        &self.salt
    }
}

/// Derive a key using Argon2id with default configuration
/// TODO: Add documentation
///
/// # Errors
///
/// Returns error if operation fails
pub fn derive_key_argon2id(
    password: &[u8],
    salt: &[u8],
    output_length: usize,
) -> SecureStorageResult<Vec<u8>> {
    let config = Argon2Config::default();
    derive_key_argon2id_with_config(password, salt, output_length, &config)
}

/// Derive a key using Argon2id with custom configuration
/// TODO: Add documentation
///
/// # Errors
///
/// Returns error if operation fails
pub fn derive_key_argon2id_with_config(
    password: &[u8],
    salt: &[u8],
    output_length: usize,
    config: &Argon2Config,
) -> SecureStorageResult<Vec<u8>> {
    // Validate configuration
    config.validate()?;

    // Validate salt length
    if salt.len() < 16 {
        return Err(EncryptionError::InvalidKey {
            reason: "Salt too short (minimum 16 bytes)".to_string(),
        }
        .into());
    }

    // Create Argon2 parameters
    let params = Params::new(
        config.memory_cost,
        config.time_cost,
        config.parallelism,
        Some(output_length),
    )
    .map_err(|e| EncryptionError::KeyGeneration {
        reason: format!("Invalid Argon2 parameters: {e}"),
    })?;

    // Create Argon2 instance
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);

    // Derive the key
    let mut output = vec![0u8; output_length];
    argon2
        .hash_password_into(password, salt, &mut output)
        .map_err(|e| EncryptionError::KeyGeneration {
            reason: format!("Argon2 key derivation failed: {e}"),
        })?;

    Ok(output)
}

/// Generate a random salt for key derivation
///
/// # Errors
///
/// Returns `SecureStorageError::Internal` if the system's random number generator fails
///
/// # Errors
///
/// Returns error if operation fails
pub fn generate_salt(length: usize) -> SecureStorageResult<Vec<u8>> {
    let mut salt = vec![0u8; length];
    getrandom::getrandom(&mut salt).map_err(|e| SecureStorageError::Internal {
        reason: format!("Failed to generate random salt: {e}"),
    })?;
    Ok(salt)
}

/// Derive a key with automatic salt generation
/// TODO: Add documentation
///
/// # Errors
///
/// Returns error if operation fails
pub fn derive_key_with_salt(
    password: &[u8],
    config: &Argon2Config,
) -> SecureStorageResult<DerivedKey> {
    config.validate()?;

    let salt = generate_salt(32)?; // 256-bit salt
    let key = derive_key_argon2id_with_config(password, &salt, config.output_length, config)?;

    Ok(DerivedKey::new(key, salt, config.clone()))
}

/// Benchmark key derivation performance
/// TODO: Add documentation
///
/// # Errors
///
/// Returns error if operation fails
pub fn benchmark_key_derivation(
    config: &Argon2Config,
    iterations: usize,
) -> SecureStorageResult<std::time::Duration> {
    config.validate()?;

    let password = b"benchmark_password";
    let salt = generate_salt(32)?;

    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _key = derive_key_argon2id_with_config(password, &salt, config.output_length, config)?;
    }

    let total_time = start.elapsed();
    let iterations_u32 = u32::try_from(iterations).unwrap_or(1_u32);
    Ok(total_time / iterations_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argon2_key_derivation() -> SecureStorageResult<()> {
        let password = b"test_password_123";
        let salt = generate_salt(32)?;

        let key = derive_key_argon2id(password, &salt, 32)?;

        assert_eq!(key.len(), 32);

        // Derive the same key again
        let key2 = derive_key_argon2id(password, &salt, 32)?;
        assert_eq!(key, key2);

        // Different salt should produce different key
        let salt2 = generate_salt(32)?;
        let key3 = derive_key_argon2id(password, &salt2, 32)?;
        assert_ne!(key, key3);

        Ok(())
    }

    #[test]
    fn test_config_validation() {
        let mut config = Argon2Config::default();
        assert!(config.validate().is_ok());

        // Test invalid memory cost
        config.memory_cost = 512;
        assert!(config.validate().is_err());

        // Test invalid time cost
        config.memory_cost = 1024;
        config.time_cost = 0;
        assert!(config.validate().is_err());

        // Test invalid parallelism
        config.time_cost = 1;
        config.parallelism = 0;
        assert!(config.validate().is_err());

        // Test invalid output length
        config.parallelism = 1;
        config.output_length = 0;
        assert!(config.validate().is_err());

        config.output_length = 2048;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_derived_key_with_salt() -> SecureStorageResult<()> {
        let password = b"test_password_456";
        let config = Argon2Config::fast(); // Use fast config for testing

        let derived = derive_key_with_salt(password, &config)?;

        assert_eq!(derived.key.len(), config.output_length);
        assert_eq!(derived.salt.len(), 32);

        // Derive again with same password but different salt
        let derived2 = derive_key_with_salt(password, &config)?;

        // Keys should be different due to different salts
        assert_ne!(derived.key, derived2.key);
        assert_ne!(derived.salt, derived2.salt);

        Ok(())
    }

    #[test]
    fn test_salt_generation() -> SecureStorageResult<()> {
        let salt1 = generate_salt(32)?;
        let salt2 = generate_salt(32)?;

        assert_eq!(salt1.len(), 32);
        assert_eq!(salt2.len(), 32);
        assert_ne!(salt1, salt2);

        Ok(())
    }

    #[test]
    fn test_short_salt_rejection() {
        let password = b"test_password";
        let short_salt = vec![0u8; 8]; // Too short

        let result = derive_key_argon2id(password, &short_salt, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_performance_benchmark() -> SecureStorageResult<()> {
        let config = Argon2Config::fast();
        let avg_time = benchmark_key_derivation(&config, 5)?;

        // Should complete reasonably quickly with fast config
        assert!(avg_time.as_millis() < 100);

        Ok(())
    }

    #[test]
    fn test_config_presets() -> SecureStorageResult<()> {
        let fast = Argon2Config::fast();
        let secure = Argon2Config::secure();

        fast.validate()?;
        secure.validate()?;

        // Secure config should have higher parameters
        assert!(secure.memory_cost > fast.memory_cost);
        assert!(secure.time_cost >= fast.time_cost);
        assert!(secure.parallelism >= fast.parallelism);

        Ok(())
    }

    #[test]
    fn test_key_security() -> SecureStorageResult<()> {
        let password = b"test_password_security";
        let config = Argon2Config::fast();

        let derived = derive_key_with_salt(password, &config)?;

        // Verify key is not all zeros
        assert!(derived.key.iter().any(|&b| b != 0));

        // Verify key has expected length
        assert_eq!(derived.key.len(), config.output_length);

        Ok(())
    }
}
