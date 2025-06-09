//! # Fixed-Size Array Operations
//!
//! Compile-time sized arrays for zero-allocation cryptographic operations.

use super::HASH_SIZE;
use super::{Hash, PublicKey, Signature, SymmetricKey};
use crate::error::{SecureStorageError, SecureStorageResult};
use zeroize::Zeroize;

/// Fixed-size cryptographic workspace
#[repr(C, align(64))]
pub struct CryptoWorkspace<const SIZE: usize> {
    /// Working buffer
    buffer: [u8; SIZE],
    /// Current position in buffer
    position: usize,
}

impl<const SIZE: usize> CryptoWorkspace<SIZE> {
    /// Create a new workspace
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: [0u8; SIZE],
            position: 0,
        }
    }

    /// Reset workspace to initial state
    pub fn reset(&mut self) {
        self.buffer.zeroize();
        self.position = 0;
    }

    /// Allocate space in workspace
    ///
    /// # Errors
    ///
    /// Returns error if insufficient space
    pub fn allocate(&mut self, size: usize) -> SecureStorageResult<&mut [u8]> {
        if self.position + size > SIZE {
            return Err(SecureStorageError::InsufficientResources {
                resource: "workspace_memory".to_string(),
                reason: format!(
                    "Required {} bytes, available {} bytes",
                    size,
                    SIZE - self.position
                ),
            });
        }

        let start = self.position;
        self.position += size;
        Ok(&mut self.buffer[start..self.position])
    }

    /// Get remaining space
    #[must_use]
    pub const fn remaining_space(&self) -> usize {
        SIZE - self.position
    }

    /// Get used space
    #[must_use]
    pub const fn used_space(&self) -> usize {
        self.position
    }

    /// Get total capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        SIZE
    }

    /// Check if workspace is full
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.position >= SIZE
    }

    /// Check if workspace is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.position == 0
    }
}

impl<const SIZE: usize> Drop for CryptoWorkspace<SIZE> {
    fn drop(&mut self) {
        self.buffer.zeroize();
    }
}

impl<const SIZE: usize> Default for CryptoWorkspace<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size key storage
#[derive(Debug, Clone)]
pub struct FixedKeyStorage<const N: usize> {
    /// Array of keys
    keys: [Option<SymmetricKey>; N],
    /// Number of stored keys
    count: usize,
}

impl<const N: usize> FixedKeyStorage<N> {
    /// Create new key storage
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keys: [None; N],
            count: 0,
        }
    }

    /// Store a key
    ///
    /// # Errors
    ///
    /// Returns error if storage is full
    pub fn store_key(&mut self, key: SymmetricKey) -> SecureStorageResult<usize> {
        if self.count >= N {
            return Err(SecureStorageError::InsufficientResources {
                resource: "key_storage".to_string(),
                reason: format!("Storage full: {}/{} keys used", self.count, N),
            });
        }

        let index = self.count;
        self.keys[index] = Some(key);
        self.count += 1;
        Ok(index)
    }

    /// Get a key by index
    ///
    /// # Errors
    ///
    /// Returns error if index is invalid
    pub fn get_key(&self, index: usize) -> SecureStorageResult<&SymmetricKey> {
        if index >= self.count {
            return Err(SecureStorageError::InvalidInput {
                field: "key_index".to_string(),
                reason: format!("Index {} out of bounds (count: {})", index, self.count),
            });
        }

        self.keys[index]
            .as_ref()
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "key".to_string(),
                identifier: format!("index_{index}"),
            })
    }

    /// Remove a key by index
    ///
    /// # Errors
    ///
    /// Returns error if index is invalid
    pub fn remove_key(&mut self, index: usize) -> SecureStorageResult<SymmetricKey> {
        if index >= self.count {
            return Err(SecureStorageError::InvalidInput {
                field: "key_index".to_string(),
                reason: format!("Index {} out of bounds (count: {})", index, self.count),
            });
        }

        let key = self.keys[index]
            .take()
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "key".to_string(),
                identifier: format!("index_{index}"),
            })?;

        // Shift remaining keys down
        for i in index..self.count - 1 {
            self.keys[i] = self.keys[i + 1].take();
        }
        self.count -= 1;

        Ok(key)
    }

    /// Get number of stored keys
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if storage is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if storage is full
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Get capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        for key in &mut self.keys[..self.count] {
            if let Some(ref mut k) = key {
                k.zeroize();
            }
            *key = None;
        }
        self.count = 0;
    }
}

impl<const N: usize> Drop for FixedKeyStorage<N> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<const N: usize> Default for FixedKeyStorage<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size signature batch
#[derive(Debug, Clone)]
pub struct SignatureBatch<const N: usize> {
    /// Array of signatures
    signatures: [Option<Signature>; N],
    /// Array of public keys
    public_keys: [Option<PublicKey>; N],
    /// Array of message hashes
    message_hashes: [Option<Hash>; N],
    /// Number of signatures in batch
    count: usize,
}

impl<const N: usize> SignatureBatch<N> {
    /// Create new signature batch
    #[must_use]
    pub const fn new() -> Self {
        Self {
            signatures: [None; N],
            public_keys: [None; N],
            message_hashes: [None; N],
            count: 0,
        }
    }

    /// Add signature to batch
    ///
    /// # Errors
    ///
    /// Returns error if batch is full
    pub fn add_signature(
        &mut self,
        signature: Signature,
        public_key: PublicKey,
        message_hash: Hash,
    ) -> SecureStorageResult<()> {
        if self.count >= N {
            return Err(SecureStorageError::InsufficientResources {
                resource: "signature_batch".to_string(),
                reason: format!("Batch full: {}/{} signatures", self.count, N),
            });
        }

        let index = self.count;
        self.signatures[index] = Some(signature);
        self.public_keys[index] = Some(public_key);
        self.message_hashes[index] = Some(message_hash);
        self.count += 1;

        Ok(())
    }

    /// Verify all signatures in batch
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub fn verify_batch(&self) -> SecureStorageResult<Vec<bool>> {
        let mut results = Vec::with_capacity(self.count);

        for i in 0..self.count {
            let signature =
                self.signatures[i]
                    .as_ref()
                    .ok_or_else(|| SecureStorageError::NotFound {
                        resource: "signature".to_string(),
                        identifier: format!("index_{i}"),
                    })?;

            let public_key =
                self.public_keys[i]
                    .as_ref()
                    .ok_or_else(|| SecureStorageError::NotFound {
                        resource: "public_key".to_string(),
                        identifier: format!("index_{i}"),
                    })?;

            let message_hash =
                self.message_hashes[i]
                    .as_ref()
                    .ok_or_else(|| SecureStorageError::NotFound {
                        resource: "message_hash".to_string(),
                        identifier: format!("index_{i}"),
                    })?;

            // Simulate signature verification
            let is_valid = verify_signature_simulation(signature, public_key, message_hash);
            results.push(is_valid);
        }

        Ok(results)
    }

    /// Get batch size
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if batch is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if batch is full
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Get capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Clear batch
    pub fn clear(&mut self) {
        for i in 0..self.count {
            self.signatures[i] = None;
            self.public_keys[i] = None;
            self.message_hashes[i] = None;
        }
        self.count = 0;
    }
}

impl<const N: usize> Default for SignatureBatch<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulate signature verification
const fn verify_signature_simulation(
    _signature: &Signature,
    _public_key: &PublicKey,
    _message_hash: &Hash,
) -> bool {
    // Simple simulation - in production, use actual cryptographic verification
    true
}

/// Fixed-size hash chain
#[derive(Debug, Clone)]
pub struct HashChain<const N: usize> {
    /// Array of hashes
    hashes: [Hash; N],
    /// Current position in chain
    position: usize,
}

impl<const N: usize> HashChain<N> {
    /// Create new hash chain with initial hash
    #[must_use]
    pub const fn new(initial_hash: Hash) -> Self {
        let mut hashes = [[0u8; HASH_SIZE]; N];
        hashes[0] = initial_hash;

        Self {
            hashes,
            position: 1,
        }
    }

    /// Add next hash to chain
    ///
    /// # Errors
    ///
    /// Returns error if chain is full
    pub fn add_hash(&mut self, data: &[u8]) -> SecureStorageResult<Hash> {
        if self.position >= N {
            return Err(SecureStorageError::InsufficientResources {
                resource: "hash_chain".to_string(),
                reason: format!("Chain full: {}/{} hashes", self.position, N),
            });
        }

        // Compute hash of previous hash + new data
        let mut input = [0u8; HASH_SIZE + 1024]; // Max data size
        if data.len() > 1024 {
            return Err(SecureStorageError::InvalidInput {
                field: "data_size".to_string(),
                reason: "Data too large for hash chain".to_string(),
            });
        }

        input[..HASH_SIZE].copy_from_slice(&self.hashes[self.position - 1]);
        input[HASH_SIZE..HASH_SIZE + data.len()].copy_from_slice(data);

        let mut new_hash = [0u8; HASH_SIZE];
        compute_hash_simulation(&input[..HASH_SIZE + data.len()], &mut new_hash);

        self.hashes[self.position] = new_hash;
        self.position += 1;

        Ok(new_hash)
    }

    /// Get hash at position
    ///
    /// # Errors
    ///
    /// Returns error if position is invalid
    pub fn get_hash(&self, index: usize) -> SecureStorageResult<&Hash> {
        if index >= self.position {
            return Err(SecureStorageError::InvalidInput {
                field: "hash_index".to_string(),
                reason: format!(
                    "Index {} out of bounds (position: {})",
                    index, self.position
                ),
            });
        }

        Ok(&self.hashes[index])
    }

    /// Get current hash (latest)
    #[must_use]
    pub const fn current_hash(&self) -> &Hash {
        &self.hashes[self.position - 1]
    }

    /// Get chain length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.position
    }

    /// Check if chain is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Check if chain is full
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.position >= N
    }

    /// Get capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<const N: usize> Drop for HashChain<N> {
    fn drop(&mut self) {
        for hash in &mut self.hashes[..self.position] {
            hash.zeroize();
        }
    }
}

/// Simulate hash computation
fn compute_hash_simulation(input: &[u8], output: &mut [u8; HASH_SIZE]) {
    // Simple hash simulation
    for (i, &byte) in input.iter().enumerate() {
        output[i % HASH_SIZE] = output[i % HASH_SIZE].wrapping_add(byte);
    }
}

/// Common workspace sizes
pub type SmallWorkspace = CryptoWorkspace<1024>; // 1KB
/// Medium workspace (4KB)
pub type MediumWorkspace = CryptoWorkspace<4096>; // 4KB
/// Large workspace (16KB)
pub type LargeWorkspace = CryptoWorkspace<16384>; // 16KB

/// Common storage sizes
pub type SmallKeyStorage = FixedKeyStorage<16>; // 16 keys
/// Medium key storage (64 keys)
pub type MediumKeyStorage = FixedKeyStorage<64>; // 64 keys
/// Large key storage (256 keys)
pub type LargeKeyStorage = FixedKeyStorage<256>; // 256 keys

/// Common batch sizes
pub type SmallSignatureBatch = SignatureBatch<8>; // 8 signatures
/// Medium signature batch (32 signatures)
pub type MediumSignatureBatch = SignatureBatch<32>; // 32 signatures
/// Large signature batch (128 signatures)
pub type LargeSignatureBatch = SignatureBatch<128>; // 128 signatures

/// Common chain sizes
pub type SmallHashChain = HashChain<32>; // 32 hashes
/// Medium hash chain (128 hashes)
pub type MediumHashChain = HashChain<128>; // 128 hashes
/// Large hash chain (512 hashes)
pub type LargeHashChain = HashChain<512>; // 512 hashes
