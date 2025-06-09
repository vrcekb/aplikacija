//! # Secure Buffer Implementation
//!
//! Memory-protected buffer for sensitive data with automatic secure wiping.

use super::{secure_wipe, MemoryProtection};
use crate::error::{SecureStorageError, SecureStorageResult};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Secure buffer with memory protection and automatic wiping
#[derive(Debug)]
/// TODO: Add documentation
pub struct SecureBuffer {
    data: Vec<u8>,
    locked: AtomicBool,
    capacity: usize,
}

impl SecureBuffer {
    /// Create a new secure buffer with specified capacity
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if memory locking fails
    /// Returns `SecureStorageError::InvalidInput` if capacity is zero
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn new(capacity: usize) -> SecureStorageResult<Self> {
        if capacity == 0 {
            return Err(SecureStorageError::InvalidInput {
                field: "capacity".to_string(),
                reason: "Capacity must be greater than zero".to_string(),
            });
        }

        let data = vec![0; capacity];

        // Lock memory to prevent swapping
        MemoryProtection::lock_memory(data.as_ptr(), capacity)?;

        let buffer = Self {
            data,
            locked: AtomicBool::new(true),
            capacity,
        };

        Ok(buffer)
    }

    /// Create a secure buffer from existing data
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if memory locking fails
    /// Returns `SecureStorageError::InvalidInput` if data is empty
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn from_data(data: Vec<u8>) -> SecureStorageResult<Self> {
        if data.is_empty() {
            return Err(SecureStorageError::InvalidInput {
                field: "data".to_string(),
                reason: "Data cannot be empty".to_string(),
            });
        }

        let capacity = data.len();

        // Lock memory to prevent swapping
        MemoryProtection::lock_memory(data.as_ptr(), capacity)?;

        let buffer = Self {
            data,
            locked: AtomicBool::new(true),
            capacity,
        };

        Ok(buffer)
    }

    /// Get the buffer capacity
    #[must_use]
    /// TODO: Add documentation
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current data length
    #[must_use]
    /// TODO: Add documentation
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    #[must_use]
    /// TODO: Add documentation
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Write data to the buffer
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::InvalidInput` if data exceeds capacity
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn write(&mut self, data: &[u8]) -> SecureStorageResult<()> {
        if data.len() > self.capacity {
            return Err(SecureStorageError::InvalidInput {
                field: "data_length".to_string(),
                reason: format!(
                    "Data length {} exceeds capacity {}",
                    data.len(),
                    self.capacity
                ),
            });
        }

        // Clear existing data
        self.data.zeroize();

        // Resize to match input data
        self.data.resize(data.len(), 0);

        // Copy new data
        self.data.copy_from_slice(data);

        Ok(())
    }

    /// Read data from the buffer
    ///
    /// # Safety
    ///
    /// The returned slice is valid only as long as the buffer exists
    /// and is not modified
    #[must_use]
    /// TODO: Add documentation
    pub fn read(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable access to buffer data
    ///
    /// # Safety
    ///
    /// Caller must ensure data integrity and security
    #[must_use]
    /// TODO: Add documentation
    pub fn read_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Append data to the buffer
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::InvalidInput` if resulting length exceeds capacity
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn append(&mut self, data: &[u8]) -> SecureStorageResult<()> {
        if self.data.len() + data.len() > self.capacity {
            return Err(SecureStorageError::InvalidInput {
                field: "data_length".to_string(),
                reason: format!(
                    "Resulting length {} exceeds capacity {}",
                    self.data.len() + data.len(),
                    self.capacity
                ),
            });
        }

        self.data.extend_from_slice(data);
        Ok(())
    }

    /// Clear the buffer securely
    pub fn clear(&mut self) {
        secure_wipe(&mut self.data);
        self.data.clear();
    }

    /// Resize the buffer
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::InvalidInput` if new size exceeds capacity
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn resize(&mut self, new_len: usize) -> SecureStorageResult<()> {
        if new_len > self.capacity {
            return Err(SecureStorageError::InvalidInput {
                field: "new_length".to_string(),
                reason: format!("New length {} exceeds capacity {}", new_len, self.capacity),
            });
        }

        // If shrinking, securely wipe the excess data
        if new_len < self.data.len() {
            let excess_start = new_len;
            secure_wipe(&mut self.data[excess_start..]);
        }

        self.data.resize(new_len, 0);
        Ok(())
    }

    /// Check if memory is locked
    #[must_use]
    /// TODO: Add documentation
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Acquire)
    }

    /// Unlock memory (for testing purposes only)
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if unlocking fails
    #[cfg(test)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn unlock_for_test(&self) -> SecureStorageResult<()> {
        if self.locked.load(Ordering::Acquire) {
            MemoryProtection::unlock_memory(self.data.as_ptr(), self.capacity)?;
            self.locked.store(false, Ordering::Release);
        }
        Ok(())
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        // Securely wipe the buffer
        secure_wipe(&mut self.data);

        // Unlock memory if it was locked
        if self.locked.load(Ordering::Acquire) {
            let _ = MemoryProtection::unlock_memory(self.data.as_ptr(), self.capacity);
        }
    }
}

// Implement Zeroize traits for additional security
impl Zeroize for SecureBuffer {
    fn zeroize(&mut self) {
        self.clear();
    }
}

impl ZeroizeOnDrop for SecureBuffer {}

// Manual Clone implementation for SecureBuffer
impl Clone for SecureBuffer {
    fn clone(&self) -> Self {
        // Create new buffer with same capacity
        let mut new_buffer = Self::new(self.capacity).unwrap_or_else(|_| {
            // Fallback to minimal buffer if allocation fails
            // For production financial applications, we cannot panic
            Self::new(1).unwrap_or_else(|_| {
                // Ultimate fallback - create empty buffer
                Self {
                    data: Vec::with_capacity(0),
                    capacity: 0,
                    locked: AtomicBool::new(false),
                }
            })
        });

        // Copy data securely
        let _ = new_buffer.write(self.read());

        new_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_buffer_creation() -> SecureStorageResult<()> {
        let buffer = SecureBuffer::new(1024)?;
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 1024);
        assert!(buffer.is_locked());
        Ok(())
    }

    #[test]
    fn test_secure_buffer_from_data() -> SecureStorageResult<()> {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = SecureBuffer::from_data(data)?;
        assert_eq!(buffer.capacity(), 5);
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.read(), &[1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn test_secure_buffer_write_read() -> SecureStorageResult<()> {
        let mut buffer = SecureBuffer::new(1024)?;
        let test_data = b"Hello, secure world!";

        buffer.write(test_data)?;
        assert_eq!(buffer.read(), test_data);
        assert_eq!(buffer.len(), test_data.len());

        Ok(())
    }

    #[test]
    fn test_secure_buffer_append() -> SecureStorageResult<()> {
        let mut buffer = SecureBuffer::new(1024)?;

        buffer.write(b"Hello")?;
        buffer.append(b", world!")?;

        assert_eq!(buffer.read(), b"Hello, world!");
        Ok(())
    }

    #[test]
    fn test_secure_buffer_clear() -> SecureStorageResult<()> {
        let mut buffer = SecureBuffer::new(1024)?;
        buffer.write(b"sensitive data")?;

        buffer.clear();
        assert!(buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_secure_buffer_resize() -> SecureStorageResult<()> {
        let mut buffer = SecureBuffer::new(1024)?;
        buffer.write(b"test data")?;

        // Shrink
        buffer.resize(4)?;
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.read(), b"test");

        // Grow
        buffer.resize(8)?;
        assert_eq!(buffer.len(), 8);

        Ok(())
    }

    #[test]
    fn test_secure_buffer_capacity_limits() {
        // Test zero capacity
        assert!(SecureBuffer::new(0).is_err());

        // Test empty data
        assert!(SecureBuffer::from_data(Vec::with_capacity(0)).is_err());
    }

    #[test]
    fn test_secure_buffer_write_overflow() -> SecureStorageResult<()> {
        let mut buffer = SecureBuffer::new(5)?;
        let large_data = vec![0xFF; 10];

        // Should fail - data too large
        assert!(buffer.write(&large_data).is_err());

        Ok(())
    }
}
