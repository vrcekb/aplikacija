//! # Memory Protection Module
//!
//! Ultra-secure memory management for `TallyIO` financial platform.
//! Provides memory locking, secure wiping, and protection against memory attacks.

use crate::error::{SecureStorageError, SecureStorageResult};
use std::sync::atomic::{AtomicBool, Ordering};

pub mod secure_buffer;

/// Memory protection configuration
#[derive(Debug, Clone)]
/// TODO: Add documentation
pub struct MemoryProtectionConfig {
    /// Enable memory locking (mlock)
    pub enable_mlock: bool,
    /// Enable secure memory wiping
    pub enable_secure_wipe: bool,
    /// Maximum locked memory size in bytes
    pub max_locked_memory: usize,
    /// Enable memory protection (mprotect)
    pub enable_mprotect: bool,
}

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enable_mlock: true,
            enable_secure_wipe: true,
            max_locked_memory: 64 * 1024 * 1024, // 64MB
            enable_mprotect: true,
        }
    }
}

/// Global memory protection state
static MEMORY_PROTECTION_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize memory protection subsystem
///
/// # Errors
///
/// Returns `SecureStorageError::MemoryProtection` if initialization fails
/// Returns `SecureStorageError::Internal` if system calls fail
///
/// # Errors
///
/// Returns error if operation fails
pub fn initialize_memory_protection(config: &MemoryProtectionConfig) -> SecureStorageResult<()> {
    if MEMORY_PROTECTION_INITIALIZED.load(Ordering::Acquire) {
        return Ok(());
    }

    // Set memory limits for locked memory
    if config.enable_mlock {
        #[cfg(unix)]
        {
            use libc::{rlimit, setrlimit, RLIMIT_MEMLOCK, RLIM_INFINITY};

            let limit = rlimit {
                rlim_cur: config.max_locked_memory as u64,
                rlim_max: RLIM_INFINITY,
            };

            let result = unsafe { setrlimit(RLIMIT_MEMLOCK, &limit) };
            if result != 0 {
                return Err(SecureStorageError::MemoryProtection {
                    operation: "setrlimit_memlock".to_string(),
                    reason: "Failed to set memory lock limits".to_string(),
                });
            }
        }
    }

    // Disable core dumps for security
    #[cfg(unix)]
    {
        use libc::{rlimit, setrlimit, RLIMIT_CORE};

        let limit = rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };

        let result = unsafe { setrlimit(RLIMIT_CORE, &limit) };
        if result != 0 {
            return Err(SecureStorageError::MemoryProtection {
                operation: "setrlimit_core".to_string(),
                reason: "Failed to disable core dumps".to_string(),
            });
        }
    }

    MEMORY_PROTECTION_INITIALIZED.store(true, Ordering::Release);
    tracing::info!("Memory protection initialized successfully");

    Ok(())
}

/// Secure memory wiping function
///
/// Uses zeroize crate for secure memory wiping
#[inline(always)]
/// TODO: Add documentation
pub fn secure_wipe(data: &mut [u8]) {
    use zeroize::Zeroize;
    data.zeroize();
}

/// Secure memory comparison function
///
/// Constant-time comparison to prevent timing attacks
///
/// # Errors
///
/// Returns `SecureStorageError::InvalidInput` if slices have different lengths
///
/// # Errors
///
/// Returns error if operation fails
pub fn secure_compare(a: &[u8], b: &[u8]) -> SecureStorageResult<bool> {
    if a.len() != b.len() {
        return Err(SecureStorageError::InvalidInput {
            field: "slice_length".to_string(),
            reason: "Slices must have equal length for secure comparison".to_string(),
        });
    }

    let mut result = 0u8;
    for (byte_a, byte_b) in a.iter().zip(b.iter()) {
        result |= byte_a ^ byte_b;
    }

    Ok(result == 0)
}

/// Memory protection utilities
pub struct MemoryProtection;

impl MemoryProtection {
    /// Lock memory pages to prevent swapping
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if mlock fails
    #[cfg(unix)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn lock_memory(ptr: *const u8, len: usize) -> SecureStorageResult<()> {
        use libc::mlock;

        let result = unsafe { mlock(ptr.cast::<libc::c_void>(), len) };
        if result != 0 {
            return Err(SecureStorageError::MemoryProtection {
                operation: "mlock".to_string(),
                reason: format!(
                    "Failed to lock memory: errno {}",
                    std::io::Error::last_os_error()
                ),
            });
        }

        Ok(())
    }

    /// Unlock memory pages
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if munlock fails
    #[cfg(unix)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn unlock_memory(ptr: *const u8, len: usize) -> SecureStorageResult<()> {
        use libc::munlock;

        let result = unsafe { munlock(ptr.cast::<libc::c_void>(), len) };
        if result != 0 {
            return Err(SecureStorageError::MemoryProtection {
                operation: "munlock".to_string(),
                reason: format!(
                    "Failed to unlock memory: errno {}",
                    std::io::Error::last_os_error()
                ),
            });
        }

        Ok(())
    }

    /// Set memory protection flags
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::MemoryProtection` if mprotect fails
    #[cfg(unix)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn protect_memory(ptr: *const u8, len: usize, read_only: bool) -> SecureStorageResult<()> {
        use libc::{mprotect, PROT_READ, PROT_WRITE};

        let prot = if read_only {
            PROT_READ
        } else {
            PROT_READ | PROT_WRITE
        };

        let result = unsafe { mprotect(ptr.cast::<libc::c_void>(), len, prot) };
        if result != 0 {
            return Err(SecureStorageError::MemoryProtection {
                operation: "mprotect".to_string(),
                reason: format!(
                    "Failed to protect memory: errno {}",
                    std::io::Error::last_os_error()
                ),
            });
        }

        Ok(())
    }

    /// Windows implementations (stubs for now)
    ///
    /// # Errors
    ///
    /// Currently always returns Ok(()) - Windows implementation pending
    #[cfg(windows)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub const fn lock_memory(_ptr: *const u8, _len: usize) -> SecureStorageResult<()> {
        // TODO: Implement VirtualLock for Windows
        Ok(())
    }

    /// Unlock memory on Windows
    ///
    /// # Errors
    ///
    /// Currently always returns Ok(()) - Windows implementation pending
    #[cfg(windows)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub const fn unlock_memory(_ptr: *const u8, _len: usize) -> SecureStorageResult<()> {
        // TODO: Implement VirtualUnlock for Windows
        Ok(())
    }

    /// Protect memory on Windows
    ///
    /// # Errors
    ///
    /// Currently always returns Ok(()) - Windows implementation pending
    #[cfg(windows)]
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub const fn protect_memory(
        _ptr: *const u8,
        _len: usize,
        _read_only: bool,
    ) -> SecureStorageResult<()> {
        // TODO: Implement VirtualProtect for Windows
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_protection_config() {
        let config = MemoryProtectionConfig::default();
        assert!(config.enable_mlock);
        assert!(config.enable_secure_wipe);
        assert_eq!(config.max_locked_memory, 64 * 1024 * 1024);
    }

    #[test]
    fn test_secure_wipe() {
        let mut data = vec![0xFF_u8; 1024];
        secure_wipe(&mut data);

        // Verify all bytes are zero
        for byte in &data {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn test_secure_compare() -> SecureStorageResult<()> {
        let a = b"hello world";
        let b = b"hello world";
        let c = b"hello rust!";

        assert!(secure_compare(a, b)?);
        assert!(!secure_compare(a, c)?);

        // Test different lengths
        let d = b"short";
        assert!(secure_compare(a, d).is_err());

        Ok(())
    }

    #[test]
    fn test_memory_protection_initialization() -> SecureStorageResult<()> {
        let config = MemoryProtectionConfig::default();
        initialize_memory_protection(&config)?;

        // Second call should succeed (already initialized)
        initialize_memory_protection(&config)?;

        Ok(())
    }
}
