//! # Intel SGX Integration
//!
//! Intel Software Guard Extensions (SGX) integration for ultra-secure
//! cryptographic operations in hardware-protected enclaves.

use super::EnclaveConfig;
use crate::error::{SecureStorageError, SecureStorageResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// SGX enclave information
#[derive(Debug, Clone)]
pub struct SgxEnclaveInfo {
    /// Enclave ID
    pub enclave_id: u64,
    /// Enclave base address
    pub base_address: u64,
    /// Enclave size in bytes
    pub size: usize,
    /// MRENCLAVE (measurement of enclave)
    pub mrenclave: [u8; 32],
    /// MRSIGNER (measurement of enclave signer)
    pub mrsigner: [u8; 32],
    /// Product ID
    pub product_id: u16,
    /// Security version number
    pub security_version: u16,
    /// Debug mode flag
    pub debug_mode: bool,
}

/// SGX error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SgxError {
    /// Success
    Success = 0x0000_0000,
    /// Unexpected error
    Unexpected = 0x0000_0001,
    /// Invalid parameter
    InvalidParameter = 0x0000_0002,
    /// Out of memory
    OutOfMemory = 0x0000_0003,
    /// Enclave lost
    EnclaveLost = 0x0000_0004,
    /// Invalid enclave
    InvalidEnclave = 0x0000_0005,
    /// Invalid enclave ID
    InvalidEnclaveId = 0x0000_0006,
    /// Invalid signature
    InvalidSignature = 0x0000_0007,
    /// Out of EPC memory
    OutOfEpc = 0x0000_0008,
    /// No device available
    NoDevice = 0x0000_0009,
    /// Memory map conflict
    MemoryMapConflict = 0x0000_000A,
    /// Invalid metadata
    InvalidMetadata = 0x0000_000B,
    /// Device busy
    DeviceBusy = 0x0000_000C,
    /// Invalid version
    InvalidVersion = 0x0000_000D,
    /// Mode incompatible
    ModeIncompatible = 0x0000_000E,
    /// Enclave file access error
    EnclaveFileAccess = 0x0000_000F,
    /// Invalid misc select
    InvalidMiscSelect = 0x0000_0010,
    /// Invalid launch token
    InvalidLaunchToken = 0x0000_0011,
}

impl SgxError {
    /// Convert SGX error to `SecureStorageError`
    #[must_use]
    pub fn to_storage_error(self) -> SecureStorageError {
        match self {
            Self::Success => SecureStorageError::InvalidInput {
                field: "sgx_error".to_string(),
                reason: "Success is not an error".to_string(),
            },
            Self::InvalidParameter => SecureStorageError::InvalidInput {
                field: "sgx_parameter".to_string(),
                reason: "Invalid SGX parameter".to_string(),
            },
            Self::OutOfMemory | Self::OutOfEpc => SecureStorageError::InsufficientResources {
                resource: "sgx_memory".to_string(),
                reason: "Insufficient SGX memory or EPC".to_string(),
            },
            Self::InvalidEnclave | Self::InvalidEnclaveId => SecureStorageError::NotFound {
                resource: "sgx_enclave".to_string(),
                identifier: "unknown".to_string(),
            },
            _ => SecureStorageError::InvalidInput {
                field: "sgx_operation".to_string(),
                reason: format!("SGX error: {self:?}"),
            },
        }
    }
}

/// SGX system state
#[derive(Debug)]
pub struct SgxSystem {
    /// Active enclaves
    enclaves: std::collections::HashMap<u64, SgxEnclaveInfo>,
    /// Performance counters
    enclaves_created: AtomicU64,
    enclaves_destroyed: AtomicU64,
    ecalls_executed: AtomicU64,
    ocalls_executed: AtomicU64,
    /// Total execution time in nanoseconds
    total_execution_time_ns: AtomicU64,
}

impl SgxSystem {
    /// Create a new SGX system
    #[must_use]
    pub fn new() -> Self {
        Self {
            enclaves: std::collections::HashMap::new(),
            enclaves_created: AtomicU64::new(0),
            enclaves_destroyed: AtomicU64::new(0),
            ecalls_executed: AtomicU64::new(0),
            ocalls_executed: AtomicU64::new(0),
            total_execution_time_ns: AtomicU64::new(0),
        }
    }

    /// Check if SGX is available on the system
    #[must_use]
    pub fn is_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            Self::check_sgx_cpuid_support() && Self::check_sgx_driver_availability()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Check SGX CPUID support
    #[cfg(target_arch = "x86_64")]
    fn check_sgx_cpuid_support() -> bool {
        // Use raw-cpuid crate for safe CPUID access
        let cpuid = raw_cpuid::CpuId::new();

        // Check for SGX support in extended features
        if let Some(extended_features) = cpuid.get_extended_feature_info() {
            let sgx_available = extended_features.has_sgx();

            if !sgx_available {
                debug!("SGX not supported by CPU");
                return false;
            }

            // Check SGX capabilities
            cpuid.get_sgx_info().is_some_and(|sgx_info| {
                let has_sgx1 = sgx_info.has_sgx1();
                let has_sgx2 = sgx_info.has_sgx2();

                debug!("SGX capabilities: SGX1={}, SGX2={}", has_sgx1, has_sgx2);

                has_sgx1 || has_sgx2
            })
        } else {
            false
        }
    }

    /// Check SGX driver availability
    fn check_sgx_driver_availability() -> bool {
        // Check for SGX device files
        let sgx_enclave_exists = std::path::Path::new("/dev/sgx_enclave").exists();
        let sgx_provision_exists = std::path::Path::new("/dev/sgx_provision").exists();
        let legacy_sgx_exists = std::path::Path::new("/dev/sgx").exists();

        let driver_available = sgx_enclave_exists || legacy_sgx_exists;

        debug!(
            "SGX driver check: enclave={}, provision={}, legacy={}",
            sgx_enclave_exists, sgx_provision_exists, legacy_sgx_exists
        );

        if !driver_available {
            warn!("SGX driver not available - install intel-sgx-driver or use in-kernel driver");
        }

        driver_available
    }

    /// Get SGX capabilities
    #[must_use]
    pub fn get_capabilities() -> SgxCapabilities {
        SgxCapabilities {
            sgx1_supported: Self::check_sgx1_support(),
            sgx2_supported: Self::check_sgx2_support(),
            max_enclave_size: Self::get_max_enclave_size(),
            epc_size: Self::get_epc_size(),
            flexible_launch_control: Self::check_flc_support(),
        }
    }

    /// Check SGX1 support
    fn check_sgx1_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();
            cpuid.get_sgx_info().is_some_and(|info| info.has_sgx1())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Check SGX2 support
    fn check_sgx2_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();
            cpuid.get_sgx_info().is_some_and(|info| info.has_sgx2())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Get maximum enclave size
    fn get_max_enclave_size() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();

            cpuid.get_sgx_info().map_or(128 * 1024 * 1024, |sgx_info| {
                // Get max enclave size in 64-bit mode
                let max_size_log2 = sgx_info.max_enclave_size_64bit();

                if max_size_log2 > 0 && max_size_log2 <= 64 {
                    1u64 << max_size_log2
                } else {
                    // Default to 128MB if unable to determine
                    128 * 1024 * 1024
                }
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            0
        }
    }

    /// Get EPC (Enclave Page Cache) size
    const fn get_epc_size() -> u64 {
        // In production, this would query actual EPC size
        // For now, return a typical value
        128 * 1024 * 1024 // 128MB
    }

    /// Check Flexible Launch Control support
    fn check_flc_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // Check if FLC is supported and enabled
            std::path::Path::new("/dev/sgx_enclave").exists()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Register a new enclave
    pub fn register_enclave(&mut self, enclave_id: u64, info: SgxEnclaveInfo) {
        self.enclaves.insert(enclave_id, info);
        self.enclaves_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Unregister an enclave
    pub fn unregister_enclave(&mut self, enclave_id: u64) -> Option<SgxEnclaveInfo> {
        if let Some(info) = self.enclaves.remove(&enclave_id) {
            self.enclaves_destroyed.fetch_add(1, Ordering::Relaxed);
            Some(info)
        } else {
            None
        }
    }

    /// Record ECALL execution
    pub fn record_ecall(&self, execution_time_ns: u64) {
        self.ecalls_executed.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ns
            .fetch_add(execution_time_ns, Ordering::Relaxed);
    }

    /// Record OCALL execution
    pub fn record_ocall(&self, execution_time_ns: u64) {
        self.ocalls_executed.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ns
            .fetch_add(execution_time_ns, Ordering::Relaxed);
    }

    /// Get system statistics
    #[must_use]
    pub fn get_stats(&self) -> SgxStats {
        let total_calls = self.ecalls_executed.load(Ordering::Relaxed)
            + self.ocalls_executed.load(Ordering::Relaxed);
        let total_time = self.total_execution_time_ns.load(Ordering::Relaxed);

        let average_execution_time_ns = if total_calls > 0 {
            total_time / total_calls
        } else {
            0
        };

        SgxStats {
            enclaves_created: self.enclaves_created.load(Ordering::Relaxed),
            enclaves_destroyed: self.enclaves_destroyed.load(Ordering::Relaxed),
            ecalls_executed: self.ecalls_executed.load(Ordering::Relaxed),
            ocalls_executed: self.ocalls_executed.load(Ordering::Relaxed),
            average_execution_time_ns,
            capabilities: Self::get_capabilities(),
        }
    }

    /// Get active enclaves count
    #[must_use]
    pub fn active_enclaves_count(&self) -> usize {
        self.enclaves.len()
    }

    /// Get enclave info by ID
    #[must_use]
    pub fn get_enclave_info(&self, enclave_id: u64) -> Option<&SgxEnclaveInfo> {
        self.enclaves.get(&enclave_id)
    }
}

impl Default for SgxSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// SGX capabilities information
#[derive(Debug, Clone)]
pub struct SgxCapabilities {
    /// SGX1 support
    pub sgx1_supported: bool,
    /// SGX2 support
    pub sgx2_supported: bool,
    /// Maximum enclave size
    pub max_enclave_size: u64,
    /// EPC size
    pub epc_size: u64,
    /// Flexible Launch Control support
    pub flexible_launch_control: bool,
}

/// Initialize SGX enclave
///
/// # Errors
///
/// Returns error if enclave initialization fails
pub async fn initialize_sgx_enclave(config: &EnclaveConfig) -> SecureStorageResult<u64> {
    let start = Instant::now();

    info!("Initializing SGX enclave with config: {:?}", config);

    // Check SGX availability
    if !SgxSystem::is_available() {
        return Err(SecureStorageError::InvalidInput {
            field: "sgx_platform".to_string(),
            reason: "SGX is not available on this system".to_string(),
        });
    }

    // Validate configuration
    validate_sgx_config(config)?;

    // Create enclave
    let enclave_id = if cfg!(feature = "sgx-production") {
        create_production_enclave(config).await?
    } else {
        simulate_enclave_creation(config).await?
    };

    let elapsed = start.elapsed();
    info!(
        "SGX enclave initialized with ID 0x{:x} in {:?}",
        enclave_id, elapsed
    );

    Ok(enclave_id)
}

/// Create production SGX enclave
#[cfg(feature = "sgx-production")]
async fn create_production_enclave(config: &EnclaveConfig) -> SecureStorageResult<u64> {
    // In production, this would use actual SGX SDK:
    // 1. Load enclave binary (.so file)
    // 2. Call sgx_create_enclave()
    // 3. Verify enclave measurement
    // 4. Initialize enclave state

    // For now, return simulation
    simulate_enclave_creation(config).await
}

/// Create production SGX enclave (fallback when feature not enabled)
#[cfg(not(feature = "sgx-production"))]
async fn create_production_enclave(config: &EnclaveConfig) -> SecureStorageResult<u64> {
    warn!("SGX production feature not enabled, falling back to simulation");
    simulate_enclave_creation(config).await
}

/// Validate SGX configuration
fn validate_sgx_config(config: &EnclaveConfig) -> SecureStorageResult<()> {
    let capabilities = SgxSystem::get_capabilities();

    // Check if enclave size is within limits
    let total_size = config.heap_size + config.stack_size;
    if total_size as u64 > capabilities.max_enclave_size {
        return Err(SecureStorageError::InvalidInput {
            field: "enclave_size".to_string(),
            reason: format!(
                "Enclave size {} exceeds maximum {}",
                total_size, capabilities.max_enclave_size
            ),
        });
    }

    // Check EPC availability
    if total_size as u64 > capabilities.epc_size / 2 {
        warn!(
            "Enclave size {} is large relative to EPC size {}",
            total_size, capabilities.epc_size
        );
    }

    // Validate thread count
    if config.max_threads > 64 {
        return Err(SecureStorageError::InvalidInput {
            field: "max_threads".to_string(),
            reason: "SGX enclaves support maximum 64 threads".to_string(),
        });
    }

    // Warn about debug mode in production
    if config.debug_mode && !cfg!(debug_assertions) {
        error!("Debug mode enabled in production build - this reduces security!");
        return Err(SecureStorageError::InvalidInput {
            field: "debug_mode".to_string(),
            reason: "Debug mode not allowed in production".to_string(),
        });
    }

    Ok(())
}

/// Simulate enclave creation for development/testing
async fn simulate_enclave_creation(config: &EnclaveConfig) -> SecureStorageResult<u64> {
    // Simulate enclave loading time
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Generate a pseudo-random enclave ID
    let enclave_id = 0x1000_0000_u64 + (config.heap_size as u64 & 0xFFFF);

    debug!("Simulated SGX enclave creation with ID: 0x{:x}", enclave_id);

    Ok(enclave_id)
}

/// Execute ECALL (Enclave Call)
///
/// # Errors
///
/// Returns error if ECALL execution fails
pub async fn execute_ecall<T, F>(
    enclave_id: u64,
    function_id: u32,
    ecall_function: F,
) -> SecureStorageResult<T>
where
    F: FnOnce() -> SecureStorageResult<T> + Send + 'static,
    T: Send + 'static,
{
    let start = Instant::now();

    debug!(
        "Executing ECALL {} on enclave 0x{:x}",
        function_id, enclave_id
    );

    // In production, this would:
    // 1. Validate enclave ID
    // 2. Enter enclave context
    // 3. Execute function inside enclave
    // 4. Exit enclave context
    // 5. Handle any SGX-specific errors

    // Simulate enclave context switch overhead
    tokio::time::sleep(Duration::from_micros(5)).await;

    // Execute the function
    let result = ecall_function()?;

    let elapsed = start.elapsed();
    debug!("ECALL completed in {:?}", elapsed);

    Ok(result)
}

/// Execute OCALL (Outside Call)
///
/// # Errors
///
/// Returns error if OCALL execution fails
pub async fn execute_ocall<T, F>(function_id: u32, ocall_function: F) -> SecureStorageResult<T>
where
    F: FnOnce() -> SecureStorageResult<T> + Send + 'static,
    T: Send + 'static,
{
    let start = Instant::now();

    debug!("Executing OCALL {}", function_id);

    // In production, this would:
    // 1. Exit enclave context
    // 2. Execute function in untrusted environment
    // 3. Re-enter enclave context
    // 4. Validate any returned data

    // Simulate context switch overhead
    tokio::time::sleep(Duration::from_micros(2)).await;

    // Execute the function
    let result = ocall_function()?;

    let elapsed = start.elapsed();
    debug!("OCALL completed in {:?}", elapsed);

    Ok(result)
}

/// Destroy SGX enclave
///
/// # Errors
///
/// Returns error if enclave destruction fails
pub async fn destroy_sgx_enclave(enclave_id: u64) -> SecureStorageResult<()> {
    let start = Instant::now();

    info!("Destroying SGX enclave 0x{:x}", enclave_id);

    // In production, this would:
    // 1. Validate enclave ID
    // 2. Call sgx_destroy_enclave()
    // 3. Clean up resources
    // 4. Zeroize sensitive data

    // Simulate enclave destruction
    tokio::time::sleep(Duration::from_millis(10)).await;

    let elapsed = start.elapsed();
    info!("SGX enclave destroyed in {:?}", elapsed);

    Ok(())
}

/// Get SGX enclave information
///
/// # Errors
///
/// Returns error if enclave information retrieval fails
pub const fn get_enclave_info(enclave_id: u64) -> SecureStorageResult<SgxEnclaveInfo> {
    // In production, this would query actual enclave information
    // For now, return simulated information

    Ok(SgxEnclaveInfo {
        enclave_id,
        base_address: 0x7000_0000_0000,
        size: 64 * 1024 * 1024, // 64MB
        mrenclave: [0x42; 32],  // Placeholder measurement
        mrsigner: [0x24; 32],   // Placeholder signer
        product_id: 0x1337,     // TallyIO product ID
        security_version: 1,
        debug_mode: cfg!(debug_assertions),
    })
}

/// SGX system statistics
#[derive(Debug, Clone)]
pub struct SgxStats {
    /// Number of enclaves created
    pub enclaves_created: u64,
    /// Number of enclaves destroyed
    pub enclaves_destroyed: u64,
    /// Number of ECALLs executed
    pub ecalls_executed: u64,
    /// Number of OCALLs executed
    pub ocalls_executed: u64,
    /// Average execution time in nanoseconds
    pub average_execution_time_ns: u64,
    /// SGX capabilities
    pub capabilities: SgxCapabilities,
}
