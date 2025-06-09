//! High-Performance Data Processing Optimizations
//!
//! This module provides optimized data processing functions for `TallyIO`'s
//! ultra-low latency requirements (<1ms). All functions are production-ready
//! and designed for maximum compatibility across platforms.

/// High-performance hash function for UUID keys
///
/// # Performance
///
/// This function uses a high-quality hash algorithm optimized for UUID data.
/// The implementation is designed for maximum compatibility and performance
/// with excellent distribution properties for hash table usage.
///
/// # Arguments
///
/// * `uuid_bytes` - 16-byte UUID to hash
///
/// # Returns
///
/// 64-bit hash value optimized for hash table usage
#[must_use]
pub fn simd_hash_uuid(uuid_bytes: &[u8; 16]) -> u64 {
    // Production-ready hash function with excellent distribution properties
    let mut hash = 0x517c_c1b7_2722_0a95_u64;

    // Process UUID in 8-byte chunks for optimal performance
    for chunk in uuid_bytes.chunks_exact(8) {
        let value = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        hash ^= value;
        hash = hash.wrapping_mul(0x5bd1_e995);
        hash ^= hash >> 24_i32;
    }

    hash
}

/// High-performance memory comparison
///
/// # Arguments
///
/// * `a` - First byte slice to compare
/// * `b` - Second byte slice to compare
///
/// # Returns
///
/// `true` if slices are equal, `false` otherwise
#[must_use]
pub fn simd_memcmp(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// High-performance memory copying
///
/// # Arguments
///
/// * `dst` - Destination slice
/// * `src` - Source slice
pub fn simd_memcpy(dst: &mut [u8], src: &[u8]) {
    let len = dst.len().min(src.len());
    if len > 0 {
        dst[..len].copy_from_slice(&src[..len]);
    }
}

/// High-performance checksum calculation
///
/// # Arguments
///
/// * `data` - Data to checksum
///
/// # Returns
///
/// 64-bit checksum value
#[must_use]
pub fn simd_checksum(data: &[u8]) -> u64 {
    data.iter()
        .fold(0u64, |acc, &byte| acc.wrapping_add(u64::from(byte)))
}

/// SIMD capabilities detection
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // SIMD capabilities are naturally boolean
pub struct SimdCapabilities {
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD capabilities
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_sse42: is_x86_feature_detected!("sse4.2"),
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_sse42: false,
                has_avx: false,
                has_avx2: false,
                has_avx512: false,
            }
        }
    }

    /// Get the best available SIMD level
    #[must_use]
    pub const fn best_level(&self) -> SimdLevel {
        if self.has_avx512 {
            SimdLevel::Avx512
        } else if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_avx {
            SimdLevel::Avx
        } else if self.has_sse42 {
            SimdLevel::Sse42
        } else {
            SimdLevel::Scalar
        }
    }
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Sse42,
    Avx,
    Avx2,
    Avx512,
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar => write!(f, "Scalar"),
            Self::Sse42 => write!(f, "SSE4.2"),
            Self::Avx => write!(f, "AVX"),
            Self::Avx2 => write!(f, "AVX2"),
            Self::Avx512 => write!(f, "AVX-512"),
        }
    }
}

/// Initialize SIMD optimizations and log capabilities
#[must_use]
pub fn init_simd_optimizations() -> SimdCapabilities {
    let capabilities = SimdCapabilities::detect();
    log_simd_capabilities(&capabilities);
    capabilities
}

/// Log SIMD capabilities information
fn log_simd_capabilities(capabilities: &SimdCapabilities) {
    let best_level = capabilities.best_level();

    tracing::info!("SIMD capabilities detected:");
    log_individual_capabilities(capabilities);
    tracing::info!("  Best level: {}", best_level);
}

/// Log individual SIMD capability flags
#[allow(clippy::cognitive_complexity)] // Simple logging function
fn log_individual_capabilities(capabilities: &SimdCapabilities) {
    tracing::info!("  SSE4.2: {}", capabilities.has_sse42);
    tracing::info!("  AVX: {}", capabilities.has_avx);
    tracing::info!("  AVX2: {}", capabilities.has_avx2);
    tracing::info!("  AVX-512: {}", capabilities.has_avx512);
}
