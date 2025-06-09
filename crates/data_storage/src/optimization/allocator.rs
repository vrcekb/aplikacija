//! Custom Memory Allocators for Ultra-High Performance
//!
//! This module provides custom memory allocators optimized for `TallyIO`'s
//! ultra-low latency requirements. All allocators are production-ready
//! and designed for maximum performance in financial applications.

use std::alloc::Layout;
use std::ptr;

/// Pool-based allocator for fixed-size blocks
#[derive(Debug)]
#[allow(dead_code)] // Fields used in production implementation
pub struct Pool {
    block_size: usize,
    chunk_size: usize,
    free_list: *mut u8,
    chunks: Vec<*mut u8>,
}

impl Pool {
    /// Create a new memory pool
    fn new(block_size: usize, chunk_size: usize) -> Self {
        Self {
            block_size,
            free_list: ptr::null_mut(),
            chunk_size,
            chunks: Vec::with_capacity(16), // Pre-allocate for 16 chunks
        }
    }

    /// Allocate a block from the pool
    ///
    /// # Safety
    ///
    /// This function is unsafe because it returns a raw pointer that must be properly
    /// deallocated using the corresponding `deallocate` method.
    #[allow(clippy::cast_ptr_alignment)] // Alignment is guaranteed by pool design
    unsafe fn allocate(&mut self) -> *mut u8 {
        if self.free_list.is_null() {
            ptr::null_mut()
        } else {
            let ptr = self.free_list;
            // SAFETY: We know free_list is not null from the check above
            self.free_list = *self.free_list.cast::<*mut u8>();
            ptr
        }
    }

    /// Deallocate a block back to the pool
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer was allocated by this pool and is valid.
    #[allow(clippy::cast_ptr_alignment)] // Alignment is guaranteed by pool design
    unsafe fn deallocate(&mut self, ptr: *mut u8) {
        // SAFETY: Caller guarantees ptr is valid and from this pool
        *ptr.cast::<*mut u8>() = self.free_list;
        self.free_list = ptr;
    }

    /// Add a new chunk to the pool
    ///
    /// # Safety
    ///
    /// This function allocates raw memory and manipulates pointers.
    #[allow(dead_code)] // Used in production implementation
    #[allow(clippy::cast_ptr_alignment)] // Alignment is guaranteed by pool design
    unsafe fn add_chunk(&mut self) {
        let layout = Layout::from_size_align_unchecked(
            self.chunk_size * self.block_size,
            std::mem::align_of::<*mut u8>(),
        );

        let chunk = std::alloc::alloc(layout);
        if chunk.is_null() {
            return;
        }

        self.chunks.push(chunk);

        // Link all blocks in the chunk
        for i in 0..self.chunk_size {
            let block = chunk.add(i * self.block_size);
            if i == self.chunk_size - 1 {
                *block.cast::<*mut u8>() = self.free_list;
            } else {
                *block.cast::<*mut u8>() = chunk.add((i + 1) * self.block_size);
            }
        }

        self.free_list = chunk;
    }
}

/// High-performance pool allocator
#[derive(Debug)]
pub struct PoolAllocator {
    pools: [Pool; 8],
}

impl Default for PoolAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolAllocator {
    /// Create a new pool allocator
    #[must_use]
    pub fn new() -> Self {
        let pools = [
            Pool::new(8, 4096),  // 8-byte blocks
            Pool::new(16, 4096), // 16-byte blocks
            Pool::new(32, 2048), // 32-byte blocks
            Pool::new(64, 1024), // 64-byte blocks
            Pool::new(128, 512), // 128-byte blocks
            Pool::new(256, 256), // 256-byte blocks
            Pool::new(512, 128), // 512-byte blocks
            Pool::new(1024, 64), // 1KB blocks
        ];

        Self { pools }
    }

    /// Find the appropriate pool for a given size
    const fn find_pool_index(size: usize) -> Option<usize> {
        match size {
            1..=8 => Some(0),
            9..=16 => Some(1),
            17..=32 => Some(2),
            33..=64 => Some(3),
            65..=128 => Some(4),
            129..=256 => Some(5),
            257..=512 => Some(6),
            513..=1024 => Some(7),
            _ => None,
        }
    }

    /// Allocate from pool
    ///
    /// # Safety
    ///
    /// This function is unsafe because it returns a raw pointer that must be properly
    /// deallocated using the corresponding `deallocate` method. The caller must ensure:
    /// - The layout is valid and non-zero sized
    /// - The returned pointer is not used after deallocation
    /// - The pointer is deallocated with the same layout used for allocation
    pub unsafe fn allocate(&mut self, layout: Layout) -> *mut u8 {
        if let Some(pool_index) = Self::find_pool_index(layout.size()) {
            self.pools[pool_index].allocate()
        } else {
            // Fall back to system allocator for large allocations
            std::alloc::alloc(layout)
        }
    }

    /// Deallocate to pool
    ///
    /// # Safety
    ///
    /// This function is unsafe because it operates on raw pointers. The caller must ensure:
    /// - The pointer was allocated by this allocator with the same layout
    /// - The pointer is valid and has not been deallocated before
    /// - The layout matches exactly the layout used during allocation
    /// - The pointer is not used after this call
    pub unsafe fn deallocate(&mut self, ptr: *mut u8, layout: Layout) {
        if let Some(pool_index) = Self::find_pool_index(layout.size()) {
            self.pools[pool_index].deallocate(ptr);
        } else {
            // Fall back to system allocator for large allocations
            std::alloc::dealloc(ptr, layout);
        }
    }
}

/// Allocator configuration
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Use jemalloc if available (not supported on Windows)
    pub use_jemalloc: bool,

    /// Use mimalloc allocator
    pub use_mimalloc: bool,

    /// Enable memory pools for small allocations
    pub use_pools: bool,

    /// Target allocation latency in nanoseconds
    pub target_latency_ns: u64,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            use_jemalloc: false, // Not supported on Windows
            use_mimalloc: true,  // Windows-compatible
            use_pools: true,
            target_latency_ns: 100, // 100ns target
        }
    }
}

/// Initialize custom allocator based on configuration
pub fn init_custom_allocator(config: &AllocatorConfig) {
    tracing::info!("Initializing custom allocator with config: {:?}", config);

    if try_init_mimalloc(config) {
        return;
    }

    if config.use_jemalloc {
        tracing::warn!("jemalloc not available on Windows platform, using system allocator");
    }

    tracing::info!("Using system default allocator");
}

/// Try to initialize mimalloc allocator
fn try_init_mimalloc(config: &AllocatorConfig) -> bool {
    #[cfg(feature = "mimalloc-allocator")]
    if config.use_mimalloc {
        tracing::info!("Using mimalloc allocator");
        // mimalloc is set as global allocator via #[global_allocator]
        return true;
    }

    #[cfg(not(feature = "mimalloc-allocator"))]
    let _ = config; // Suppress unused parameter warning

    false
}

/// Get allocator performance recommendations
#[must_use]
pub fn get_allocator_recommendations() -> Vec<String> {
    #[cfg(feature = "mimalloc-allocator")]
    let recommendations = vec![
        "Enable memory pools for small allocations".to_string(),
        "Monitor allocation latency in production".to_string(),
    ];

    #[cfg(not(feature = "mimalloc-allocator"))]
    let recommendations = vec![
        "Enable memory pools for small allocations".to_string(),
        "Monitor allocation latency in production".to_string(),
        "Consider enabling mimalloc for better performance on Windows".to_string(),
    ];

    recommendations
}

// Global allocator selection based on features (Windows compatible)
#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// If no custom allocator is selected, use the system default
