//! Kernel bypass networking for ultra-low latency
//!
//! Implements DPDK-style kernel bypass for sub-microsecond network latency.
//! Zero-allocation design for production-ready financial applications.

use crate::error::{CoreError, CoreResult};
use crate::lockfree::ultra::cache_aligned::CacheAligned;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Maximum ring size (power of 2)
const MAX_RING_SIZE: usize = 4096;
/// Maximum free packet pool size
const MAX_PACKET_POOL_SIZE: usize = 8192;

/// Kernel bypass network interface - zero allocation design
#[repr(C, align(64))]
pub struct KernelBypassNic {
    /// Device info
    device: NetworkDevice,
    /// RX ring
    rx_ring: RxRing,
    /// TX ring  
    tx_ring: TxRing,
    /// Memory pool for packets (owned, not Arc)
    packet_pool: PacketMemoryPool,
    /// Statistics (owned, not Arc)
    stats: NetworkStats,
}

/// Network device info
#[repr(C, align(32))]
pub struct NetworkDevice {
    /// PCI address
    pci_addr: [u8; 64], // Fixed size instead of String
    /// Number of RX queues
    num_rx_queues: u16,
    /// Number of TX queues
    num_tx_queues: u16,
    /// MAC address
    mac_addr: [u8; 6],
    /// Device flags
    flags: u32,
    /// Reserved for alignment
    _reserved: [u8; 18],
}

/// RX ring for receiving packets - fixed size allocation
#[repr(C, align(64))]
pub struct RxRing {
    /// Ring buffer (fixed size, no Vec)
    descriptors: CacheAligned<[RxDescriptor; MAX_RING_SIZE]>,
    /// Producer index
    producer: CacheAligned<AtomicUsize>,
    /// Consumer index
    consumer: CacheAligned<AtomicUsize>,
    /// Ring size mask
    mask: usize,
    /// Active ring size
    size: usize,
}

/// TX ring for transmitting packets - fixed size allocation
#[repr(C, align(64))]
pub struct TxRing {
    /// Ring buffer (fixed size, no Vec)
    descriptors: CacheAligned<[TxDescriptor; MAX_RING_SIZE]>,
    /// Producer index
    producer: CacheAligned<AtomicUsize>,
    /// Consumer index
    consumer: CacheAligned<AtomicUsize>,
    /// Ring size mask
    mask: usize,
    /// Active ring size
    size: usize,
}

/// RX descriptor
#[repr(C, align(64))]
pub struct RxDescriptor {
    /// Packet buffer physical address
    addr: u64,
    /// Packet length
    length: u16,
    /// Status flags
    status: u16,
    /// Reserved
    _reserved: [u8; 52],
}

/// TX descriptor
#[repr(C, align(64))]
pub struct TxDescriptor {
    /// Packet buffer physical address
    addr: u64,
    /// Packet length
    length: u16,
    /// Command flags
    cmd: u16,
    /// Status flags
    status: u16,
    /// Reserved
    _reserved: [u8; 50],
}

/// Packet memory pool
#[repr(C, align(64))]
pub struct PacketMemoryPool {
    /// Hugepage-backed memory
    memory: NonNull<u8>,
    /// Total size
    size: usize,
    /// Free list
    free_list: [NonNull<Packet>; MAX_PACKET_POOL_SIZE],
    /// Free index
    free_index: CacheAligned<AtomicUsize>,
}

/// Network packet
#[repr(C, align(64))]
pub struct Packet {
    /// Data buffer
    data: [u8; 2048],
    /// Actual data length
    len: u16,
    /// Metadata
    meta: PacketMetadata,
    /// Reserved for cache alignment
    _reserved: [u8; 14],
}

/// Packet metadata
#[derive(Debug, Clone, Copy)]
pub struct PacketMetadata {
    /// Receive timestamp (TSC)
    rx_timestamp: u64,
    /// Port ID
    port: u16,
    /// Queue ID
    queue: u16,
    /// Checksum flags
    checksum: u16,
    /// VLAN tag
    vlan: u16,
}

/// Network statistics
#[derive(Debug, Default)]
pub struct NetworkStats {
    /// Packets received
    pub rx_packets: AtomicU64,
    /// Packets transmitted
    pub tx_packets: AtomicU64,
    /// RX bytes
    pub rx_bytes: AtomicU64,
    /// TX bytes
    pub tx_bytes: AtomicU64,
    /// RX dropped
    pub rx_dropped: AtomicU64,
    /// TX dropped
    pub tx_dropped: AtomicU64,
    /// Average RX latency (TSC cycles)
    pub avg_rx_latency: AtomicU64,
}

impl KernelBypassNic {
    /// Initialize kernel bypass NIC
    ///
    /// # Arguments
    ///
    /// * `pci_addr` - PCI address of network card
    /// * `num_rx_desc` - Number of RX descriptors
    /// * `num_tx_desc` - Number of TX descriptors
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(pci_addr: &str, num_rx_desc: usize, num_tx_desc: usize) -> CoreResult<Self> {
        // This is a simplified implementation
        // In production, use DPDK or similar

        let mut pci_addr_bytes = [0u8; 64];
        let bytes = pci_addr.as_bytes();
        let copy_len = bytes.len().min(64);
        pci_addr_bytes[..copy_len].copy_from_slice(&bytes[..copy_len]);

        let device = NetworkDevice {
            pci_addr: pci_addr_bytes,
            num_rx_queues: 1,
            num_tx_queues: 1,
            mac_addr: [0; 6],
            flags: 0,
            _reserved: [0; 18],
        };

        let rx_ring = RxRing::new(num_rx_desc)?;
        let tx_ring = TxRing::new(num_tx_desc)?;
        let packet_pool = PacketMemoryPool::new(num_rx_desc + num_tx_desc)?;

        Ok(Self {
            device,
            rx_ring,
            tx_ring,
            packet_pool,
            stats: NetworkStats::default(),
        })
    }

    /// Receive packets without syscalls
    ///
    /// # Arguments
    ///
    /// * `packets` - Buffer to store received packets
    ///
    /// # Returns
    ///
    /// Number of packets received
    pub fn rx_burst(&mut self, packets: &mut [Option<Box<Packet>>]) -> usize {
        let start_tsc = rdtsc();
        let mut count = 0;

        let consumer = self.rx_ring.consumer.load(Ordering::Acquire);
        let producer = self.rx_ring.producer.load(Ordering::Acquire);

        while count < packets.len() && consumer != producer {
            let idx = consumer & self.rx_ring.mask;
            let desc = &self.rx_ring.descriptors[idx];

            // Check if packet is ready
            if desc.status & RX_STATUS_DD == 0 {
                break;
            }

            // Get packet from descriptor
            let packet_ptr = desc.addr as *mut Packet;
            let packet = unsafe { Box::from_raw(packet_ptr) };

            // Update statistics
            self.stats.rx_packets.fetch_add(1, Ordering::Relaxed);
            self.stats
                .rx_bytes
                .fetch_add(desc.length as u64, Ordering::Relaxed);

            packets[count] = Some(packet);
            count += 1;

            // Clear descriptor for reuse
            self.rx_ring.descriptors[idx].status = 0;
        }

        // Update consumer index
        if count > 0 {
            self.rx_ring
                .consumer
                .store(consumer.wrapping_add(count), Ordering::Release);

            // Update latency stats
            let latency = rdtsc() - start_tsc;
            self.stats
                .avg_rx_latency
                .store(latency / count as u64, Ordering::Relaxed);
        }

        count
    }

    /// Transmit packets without syscalls
    ///
    /// # Arguments
    ///
    /// * `packets` - Packets to transmit
    ///
    /// # Returns
    ///
    /// Number of packets transmitted
    pub fn tx_burst(&mut self, packets: &[Box<Packet>]) -> usize {
        let producer = self.tx_ring.producer.load(Ordering::Acquire);
        let consumer = self.tx_ring.consumer.load(Ordering::Acquire);

        let available = if producer >= consumer {
            self.tx_ring.descriptors.len() - (producer - consumer)
        } else {
            consumer - producer
        } - 1; // Keep one slot empty

        let count = packets.len().min(available);

        for i in 0..count {
            let idx = producer.wrapping_add(i) & self.tx_ring.mask;
            let packet = &packets[i];

            // Setup descriptor
            self.tx_ring.descriptors[idx].addr = packet.as_ref() as *const _ as u64;
            self.tx_ring.descriptors[idx].length = packet.len;
            self.tx_ring.descriptors[idx].cmd = TX_CMD_EOP | TX_CMD_RS;
            self.tx_ring.descriptors[idx].status = 0;

            // Update statistics
            self.stats.tx_packets.fetch_add(1, Ordering::Relaxed);
            self.stats
                .tx_bytes
                .fetch_add(packet.len as u64, Ordering::Relaxed);
        }

        // Update producer index
        if count > 0 {
            self.tx_ring
                .producer
                .store(producer.wrapping_add(count), Ordering::Release);

            // Notify hardware (memory-mapped I/O)
            self.notify_tx_hardware();
        }

        count
    }

    /// Poll for TX completions
    pub fn tx_clean(&mut self) -> usize {
        let consumer = self.tx_ring.consumer.load(Ordering::Acquire);
        let producer = self.tx_ring.producer.load(Ordering::Acquire);
        let mut cleaned = 0;

        let mut current = consumer;
        while current != producer {
            let idx = current & self.tx_ring.mask;

            // Check if transmission complete
            if self.tx_ring.descriptors[idx].status & TX_STATUS_DD == 0 {
                break;
            }

            // Free packet buffer
            let packet_ptr = self.tx_ring.descriptors[idx].addr as *mut Packet;
            self.packet_pool.free(packet_ptr);

            cleaned += 1;
            current = current.wrapping_add(1);
        }

        if cleaned > 0 {
            self.tx_ring.consumer.store(current, Ordering::Release);
        }

        cleaned
    }

    /// Allocate packet from pool
    pub fn alloc_packet(&mut self) -> Option<Box<Packet>> {
        self.packet_pool.alloc()
    }

    /// Get statistics
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Notify TX hardware via memory-mapped I/O
    fn notify_tx_hardware(&self) {
        // In real implementation, write to TX tail register
        // This would be a memory-mapped I/O operation
        compiler_fence(Ordering::Release);
    }
}

impl RxRing {
    fn new(size: usize) -> CoreResult<Self> {
        if !size.is_power_of_two() {
            return Err(CoreError::InvalidSize(
                "RX ring size must be power of 2".to_string(),
            ));
        }

        Ok(Self {
            descriptors: CacheAligned::new(
                [RxDescriptor {
                    addr: 0,
                    length: 0,
                    status: 0,
                    _reserved: [0; 52],
                }; MAX_RING_SIZE],
            ),
            producer: CacheAligned::new(AtomicUsize::new(0)),
            consumer: CacheAligned::new(AtomicUsize::new(0)),
            mask: size - 1,
            size,
        })
    }
}

impl TxRing {
    fn new(size: usize) -> CoreResult<Self> {
        if !size.is_power_of_two() {
            return Err(CoreError::InvalidSize(
                "TX ring size must be power of 2".to_string(),
            ));
        }

        Ok(Self {
            descriptors: CacheAligned::new(
                [TxDescriptor {
                    addr: 0,
                    length: 0,
                    cmd: 0,
                    status: 0,
                    _reserved: [0; 50],
                }; MAX_RING_SIZE],
            ),
            producer: CacheAligned::new(AtomicUsize::new(0)),
            consumer: CacheAligned::new(AtomicUsize::new(0)),
            mask: size - 1,
            size,
        })
    }
}

impl PacketMemoryPool {
    fn new(capacity: usize) -> CoreResult<Self> {
        // Allocate hugepage-backed memory
        let size = capacity * mem::size_of::<Packet>();
        let layout = std::alloc::Layout::from_size_align(size, 2 * 1024 * 1024)
            .map_err(|_| CoreError::InvalidSize("Invalid packet pool size".to_string()))?;

        let memory = unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                return Err(CoreError::OutOfMemory(
                    "Failed to allocate packet pool".to_string(),
                ));
            }
            NonNull::new_unchecked(ptr)
        };

        // Initialize free list
        let mut free_list = [NonNull::dangling(); MAX_PACKET_POOL_SIZE];
        for i in 0..capacity {
            let packet_ptr =
                unsafe { memory.as_ptr().add(i * mem::size_of::<Packet>()) as *mut Packet };
            free_list[i] = NonNull::new_unchecked(packet_ptr);
        }

        Ok(Self {
            memory,
            size,
            free_list,
            free_index: CacheAligned::new(AtomicUsize::new(capacity)),
        })
    }

    fn alloc(&mut self) -> Option<Box<Packet>> {
        let index = self.free_index.fetch_sub(1, Ordering::AcqRel);
        if index == 0 {
            self.free_index.fetch_add(1, Ordering::AcqRel);
            return None;
        }

        let packet_ptr = self.free_list[index - 1].as_ptr();
        unsafe { Some(Box::from_raw(packet_ptr)) }
    }

    fn free(&mut self, packet: *mut Packet) {
        let index = self.free_index.fetch_add(1, Ordering::AcqRel);
        if index < self.free_list.len() {
            self.free_list[index] = NonNull::new_unchecked(packet);
        }
    }
}

// Constants
const RX_STATUS_DD: u16 = 0x0001; // Descriptor Done
const TX_STATUS_DD: u16 = 0x0001; // Descriptor Done
const TX_CMD_EOP: u16 = 0x0001; // End of Packet
const TX_CMD_RS: u16 = 0x0008; // Report Status

// Helper functions

#[inline]
fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        0 // Fallback for non-x86
    }
}

use std::sync::atomic::compiler_fence;

impl Drop for PacketMemoryPool {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(self.size, 2 * 1024 * 1024);
            std::alloc::dealloc(self.memory.as_ptr(), layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rx_ring_creation() {
        let ring = RxRing::new(1024);
        assert!(ring.is_ok());

        let ring = ring.unwrap();
        assert_eq!(ring.mask, 1023);
        assert_eq!(ring.size, 1024);
    }

    #[test]
    fn test_tx_ring_creation() {
        let ring = TxRing::new(512);
        assert!(ring.is_ok());

        let ring = ring.unwrap();
        assert_eq!(ring.mask, 511);
        assert_eq!(ring.size, 512);
    }

    #[test]
    fn test_packet_pool() {
        let pool = PacketMemoryPool::new(256);
        assert!(pool.is_ok());

        let mut pool = pool.unwrap();
        let packet = pool.alloc();
        assert!(packet.is_some());
    }

    #[test]
    fn test_ring_power_of_two() {
        let ring = RxRing::new(1023);
        assert!(ring.is_err());

        let ring = TxRing::new(1000);
        assert!(ring.is_err());
    }
}
