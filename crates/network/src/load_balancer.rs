//! Load Balancer Module
//!
//! High-performance load balancing with multiple strategies and health checking.
//! Optimized for financial applications requiring high availability and low latency.

use crate::config::LoadBalancingStrategy;
use crate::error::{NetworkError, NetworkResult};
use crate::types::Endpoint;
use async_trait::async_trait;
use std::sync::Arc;

/// Load balancer trait for different strategies
#[async_trait]
pub trait LoadBalancerTrait: Send + Sync {
    /// Select next endpoint based on strategy
    ///
    /// # Errors
    /// Returns error if no healthy endpoints are available
    async fn select_endpoint(&self) -> NetworkResult<Endpoint>;

    /// Mark endpoint as healthy
    async fn mark_healthy(&self, endpoint: &Endpoint);

    /// Mark endpoint as unhealthy
    async fn mark_unhealthy(&self, endpoint: &Endpoint);

    /// Get load balancer statistics
    fn stats(&self) -> LoadBalancerStats;

    /// Check if load balancer is healthy
    fn is_healthy(&self) -> bool;
}

/// Load balancer statistics
#[derive(Debug, Clone)]
pub struct LoadBalancerStats {
    /// Total number of endpoints
    pub total_endpoints: u32,
    /// Number of healthy endpoints
    pub healthy_endpoints: u32,
    /// Number of unhealthy endpoints
    pub unhealthy_endpoints: u32,
    /// Total requests routed
    pub total_requests: u64,
    /// Requests per endpoint
    pub requests_per_endpoint: std::collections::HashMap<String, u64>,
    /// Current strategy
    pub strategy: LoadBalancingStrategy,
}

impl Default for LoadBalancerStats {
    fn default() -> Self {
        Self {
            total_endpoints: 0,
            healthy_endpoints: 0,
            unhealthy_endpoints: 0,
            total_requests: 0,
            requests_per_endpoint: std::collections::HashMap::new(),
            strategy: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancer implementation
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    endpoints: Arc<std::sync::RwLock<Vec<Endpoint>>>,
    stats: Arc<std::sync::RwLock<LoadBalancerStats>>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(strategy: LoadBalancingStrategy, endpoints: Vec<Endpoint>) -> Self {
        let mut stats = LoadBalancerStats::default();
        stats.strategy = strategy;
        stats.total_endpoints = endpoints.len() as u32;
        stats.healthy_endpoints = endpoints.len() as u32; // Assume all healthy initially

        Self {
            strategy,
            endpoints: Arc::new(std::sync::RwLock::new(endpoints)),
            stats: Arc::new(std::sync::RwLock::new(stats)),
        }
    }

    /// Add endpoint
    pub fn add_endpoint(&self, endpoint: Endpoint) {
        if let Ok(mut endpoints) = self.endpoints.write() {
            endpoints.push(endpoint);
            
            if let Ok(mut stats) = self.stats.write() {
                stats.total_endpoints = endpoints.len() as u32;
                stats.healthy_endpoints += 1;
            }
        }
    }

    /// Remove endpoint
    pub fn remove_endpoint(&self, url: &str) {
        if let Ok(mut endpoints) = self.endpoints.write() {
            endpoints.retain(|e| e.url != url);
            
            if let Ok(mut stats) = self.stats.write() {
                stats.total_endpoints = endpoints.len() as u32;
                stats.requests_per_endpoint.remove(url);
            }
        }
    }
}

#[async_trait]
impl LoadBalancerTrait for LoadBalancer {
    async fn select_endpoint(&self) -> NetworkResult<Endpoint> {
        let endpoints = self.endpoints.read().map_err(|_| {
            NetworkError::internal("Failed to acquire endpoints lock")
        })?;

        if endpoints.is_empty() {
            return Err(NetworkError::LoadBalancer {
                strategy: format!("{:?}", self.strategy),
                message: "No endpoints available".to_string(),
                available_endpoints: 0,
            });
        }

        // For now, implement simple round-robin
        // In full implementation, this would use the configured strategy
        let index = {
            let mut stats = self.stats.write().map_err(|_| {
                NetworkError::internal("Failed to acquire stats lock")
            })?;
            let index = (stats.total_requests as usize) % endpoints.len();
            stats.total_requests += 1;
            index
        };

        let endpoint = endpoints[index].clone();
        
        // Update per-endpoint stats
        if let Ok(mut stats) = self.stats.write() {
            *stats.requests_per_endpoint.entry(endpoint.url.clone()).or_insert(0) += 1;
        }

        Ok(endpoint)
    }

    async fn mark_healthy(&self, _endpoint: &Endpoint) {
        // Implementation would update endpoint health status
        // For now, this is a stub
    }

    async fn mark_unhealthy(&self, _endpoint: &Endpoint) {
        // Implementation would update endpoint health status
        // For now, this is a stub
    }

    fn stats(&self) -> LoadBalancerStats {
        self.stats.read().map_or_else(
            |_| LoadBalancerStats::default(),
            |stats| stats.clone()
        )
    }

    fn is_healthy(&self) -> bool {
        let stats = self.stats();
        stats.healthy_endpoints > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Endpoint;

    #[test]
    fn test_load_balancer_creation() {
        let endpoints = vec![
            Endpoint::new("https://api1.example.com"),
            Endpoint::new("https://api2.example.com"),
        ];
        
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, endpoints);
        let stats = lb.stats();
        
        assert_eq!(stats.total_endpoints, 2);
        assert_eq!(stats.healthy_endpoints, 2);
        assert_eq!(stats.strategy, LoadBalancingStrategy::RoundRobin);
    }

    #[tokio::test]
    async fn test_endpoint_selection() {
        let endpoints = vec![
            Endpoint::new("https://api1.example.com"),
            Endpoint::new("https://api2.example.com"),
        ];
        
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, endpoints);
        
        // Test multiple selections
        let endpoint1 = lb.select_endpoint().await.unwrap();
        let endpoint2 = lb.select_endpoint().await.unwrap();
        
        // Should round-robin between endpoints
        assert_ne!(endpoint1.url, endpoint2.url);
        
        let stats = lb.stats();
        assert_eq!(stats.total_requests, 2);
    }

    #[test]
    fn test_endpoint_management() {
        let endpoints = vec![Endpoint::new("https://api1.example.com")];
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, endpoints);
        
        // Add endpoint
        lb.add_endpoint(Endpoint::new("https://api2.example.com"));
        let stats = lb.stats();
        assert_eq!(stats.total_endpoints, 2);
        
        // Remove endpoint
        lb.remove_endpoint("https://api1.example.com");
        let stats = lb.stats();
        assert_eq!(stats.total_endpoints, 1);
    }

    #[tokio::test]
    async fn test_no_endpoints_error() {
        let lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin, vec![]);
        let result = lb.select_endpoint().await;
        
        assert!(result.is_err());
        if let Err(NetworkError::LoadBalancer { available_endpoints, .. }) = result {
            assert_eq!(available_endpoints, 0);
        } else {
            panic!("Expected LoadBalancer error");
        }
    }
}
