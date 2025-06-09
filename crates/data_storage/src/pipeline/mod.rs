//! Pipeline Processing Module
//!
//! High-performance data processing pipeline for `TallyIO` financial data.
//! Provides real-time ingestion, transformation, and validation of MEV opportunities.

pub mod batch_processor;
pub mod ingestion;
pub mod transformation;
pub mod validation;

pub use batch_processor::{BatchConfig, BatchProcessor, BatchResult};
pub use ingestion::{DataIngestion, ProcessedData, RawData};
pub use transformation::{DataTransformation, TransformationRules, TransformedData};
pub use validation::{DataValidation, ValidationResult, ValidationRules};

use crate::error::DataStorageResult;

/// Pipeline stage trait
#[async_trait::async_trait]
pub trait PipelineStage<T>: Send + Sync {
    /// Process data through this pipeline stage
    async fn process(&self, data: T) -> DataStorageResult<T>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Get stage metrics
    async fn metrics(&self) -> DataStorageResult<PipelineMetrics>;
}

/// Pipeline metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    /// Stage name
    pub stage_name: String,

    /// Number of items processed
    pub items_processed: u64,

    /// Number of items failed
    pub items_failed: u64,

    /// Average processing time in microseconds
    pub avg_processing_time_us: u64,

    /// Total processing time in microseconds
    pub total_processing_time_us: u64,
}

impl PipelineMetrics {
    /// Create new pipeline metrics
    #[must_use]
    pub fn new(stage_name: String) -> Self {
        Self {
            stage_name,
            items_processed: 0,
            items_failed: 0,
            avg_processing_time_us: 0,
            total_processing_time_us: 0,
        }
    }
}
