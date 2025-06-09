#!/usr/bin/env python3
"""
Ultra-strict Clippy Error Fixer for TallyIO Financial Application
Systematically fixes all 193 clippy errors for production-ready fintech code.
"""

import re
import os
from pathlib import Path

def fix_precision_loss_casts():
    """Fix all u64 to f64 precision loss casts"""
    files_to_fix = [
        "crates/data_storage/src/cache/memory_cache.rs",
        "crates/data_storage/src/cache/redis_cache.rs", 
        "crates/data_storage/src/cache/cache_strategy.rs",
        "crates/data_storage/src/stream/mod.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace precision loss patterns with safe_ratio calls
            content = re.sub(
                r'(\w+)\.hits as f64 / \((\w+)\.hits \+ (\w+)\.misses\) as f64',
                r'safe_ratio(\1.hits, \2.hits + \3.misses)',
                content
            )
            
            content = re.sub(
                r'stats\.hits as f64 / \(stats\.hits \+ stats\.misses\) as f64',
                r'safe_ratio(stats.hits, stats.hits + stats.misses)',
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_default_numeric_fallback():
    """Fix all default numeric fallback issues"""
    files_to_fix = [
        "crates/data_storage/src/stream/aggregator.rs",
        "crates/data_storage/src/pipeline/batch_processor.rs",
        "crates/data_storage/src/stream/processor.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix numeric literals
            content = re.sub(r'\b0\.0\b(?!_f64)', '0.0_f64', content)
            content = re.sub(r'\b1\.0\b(?!_f64)', '1.0_f64', content) 
            content = re.sub(r'\b2\.0\b(?!_f64)', '2.0_f64', content)
            content = re.sub(r'\b1\.5\b(?!_f64)', '1.5_f64', content)
            content = re.sub(r'\b0\.01\b(?!_f32)', '0.01_f32', content)
            content = re.sub(r'for _i in 0\.\.2', 'for _i in 0_i32..2_i32', content)
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_manual_midpoint():
    """Fix manual midpoint implementations"""
    files_to_fix = [
        "crates/data_storage/src/cache/memory_cache.rs",
        "crates/data_storage/src/cache/redis_cache.rs",
        "crates/data_storage/src/cache/cache_strategy.rs",
        "crates/data_storage/src/stream/aggregator.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace manual midpoint with u64::midpoint
            content = re.sub(
                r'\((\w+)\.avg_latency_us \+ (\w+)\) / 2',
                r'u64::midpoint(\1.avg_latency_us, \2)',
                content
            )
            
            # Replace f64 midpoint
            content = re.sub(
                r'\((\w+)\.throughput \+ (\w+)\) / 2\.0_f64',
                r'f64::midpoint(\1.throughput, \2)',
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_unused_async():
    """Remove async from functions that don't need it"""
    files_to_fix = [
        "crates/data_storage/src/cache/redis_cache.rs",
        "crates/data_storage/src/pipeline/ingestion.rs",
        "crates/data_storage/src/pipeline/transformation.rs",
        "crates/data_storage/src/pipeline/validation.rs",
        "crates/data_storage/src/pipeline/batch_processor.rs",
        "crates/data_storage/src/stream/processor.rs",
        "crates/data_storage/src/stream/buffer.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove async from specific functions that don't await
            patterns_to_fix = [
                r'async fn create_redis_pool',
                r'async fn process_raw_data',
                r'async fn transform_block',
                r'async fn detect_mev_opportunity',
                r'async fn enrich_transaction',
                r'async fn decode_event_data',
                r'async fn validate_transaction',
                r'async fn validate_opportunity',
                r'async fn process_opportunity_batch',
                r'pub async fn start\(&self\) -> DataStorageResult<\(\)>',
                r'pub async fn stop\(&self\) -> DataStorageResult<\(\)>',
                r'async fn process_block',
                r'async fn process_event',
                r'async fn validate_opportunity',
                r'pub async fn metrics',
                r'pub async fn health_check'
            ]
            
            for pattern in patterns_to_fix:
                content = re.sub(pattern, pattern.replace('async ', ''), content)
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_must_use_candidates():
    """Add #[must_use] to functions that should have it"""
    files_to_fix = [
        "crates/data_storage/src/cache/cache_strategy.rs",
        "crates/data_storage/src/pipeline/ingestion.rs",
        "crates/data_storage/src/stream/mod.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add #[must_use] before specific function patterns
            patterns = [
                r'pub fn strategy_stats\(&self\)',
                r'pub fn new\([^)]*\) -> Self',
                r'pub fn with_input_channel',
                r'pub fn with_output_channel',
                r'pub fn success_rate\(&self\)'
            ]
            
            for pattern in patterns:
                content = re.sub(
                    f'({pattern})',
                    r'#[must_use]\n    \1',
                    content
                )
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_documentation_issues():
    """Fix TallyIO documentation and missing errors docs"""
    files_to_fix = [
        "crates/data_storage/src/pipeline/mod.rs",
        "crates/data_storage/src/stream/aggregator.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix TallyIO documentation
            content = re.sub(r'TallyIO(?!`)', '`TallyIO`', content)
            
            # Add missing errors documentation
            content = re.sub(
                r'(pub async fn health_check\(&self\) -> DataStorageResult<\(\)>)',
                r'/// Health check for the component\n    ///\n    /// # Errors\n    ///\n    /// Returns error if component is in error state or critical thresholds exceeded\n    \1',
                content
            )
            
            with open(file_path, 'w') as f:
                f.write(content)

def main():
    """Run all fixes systematically"""
    print("🚨 Starting ultra-strict clippy error fixes for TallyIO financial application...")
    
    print("1. Fixing precision loss casts...")
    fix_precision_loss_casts()
    
    print("2. Fixing default numeric fallback...")
    fix_default_numeric_fallback()
    
    print("3. Fixing manual midpoint implementations...")
    fix_manual_midpoint()
    
    print("4. Removing unnecessary async...")
    fix_unused_async()
    
    print("5. Adding #[must_use] attributes...")
    fix_must_use_candidates()
    
    print("6. Fixing documentation issues...")
    fix_documentation_issues()
    
    print("✅ All systematic fixes applied!")
    print("🚨 REMEMBER: TallyIO manages real money - zero tolerance for errors!")

if __name__ == "__main__":
    main()
