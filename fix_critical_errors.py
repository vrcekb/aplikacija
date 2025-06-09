#!/usr/bin/env python3
"""
Critical Error Fixer for TallyIO - Fixes only compilation errors
"""

import re
import os

def fix_await_errors():
    """Remove .await from non-async function calls"""
    files_to_fix = [
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
            
            # Remove .await from specific function calls
            content = re.sub(r'\.process_raw_data\(([^)]+)\)\.await', r'.process_raw_data(\1)', content)
            content = re.sub(r'\.detect_mev_opportunity\(([^)]+)\)\.await', r'.detect_mev_opportunity(\1)', content)
            content = re.sub(r'\.enrich_transaction\(([^)]+)\)\.await', r'.enrich_transaction(\1)', content)
            content = re.sub(r'\.decode_event_data\(([^)]+)\)\.await', r'.decode_event_data(\1)', content)
            content = re.sub(r'\.validate_transaction\(([^)]+)\)\.await', r'.validate_transaction(\1)', content)
            content = re.sub(r'\.validate_opportunity\(([^)]+)\)\.await', r'.validate_opportunity(\1)', content)
            content = re.sub(r'\.process_opportunity_batch\(([^)]+)\)\.await', r'.process_opportunity_batch(\1)', content)
            content = re.sub(r'\.process_block\(([^)]+)\)\.await', r'.process_block(\1)', content)
            content = re.sub(r'\.process_event\(([^)]+)\)\.await', r'.process_event(\1)', content)
            
            # Fix specific lines that are broken
            content = re.sub(r'self\.len\(\)\.await', 'self.len()', content)
            content = re.sub(r'self\.utilization\(\)\.await', 'self.utilization()', content)
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_numeric_literals():
    """Fix numeric literals with proper suffixes"""
    files_to_fix = [
        "crates/data_storage/src/stream/processor.rs",
        "crates/data_storage/src/stream/aggregator.rs",
        "crates/data_storage/src/pipeline/batch_processor.rs"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix specific broken line
            content = re.sub(r'0\.01_fPS', '0.01_f64', content)
            content = re.sub(r'for _i in 0\.\.2', 'for _i in 0_i32..2_i32', content)
            
            with open(file_path, 'w') as f:
                f.write(content)

def main():
    """Run critical fixes only"""
    print("🚨 Fixing critical compilation errors...")
    
    print("1. Fixing .await errors...")
    fix_await_errors()
    
    print("2. Fixing numeric literals...")
    fix_numeric_literals()
    
    print("✅ Critical fixes applied!")

if __name__ == "__main__":
    main()
