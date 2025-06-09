#!/usr/bin/env python3
"""
Script to fix Vec::new() patterns in TallyIO codebase.
Replaces Vec::new() with Vec::with_capacity() where appropriate.
"""

import os
import re
import sys
from pathlib import Path

def fix_vec_new_in_file(file_path):
    """Fix Vec::new() patterns in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Simple Vec::new() -> Vec::with_capacity(0)
        # But be smart about it - look for context clues
        
        # Pattern for Vec::new() in struct initialization
        content = re.sub(
            r'(\w+):\s*Vec::new\(\),',
            r'\1: Vec::with_capacity(0),',
            content
        )
        
        # Pattern for let mut var = Vec::new() with known capacity hints
        # Look for patterns like "for _ in 0..N" or similar
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            if 'Vec::new()' in line and 'let mut' in line:
                # Look ahead for capacity hints
                capacity = 0
                
                # Look for for loops in next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j]
                    
                    # Pattern: for _ in 0..N
                    match = re.search(r'for\s+\w+\s+in\s+\d+\.\.(\d+)', next_line)
                    if match:
                        capacity = int(match.group(1))
                        break
                    
                    # Pattern: for _ in 0..=N
                    match = re.search(r'for\s+\w+\s+in\s+\d+\.\.=(\d+)', next_line)
                    if match:
                        capacity = int(match.group(1)) + 1
                        break
                    
                    # Pattern: while let Ok/Some
                    if 'while let' in next_line:
                        capacity = 100  # reasonable default
                        break
                
                if capacity > 0:
                    line = line.replace('Vec::new()', f'Vec::with_capacity({capacity})')
                else:
                    line = line.replace('Vec::new()', 'Vec::with_capacity(10)')
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # Final fallback: any remaining Vec::new() -> Vec::with_capacity(0)
        content = re.sub(r'Vec::new\(\)', 'Vec::with_capacity(0)', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Vec::new() patterns in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function."""
    # Target directories
    target_dirs = [
        "crates/secure_storage/src",
    ]
    
    files_fixed = 0
    
    for target_dir in target_dirs:
        target_path = Path(target_dir)
        if not target_path.exists():
            print(f"Directory {target_path} does not exist")
            continue
        
        # Find all Rust files
        rust_files = list(target_path.rglob("*.rs"))
        
        print(f"Processing {len(rust_files)} files in {target_dir}...")
        
        for file_path in rust_files:
            if fix_vec_new_in_file(file_path):
                files_fixed += 1
    
    print(f"\n✅ Fixed Vec::new() patterns in {files_fixed} files")
    return 0

if __name__ == "__main__":
    sys.exit(main())
