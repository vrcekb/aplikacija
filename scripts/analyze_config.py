#!/usr/bin/env python3
"""
TallyIO Config Analyzer - Analyze only config.rs
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_config.py <config_file_path>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"Error: File {config_file} does not exist")
        sys.exit(1)
    
    if not config_file.endswith('config.rs'):
        print(f"Error: This script is only for config.rs files")
        sys.exit(1)
    
    print("🎯 TallyIO Config Analyzer")
    print(f"📁 Analyzing: {config_file}")
    print("-" * 60)
    
    # Run the main analyzer on just this file
    try:
        # Set UTF-8 encoding for Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run([
            'python', 'scripts/code.py', config_file
        ], capture_output=True, text=True, timeout=300, env=env, encoding='utf-8')
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
