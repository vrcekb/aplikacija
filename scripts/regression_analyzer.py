#!/usr/bin/env python3
"""
TallyIO Performance Regression Analyzer
Analyzes Criterion benchmark results and detects performance regressions
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

class BenchmarkResult:
    """Represents a single benchmark result"""
    def __init__(self, name: str, mean_ns: float, median_ns: float, std_dev: float):
        self.name = name
        self.mean_ns = mean_ns
        self.median_ns = median_ns
        self.std_dev = std_dev
        
    def __repr__(self):
        return f"BenchmarkResult({self.name}, {self.mean_ns:.1f}ns)"

class RegressionAnalyzer:
    """Analyzes benchmark results for performance regressions"""
    
    # Performance targets (nanoseconds)
    TARGETS = {
        "spsc_queue": 100,           # Ultra-low latency target
        "ring_buffer": 100,          # Ultra-low latency target  
        "ultra_low_latency": 100,    # Must maintain
        "memory_usage_1024": 20000,  # 20µs target
        "memory_usage_4096": 60000,  # 60µs target
        "mev_detection": 200000,     # 200µs - maintain competitive
        "latency_under_load": 1000000, # 1ms - critical requirement
    }
    
    # Regression thresholds (percentage)
    REGRESSION_THRESHOLD = 10.0  # 10% regression is concerning
    CRITICAL_THRESHOLD = 25.0    # 25% regression is critical
    
    def __init__(self, criterion_dir: Path):
        self.criterion_dir = criterion_dir
        self.results: Dict[str, BenchmarkResult] = {}
        
    def load_results(self) -> bool:
        """Load all benchmark results from Criterion output"""
        print(f"🔍 Loading benchmark results from {self.criterion_dir}")
        
        if not self.criterion_dir.exists():
            print(f"❌ Criterion directory not found: {self.criterion_dir}")
            return False
            
        loaded_count = 0
        
        # Walk through all benchmark directories
        for bench_dir in self.criterion_dir.iterdir():
            if not bench_dir.is_dir():
                continue
                
            # Look for estimates.json files
            estimates_files = list(bench_dir.rglob("estimates.json"))
            
            for estimates_file in estimates_files:
                if self._load_single_result(estimates_file):
                    loaded_count += 1
                    
        print(f"✅ Loaded {loaded_count} benchmark results")
        return loaded_count > 0
        
    def _load_single_result(self, estimates_file: Path) -> bool:
        """Load a single benchmark result from estimates.json"""
        try:
            with open(estimates_file, 'r') as f:
                data = json.load(f)
                
            # Extract benchmark name from path
            # e.g., spsc_queue/enqueue_dequeue/1024/base/estimates.json
            path_parts = estimates_file.parts
            bench_name = self._extract_benchmark_name(path_parts)
            
            if not bench_name:
                return False
                
            # Extract performance metrics
            mean_ns = data["mean"]["point_estimate"]
            median_ns = data["median"]["point_estimate"] 
            std_dev = data["std_dev"]["point_estimate"]
            
            result = BenchmarkResult(bench_name, mean_ns, median_ns, std_dev)
            self.results[bench_name] = result
            
            return True
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"⚠️  Failed to load {estimates_file}: {e}")
            return False
            
    def _extract_benchmark_name(self, path_parts: Tuple[str, ...]) -> Optional[str]:
        """Extract meaningful benchmark name from file path"""
        # Find criterion directory index
        try:
            criterion_idx = path_parts.index("criterion")
        except ValueError:
            return None
            
        # Extract relevant parts after criterion
        relevant_parts = path_parts[criterion_idx + 1:]
        
        if len(relevant_parts) < 2:
            return None
            
        # Handle different benchmark structures
        bench_group = relevant_parts[0]
        
        if len(relevant_parts) >= 3:
            bench_function = relevant_parts[1]
            bench_param = relevant_parts[2]
            
            # Create meaningful name
            if bench_param.isdigit():
                return f"{bench_group}_{bench_param}"
            else:
                return f"{bench_group}_{bench_function}_{bench_param}"
        else:
            return bench_group
            
    def analyze_regressions(self) -> Dict[str, Dict]:
        """Analyze results for performance regressions"""
        print("\n📊 Performance Regression Analysis")
        print("=" * 50)
        
        analysis = {
            "critical_regressions": [],
            "concerning_regressions": [],
            "target_violations": [],
            "good_performance": [],
        }
        
        for name, result in self.results.items():
            # Check against targets
            target_violation = self._check_target_violation(name, result)
            if target_violation:
                analysis["target_violations"].append(target_violation)
                
            # For now, we don't have historical data for regression detection
            # This would require storing previous results
            
        return analysis
        
    def _check_target_violation(self, name: str, result: BenchmarkResult) -> Optional[Dict]:
        """Check if result violates performance targets"""
        # Find matching target
        target_ns = None
        target_name = None
        
        for target_key, target_value in self.TARGETS.items():
            if target_key in name.lower():
                target_ns = target_value
                target_name = target_key
                break
                
        if target_ns is None:
            return None
            
        if result.mean_ns > target_ns:
            violation_ratio = result.mean_ns / target_ns
            severity = "CRITICAL" if violation_ratio > 10 else "HIGH" if violation_ratio > 5 else "MEDIUM"
            
            return {
                "name": name,
                "target": target_ns,
                "actual": result.mean_ns,
                "ratio": violation_ratio,
                "severity": severity,
                "target_name": target_name,
            }
            
        return None
        
    def generate_report(self, analysis: Dict) -> str:
        """Generate detailed regression report"""
        report = []
        report.append("# 🚨 TallyIO Performance Regression Report")
        report.append(f"**Generated:** {self._get_timestamp()}")
        report.append("")
        
        # Summary
        total_violations = len(analysis["target_violations"])
        critical_count = sum(1 for v in analysis["target_violations"] if v["severity"] == "CRITICAL")
        
        if total_violations == 0:
            report.append("## ✅ EXCELLENT - No Performance Issues Detected")
        else:
            report.append(f"## ❌ PERFORMANCE ISSUES DETECTED")
            report.append(f"- **Total Violations:** {total_violations}")
            report.append(f"- **Critical Issues:** {critical_count}")
            
        report.append("")
        
        # Target violations
        if analysis["target_violations"]:
            report.append("## 🎯 Performance Target Violations")
            report.append("")
            
            for violation in sorted(analysis["target_violations"], key=lambda x: x["ratio"], reverse=True):
                severity_emoji = "🔥" if violation["severity"] == "CRITICAL" else "⚠️"
                report.append(f"### {severity_emoji} {violation['name']}")
                report.append(f"- **Target:** {violation['target']:,.0f} ns")
                report.append(f"- **Actual:** {violation['actual']:,.0f} ns")
                report.append(f"- **Violation:** {violation['ratio']:.1f}x slower than target")
                report.append(f"- **Severity:** {violation['severity']}")
                report.append("")
                
        # Recommendations
        report.append("## 🔧 Immediate Actions Required")
        
        critical_violations = [v for v in analysis["target_violations"] if v["severity"] == "CRITICAL"]
        if critical_violations:
            report.append("### 🔥 CRITICAL - Fix within 24h")
            for v in critical_violations:
                report.append(f"- **{v['name']}**: {v['ratio']:.1f}x slower than target")
                
        high_violations = [v for v in analysis["target_violations"] if v["severity"] == "HIGH"]
        if high_violations:
            report.append("### ⚠️ HIGH PRIORITY - Fix within 48h")
            for v in high_violations:
                report.append(f"- **{v['name']}**: {v['ratio']:.1f}x slower than target")
                
        return "\n".join(report)
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def print_summary(self, analysis: Dict):
        """Print analysis summary to console"""
        violations = analysis["target_violations"]
        
        if not violations:
            print("🎉 ALL PERFORMANCE TARGETS MET!")
            print("✅ No regressions detected")
            return
            
        print(f"❌ PERFORMANCE ISSUES DETECTED: {len(violations)} violations")
        print("")
        
        for violation in sorted(violations, key=lambda x: x["ratio"], reverse=True):
            severity_emoji = "🔥" if violation["severity"] == "CRITICAL" else "⚠️"
            print(f"{severity_emoji} {violation['name']}: {violation['actual']:,.0f}ns "
                  f"(target: {violation['target']:,.0f}ns) - {violation['ratio']:.1f}x slower")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze TallyIO benchmark results for regressions")
    parser.add_argument("--criterion-dir", type=Path, default="target/criterion",
                       help="Path to Criterion output directory")
    parser.add_argument("--output", type=Path, help="Output report file")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Exit with error code if regressions detected")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RegressionAnalyzer(args.criterion_dir)
    
    # Load results
    if not analyzer.load_results():
        print("❌ Failed to load benchmark results")
        sys.exit(1)
        
    # Analyze
    analysis = analyzer.analyze_regressions()
    
    # Print summary
    analyzer.print_summary(analysis)
    
    # Generate report
    if args.output:
        report = analyzer.generate_report(analysis)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Detailed report saved to: {args.output}")
        
    # Exit with error if regressions detected and flag is set
    if args.fail_on_regression and analysis["target_violations"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
