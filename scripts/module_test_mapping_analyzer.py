#!/usr/bin/env python3
"""
TallyIO Module Test Mapping Analyzer - Enhanced Version
==================================================

Verifies that all modules are properly included in the appropriate test categories:
- Security modules → security_tests.rs and advanced_security_tests.rs
- Economic/MEV modules → economic_tests.rs
- State management modules → state_consistency_tests.rs
- Performance critical modules → timing_tests.rs and advanced_timing_tests.rs
- Resilience modules → recovery_resilience_tests.rs
- Stability modules → stability_tests.rs
- Monitoring modules → monitoring_observability_tests.rs
- Performance regression modules → performance_regression_tests.rs

Features:
- Detection of missing modules from appropriate test categories
- Analysis of test coverage data (requires cargo tarpaulin)
- Checking for crate-specific tests
- Analysis of test quality (checks for edge cases, panic tests, etc.)
- Verification of performance requirements (<0.1ms for critical paths)

Usage: python scripts/module_test_mapping_analyzer.py [--coverage] [--quality]

Options:
  --coverage    Run coverage analysis (requires cargo tarpaulin)
  --quality     Run test quality analysis
  --performance Check performance test compliance
  --all         Run all analyses
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from datetime import datetime

@dataclass
class ModuleMapping:
    """Mapping of modules to their expected test categories."""
    security_modules: Set[str]
    advanced_security_modules: Set[str]
    economic_modules: Set[str]
    state_modules: Set[str]
    timing_modules: Set[str]
    advanced_timing_modules: Set[str]
    performance_regression_modules: Set[str]
    monitoring_modules: Set[str]
    stability_modules: Set[str]
    recovery_modules: Set[str]
    # Keep other existing modules

@dataclass
class TestInclusion:
    """Which modules are actually included in each test file."""
    security_tests: Set[str]
    advanced_security_tests: Set[str]
    economic_tests: Set[str]
    state_tests: Set[str]
    timing_tests: Set[str]
    advanced_timing_tests: Set[str]
    performance_regression_tests: Set[str]
    monitoring_tests: Set[str]
    stability_tests: Set[str]
    recovery_tests: Set[str]
    # Keep other existing tests

class TestQualityLevel(Enum):
    """Quality levels for tests."""
    EXCELLENT = 4   # Complete coverage, edge cases, performance validation
    GOOD = 3        # Good coverage, some edge cases
    ADEQUATE = 2    # Basic coverage, few edge cases
    MINIMAL = 1     # Minimal coverage, no edge cases
    INSUFFICIENT = 0 # Needs improvement

@dataclass
class TestCoverage:
    """Code coverage data for modules."""
    module_name: str
    line_coverage: float  # Percentage of lines covered
    branch_coverage: float  # Percentage of branches covered
    has_unit_tests: bool
    has_integration_tests: bool
    has_performance_tests: bool
    has_fuzzing_tests: bool
    quality_level: TestQualityLevel

@dataclass
class PerformanceMetrics:
    """Performance metrics for critical modules."""
    module_name: str
    avg_latency_ms: float
    max_latency_ms: float
    meets_requirements: bool  # True if max_latency < 0.1ms for critical paths
    has_zero_allocations: bool = False  # True if critical path has zero heap allocations
    has_lock_free_impl: bool = False  # True if implements lock-free algorithms
    memory_efficiency: float = 0.0  # Memory efficiency score (lower is better)

@dataclass
class CrateTestStatus:
    """Test status for a specific crate."""
    crate_name: str
    has_unit_tests: bool
    has_integration_tests: bool
    has_doc_tests: bool
    has_benchmarks: bool

@dataclass
class MissingModules:
    """Modules missing from their expected test categories."""
    missing_security: Set[str]
    missing_advanced_security: Set[str]
    missing_economic: Set[str]
    missing_state: Set[str]
    missing_timing: Set[str]
    missing_advanced_timing: Set[str]
    missing_performance_regression: Set[str]
    missing_monitoring: Set[str]
    missing_stability: Set[str]
    missing_recovery: Set[str]

@dataclass
class CrateTestStatus:
    """Represents test status for a single crate."""
    crate_name: str
    has_unit_tests: bool
    has_integration_tests: bool
    has_doc_tests: bool
    has_benchmarks: bool

class ModuleTestMappingAnalyzer:
    """Analyzes module inclusion, coverage, and quality in TallyIO test categories."""
    # Constants for TallyIO specific requirements
    CRITICAL_PATH_MAX_LATENCY_MS = 0.1  # Maximum latency for critical paths in ms
    MIN_LINE_COVERAGE_PERCENTAGE = 95.0  # Minimum line coverage percentage for critical modules
    MIN_BRANCH_COVERAGE_PERCENTAGE = 90.0  # Minimum branch coverage percentage for critical modules

    def __init__(self, project_root: str = ".", run_coverage: bool = False, run_quality: bool = False, run_performance: bool = False):
        self.project_root = Path(project_root)
        
        # Analysis flags
        self.run_coverage = run_coverage
        self.run_quality = run_quality
        self.run_performance = run_performance
        
        # Analysis results
        self.module_coverage = {}  # Dict[str, TestCoverage]
        self.crate_test_status = {}  # Dict[str, CrateTestStatus]
        self.performance_metrics = {}  # Dict[str, PerformanceMetrics]

        # Define comprehensive module categories based on TallyIO codebase analysis
        self.module_categories = {
            # Security-related modules (enhanced with actual TallyIO modules)
            'security': {
                'security', 'auth', 'crypto', 'key', 'signature', 'validation',
                'protection', 'guard', 'access', 'permission', 'audit', 'error',
                'critical', 'safety', 'risk', 'compliance', 'threat', 'vulnerability'
            },

            # Economic/MEV-related modules (enhanced with actual TallyIO modules)
            'economic': {
                'mev', 'arbitrage', 'liquidation', 'opportunity', 'profit',
                'economics', 'trading', 'swap', 'dex', 'price', 'slippage',
                'sandwich', 'frontrun', 'backrun', 'flash_loan', 'analyzer',
                'filter', 'watcher', 'mempool', 'defi', 'yield', 'farming',
                'cross_chain', 'bridge', 'amm', 'orderbook', 'market_making'
            },

            # State management modules (enhanced with actual TallyIO modules)
            'state': {
                'state', 'storage', 'database', 'cache', 'sync', 'consistency',
                'transaction', 'mempool', 'global', 'local', 'persistence',
                'checkpoint', 'rollback', 'recovery', 'backup', 'restore'
            },

            # Performance/timing critical modules (enhanced with actual TallyIO modules)
            'timing': {
                'engine', 'executor', 'scheduler', 'worker', 'optimization',
                'performance', 'cpu', 'memory', 'simd', 'lock_free', 'affinity',
                'latency', 'benchmark', 'utils', 'time', 'hash', 'memory_pool',
                'cpu_affinity', 'prelude', 'types', 'result', 'config'
            },

            # Advanced categories (enhanced with actual TallyIO modules)
            'advanced_security': {
                'advanced_security', 'penetration', 'encryption', 'authentication', 'threat_model',
                'vulnerability', 'exploit', 'hardening', 'defense', 'attack_vector',
                'compliance', 'audit_trail', 'forensics', 'intrusion_detection'
            },
            'advanced_timing': {
                'advanced_timing', 'concurrency', 'parallelism', 'async', 'await', 'throughput',
                'high_performance', 'streaming', 'real_time', 'tokio', 'futures',
                'rayon', 'crossbeam', 'atomic', 'lock_free'
            },
            'performance_regression': {
                'regression', 'benchmark', 'profile', 'flamegraph', 'bottleneck', 'slowdown',
                'degradation', 'perf_test', 'criterion', 'pprof', 'valgrind', 'heaptrack'
            },
            'monitoring': {
                'monitor', 'metrics', 'observability', 'telemetry', 'dashboard', 'alert',
                'log', 'trace', 'span', 'prometheus', 'grafana', 'metrics_collector',
                'prometheus_exporter', 'stability_monitor', 'health_check'
            },
            'stability': {
                'stability', 'robustness', 'reliability', 'uptime', 'availability',
                'fault_tolerance', 'failover', 'circuit_breaker', 'retry', 'timeout',
                'graceful_degradation', 'load_balancing'
            },
            'recovery': {
                'recovery', 'resilience', 'backup', 'restore', 'checkpoint', 'rollback',
                'disaster', 'failover', 'high_availability', 'replication', 'consensus'
            }
        }

        # Universal test files (cross-cutting concerns across multiple crates)
        self.universal_test_files = {
            'security': 'tests/security_tests.rs',
            'advanced_security': 'tests/advanced_security_tests.rs',
            'economic': 'tests/economic_tests.rs',
            'state': 'tests/state_consistency_tests.rs',
            'timing': 'tests/timing_tests.rs',
            'advanced_timing': 'tests/advanced_timing_tests.rs',
            'performance_regression': 'tests/performance_regression_tests.rs',
            'chaos': 'tests/chaos_engineering_tests.rs',
            'market': 'tests/market_simulation_tests.rs',
            'fuzzing': 'tests/fuzzing_tests.rs',
            'advanced_fuzzing': 'tests/advanced_fuzzing_tests.rs',
            'e2e': 'tests/testnet_e2e_tests.rs',
            'real_testnet': 'tests/real_testnet_integration_tests.rs',
            'integration': 'tests/integration_test.rs',
            'monitoring': 'tests/monitoring_observability_tests.rs',
            'stability': 'tests/stability_tests.rs',
            'recovery': 'tests/recovery_resilience_tests.rs'
        }

        # Note: Crate-specific tests are in crates/*/tests/ and test only that crate's functionality

    def find_all_modules(self) -> Set[str]:
        """Find all Rust modules in the codebase with comprehensive discovery."""
        modules = set()

        # Enhanced search patterns for comprehensive module discovery
        source_patterns = [
            "crates/*/src/**/*.rs",
            "crates/*/src/*.rs",
            "src/**/*.rs",
            "src/*.rs"
        ]

        for pattern in source_patterns:
            for rust_file in self.project_root.glob(pattern):
                # Skip lib.rs, mod.rs, and test files but include them for mod declarations
                if rust_file.name in ['lib.rs', 'mod.rs'] or 'test' in str(rust_file):
                    # Still extract mod declarations from these files
                    try:
                        content = rust_file.read_text(encoding='utf-8')
                        mod_matches = re.findall(r'(?:pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                        modules.update(mod_matches)
                    except Exception:
                        pass
                    continue

                # Extract module name from file path with full path context
                try:
                    # Get relative path from project root
                    rel_path = rust_file.relative_to(self.project_root)
                    path_parts = list(rel_path.parts)

                    # Build hierarchical module name
                    if 'crates' in path_parts:
                        crates_idx = path_parts.index('crates')
                        if crates_idx + 2 < len(path_parts) and path_parts[crates_idx + 2] == 'src':
                            # crates/crate_name/src/module.rs -> crate_name::module
                            crate_name = path_parts[crates_idx + 1]
                            module_parts = path_parts[crates_idx + 3:]

                            # Remove .rs extension from last part
                            if module_parts and module_parts[-1].endswith('.rs'):
                                module_parts[-1] = module_parts[-1][:-3]

                            # Create full module path
                            if module_parts:
                                full_module = f"{crate_name}::{'/'.join(module_parts)}"
                                modules.add(full_module)
                                # Also add just the module name
                                modules.add(module_parts[-1])
                    else:
                        # For src/ files, just use the filename
                        module_name = rust_file.stem
                        modules.add(module_name)

                    # Also extract module names from file content
                    content = rust_file.read_text(encoding='utf-8')
                    # Find mod declarations
                    mod_matches = re.findall(r'(?:pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                    modules.update(mod_matches)

                    # Find use statements to discover more modules
                    use_matches = re.findall(r'use\s+(?:crate::)?([a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)*)', content)
                    for use_match in use_matches:
                        # Split by :: and add each part
                        parts = use_match.split('::')
                        modules.update(parts)

                except Exception as e:
                    # Fallback to simple filename
                    modules.add(rust_file.stem)

        return modules

    def analyze_coverage(self) -> Dict[str, TestCoverage]:
        """Analyze test coverage using LLVM source-based code coverage."""
        if not self.run_coverage:
            print("  Skipping coverage analysis")
            return {}
            
        print("  Running tests with LLVM coverage instrumentation...")
        try:
            # Create directory for coverage data
            coverage_dir = self.project_root / ".coverage"
            coverage_dir.mkdir(exist_ok=True)
            
            # Set LLVM profile environment variables
            profile_dir = coverage_dir / "profraw"
            profile_dir.mkdir(exist_ok=True)
            profile_data = str(profile_dir / "tallyio-%p-%m.profraw")
            
            # Define environment variables for LLVM coverage
            env = os.environ.copy()
            env["RUSTFLAGS"] = "-Cinstrument-coverage"
            env["LLVM_PROFILE_FILE"] = profile_data
            
            # Run cargo test to generate coverage data
            print("  Running cargo test with coverage instrumentation...")
            subprocess.run(
                ["cargo", "test", "--all-features"], 
                env=env, check=False, cwd=self.project_root
            )
            
            # Merge profraw files
            merged_profdata = coverage_dir / "coverage.profdata"
            profraw_files = list(profile_dir.glob("*.profraw"))
            
            if not profraw_files:
                print("  No coverage data generated. Tests may have failed.")
                return {}
                
            print(f"  Processing {len(profraw_files)} coverage data files...")
            
            # Find llvm-profdata tool
            profdata_cmd = None
            for cmd in ["llvm-profdata", "llvm-profdata-14", "llvm-profdata-15", "llvm-profdata-16"]:
                try:
                    result = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        profdata_cmd = cmd
                        break
                except FileNotFoundError:
                    continue
                    
            if not profdata_cmd:
                print("  Error: llvm-profdata tool not found")
                return {}
                
            # Merge profraw files
            subprocess.run(
                [profdata_cmd, "merge", "-sparse", *[str(f) for f in profraw_files], "-o", str(merged_profdata)],
                check=False, cwd=self.project_root
            )
            
            # Find llvm-cov tool
            cov_cmd = None
            for cmd in ["llvm-cov", "llvm-cov-14", "llvm-cov-15", "llvm-cov-16"]:
                try:
                    result = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        cov_cmd = cmd
                        break
                except FileNotFoundError:
                    continue
                    
            if not cov_cmd:
                print("  Error: llvm-cov tool not found")
                return {}
                
            # Get coverage report in JSON format
            coverage_json = coverage_dir / "coverage-report.json"
            target_dir = self.project_root / "target" / "debug"
            test_binaries = []
            
            # Find test binaries
            for entry in target_dir.glob("deps/*"):
                if entry.is_file() and entry.stat().st_mode & 0o111 and not entry.name.endswith(".d"):
                    # Check if it's a test binary
                    try:
                        result = subprocess.run(
                            [str(entry), "--list", "--format=json"], 
                            capture_output=True, text=True, check=False
                        )
                        if result.returncode == 0 and "tests" in result.stdout:
                            test_binaries.append(str(entry))
                    except:
                        continue
                        
            if not test_binaries:
                print("  Error: No test binaries found")
                return {}
                
            # Generate coverage report
            subprocess.run(
                [cov_cmd, "export", "-instr-profile", str(merged_profdata), 
                 *["-object" for _ in test_binaries], *test_binaries,
                 "-format=text", f"-output-dir={coverage_dir}"],
                check=False, cwd=self.project_root
            )
            
            # Parse coverage files to extract data
            coverage_files = list(coverage_dir.glob("coverage-*.txt"))
            
            if not coverage_files:
                print("  Error: No coverage output files generated")
                return {}
                
            # Process coverage data
            for file_path in coverage_files:
                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        
                    # Extract module info
                    current_file = None
                    current_module = None
                    covered_lines = 0
                    total_lines = 0
                    branch_hits = 0
                    branch_total = 0
                    
                    for line in lines:
                        line = line.strip()
                        
                        # New file entry
                        if line.startswith("File "):
                            # Process previous module if exists
                            if current_module and total_lines > 0:
                                # Calculate percentages
                                line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                                branch_coverage = (branch_hits / branch_total * 100) if branch_total > 0 else 0
                                
                                # Determine test types
                                has_unit_tests = self._has_unit_tests(Path(current_file))
                                package_name = self._extract_package_name(current_file)
                                has_integration_tests = self._check_for_integration_tests(package_name, current_module)
                                has_performance_tests = self._check_for_performance_tests(current_module)
                                has_fuzzing_tests = self._check_for_fuzzing_tests(current_module)
                                
                                # Determine test quality level
                                quality_level = self._determine_test_quality(
                                    line_coverage, branch_coverage,
                                    has_unit_tests, has_integration_tests,
                                    has_performance_tests, has_fuzzing_tests
                                )
                                
                                # Create TestCoverage object
                                self.module_coverage[current_module] = TestCoverage(
                                    module_name=current_module,
                                    line_coverage=line_coverage,
                                    branch_coverage=branch_coverage,
                                    has_unit_tests=has_unit_tests,
                                    has_integration_tests=has_integration_tests,
                                    has_performance_tests=has_performance_tests,
                                    has_fuzzing_tests=has_fuzzing_tests,
                                    quality_level=quality_level
                                )
                            
                            # Reset counters for new file
                            current_file = line[5:].strip()
                            current_module = self._extract_module_name(current_file)
                            covered_lines = 0
                            total_lines = 0
                            branch_hits = 0
                            branch_total = 0
                            
                        # Line coverage data
                        elif current_module and line and line[0].isdigit():
                            parts = line.split('|')
                            if len(parts) >= 2:
                                count = parts[1].strip()
                                if count.isdigit() or count == '0':
                                    total_lines += 1
                                    if int(count) > 0:
                                        covered_lines += 1
                                        
                        # Branch coverage data (simplified)
                        elif current_module and "branch" in line.lower() and "%" in line:
                            try:
                                branch_parts = line.split('%')[0].strip().split()
                                branch_percent = float(branch_parts[-1])
                                branch_coverage = branch_percent
                                # Estimate branch hits/total from percentage
                                branch_total = 100
                                branch_hits = int(branch_percent)
                            except:
                                pass
                                
                    # Process last module
                    if current_module and total_lines > 0:
                        # Calculate percentages
                        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                        branch_coverage = (branch_hits / branch_total * 100) if branch_total > 0 else 0
                        
                        # Determine test types
                        has_unit_tests = self._has_unit_tests(Path(current_file))
                        package_name = self._extract_package_name(current_file)
                        has_integration_tests = self._check_for_integration_tests(package_name, current_module)
                        has_performance_tests = self._check_for_performance_tests(current_module)
                        has_fuzzing_tests = self._check_for_fuzzing_tests(current_module)
                        
                        # Determine test quality level
                        quality_level = self._determine_test_quality(
                            line_coverage, branch_coverage,
                            has_unit_tests, has_integration_tests,
                            has_performance_tests, has_fuzzing_tests
                        )
                        
                        # Create TestCoverage object
                        self.module_coverage[current_module] = TestCoverage(
                            module_name=current_module,
                            line_coverage=line_coverage,
                            branch_coverage=branch_coverage,
                            has_unit_tests=has_unit_tests,
                            has_integration_tests=has_integration_tests,
                            has_performance_tests=has_performance_tests,
                            has_fuzzing_tests=has_fuzzing_tests,
                            quality_level=quality_level
                        )
                except Exception as e:
                    print(f"  Error processing coverage file {file_path}: {e}")
            
            print(f"  Analyzed coverage for {len(self.module_coverage)} modules")
        except Exception as e:
            print(f"  Error analyzing coverage: {e}")
            
        return self.module_coverage
        
    def _extract_package_name(self, file_path: str) -> str:
        """Extract crate name from file path."""
        try:
            path = Path(file_path)
            if "crates" in path.parts:
                crates_idx = path.parts.index("crates")
                if crates_idx + 1 < len(path.parts):
                    return path.parts[crates_idx + 1]
            return "tallyio"
        except:
            return "tallyio"
            
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        try:
            path = Path(file_path)
            rel_path = path.relative_to(self.project_root) if str(self.project_root) in file_path else path
            
            parts = list(rel_path.parts)
            if "src" in parts:
                src_index = parts.index("src")
                # Get module name components after src
                module_parts = parts[src_index+1:]
                
                # Handle lib.rs, mod.rs special cases
                if module_parts[-1] == "lib.rs":
                    module_parts = module_parts[:-1]
                elif module_parts[-1] == "mod.rs":
                    module_parts = module_parts[:-1]
                else:
                    # Remove .rs extension from last part
                    module_parts[-1] = module_parts[-1][:-3] if module_parts[-1].endswith(".rs") else module_parts[-1]
                    
                # Create module name path
                module_name = "::".join(module_parts)
                
                # If empty (e.g., src/lib.rs), use crate name
                if not module_name:
                    if "crates" in parts:
                        crates_idx = parts.index("crates")
                        if crates_idx + 1 < len(parts):
                            return parts[crates_idx + 1]
                    return "tallyio"
                    
                return module_name
            elif path.name.endswith(".rs"):
                # For root files, use filename without extension
                return path.stem
            else:
                # Default case
                return "unknown"
        except Exception as e:
            print(f"  Error extracting module name from {file_path}: {e}")
            return "unknown"

    def _has_unit_tests(self, file_path: Path) -> bool:
        """Check if the file has unit tests with enhanced detection."""
        try:
            if not file_path.exists():
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

                # Enhanced test detection patterns
                test_patterns = [
                    "#[test]",
                    "#[cfg(test)]",
                    "mod tests {",
                    "mod test {",
                    "#[tokio::test]",
                    "#[async_test]",
                    "fn test_",
                    "#[should_panic]",
                    "#[ignore]",
                    "assert_eq!",
                    "assert_ne!",
                    "assert!",
                    "panic!",
                    "Result<(), "
                ]

                return any(pattern in content for pattern in test_patterns)
        except:
            return False
            
    def _check_for_integration_tests(self, package_name: str, module_name: str) -> bool:
        """Check if the module has integration tests."""
        # Check for integration tests in tests directory
        tests_dir = self.project_root / "crates" / package_name / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.glob("**/*.rs"):
                try:
                    with open(test_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if module_name in content:
                            return True
                except:
                    continue
                    
        # Also check universal integration tests
        integration_tests = self.project_root / "tests" / "integration_test.rs"
        if integration_tests.exists():
            try:
                with open(integration_tests, "r", encoding="utf-8") as f:
                    content = f.read()
                    return module_name in content
            except:
                pass
                
        return False
        
    def _check_for_performance_tests(self, module_name: str) -> bool:
        """Check if the module has performance tests with comprehensive benchmark discovery."""
        # Check performance test files
        performance_files = [
            self.project_root / "tests" / "timing_tests.rs",
            self.project_root / "tests" / "advanced_timing_tests.rs",
            self.project_root / "tests" / "performance_regression_tests.rs"
        ]

        for perf_file in performance_files:
            if perf_file.exists():
                try:
                    with open(perf_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if self._contains_module_reference(content, module_name):
                            return True
                except:
                    continue

        # Check workspace-level benchmarks
        benchmark_dir = self.project_root / "benches"
        if benchmark_dir.exists():
            for bench_file in benchmark_dir.glob("**/*.rs"):
                try:
                    with open(bench_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if self._contains_module_reference(content, module_name):
                            return True
                except:
                    pass

        # Check all crate-specific benchmarks (comprehensive discovery)
        crates_dir = self.project_root / "crates"
        if crates_dir.exists():
            for crate_dir in crates_dir.iterdir():
                if not crate_dir.is_dir():
                    continue

                crate_bench_dir = crate_dir / "benches"
                if crate_bench_dir.exists():
                    for bench_file in crate_bench_dir.glob("**/*.rs"):
                        try:
                            with open(bench_file, "r", encoding="utf-8") as f:
                                content = f.read()
                                if self._contains_module_reference(content, module_name):
                                    return True
                        except:
                            pass

        # Also check for specific benchmark patterns in module names
        benchmark_patterns = ['bench', 'benchmark', 'perf', 'performance', 'timing', 'latency']
        module_lower = module_name.lower()
        for pattern in benchmark_patterns:
            if pattern in module_lower:
                return True

        return False

    def _contains_module_reference(self, content: str, module_name: str) -> bool:
        """Enhanced module reference detection in file content."""
        if not content or not module_name:
            return False

        # Direct module name match
        if module_name in content:
            return True

        # Handle hierarchical module names (e.g., "core::engine")
        if "::" in module_name:
            parts = module_name.split("::")
            # Check if all parts are mentioned
            if all(part in content for part in parts):
                return True
            # Check for use statements with the full path
            full_path_patterns = [
                f"use {module_name}",
                f"use crate::{module_name}",
                f"use super::{module_name}",
                f"{module_name}::",
                f"crate::{module_name}::"
            ]
            if any(pattern in content for pattern in full_path_patterns):
                return True

        # Check for common Rust patterns
        patterns = [
            f"mod {module_name}",
            f"use {module_name}",
            f"use crate::{module_name}",
            f"use super::{module_name}",
            f"{module_name}::",
            f"crate::{module_name}::",
            f"super::{module_name}::",
            f"fn test_{module_name}",
            f"fn bench_{module_name}",
            f"#[test] {module_name}",
            f"#[bench] {module_name}"
        ]

        return any(pattern in content for pattern in patterns)

    def _check_for_fuzzing_tests(self, module_name: str) -> bool:
        """Check if the module has fuzzing tests."""
        # Look for fuzzing tests in tests/fuzzing/
        fuzzing_dir = self.project_root / "tests" / "fuzzing"
        if not fuzzing_dir.exists():
            return False
            
        # Check if any fuzzing test mentions the module
        for test_file in fuzzing_dir.glob("**/*.rs"):
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if self._contains_module_reference(content, module_name):
                        return True
            except:
                pass
        
        # Also check fuzz directory if it exists
        fuzz_dir = self.project_root / "fuzz" / "fuzz_targets"
        if fuzz_dir.exists():
            for fuzz_file in fuzz_dir.glob("**/*.rs"):
                try:
                    with open(fuzz_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if self._contains_module_reference(content, module_name):
                            return True
                except:
                    pass
                    
        return False
        
    def _analyze_optimization_practices(self) -> Dict[str, PerformanceMetrics]:
        """Analyze critical optimization practices for ultra-low latency.
        
        Checks for:
        - Zero allocation design in critical paths
        - Lock-free algorithm implementation
        - Memory efficiency and optimization
        - Performance annotations and attributes
        """
        print("  Analyzing critical optimization practices...")
        
        results = {}
        
        # Najprej identificiramo kritične poti - bodisi iz arhitekturne dokumentacije ali po
        # dogovorjenih imenskih vzorcih, ki nakazujejo latency-critical komponente
        critical_paths = [
            "core::engine", "core::optimization", "strategies::mev",
            "blockchain::transaction", "blockchain::mempool", "risk::precheck",
            "core::state", "core::engine", "risk::analysis", "blockchain::blocks",
            "strategies::arbitrage", "blockchain::dex", "strategies::liquidation"
        ]
        
        # Scan za simšnje kritične module, ki jih morda zgoraj nismo identificirali
        for module in self.find_all_modules():
            module_lower = module.lower()
            for kw in ["latency", "atomic", "concurrent", "memory", "engine", "throughput", "metrics", "stability"]:
                if kw in module_lower and module not in critical_paths:
                    critical_paths.append(module)
                    break
                    
        print(f"  Identified {len(critical_paths)} potential critical path modules")
        
        # Najprej izvedemo pregled benchmark rezultatov (pred analizo datotek),
        # da imamo osnovo za primerjavo pri analizi kode
        benchmark_results = {}
        benchmark_dir = self.project_root / "target" / "criterion"
        
        if benchmark_dir.exists():
            for bench_dir in benchmark_dir.glob("*/*/new/estimates.json"):
                try:
                    with open(bench_dir, "r", encoding="utf-8") as f:
                        bench_data = json.load(f)
                        
                    # Dobimo ime benchmarka iz poti (najpogosteje ime modula ali funkcije)
                    bench_name = bench_dir.parent.parent.parent.name
                    
                    # Izločimo latenco v milisekundah
                    avg_latency_ms = bench_data.get("mean", {}).get("point_estimate", 1.0) / 1_000_000  # ns to ms
                    max_latency_ms = bench_data.get("max", {}).get("point_estimate", 10.0) / 1_000_000  # ns to ms
                    
                    # Preverimo, ali je pod 0.1ms za kritične poti
                    meets_requirements = max_latency_ms < 0.1
                    
                    benchmark_results[bench_name] = (avg_latency_ms, max_latency_ms, meets_requirements)
                except (json.JSONDecodeError, FileNotFoundError, KeyError):
                    continue
        
        # Zdaj pa analiziramo datoteke kritičnih modulov
        for module_path in critical_paths:
            parts = module_path.split("::") 
            # Pretvorimo v pot datotečnega sistema
            if len(parts) >= 1:
                # Poskusimo z največjo verjetnostjo najti datoteko modula
                locations = [
                    # Za normalne crate module (npr. core::engine)
                    self.project_root / "crates" / parts[0] / "src" / "/".join(parts[1:]) if len(parts) > 1 else "",
                    # Za src/lib.rs datoteke
                    self.project_root / "crates" / parts[0] / "src",
                    # Za primere, ko je modul neposredno v korenskem imeniku
                    self.project_root / "src" / "/".join(parts)
                ]
                
                found = False
                for base_path in locations:
                    if not base_path or not base_path.exists():
                        continue
                        
                    # Preverimo različne možne datoteke
                    for file_pattern in ["*.rs", "mod.rs", "lib.rs"]:
                        # Če je to direktorij, iščemo datoteke v njem
                        if base_path.is_dir():
                            for file_path in base_path.glob(file_pattern):
                                if file_path.exists() and file_path.is_file():
                                    # Ustvarimo kratko ime modula za ujemanje z benchmark rezultati
                                    short_module_name = module_path.split("::")[-1]
                                    
                                    # Poskusimo najti ustrezne benchmark rezultate
                                    bench_key = None
                                    for bench_name in benchmark_results:
                                        if short_module_name in bench_name:
                                            bench_key = bench_name
                                            break
                                            
                                    # Uporabimo benchmark podatke, če so na voljo, drugače privzete vrednosti
                                    if bench_key:
                                        avg_latency, max_latency, meets_req = benchmark_results[bench_key]
                                    else:
                                        avg_latency, max_latency, meets_req = 0.5, 1.0, False
                                        
                                    # Izvedemo analizo optimizacij datoteke
                                    optimization_results = self._analyze_file_for_optimizations(file_path)
                                    
                                    # Ustvarimo končni objekt metrik
                                    metrics = PerformanceMetrics(
                                        module_name=module_path,
                                        avg_latency_ms=avg_latency,
                                        max_latency_ms=max_latency,
                                        meets_requirements=meets_req or (max_latency < 0.1),
                                        has_zero_allocations=optimization_results.get("zero_allocation", False),
                                        has_lock_free_impl=optimization_results.get("lock_free", False),
                                        memory_efficiency=self._calculate_memory_score(optimization_results)
                                    )
                                    
                                    results[module_path] = metrics
                                    found = True
                                    break
                        # Če je to datoteka, jo analizirajmo neposredno
                        elif base_path.is_file() and any(base_path.name.endswith(ext) for ext in [".rs"]):
                            # Podobna logika kot zgoraj, le da neposredno uporabimo base_path kot file_path
                            short_module_name = module_path.split("::")[-1]
                            
                            bench_key = None
                            for bench_name in benchmark_results:
                                if short_module_name in bench_name:
                                    bench_key = bench_name
                                    break
                                    
                            if bench_key:
                                avg_latency, max_latency, meets_req = benchmark_results[bench_key]
                            else:
                                avg_latency, max_latency, meets_req = 0.5, 1.0, False
                                
                            optimization_results = self._analyze_file_for_optimizations(base_path)
                            
                            metrics = PerformanceMetrics(
                                module_name=module_path,
                                avg_latency_ms=avg_latency,
                                max_latency_ms=max_latency,
                                meets_requirements=meets_req or (max_latency < 0.1),
                                has_zero_allocations=optimization_results.get("zero_allocation", False),
                                has_lock_free_impl=optimization_results.get("lock_free", False),
                                memory_efficiency=self._calculate_memory_score(optimization_results)
                            )
                            
                            results[module_path] = metrics
                            found = True
                    
                    if found:
                        break
        
        return results

    def analyze_crate_test_status(self) -> Dict[str, CrateTestStatus]:
        """Analyze test status for each crate including benchmarks."""
        crate_status = {}

        # Analyze workspace-level crates
        crates_dir = self.project_root / "crates"
        if crates_dir.exists():
            for crate_dir in crates_dir.iterdir():
                if not crate_dir.is_dir():
                    continue

                crate_name = crate_dir.name

                # Check for unit tests (in src/ files)
                has_unit_tests = False
                src_dir = crate_dir / "src"
                if src_dir.exists():
                    for src_file in src_dir.glob("**/*.rs"):
                        if self._has_unit_tests(src_file):
                            has_unit_tests = True
                            break

                # Check for integration tests (in tests/ directory)
                has_integration_tests = False
                tests_dir = crate_dir / "tests"
                if tests_dir.exists() and any(tests_dir.glob("*.rs")):
                    has_integration_tests = True

                # Check for doc tests (look for /// examples in src files)
                has_doc_tests = False
                if src_dir.exists():
                    for src_file in src_dir.glob("**/*.rs"):
                        try:
                            with open(src_file, "r", encoding="utf-8") as f:
                                content = f.read()
                                if "///" in content and ("```" in content or "# " in content):
                                    has_doc_tests = True
                                    break
                        except:
                            continue

                # Check for benchmarks (in benches/ directory)
                has_benchmarks = False
                benches_dir = crate_dir / "benches"
                if benches_dir.exists() and any(benches_dir.glob("*.rs")):
                    has_benchmarks = True

                crate_status[crate_name] = CrateTestStatus(
                    crate_name=crate_name,
                    has_unit_tests=has_unit_tests,
                    has_integration_tests=has_integration_tests,
                    has_doc_tests=has_doc_tests,
                    has_benchmarks=has_benchmarks
                )

        return crate_status

    def _calculate_memory_score(self, optimization_results: Dict[str, bool]) -> float:
        """Izračun skupne ocene učinkovitosti pomnilnika na podlagi več optimizacijskih atributov."""
        score = 0.0
        
        # Prealokacija kapacitete
        if optimization_results.get("capacity_preallocated", False):
            score += 0.5
        
        # Optimizacija pomnilniške postavitve
        if optimization_results.get("memory_layout_optimized", False):
            score += 0.5
        
        # Splošna učinkovitost pomnilnika
        if optimization_results.get("memory_efficient", False):
            score += 0.5
        
        # Inlining funkcij
        if optimization_results.get("function_inlined", False):
            score += 0.5
            
        return score
    
    def _analyze_file_for_optimizations(self, file_path: Path) -> Dict[str, bool]:
        """Analyze a source file for specific optimization practices."""
        results = {
            "zero_allocation": False,
            "lock_free": False,
            "memory_efficient": False,
            "memory_layout_optimized": False,
            "function_inlined": False,
            "capacity_preallocated": False
        }
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Check for zero allocation patterns (razširjen seznam vzorcev)
                zero_alloc_patterns = [
                    "#[no_std]", 
                    "Bumpalo", 
                    "bumpalo::Bump",
                    "bump_alloc",
                    "#[global_allocator]",
                    "#[allocator_trait]",
                    "static BUFFER",
                    "no_heap_allocation",
                    "zero_alloc",
                    "arena_alloc",
                    "&mut self.arena",
                    "::Bump::new",
                    "#[no_heap]",
                    "alloca::",
                    "no_dynamic_allocation",
                    "staticvec::",
                    "smallvec::",
                    "tinyvec::",
                    "[repr(C)][repr(align(64))]", # Verjetno za poravnano alokacijo
                    "stack_buffer",
                    "alloc_guard"
                ]
                
                for pattern in zero_alloc_patterns:
                    if pattern in content:
                        results["zero_allocation"] = True
                        break
                    
                # Check for lock-free implementation (razširjen seznam vzorcev)
                lock_free_patterns = [
                    "std::sync::atomic",
                    "AtomicUsize",
                    "AtomicBool",
                    "AtomicI64",
                    "AtomicU64",
                    "AtomicU32",
                    "AtomicI32",
                    "crossbeam",
                    "crossbeam::epoch",
                    "crossbeam_epoch",
                    "lock_free",
                    "wait_free",
                    "nonblocking",
                    "compare_exchange",
                    "compare_exchange_weak",
                    "fetch_add",
                    "fetch_sub",
                    "fetch_and",
                    "fetch_or",
                    "fetch_xor",
                    "load(Ordering::",
                    "store(.*Ordering::",
                    "flize::",
                    "lockfree::",
                    "parking_lot::Mutex", # Boljši od std::sync::Mutex, a še vedno lock
                    "atomic_refcell",
                    "RwLock::try_read", # Poskuša pridobiti lock brez blokiranja
                    "RwLock::try_write",
                    "spsc_queue",
                    "mpsc_queue",
                    "evmap::",
                    "dashmap::"
                ]
                
                for pattern in lock_free_patterns:
                    if pattern in content:
                        results["lock_free"] = True
                        break
                    
                # Check for memory layout optimization
                memory_layout_patterns = [
                    "#[repr(C)]", 
                    "#[repr(packed)]",
                    "#[repr(transparent)]",
                    "#[repr(align(",
                    "cache_aligned",
                    "cache_line_size",
                    "cache_padded",
                    "false_sharing",
                    "prefetch",
                    "Prefetch",
                    "memory_order",
                    "layout::Layout",
                    "std::alloc::Layout"
                ]
                
                for pattern in memory_layout_patterns:
                    if pattern in content:
                        results["memory_layout_optimized"] = True
                        break
                    
                # Check for function inlining
                inline_patterns = [
                    "#[inline]",
                    "#[inline(always)]",
                    "#[cold]", # Za funkcije, ki se redko kličejo
                    "#[hot]" # Za funkcije, ki se pogosto kličejo
                ]
                
                for pattern in inline_patterns:
                    if pattern in content:
                        results["function_inlined"] = True
                        break
                    
                # Check for pre-allocated capacity
                capacity_patterns = [
                    "with_capacity",
                    "reserve",
                    "reserve_exact",
                    "capacity()",
                    "set_capacity",
                    "preallocate",
                    "shrink_to_fit",
                    "shrink_to"
                ]
                
                for pattern in capacity_patterns:
                    if pattern in content:
                        results["capacity_preallocated"] = True
                        break
                    
                # Check for memory efficiency patterns
                efficiency_patterns = [
                    "Vec::with_capacity", 
                    "HashMap::with_capacity",
                    "HashSet::with_capacity",
                    "BTreeMap::",
                    "BTreeSet::",
                    "std::alloc::Allocator",
                    "#[derive(Copy)]",
                    "Box::leak",
                    "Box::into_raw",
                    "as_ptr",
                    "NonNull",
                    "no_copy",
                    "zero_copy",
                    "pool_allocator",
                    "slab_allocator",
                    "memmap::",
                    "mmap::"
                ]
                
                for pattern in efficiency_patterns:
                    if pattern in content:
                        results["memory_efficient"] = True
                        break
                        
                # Negativni vzorci - če zaznamo te, je manj verjetno, da koda res uporablja zero-allocation ali lock-free zasnovo
                negative_patterns = [
                    "Box::new",
                    "Vec::new()", # Brez with_capacity
                    "HashMap::new()", # Brez with_capacity
                    "std::sync::Mutex",
                    "std::sync::RwLock",
                    "parking_lot::RwLock",
                    "unwrap()",
                    "expect(",
                    ".clone()"
                ]
                
                # Preverimo, če je to test datoteka (test datoteke bomo ignorirali za negativne vzorce)
                is_test = "#[test]" in content or "#[bench]" in content or "benches/" in str(file_path) or "tests/" in str(file_path)
                
                if not is_test:
                    for pattern in negative_patterns:
                        # Zmanjšamo zaupanje v rezultate, če najdemo negativne vzorce
                        if pattern in content:
                            if pattern in ["Box::new", "Vec::new()", "HashMap::new()"]:
                                results["zero_allocation"] = False
                            if pattern in ["std::sync::Mutex", "std::sync::RwLock", "parking_lot::RwLock"]:
                                results["lock_free"] = False
        except Exception as e:
            print(f"  Could not analyze file {file_path}: {e}")
            
        return results
    
    # Metoda _analyze_file_optimizations je bila zamenjana z _analyze_file_for_optimizations
        
    def _determine_test_quality(self, line_coverage: float, branch_coverage: float,
                                has_unit_tests: bool, has_integration_tests: bool,
                                has_performance_tests: bool, has_fuzzing_tests: bool) -> TestQualityLevel:
        """Determine test quality level based on coverage and test types.
        
        For financial applications with ultra-low latency requirements, we have stricter
        quality standards, especially for critical path modules.
        """
        score = 0.0
        
        # Coverage scoring - stricter requirements for financial applications
        if line_coverage >= self.MIN_LINE_COVERAGE_PERCENTAGE:
            score += 1.5
        elif line_coverage >= 85:  # Higher threshold for good coverage
            score += 1.0
        elif line_coverage >= 70:  # Higher minimum acceptable threshold
            score += 0.5
            
        if branch_coverage >= self.MIN_BRANCH_COVERAGE_PERCENTAGE:
            score += 1.5
        elif branch_coverage >= 80:  # Higher threshold for good branch coverage
            score += 1.0
        elif branch_coverage >= 60:  # Higher minimum acceptable threshold
            score += 0.5
            
        # Test type scoring - all test types are important for financial applications
        if has_unit_tests:
            score += 0.5
        if has_integration_tests:
            score += 0.75  # Integration tests are more valuable for system integrity
        if has_performance_tests:
            score += 1.0  # Critical for ensuring sub-millisecond performance
        if has_fuzzing_tests:
            score += 1.25  # Essential for security and robustness in financial apps
            
        # Determine level based on score - stricter thresholds
        if score >= 5.0:  # Higher threshold for excellence
            return TestQualityLevel.EXCELLENT
        elif score >= 4.0:  # Higher threshold for good quality
            return TestQualityLevel.GOOD
        elif score >= 3.0:  # Higher threshold for adequacy
            return TestQualityLevel.ADEQUATE
        elif score >= 2.0:  # Higher threshold for minimal quality
            return TestQualityLevel.MINIMAL
        else:
            return TestQualityLevel.INSUFFICIENT
            
    def analyze_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Analyze performance metrics from benchmark results."""
        if not self.run_performance:
            print("  Skipping performance analysis")
            return {}
            
        print("  Analyzing performance metrics...")
        
        # Automatically run benchmarks
        print("  Running cargo bench to execute benchmarks...")
        try:
            # Set up environment for benchmarking
            env = os.environ.copy()
            
            # Run cargo bench
            result = subprocess.run(
                ["cargo", "bench", "--all-features"],
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print(f"  Warning: Benchmark execution failed with code {result.returncode}")
                print(f"  Error: {result.stderr}")
        except Exception as e:
            print(f"  Error running benchmarks: {e}")
        
        # Also analyze critical optimization practices for ultra-low latency
        optimization_analysis = self._analyze_optimization_practices()
        
        # Look for benchmark results
        benchmark_results = self.project_root / "target" / "criterion"
        
        if not benchmark_results.exists():
            print("  Warning: No benchmark results found after execution.")
            return optimization_analysis or {}
            
        # Process benchmark results
        for benchmark_dir in benchmark_results.glob("*/*/new/estimates.json"):
            try:
                # Extract module name from path
                parts = list(benchmark_dir.parts)
                criterion_index = parts.index("criterion")
                if criterion_index + 1 < len(parts):
                    module_name = parts[criterion_index + 1]
                    
                    # Read benchmark data
                    with open(benchmark_dir, "r") as f:
                        benchmark_data = json.load(f)
                        
                    # Extract metrics (convert from ns to ms)
                    avg_latency_ms = benchmark_data.get("mean", {}).get("point_estimate", 0) / 1_000_000
                    max_latency_ms = benchmark_data.get("max", {}).get("point_estimate", 0) / 1_000_000
                    
                    # Check if it meets requirements for critical paths
                    meets_requirements = max_latency_ms <= self.CRITICAL_PATH_MAX_LATENCY_MS
                    
                    self.performance_metrics[module_name] = PerformanceMetrics(
                        module_name=module_name,
                        avg_latency_ms=avg_latency_ms,
                        max_latency_ms=max_latency_ms,
                        meets_requirements=meets_requirements
                    )
            except (ValueError, IndexError, KeyError, json.JSONDecodeError, FileNotFoundError):
                continue
                
        print(f"  Analyzed performance metrics for {len(self.performance_metrics)} modules")
        return self.performance_metrics
        
    def analyze_crate_tests(self) -> Dict[str, CrateTestStatus]:
        """Analyze test structure for each crate."""
        if not self.run_quality:
            print("  Skipping crate test analysis")
            return {}
            
        print("  Analyzing crate test structure...")
        crates_dir = self.project_root / "crates"
        
        for crate_dir in crates_dir.glob("*"):
            if not crate_dir.is_dir():
                continue
                
            crate_name = crate_dir.name
            
            # Check for different test types
            has_unit_tests = False
            has_integration_tests = False
            has_doc_tests = False
            has_benchmarks = False
            
            # Check for unit tests
            src_dir = crate_dir / "src"
            if src_dir.exists():
                for src_file in src_dir.glob("**/*.rs"):
                    try:
                        with open(src_file, "r", encoding="utf-8") as f:
                            content = f.read()
                            if "#[test]" in content or "mod tests {" in content:
                                has_unit_tests = True
                                break
                    except:
                        continue
                        
            # Check for integration tests
            tests_dir = crate_dir / "tests"
            has_integration_tests = tests_dir.exists() and len(list(tests_dir.glob("**/*.rs"))) > 0
            
            # Check for doc tests
            has_doc_tests = False
            lib_rs = crate_dir / "src" / "lib.rs"
            if lib_rs.exists():
                try:
                    with open(lib_rs, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "```rust" in content or "```no_run" in content:
                            has_doc_tests = True
                except:
                    pass
                    
            # Check for benchmarks
            benches_dir = crate_dir / "benches"
            has_benchmarks = benches_dir.exists() and len(list(benches_dir.glob("**/*.rs"))) > 0
            
            self.crate_test_status[crate_name] = CrateTestStatus(
                crate_name=crate_name,
                has_unit_tests=has_unit_tests,
                has_integration_tests=has_integration_tests,
                has_doc_tests=has_doc_tests,
                has_benchmarks=has_benchmarks
            )
            
        print(f"  Analyzed test structure for {len(self.crate_test_status)} crates")
        return self.crate_test_status

    def categorize_modules(self, modules: Set[str]) -> ModuleMapping:
        """Categorize modules based on their names and functionality."""
        security_modules = set()
        advanced_security_modules = set()
        economic_modules = set()
        state_modules = set()
        timing_modules = set()
        advanced_timing_modules = set()
        performance_regression_modules = set()
        monitoring_modules = set()
        stability_modules = set()
        recovery_modules = set()

        # External crates to exclude (these are dependencies, not our modules)
        external_crates = {
            'core_affinity', 'affinity', 'num_cpus', 'memory', 'libc', 'tokio',
            'crossbeam', 'rayon', 'parking_lot', 'serde', 'chrono', 'uuid',
            'log', 'thiserror', 'anyhow', 'reqwest', 'hyper', 'ethers', 'web3'
        }

        for module in modules:
            module_lower = module.lower()

            # Skip external crates
            if module in external_crates or module_lower in external_crates:
                continue

            # Check each category
            for keyword in self.module_categories['security']:
                if keyword in module_lower:
                    security_modules.add(module)
                    break

            for keyword in self.module_categories['advanced_security']:
                if keyword in module_lower:
                    advanced_security_modules.add(module)
                    break

            for keyword in self.module_categories['economic']:
                if keyword in module_lower:
                    economic_modules.add(module)
                    break

            for keyword in self.module_categories['state']:
                if keyword in module_lower:
                    state_modules.add(module)
                    break

            for keyword in self.module_categories['timing']:
                if keyword in module_lower:
                    timing_modules.add(module)
                    break

            for keyword in self.module_categories['advanced_timing']:
                if keyword in module_lower:
                    advanced_timing_modules.add(module)
                    break

            for keyword in self.module_categories['performance_regression']:
                if keyword in module_lower:
                    performance_regression_modules.add(module)
                    break

            for keyword in self.module_categories['monitoring']:
                if keyword in module_lower:
                    monitoring_modules.add(module)
                    break

            for keyword in self.module_categories['stability']:
                if keyword in module_lower:
                    stability_modules.add(module)
                    break

            for keyword in self.module_categories['recovery']:
                if keyword in module_lower:
                    recovery_modules.add(module)
                    break

        return ModuleMapping(
            security_modules=security_modules,
            advanced_security_modules=advanced_security_modules,
            economic_modules=economic_modules,
            state_modules=state_modules,
            timing_modules=timing_modules,
            advanced_timing_modules=advanced_timing_modules,
            performance_regression_modules=performance_regression_modules,
            monitoring_modules=monitoring_modules,
            stability_modules=stability_modules,
            recovery_modules=recovery_modules
        )

    def find_included_modules(self) -> TestInclusion:
        """Find which modules are actually included in each test file."""
        security_tests = set()
        advanced_security_tests = set()
        economic_tests = set()
        state_tests = set()
        timing_tests = set()
        advanced_timing_tests = set()
        performance_regression_tests = set()
        monitoring_tests = set()
        stability_tests = set()
        recovery_tests = set()

        # Check each universal test file
        for test_type, test_file_path in self.universal_test_files.items():
            test_file = self.project_root / test_file_path

            if not test_file.exists():
                print(f"⚠️  Test file not found: {test_file}")
                continue

            try:
                content = test_file.read_text(encoding='utf-8')

                # Find use statements and module references (enhanced)
                use_matches = re.findall(r'use\s+(?:crate::)?([a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)*)', content)
                mod_matches = re.findall(r'mod\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)

                # Find test function names that indicate module testing
                test_function_matches = re.findall(r'fn\s+test_([a-zA-Z_][a-zA-Z0-9_]*)', content)

                # Find module references in comments and test names
                module_comment_matches = re.findall(r'(?:Test|test)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)*)', content)

                # Find direct module name mentions in the entire file (enhanced)
                word_matches = []
                for line in content.split('\n'):  # Check all lines
                    # Look for specific patterns that indicate module testing
                    if any(keyword in line.lower() for keyword in ['test', 'module', 'core::', 'crate::']):
                        # Find full module paths with :: or /
                        module_paths = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:[::/][a-zA-Z_][a-zA-Z0-9_]*)*)\b', line)
                        word_matches.extend(module_paths)

                        # Also find individual words for backward compatibility
                        words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
                        word_matches.extend(words)

                # Flatten use matches to get individual module parts
                flattened_use_matches = []
                for use_match in use_matches:
                    parts = use_match.split('::')
                    flattened_use_matches.extend(parts)

                all_mentions = set(flattened_use_matches + mod_matches + test_function_matches + module_comment_matches + word_matches)

                # Categorize based on test type
                if test_type == 'security':
                    security_tests.update(all_mentions)
                elif test_type == 'advanced_security':
                    advanced_security_tests.update(all_mentions)
                elif test_type == 'economic':
                    economic_tests.update(all_mentions)
                elif test_type == 'state':
                    state_tests.update(all_mentions)
                elif test_type == 'timing':
                    timing_tests.update(all_mentions)
                elif test_type == 'advanced_timing':
                    advanced_timing_tests.update(all_mentions)
                elif test_type == 'performance_regression':
                    performance_regression_tests.update(all_mentions)
                elif test_type == 'monitoring':
                    monitoring_tests.update(all_mentions)
                elif test_type == 'stability':
                    stability_tests.update(all_mentions)
                elif test_type == 'recovery':
                    recovery_tests.update(all_mentions)
                elif test_type == 'integration':
                    # Integration tests cover all categories
                    # Add specific modules that are tested in integration_test.rs
                    integration_modules = {
                        'engine', 'executor', 'TallyEngine', 'error', 'CoreError',
                        'CriticalError', 'utils', 'affinity', 'memory', 'hash',
                        'validation', 'time', 'LatencyTimer', 'mempool', 'watcher',
                        'MempoolWatcher', 'MempoolEvent', 'analyzer', 'MempoolAnalyzer',
                        'transaction', 'Transaction', 'ProcessingResult', 'filter',
                        'MempoolFilter', 'FilterConfig', 'TransactionFilter'
                    }

                    # Distribute integration modules to appropriate categories
                    for module in integration_modules:
                        module_lower = module.lower()

                        # Check which category this module belongs to
                        for keyword in self.module_categories['security']:
                            if keyword in module_lower:
                                security_tests.add(module)
                                break

                        for keyword in self.module_categories['advanced_security']:
                            if keyword in module_lower:
                                advanced_security_tests.add(module)
                                break

                        for keyword in self.module_categories['economic']:
                            if keyword in module_lower:
                                economic_tests.add(module)
                                break

                        for keyword in self.module_categories['state']:
                            if keyword in module_lower:
                                state_tests.add(module)
                                break

                        for keyword in self.module_categories['timing']:
                            if keyword in module_lower:
                                timing_tests.add(module)
                                break

                        for keyword in self.module_categories['advanced_timing']:
                            if keyword in module_lower:
                                advanced_timing_tests.add(module)
                                break

                        for keyword in self.module_categories['performance_regression']:
                            if keyword in module_lower:
                                performance_regression_tests.add(module)
                                break

                        for keyword in self.module_categories['monitoring']:
                            if keyword in module_lower:
                                monitoring_tests.add(module)
                                break

                        for keyword in self.module_categories['stability']:
                            if keyword in module_lower:
                                stability_tests.add(module)
                                break

                        for keyword in self.module_categories['recovery']:
                            if keyword in module_lower:
                                recovery_tests.add(module)
                                break

            except Exception as e:
                print(f"⚠️  Could not read {test_file}: {e}")

        return TestInclusion(
            security_tests=security_tests,
            advanced_security_tests=advanced_security_tests,
            economic_tests=economic_tests,
            state_tests=state_tests,
            timing_tests=timing_tests,
            advanced_timing_tests=advanced_timing_tests,
            performance_regression_tests=performance_regression_tests,
            monitoring_tests=monitoring_tests,
            stability_tests=stability_tests,
            recovery_tests=recovery_tests
        )

    def find_missing_modules(self, expected: ModuleMapping, actual: TestInclusion) -> MissingModules:
        """Find modules that should be tested but are missing from test files."""

        def check_module_coverage(expected_modules, actual_tests):
            """Check if modules are covered, considering both full and partial paths."""
            missing = set()
            for module in expected_modules:
                # Check exact match first
                if module in actual_tests:
                    continue

                # Check if any part of the module path is found
                module_parts = module.split('::')
                found = False

                # Check for partial matches (e.g., 'engine/executor' matches 'core::engine/executor')
                for actual in actual_tests:
                    if '/' in actual and '/' in module:
                        # Compare path parts
                        actual_path = actual.split('::')[-1] if '::' in actual else actual
                        module_path = module.split('::')[-1] if '::' in module else module
                        if actual_path == module_path:
                            found = True
                            break

                    # Check if module name appears in actual test mentions
                    if module.split('::')[-1] in actual or module.split('/')[-1] in actual:
                        found = True
                        break

                if not found:
                    missing.add(module)
            return missing

        missing_security = check_module_coverage(expected.security_modules, actual.security_tests)
        missing_advanced_security = check_module_coverage(expected.advanced_security_modules, actual.advanced_security_tests)
        missing_economic = check_module_coverage(expected.economic_modules, actual.economic_tests)
        missing_state = check_module_coverage(expected.state_modules, actual.state_tests)
        missing_timing = check_module_coverage(expected.timing_modules, actual.timing_tests)
        missing_advanced_timing = check_module_coverage(expected.advanced_timing_modules, actual.advanced_timing_tests)
        missing_performance_regression = check_module_coverage(expected.performance_regression_modules, actual.performance_regression_tests)
        missing_monitoring = check_module_coverage(expected.monitoring_modules, actual.monitoring_tests)
        missing_stability = check_module_coverage(expected.stability_modules, actual.stability_tests)
        missing_recovery = check_module_coverage(expected.recovery_modules, actual.recovery_tests)

        return MissingModules(
            missing_security=missing_security,
            missing_advanced_security=missing_advanced_security,
            missing_economic=missing_economic,
            missing_state=missing_state,
            missing_timing=missing_timing,
            missing_advanced_timing=missing_advanced_timing,
            missing_performance_regression=missing_performance_regression,
            missing_monitoring=missing_monitoring,
            missing_stability=missing_stability,
            missing_recovery=missing_recovery
        )

    def generate_report(self, expected: ModuleMapping, actual: TestInclusion, missing: MissingModules) -> str:
        """Generate detailed module mapping report."""
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.append("=" * 70)
        report.append("🧩 TALLYIO MODULE TEST MAPPING ANALYSIS")
        report.append("=" * 70)
        report.append("")
        report.append("📋 TEST ORGANIZATION")
        report.append("-" * 20)
        report.append("• Universal tests (tests/): Cross-cutting concerns across multiple crates")
        report.append("• Crate-specific tests (crates/*/tests/): Individual crate functionality")
        report.append("• This analysis focuses on universal tests only")
        report.append("")

        # Summary
        total_expected = (len(expected.security_modules) + len(expected.advanced_security_modules) +
                          len(expected.economic_modules) + len(expected.state_modules) +
                          len(expected.timing_modules) + len(expected.advanced_timing_modules) +
                          len(expected.performance_regression_modules) + len(expected.monitoring_modules) +
                          len(expected.stability_modules) + len(expected.recovery_modules))
                          
        total_missing = (len(missing.missing_security) + len(missing.missing_advanced_security) +
                         len(missing.missing_economic) + len(missing.missing_state) +
                         len(missing.missing_timing) + len(missing.missing_advanced_timing) +
                         len(missing.missing_performance_regression) + len(missing.missing_monitoring) +
                         len(missing.missing_stability) + len(missing.missing_recovery))

        coverage_percent = ((total_expected - total_missing) / total_expected * 100) if total_expected > 0 else 100

        report.append("📊 SUMMARY")
        report.append("-" * 20)
        report.append(f"Total categorized modules: {total_expected}")
        report.append(f"Missing from tests: {total_missing}")
        report.append(f"Module inclusion rate: {coverage_percent:.1f}%")
        report.append("")

        # Status
        if total_missing == 0:
            report.append("✅ PERFECT: All modules are properly included in tests")
        elif total_missing <= 3:
            report.append("⚠️  GOOD: Only a few modules missing from tests")
        else:
            report.append("❌ NEEDS ATTENTION: Many modules missing from tests")
        report.append("")

        # Detailed breakdown
        categories = [
            ("SECURITY", expected.security_modules, missing.missing_security, "tests/security_tests.rs"),
            ("ADVANCED SECURITY", expected.advanced_security_modules, missing.missing_advanced_security, "tests/advanced_security_tests.rs"),
            ("ECONOMIC", expected.economic_modules, missing.missing_economic, "tests/economic_tests.rs"),
            ("STATE", expected.state_modules, missing.missing_state, "tests/state_consistency_tests.rs"),
            ("TIMING", expected.timing_modules, missing.missing_timing, "tests/timing_tests.rs"),
            ("ADVANCED TIMING", expected.advanced_timing_modules, missing.missing_advanced_timing, "tests/advanced_timing_tests.rs"),
            ("PERFORMANCE REGRESSION", expected.performance_regression_modules, missing.missing_performance_regression, "tests/performance_regression_tests.rs"),
            ("MONITORING", expected.monitoring_modules, missing.missing_monitoring, "tests/monitoring_observability_tests.rs"),
            ("STABILITY", expected.stability_modules, missing.missing_stability, "tests/stability_tests.rs"),
            ("RECOVERY", expected.recovery_modules, missing.missing_recovery, "tests/recovery_resilience_tests.rs")
        ]

        for category_name, expected_modules, missing_modules, test_file in categories:
            if expected_modules:
                report.append(f"🔍 {category_name} MODULES")
                report.append("-" * (len(category_name) + 10))
                report.append(f"Expected in {test_file}: {len(expected_modules)}")
                report.append(f"Missing: {len(missing_modules)}")

                if missing_modules:
                    report.append("Missing modules:")
                    for module in sorted(missing_modules):
                        report.append(f"  ❌ {module}")
                else:
                    report.append("✅ All modules properly included")
                report.append("")

        # Recommendations
        if total_missing > 0:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 20)
            report.append("1. Add missing modules to their respective universal test files")
            report.append("2. Create test cases for each missing module")
            report.append("3. Consider if module needs crate-specific tests instead")
            report.append("4. Verify module categorization is correct")
            report.append("5. Run this analyzer after adding new modules")
            report.append("")
        else:
            report.append("📝 NOTE")
            report.append("-" * 8)
            report.append("• Crate-specific tests should be in crates/*/tests/")
            report.append("• Universal tests cover cross-cutting concerns")
            report.append("• Both types of tests are important for complete coverage")
            report.append("")
            
        # Add coverage report if available
        if self.module_coverage:
            report.append("📈 TEST COVERAGE ANALYSIS")
            report.append("=" * 25)
            report.append("")
            
            # Calculate overall statistics
            avg_line_coverage = sum(cov.line_coverage for cov in self.module_coverage.values()) / len(self.module_coverage) if self.module_coverage else 0
            avg_branch_coverage = sum(cov.branch_coverage for cov in self.module_coverage.values()) / len(self.module_coverage) if self.module_coverage else 0
            
            # Count modules by quality level
            quality_counts = {level: 0 for level in TestQualityLevel}
            for cov in self.module_coverage.values():
                quality_counts[cov.quality_level] = quality_counts.get(cov.quality_level, 0) + 1
                
            report.append("📊 OVERALL COVERAGE SUMMARY")
            report.append("-" * 25)
            report.append(f"Modules analyzed: {len(self.module_coverage)}")
            report.append(f"Average line coverage: {avg_line_coverage:.1f}%")
            report.append(f"Average branch coverage: {avg_branch_coverage:.1f}%")
            report.append(f"Modules with excellent test quality: {quality_counts.get(TestQualityLevel.EXCELLENT, 0)}")
            report.append(f"Modules with good test quality: {quality_counts.get(TestQualityLevel.GOOD, 0)}")
            report.append(f"Modules with adequate test quality: {quality_counts.get(TestQualityLevel.ADEQUATE, 0)}")
            report.append(f"Modules with minimal test quality: {quality_counts.get(TestQualityLevel.MINIMAL, 0)}")
            report.append(f"Modules with insufficient test quality: {quality_counts.get(TestQualityLevel.INSUFFICIENT, 0)}")
            report.append("")
            
            # Coverage status
            if avg_line_coverage >= self.MIN_LINE_COVERAGE_PERCENTAGE and avg_branch_coverage >= self.MIN_BRANCH_COVERAGE_PERCENTAGE:
                report.append(f"✅ MEETS COVERAGE REQUIREMENTS: >{self.MIN_LINE_COVERAGE_PERCENTAGE}% line, >{self.MIN_BRANCH_COVERAGE_PERCENTAGE}% branch")
            else:
                report.append(f"❌ BELOW COVERAGE REQUIREMENTS: >{self.MIN_LINE_COVERAGE_PERCENTAGE}% line, >{self.MIN_BRANCH_COVERAGE_PERCENTAGE}% branch")
            report.append("")
            
            # Modules with low coverage
            low_coverage_modules = [m for m, cov in self.module_coverage.items() 
                                if cov.line_coverage < self.MIN_LINE_COVERAGE_PERCENTAGE 
                                or cov.branch_coverage < self.MIN_BRANCH_COVERAGE_PERCENTAGE]
            
            if low_coverage_modules:
                report.append("🔻 MODULES WITH INSUFFICIENT COVERAGE")
                report.append("-" * 35)
                for module in sorted(low_coverage_modules):
                    cov = self.module_coverage[module]
                    report.append(f"  • {module}: Line {cov.line_coverage:.1f}%, Branch {cov.branch_coverage:.1f}%")
                report.append("")
            
            # Quality recommendations
            report.append("💡 COVERAGE RECOMMENDATIONS")
            report.append("-" * 25)
            report.append("1. Focus on improving test coverage for modules with < 80% coverage")
            report.append("2. Add edge case tests to improve branch coverage")
            report.append("3. Consider property-based testing for complex modules")
            report.append("4. Aim for 100% coverage on critical security and economic modules")
            report.append("5. Regularly run coverage analysis during development")
            report.append("")
            
        # Add crate test structure report if available
        if self.crate_test_status:
            report.append("📦 CRATE TEST STRUCTURE ANALYSIS")
            report.append("=" * 30)
            report.append("")
            
            # Summary statistics
            unit_test_count = sum(1 for status in self.crate_test_status.values() if status.has_unit_tests)
            integration_test_count = sum(1 for status in self.crate_test_status.values() if status.has_integration_tests)
            doc_test_count = sum(1 for status in self.crate_test_status.values() if status.has_doc_tests)
            benchmark_count = sum(1 for status in self.crate_test_status.values() if status.has_benchmarks)
            
            report.append("📊 CRATE TEST SUMMARY")
            report.append("-" * 20)
            report.append(f"Total crates analyzed: {len(self.crate_test_status)}")
            report.append(f"Crates with unit tests: {unit_test_count} ({unit_test_count/len(self.crate_test_status)*100:.1f}%)")
            report.append(f"Crates with integration tests: {integration_test_count} ({integration_test_count/len(self.crate_test_status)*100:.1f}%)")
            report.append(f"Crates with doc tests: {doc_test_count} ({doc_test_count/len(self.crate_test_status)*100:.1f}%)")
            report.append(f"Crates with benchmarks: {benchmark_count} ({benchmark_count/len(self.crate_test_status)*100:.1f}%)")
            report.append("")
            
            # Missing test types
            missing_unit_tests = [crate for crate, status in self.crate_test_status.items() if not status.has_unit_tests]
            missing_integration_tests = [crate for crate, status in self.crate_test_status.items() if not status.has_integration_tests]
            
            if missing_unit_tests or missing_integration_tests:
                report.append("⚠️ CRATES MISSING ESSENTIAL TEST TYPES")
                report.append("-" * 35)
                
                if missing_unit_tests:
                    report.append("Crates missing unit tests:")
                    for crate in sorted(missing_unit_tests):
                        report.append(f"  • {crate}")
                    report.append("")
                    
                if missing_integration_tests:
                    report.append("Crates missing integration tests:")
                    for crate in sorted(missing_integration_tests):
                        report.append(f"  • {crate}")
                    report.append("")
            
            # Recommendations
            report.append("💡 CRATE TEST RECOMMENDATIONS")
            report.append("-" * 28)
            report.append("1. Ensure every crate has both unit and integration tests")
            report.append("2. Add doc tests for all public APIs to serve as documentation examples")
            report.append("3. Implement benchmarks for performance-critical crates")
            report.append("4. Consider property-based testing for complex data structures")
            report.append("5. Organize test directories consistently across all crates")
            report.append("")
        
        # Add performance metrics report if available
        if self.performance_metrics:
            report.append("⚡ PERFORMANCE METRICS ANALYSIS")
            report.append("=" * 30)
            report.append("")
            
            # Calculate summary stats
            meets_req_count = sum(1 for metric in self.performance_metrics.values() if metric.meets_requirements)
            fails_req_count = len(self.performance_metrics) - meets_req_count
            
            report.append("📊 PERFORMANCE SUMMARY")
            report.append("-" * 22)
            report.append(f"Critical path modules analyzed: {len(self.performance_metrics)}")
            report.append(f"Modules meeting latency requirements: {meets_req_count}")
            report.append(f"Modules failing latency requirements: {fails_req_count}")
            report.append(f"Latency requirement: < {self.CRITICAL_PATH_MAX_LATENCY_MS} ms for critical paths")
            report.append("")
            
            # Status
            if fails_req_count == 0:
                report.append("✅ ALL CRITICAL MODULES MEET LATENCY REQUIREMENTS")
            else:
                report.append("❌ SOME CRITICAL MODULES EXCEED LATENCY REQUIREMENTS")
            report.append("")
            
            # Sort by latency (slowest first)
            sorted_metrics = sorted(self.performance_metrics.items(), 
                                   key=lambda x: x[1].max_latency_ms, 
                                   reverse=True)
            
            # Detailed metrics for critical modules
            report.append("🔍 CRITICAL PATH LATENCY DETAILS")
            report.append("-" * 30)
            for module, metric in sorted_metrics:
                status = "✅" if metric.meets_requirements else "❌"
                report.append(f"{status} {module}: Avg {metric.avg_latency_ms:.3f} ms, Max {metric.max_latency_ms:.3f} ms")
            report.append("")
            
            # Recommendations for performance improvements
            if fails_req_count > 0:
                report.append("💡 PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
                report.append("-" * 38)
                report.append("1. Profile modules failing latency requirements")
                report.append("2. Look for contention, locks, or memory allocations in hot paths")
                report.append("3. Consider using more aggressive parallelism or lock-free structures")
                report.append("4. Minimize blocking operations and external calls in critical paths")
                report.append("5. Benchmark after each optimization to verify improvements")
                report.append("")

        # Add optimization practices analysis if available
        if self.performance_metrics:
            report.append("⚡ CRITICAL OPTIMIZATION ANALYSIS")
            report.append("=" * 30)
            report.append("")
            
            # Count modules by optimization metrics
            meets_latency_req = sum(1 for m in self.performance_metrics.values() if m.meets_requirements)
            has_zero_alloc = sum(1 for m in self.performance_metrics.values() if m.has_zero_allocations)
            has_lock_free = sum(1 for m in self.performance_metrics.values() if m.has_lock_free_impl)
            
            total_critical = len(self.performance_metrics)
            
            report.append("📊 ULTRA-LOW LATENCY REQUIREMENTS")
            report.append("-" * 30)
            report.append(f"Critical path modules analyzed: {total_critical}")
            report.append(f"Modules meeting latency requirements (<0.1ms): {meets_latency_req} ({meets_latency_req/total_critical*100:.1f}%)")
            report.append(f"Modules with zero-allocation design: {has_zero_alloc} ({has_zero_alloc/total_critical*100:.1f}%)")
            report.append(f"Modules with lock-free implementation: {has_lock_free} ({has_lock_free/total_critical*100:.1f}%)")
            report.append("")
            
            # Status of critical optimization
            optimization_percentage = meets_latency_req / total_critical * 100
            if optimization_percentage >= 95:
                report.append("✅ EXCELLENT: Critical paths are highly optimized for ultra-low latency")
            elif optimization_percentage >= 80:
                report.append("✅ GOOD: Most critical paths meet performance requirements")
            elif optimization_percentage >= 60:
                report.append("⚠️  NEEDS IMPROVEMENT: Several critical paths require optimization")
            else:
                report.append("❌ CRITICAL ISSUE: Most critical paths do not meet performance requirements")
            report.append("")
            
            # Detailed breakdown of critical modules
            report.append("🔍 CRITICAL PATH MODULES BREAKDOWN")
            report.append("-" * 30)
            
            # Sort modules by performance status (non-compliant first)
            sorted_modules = sorted(self.performance_metrics.items(), 
                                   key=lambda x: (x[1].meets_requirements, x[1].max_latency_ms))
            
            for module_name, metrics in sorted_modules:
                status = "✅" if metrics.meets_requirements else "❌"
                zero_alloc = "✓" if metrics.has_zero_allocations else "✗"
                lock_free = "✓" if metrics.has_lock_free_impl else "✗"
                memory_score = metrics.memory_efficiency
                
                report.append(f"{status} {module_name}")
                report.append(f"   Max latency: {metrics.max_latency_ms:.3f}ms (target: <0.1ms)")
                report.append(f"   Zero allocations: {zero_alloc} | Lock-free: {lock_free} | Memory score: {memory_score:.1f}/2.0")
                
                # Recommendations for non-compliant modules
                if not metrics.meets_requirements:
                    report.append("   OPTIMIZATION RECOMMENDATIONS:")
                    if not metrics.has_zero_allocations:
                        report.append("   - Implement zero-allocation design using arena allocators (Bumpalo)")
                    if not metrics.has_lock_free_impl:
                        report.append("   - Replace mutexes with lock-free data structures (e.g., AtomicUsize, crossbeam)")
                    if metrics.memory_efficiency < 1.0:
                        report.append("   - Optimize memory usage (pre-allocated capacity, avoid Box/heap allocations)")
                    if metrics.max_latency_ms >= 0.5:
                        report.append("   - Profile and optimize hot paths to reduce latency below 0.1ms")
                    report.append("   - Add #[inline] attributes to critical functions")
                report.append("")
            
            # Financial application specific recommendations
            report.append("💰 FINANCIAL APPLICATION RECOMMENDATIONS")
            report.append("-" * 30)
            report.append("As a financial application with ultra-low latency requirements (<0.1ms):")
            report.append("1. ALL critical path modules MUST implement zero-allocation design")
            report.append("2. ALL critical paths MUST use lock-free algorithms instead of mutexes")
            report.append("3. Latency-sensitive code MUST be benchmarked with <1ms target")
            report.append("4. Memory layout optimization is essential for cache efficiency")
            report.append("5. Add CPU affinity for critical threads")
            report.append("6. Implement data prefetching where possible")
            report.append("7. Use inline assembly for performance-critical sections if necessary")
            report.append("8. Run cargo clippy with --all-targets --all-features for performance warnings")
            report.append("")

        # Timestamp and save report
        report.append(f"Generated on: {timestamp}")
        
        return "\n".join(report)

def main():
    """Main entry point."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TallyIO Module Test Mapping Analyzer")
    parser.add_argument("-p", "--project-root", default=".", help="Root directory of the TallyIO project")
    parser.add_argument("-c", "--coverage", action="store_true", help="Run test coverage analysis")
    parser.add_argument("-q", "--quality", action="store_true", help="Run test quality analysis")
    parser.add_argument("-f", "--performance", action="store_true", help="Run performance metrics analysis")
    parser.add_argument("-a", "--all", action="store_true", help="Run all analyses")
    parser.add_argument("-o", "--output", default="module_test_mapping_report.txt", help="Output file for the report")
    args = parser.parse_args()
    
    # Initialize analyzer with command line options
    analyzer = ModuleTestMappingAnalyzer(
        project_root=args.project_root,
        run_coverage=args.coverage or args.all,
        run_quality=args.quality or args.all,
        run_performance=args.performance or args.all
    )

    print(" Analyzing TallyIO module test mapping...")

    # Find all modules
    all_modules = analyzer.find_all_modules()
    print(f" Found {len(all_modules)} modules")

    # Categorize modules based on functionality
    expected_mapping = analyzer.categorize_modules(all_modules)

    # Find modules that are actually included in test files
    actual_inclusion = analyzer.find_included_modules()

    # Find modules that are missing from test files
    missing_modules = analyzer.find_missing_modules(expected_mapping, actual_inclusion)
    
    # Run additional analyses if requested
    if args.coverage or args.all:
        analyzer.analyze_coverage()
        
    if args.quality or args.all:
        analyzer.analyze_crate_tests()
        
    if args.performance or args.all:
        analyzer.analyze_performance_metrics()

    # Generate detailed report
    report = analyzer.generate_report(expected_mapping, actual_inclusion, missing_modules)

    # Save report to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📝 Report saved to {args.output}")
    print(report)
    
    # Exit with error if modules are missing
    total_missing = (len(missing_modules.missing_security) + len(missing_modules.missing_advanced_security) +
                    len(missing_modules.missing_economic) + len(missing_modules.missing_state) +
                    len(missing_modules.missing_timing) + len(missing_modules.missing_advanced_timing) +
                    len(missing_modules.missing_performance_regression) + len(missing_modules.missing_monitoring) +
                    len(missing_modules.missing_stability) + len(missing_modules.missing_recovery))

    if total_missing > 0:
        print(f"\n❌ {total_missing} modules are missing from their expected test files")
        sys.exit(1)
    else:
        print(f"\n✅ All modules are properly included in their test categories")

if __name__ == "__main__":
    main()
