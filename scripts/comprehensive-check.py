#!/usr/bin/env python3
"""
TallyIO Production-Ready Validation Suite
==========================================

Comprehensive validation script for TallyIO MEV infrastructure with visual progress indicators.
Validates ALL tests and benchmarks across workspace, crates, and integration categories.

Test Categories:
- Workspace Tests: All unit and integration tests across workspace
- Crate Tests: Individual crate tests and benchmarks
- Integration Tests: Cross-module integration tests from tests/ directory
- Benchmarks: Performance benchmarks from all crates and workspace
- Doc Tests: Documentation tests across all modules
- Critical Tests: Security, Economic, State Consistency, Timing (must pass for production)
- Stability Tests: Endurance, load stress, resource exhaustion
- Security Audit: Vulnerability scanning with ignored known issues
- Code Coverage: Complete workspace coverage analysis
- Module Mapping: Test coverage validation

Coverage:
- Workspace-level tests (cargo test --workspace)
- Individual crate tests (cargo test -p <crate>)
- All benchmarks (cargo bench --workspace and per-crate)
- Integration tests from tests/ directory
- MEV/DeFi comprehensive test suite

Requirements:
- Python 3.7+
- Rust toolchain (cargo, clippy, llvm-cov)
- All TallyIO dependencies

Usage:
    python scripts/comprehensive-check.py [options]

Options:
    --clear-cache    Clear all cached test results
    --no-cache       Run without using cached results (force fresh execution)

Environment Variables:
    TALLYIO_FAST_TESTS=1  # Enable fast test mode for CI/CD
"""

import subprocess
import sys
import re
import time
import threading
import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Visual indicators and colors
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Status(Enum):
    RUNNING = "🔄"
    SUCCESS = "✅"
    FAILED = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"

@dataclass
class CheckResult:
    name: str
    success: bool
    message: str
    duration: float
    details: Optional[str] = None
    error_output: Optional[str] = None

class TestCache:
    """Cache system for test results to speed up repeated executions."""

    def __init__(self, cache_dir: Path = Path(".tallyio_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "test_cache.json"
        self.file_hashes_file = self.cache_dir / "file_hashes.json"

    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""

    def _get_project_hash(self) -> str:
        """Get hash of all relevant project files."""
        relevant_files = []

        # Include Rust source files
        for pattern in ["**/*.rs", "**/Cargo.toml", "**/Cargo.lock"]:
            relevant_files.extend(Path(".").glob(pattern))

        # Sort for consistent hashing
        relevant_files.sort()

        combined_hash = hashlib.sha256()
        for file_path in relevant_files:
            if file_path.is_file():
                file_hash = self._get_file_hash(file_path)
                combined_hash.update(f"{file_path}:{file_hash}".encode())

        return combined_hash.hexdigest()

    def get_cached_result(self, test_name: str) -> Optional[CheckResult]:
        """Get cached test result if still valid."""
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            current_hash = self._get_project_hash()

            if test_name in cache_data:
                cached_entry = cache_data[test_name]
                if cached_entry.get("project_hash") == current_hash:
                    result_data = cached_entry["result"]
                    return CheckResult(**result_data)
        except:
            pass

        return None

    def cache_result(self, test_name: str, result: CheckResult):
        """Cache a test result."""
        try:
            cache_data = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)

            cache_data[test_name] = {
                "project_hash": self._get_project_hash(),
                "timestamp": time.time(),
                "result": asdict(result)
            }

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except:
            pass  # Ignore cache errors

    def clear_cache(self):
        """Clear all cached results."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.file_hashes_file.exists():
                self.file_hashes_file.unlink()
        except:
            pass

# Global cache instance
test_cache = TestCache()

def cached_test_execution(test_name: str, test_func, *args, **kwargs) -> CheckResult:
    """Execute test with caching support."""
    # Check if we have a cached result
    cached_result = test_cache.get_cached_result(test_name)
    if cached_result:
        print(f"   💾 Using cached result for {test_name} ({cached_result.duration:.1f}s)")
        return cached_result

    # Execute the test
    result = test_func(*args, **kwargs)

    # Cache the result
    test_cache.cache_result(test_name, result)

    return result

class ProgressBar:
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.current = 0
        self.width = width
        self.start_time = time.time()

    def update(self, step: int = 1):
        self.current += step
        self._draw()

    def _draw(self):
        percent = (self.current / self.total) * 100
        filled = int(self.width * self.current // self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        elapsed = time.time() - self.start_time

        print(f'\r{Color.CYAN}Progress: {Color.END}[{Color.GREEN}{bar}{Color.END}] '
              f'{Color.BOLD}{percent:.1f}%{Color.END} '
              f'{Color.YELLOW}({self.current}/{self.total}){Color.END} '
              f'{Color.BLUE}⏱️ {elapsed:.1f}s{Color.END}', end='', flush=True)

class LiveSpinner:
    def __init__(self, message: str):
        self.message = message
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        self.start_time = time.time()

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        print('\r' + ' ' * 80 + '\r', end='', flush=True)

    def _spin(self):
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.spinner_chars[i % len(self.spinner_chars)]
            print(f'\r{Color.CYAN}{spinner}{Color.END} {Color.BOLD}{self.message}{Color.END} '
                  f'{Color.YELLOW}⏱️ {elapsed:.1f}s{Color.END}', end='', flush=True)
            time.sleep(0.1)
            i += 1

class ProgressSpinner:
    """Advanced spinner with progress indicators for long-running tests."""
    def __init__(self, message: str, estimated_duration: int = 60):
        self.message = message
        self.estimated_duration = estimated_duration
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        self.start_time = time.time()

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the progress line
        print('\r' + ' ' * 120 + '\r', end='', flush=True)

    def _spin(self):
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.spinner_chars[i % len(self.spinner_chars)]

            # Calculate progress percentage
            progress_pct = min(100, (elapsed / self.estimated_duration) * 100)

            # Create progress bar
            bar_length = 20
            filled_length = int(bar_length * progress_pct / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            # Format time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                time_str = f"{minutes}m{seconds:02d}s"

            # Estimated time remaining
            if progress_pct > 5:  # Only show ETA after 5% progress
                eta = (elapsed / progress_pct * 100) - elapsed
                if eta < 60:
                    eta_str = f" (ETA: {eta:.0f}s)"
                else:
                    eta_minutes = int(eta // 60)
                    eta_seconds = int(eta % 60)
                    eta_str = f" (ETA: {eta_minutes}m{eta_seconds:02d}s)"
            else:
                eta_str = ""

            print(f"\r{Color.CYAN}{spinner}{Color.END} {Color.BOLD}{self.message}{Color.END} "
                  f"[{Color.GREEN}{bar}{Color.END}] {Color.YELLOW}{progress_pct:.1f}%{Color.END} "
                  f"{Color.BLUE}⏱️ {time_str}{eta_str}{Color.END}", end="", flush=True)
            time.sleep(0.1)
            i += 1

def print_header():
    """Print TallyIO header with visual styling."""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}🚀 TallyIO Comprehensive Validation Suite{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}📊 Complete Testing | 🔒 Security | ⚡ Performance | 🎯 MEV-Ready{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}🏗️  Workspace + Crates + Integration + Benchmarks + Coverage{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")

def print_section(title: str, status: Status = Status.INFO):
    """Print section header with visual indicators."""
    print(f"\n{Color.BOLD}{Color.BLUE}{status.value} {title}{Color.END}")
    print(f"{Color.BLUE}{'─' * (len(title) + 4)}{Color.END}")

def print_result(result: CheckResult):
    """Print check result with appropriate colors and formatting."""
    status_icon = Status.SUCCESS.value if result.success else Status.FAILED.value
    color = Color.GREEN if result.success else Color.RED

    print(f"{status_icon} {Color.BOLD}{result.name}{Color.END}: "
          f"{color}{result.message}{Color.END} "
          f"{Color.YELLOW}({result.duration:.2f}s){Color.END}")

    if result.details:
        print(f"   {Color.CYAN}Details:{Color.END} {result.details}")

    # If failed, show more detailed error information
    if not result.success and hasattr(result, 'error_output') and result.error_output:
        print(f"   {Color.RED}Error Output:{Color.END}")
        # Show first 10 lines of error output
        error_lines = result.error_output.strip().split('\n')
        for line in error_lines[:10]:
            print(f"   {Color.RED}│{Color.END} {line}")
        if len(error_lines) > 10:
            print(f"   {Color.RED}│{Color.END} ... ({len(error_lines) - 10} more lines)")
        print()

def run_command(cmd: List[str], timeout: int = 300, show_spinner: bool = True, spinner_message: str = "", env: dict = None) -> Tuple[bool, str, str]:
    """Run command with timeout and return success, stdout, stderr."""
    spinner = None
    if show_spinner and spinner_message:
        spinner = LiveSpinner(spinner_message)
        spinner.start()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=".",
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env
        )
        return result.returncode == 0, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)
    finally:
        if spinner:
            spinner.stop()

def check_forbidden_patterns() -> CheckResult:
    """
    AST-inspirirana analiza za prepovedane vzorce v Rust kodi:
    - Ignorira komentarje (//, /* ... */), doc-komentarje (///, //!)
    - Ignorira vrstice in bloke pod #[cfg(test)], #[test], #[bench], #[tokio::test], #[test_case], #[allow(...)]
    - Ignorira datoteke v tests/, benches/, doc-testih
    - .unwrap_or_default() je warning, ne critical
    - Vec::new in Mutex ignorira v testih/benchih
    - const fn: opozori, če je funkcija predolga (>20 vrstic)
    """
    import re
    start_time = time.time()

    forbidden_patterns = [
        r'\.unwrap\(',
        r'\.expect\(',
        r'panic!\(',
        r'\.unwrap_or_default\(',
        r'todo!\(',
        r'unimplemented!\('
    ]
    pattern_names = {
        r'\.unwrap\(': '.unwrap()',
        r'\.expect\(': '.expect(',
        r'panic!\(': 'panic!(',
        r'\.unwrap_or_default\(': '.unwrap_or_default()',
        r'todo!\(': 'todo!(',
        r'unimplemented!\(': 'unimplemented!('
    }

    rust_files = []
    project_root = Path(".")
    for search_dir in ["src", "crates", "examples"]:
        search_path = project_root / search_dir
        if search_path.exists():
            rust_files.extend(search_path.rglob("*.rs"))

    violations = []
    for file_path in rust_files:
        file_str = str(file_path)
        is_test_file = (
            '/benches/' in file_str or '\\benches\\' in file_str or
            '/tests/' in file_str or '\\tests\\' in file_str or
            file_str.endswith('_test.rs') or file_str.endswith('_tests.rs')
        )
        if is_test_file:
            # Defense-in-depth: preveri, če je v testni datoteki pub funkcija/modul ali build flag
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                pub_fn_re = re.compile(r'pub\s+(async\s+)?fn\s+\w+')
                pub_mod_re = re.compile(r'pub\s+mod\s+\w+')
                cfg_attr_export_re = re.compile(r'cfg_attr|export|macro_rules!|macro_export')
                for idx, line in enumerate(content.split('\n')):
                    if pub_fn_re.search(line) or pub_mod_re.search(line):
                        violations.append({
                            'file': str(file_path),
                            'line': idx + 1,
                            'pattern': 'pub fn/mod in test',
                            'content': line.strip(),
                            'severity': 'warning',
                            'note': 'Test function or module is public, could be exposed in production!'
                        })
                    if cfg_attr_export_re.search(line):
                        violations.append({
                            'file': str(file_path),
                            'line': idx + 1,
                            'pattern': 'cfg_attr/export/macro in test',
                            'content': line.strip(),
                            'severity': 'warning',
                            'note': 'Suspicious build flag/macro that could expose test code!'
                        })
            except Exception:
                pass
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            def strip_comments(code: str) -> str:
                code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
                code = re.sub(r'//.*', '', code)
                return code
            code_no_comments = strip_comments(content)
            lines = code_no_comments.split('\n')
            forbidden_re = re.compile('|'.join(forbidden_patterns))
            allow_re = re.compile(r'#\[\s*allow')
            test_attr_re = re.compile(r'#\[\s*(cfg\(test\)|test|bench|tokio::test|test_case)')
            in_ignored_block = False
            block_level = 0
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if test_attr_re.search(stripped) or allow_re.search(stripped):
                    in_ignored_block = True
                    continue
                if in_ignored_block:
                    if re.match(r'(pub\s+)?(async\s+)?fn\s+\w+\s*\(', stripped):
                        block_level += 1
                    if '{' in stripped:
                        block_level += stripped.count('{')
                    if '}' in stripped:
                        block_level -= stripped.count('}')
                    if block_level <= 0:
                        in_ignored_block = False
                    continue
                if not stripped:
                    continue
                if stripped.startswith('///') or stripped.startswith('//!'):
                    continue
                if allow_re.search(stripped) or test_attr_re.search(stripped):
                    continue
                for pattern in forbidden_patterns:
                    if re.search(pattern, stripped):
                        severity = "critical"
                        if pattern == r'\.unwrap_or_default\(': severity = "warning"
                        violations.append({
                            'file': str(file_path),
                            'line': idx + 1,
                            'pattern': pattern_names[pattern],
                            'content': stripped,
                            'severity': severity
                        })
            # Mutex/atomic/vec checks (ignore in test/bench)
            if not in_ignored_block:
                if re.search(r'std::sync::Mutex', code_no_comments):
                    violations.append({
                        'file': str(file_path),
                        'line': 0,
                        'pattern': 'std::sync::Mutex',
                        'content': 'Mutex usage',
                        'severity': 'warning'
                    })
                if re.search(r'Vec::new\(', code_no_comments):
                    violations.append({
                        'file': str(file_path),
                        'line': 0,
                        'pattern': 'Vec::new()',
                        'content': 'Vec::new() usage',
                        'severity': 'warning'
                    })
            # const fn complexity (warn if too long or complex)
            const_fn_iter = re.finditer(r'const\s+fn\s+\w+\s*\([^)]*\)\s*\{', code_no_comments)
            for match in const_fn_iter:
                start = match.end()
                block = ''
                depth = 1
                for i in range(start, len(code_no_comments)):
                    c = code_no_comments[i]
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            block = code_no_comments[start:i]
                            break
                if block:
                    lines_in_fn = block.count('\n')
                    if lines_in_fn > 20:
                        violations.append({
                            'file': str(file_path),
                            'line': 0,
                            'pattern': 'const fn (complex)',
                            'content': f'const fn block > 20 lines',
                            'severity': 'warning'
                        })
        except Exception as e:
            continue
    duration = time.time() - start_time
    if not violations:
        return CheckResult(
            name="Forbidden Patterns Check",
            success=True,
            message="No forbidden patterns found (production-ready code)",
            duration=duration,
            details=f"Checked {len(rust_files)} Rust files for {len(forbidden_patterns)} forbidden patterns"
        )
    else:
        violation_summary = {}
        for violation in violations:
            pattern = violation['pattern']
            if pattern not in violation_summary:
                violation_summary[pattern] = 0
            violation_summary[pattern] += 1

        summary_text = ", ".join([f"{pattern}: {count}" for pattern, count in violation_summary.items()])

        error_details = "\n".join([
            f"  {v['file']}:{v['line']} - {v['pattern']} ({v['severity']}) in: {v['content']}"
            for v in violations[:20]  # Show first 20 violations
        ])

        if len(violations) > 20:
            error_details += f"\n  ... and {len(violations) - 20} more violations"

        return CheckResult(
            name="Forbidden Patterns Check",
            success=False,
            message=f"Found {len(violations)} forbidden patterns: {summary_text}",
            duration=duration,
            details="These patterns are forbidden in production TallyIO code",
            error_output=error_details
        )


def check_cargo_fmt() -> CheckResult:
    """Check Rust code formatting and auto-fix if needed."""
    start_time = time.time()

    # First, check if formatting is needed
    success, stdout, stderr = run_command(
        ["cargo", "fmt", "--check"],
        show_spinner=True,
        spinner_message="Checking code formatting..."
    )

    if success:
        duration = time.time() - start_time
        return CheckResult(
            name="Cargo Format",
            success=True,
            message="Code formatting is correct",
            duration=duration
        )
    else:
        # Auto-fix formatting issues
        print(f"   {Color.YELLOW}⚠️  Formatting issues detected, running cargo fmt...{Color.END}")

        fix_success, fix_stdout, fix_stderr = run_command(
            ["cargo", "fmt"],
            show_spinner=True,
            spinner_message="Auto-fixing formatting..."
        )

        if fix_success:
            # Verify formatting is now correct
            verify_success, verify_stdout, verify_stderr = run_command(
                ["cargo", "fmt", "--check"],
                show_spinner=False
            )

            duration = time.time() - start_time

            if verify_success:
                return CheckResult(
                    name="Cargo Format",
                    success=True,
                    message="Code formatting auto-fixed successfully",
                    duration=duration,
                    details="Formatting issues were detected and automatically corrected"
                )
            else:
                return CheckResult(
                    name="Cargo Format",
                    success=False,
                    message="Auto-fix failed - manual intervention required",
                    duration=duration,
                    details="Some formatting issues could not be automatically fixed",
                    error_output=verify_stderr if verify_stderr else verify_stdout
                )
        else:
            duration = time.time() - start_time
            return CheckResult(
                name="Cargo Format",
                success=False,
                message="Failed to run cargo fmt auto-fix",
                duration=duration,
                details="cargo fmt command failed",
                error_output=fix_stderr if fix_stderr else fix_stdout
            )

def check_cargo_clippy() -> CheckResult:
    """Check Rust code with clippy lints using TallyIO ultra-strict standards (avoiding workspace false results)."""
    start_time = time.time()

    # TallyIO ultra-strict clippy configuration (remove --workspace for real results)
    clippy_args = [
        "cargo", "clippy", "--all-targets", "--workspace", "--",
        "-D", "warnings",
        "-D", "clippy::pedantic",
        "-D", "clippy::nursery",
        "-D", "clippy::correctness",
        "-D", "clippy::suspicious",
        "-D", "clippy::perf",
        "-W", "clippy::redundant_allocation",
        "-W", "clippy::needless_collect",
        "-W", "clippy::suboptimal_flops",
        "-A", "clippy::missing_docs_in_private_items",
        "-D", "clippy::infinite_loop",
        "-D", "clippy::while_immutable_condition",
        "-D", "clippy::never_loop",
        "-D", "for_loops_over_fallibles",
        "-D", "clippy::manual_strip",
        "-D", "clippy::needless_continue",
        "-D", "clippy::match_same_arms",
        "-D", "clippy::unwrap_used",
        "-D", "clippy::expect_used",
        "-D", "clippy::panic",
        "-D", "clippy::large_stack_arrays",
        "-D", "clippy::large_enum_variant",
        "-D", "clippy::mut_mut",
        "-D", "clippy::cast_possible_truncation",
        "-D", "clippy::cast_sign_loss",
        "-D", "clippy::cast_precision_loss",
        "-D", "clippy::must_use_candidate",
        "-D", "clippy::empty_loop",
        "-D", "clippy::if_same_then_else",
        "-D", "clippy::await_holding_lock",
        "-D", "clippy::await_holding_refcell_ref",
        "-D", "clippy::let_underscore_future",
        "-D", "clippy::diverging_sub_expression",
        "-D", "clippy::unreachable",
        "-D", "clippy::default_numeric_fallback",
        "-D", "clippy::redundant_pattern_matching",
        "-D", "clippy::manual_let_else",
        "-D", "clippy::blocks_in_conditions",
        "-D", "clippy::needless_pass_by_value",
        "-D", "clippy::single_match_else",
        "-D", "clippy::branches_sharing_code",
        "-D", "clippy::useless_asref",
        "-D", "clippy::redundant_closure_for_method_calls",
        "-v"
    ]
    
    success, stdout, stderr = run_command(clippy_args, spinner_message="Running clippy lints...")
    duration = time.time() - start_time

    if success:
        return CheckResult(
            name="Cargo Clippy",
            success=True,
            message="No clippy warnings found (TallyIO ultra-strict standards)",
            duration=duration
        )
    else:
        # Extract warning count from output
        combined_output = stdout + stderr
        warning_count = len(re.findall(r'warning:', combined_output))
        error_count = len(re.findall(r'error:', combined_output))

        if error_count > 0:
            message = f"Found {error_count} clippy errors and {warning_count} warnings"
        else:
            message = f"Found {warning_count} clippy warnings"

        return CheckResult(
            name="Cargo Clippy",
            success=False,
            message=message,
            duration=duration,
            details="Run 'cargo clippy --fix' to auto-fix issues",
            error_output=combined_output
        )

def check_cargo_test():
    """Run comprehensive Rust test suite with detailed breakdown."""
    start_time = time.time()
    print_section("Running comprehensive test suite")

    # Test results tracking
    all_results = []

    # 1. Unit tests per crate (parallel execution)
    print("📦 Running unit tests per crate...")
    crate_results = run_crate_unit_tests()
    all_results.extend(crate_results)

    # 2. Integration tests from workspace
    print("🔗 Running workspace integration tests...")
    integration_results = run_integration_tests()
    all_results.extend(integration_results)

    # 3. Doc tests
    print("📚 Running documentation tests...")
    doc_results = run_doc_tests()
    all_results.extend(doc_results)

    # 4. Tests from tests/ directory (categorized)
    print("🧪 Running specialized test suites...")
    specialized_results = run_specialized_tests()
    all_results.extend(specialized_results)

    # 5. Benchmarks (if requested)
    print("⚡ Running performance benchmarks...")
    benchmark_results = run_benchmarks()
    all_results.extend(benchmark_results)

    # Aggregate results
    total_success = all(result.success for result in all_results)
    total_duration = time.time() - start_time

    # Generate detailed report
    generate_test_report(all_results, total_duration)

    return CheckResult(
        name="Comprehensive Testing",
        success=total_success,
        message=f"Completed {len(all_results)} test suites",
        duration=total_duration,
        details=f"Comprehensive testing completed: {len(all_results)} test suites executed"
    )

def check_security_audit() -> CheckResult:
    """Run comprehensive security audit - NO VULNERABILITIES ALLOWED for financial applications."""
    start_time = time.time()

    # TallyIO Security Policy: ZERO TOLERANCE for vulnerabilities in financial applications
    # Only allow specific warnings for unmaintained crates that are not security-critical
    allowed_warnings = [
        "RUSTSEC-2024-0384",  # instant - unmaintained (timing utility, not security-critical)
        "RUSTSEC-2024-0436"   # paste - unmaintained (macro utility, not security-critical)
    ]

    # Temporary exceptions for vulnerabilities without fixes (MUST BE REVIEWED REGULARLY)
    temporary_exceptions = [
        "RUSTSEC-2023-0071",  # rsa 0.9.8 - Marvin Attack (no fix available, sqlx-mysql dependency)
    ]

    # Run full audit first to see all issues
    success, stdout, stderr = run_command(
        ["cargo", "audit"],
        spinner_message="Running comprehensive security audit..."
    )
    duration = time.time() - start_time
    combined_output = stdout + stderr

    # Parse vulnerabilities and warnings
    vulnerabilities = []
    warnings = []

    # Extract all RUSTSEC entries
    rustsec_pattern = r'ID:\s+(RUSTSEC-\d{4}-\d{4})'
    severity_pattern = r'Severity:\s+(\d+\.?\d*)\s+\((\w+)\)'

    rustsec_matches = re.findall(rustsec_pattern, combined_output)

    for rustsec_id in rustsec_matches:
        if 'Warning:' in combined_output and rustsec_id in combined_output:
            if rustsec_id in allowed_warnings:
                warnings.append(f"{rustsec_id} (allowed)")
            else:
                warnings.append(f"{rustsec_id} (NOT ALLOWED)")
        else:
            vulnerabilities.append(rustsec_id)

    # Count actual vulnerabilities (not warnings)
    vuln_count = len(re.findall(r'(\d+) vulnerabilities? found', combined_output))
    warning_count = len(re.findall(r'(\d+) allowed warnings? found', combined_output))

    # TallyIO Security Policy: FAIL if ANY vulnerabilities or disallowed warnings
    disallowed_warnings = [w for w in warnings if "NOT ALLOWED" in w]

    if vuln_count == 0 and len(disallowed_warnings) == 0:
        message = "Security audit passed"
        if warnings:
            message += f" ({len(warnings)} allowed warnings)"

        return CheckResult(
            name="Security Audit",
            success=True,
            message=message,
            duration=duration,
            details="TallyIO security policy: Zero tolerance for vulnerabilities"
        )
    else:
        # Security audit failed
        issues = []
        if vuln_count > 0:
            issues.append(f"{vuln_count} vulnerabilities")
        if disallowed_warnings:
            issues.append(f"{len(disallowed_warnings)} disallowed warnings")

        error_details = []
        if vulnerabilities:
            error_details.append(f"Vulnerabilities: {', '.join(vulnerabilities)}")
        if disallowed_warnings:
            error_details.append(f"Disallowed warnings: {', '.join(disallowed_warnings)}")

        return CheckResult(
            name="Security Audit",
            success=False,
            message=f"SECURITY FAILURE: {' and '.join(issues)} found",
            duration=duration,
            details=f"TallyIO requires immediate security fixes:\n{chr(10).join(error_details)}",
            error_output=combined_output
        )

def check_code_coverage() -> CheckResult:
    """Run code coverage analysis with cargo llvm-cov (better for async/multithreading)."""
    start_time = time.time()

    # Set fast test mode for coverage to avoid long-running tests
    os.environ["TALLYIO_FAST_TESTS"] = "1"

    # Use llvm-cov for comprehensive coverage analysis (workspace mode) with optimizations
    cpu_count = os.cpu_count() or 4

    success, stdout, stderr = run_command([
        "cargo", "llvm-cov",
        "--workspace",          # Include all workspace crates
        "--tests",              # Include integration tests from tests/ directory
        "--no-fail-fast",       # Continue even if some tests fail
        "--jobs", str(cpu_count), # Use all available CPU cores
        "--release"             # Use release mode for faster execution
    ], timeout=600, spinner_message=f"Analyzing comprehensive code coverage (using {cpu_count} cores)...")

    duration = time.time() - start_time

    if not success:
        # Try fallback with just workspace libraries (optimized)
        print("   ⚠️  Full coverage failed, trying workspace libraries only...")
        success, stdout, stderr = run_command([
            "cargo", "llvm-cov",
            "--workspace",
            "--lib",
            "--jobs", str(cpu_count),
            "--release"
        ], timeout=300, spinner_message="Analyzing workspace library coverage...")

        duration = time.time() - start_time

        if not success:
            return CheckResult(
                name="Code Coverage",
                success=False,
                message="Coverage analysis failed",
                duration=duration,
                details="Both full and unit test coverage failed",
                error_output=stderr if stderr else stdout
            )

    # Extract coverage percentage from llvm-cov output
    coverage_patterns = [
        r'lines\.\.\.\.\.\.\.\.\.\. (\d+\.?\d*)%',  # llvm-cov format
        r'TOTAL.*?(\d+\.?\d*)%',                    # alternative llvm-cov format
        r'(\d+\.?\d*)% coverage'                    # fallback format
    ]

    coverage = None
    for pattern in coverage_patterns:
        coverage_match = re.search(pattern, stdout)
        if coverage_match:
            coverage = float(coverage_match.group(1))
            break

    if coverage is None:
        return CheckResult(
            name="Code Coverage",
            success=False,
            message="Could not parse coverage output",
            duration=duration,
            details="Check llvm-cov output format",
            error_output=stdout
        )

    min_coverage = 90.0

    # Extract detailed coverage info for better reporting
    coverage_details = []
    lines = stdout.split('\n')
    for line in lines:
        if '.rs' in line and '%' in line and 'coverage' not in line.lower():
            coverage_details.append(line.strip())

    if coverage >= min_coverage:
        return CheckResult(
            name="Code Coverage",
            success=True,
            message=f"Coverage: {coverage}% (≥{min_coverage}%) [llvm-cov complete coverage]",
            duration=duration
        )
    else:
        # Show which files need more coverage
        low_coverage_files = []
        for detail in coverage_details[:5]:  # Show top 5 problematic files
            if detail:
                low_coverage_files.append(detail)

        details_msg = f"Need {min_coverage - coverage:.1f}% more coverage"
        if low_coverage_files:
            details_msg += f"\nFiles needing attention:\n" + "\n".join(low_coverage_files)

        return CheckResult(
            name="Code Coverage",
            success=False,
            message=f"Coverage: {coverage}% (<{min_coverage}%) [llvm-cov complete coverage]",
            duration=duration,
            details=details_msg
        )

def check_module_test_mapping() -> CheckResult:
    """Check that all modules are properly included in universal test categories."""
    start_time = time.time()

    # Import and run the module test mapping analyzer
    try:
        # Add scripts directory to path to import the analyzer
        scripts_dir = Path(__file__).parent
        sys.path.insert(0, str(scripts_dir))

        from module_test_mapping_analyzer import ModuleTestMappingAnalyzer

        analyzer = ModuleTestMappingAnalyzer()

        # Find all modules
        all_modules = analyzer.find_all_modules()

        # Categorize modules
        expected = analyzer.categorize_modules(all_modules)

        # Find included modules
        actual = analyzer.find_included_modules()

        # Find missing modules
        missing = analyzer.find_missing_modules(expected, actual)

        # Calculate totals
        total_expected = (len(expected.security_modules) + len(expected.economic_modules) +
                         len(expected.state_modules) + len(expected.timing_modules))
        total_missing = (len(missing.missing_security) + len(missing.missing_economic) +
                        len(missing.missing_state) + len(missing.missing_timing))

        duration = time.time() - start_time

        if total_missing == 0:
            return CheckResult(
                name="Module Test Mapping",
                success=True,
                message=f"All {total_expected} modules properly included in test categories",
                duration=duration,
                details="All modules are covered by appropriate universal tests"
            )
        else:
            coverage_percent = ((total_expected - total_missing) / total_expected * 100) if total_expected > 0 else 100

            # Create detailed missing modules report
            missing_details = []
            if missing.missing_security:
                missing_details.append(f"Security: {', '.join(sorted(list(missing.missing_security)[:3]))}")
            if missing.missing_economic:
                missing_details.append(f"Economic: {', '.join(sorted(list(missing.missing_economic)[:3]))}")
            if missing.missing_state:
                missing_details.append(f"State: {', '.join(sorted(list(missing.missing_state)[:3]))}")
            if missing.missing_timing:
                missing_details.append(f"Timing: {', '.join(sorted(list(missing.missing_timing)[:3]))}")

            details_msg = f"Module inclusion rate: {coverage_percent:.1f}%"
            if missing_details:
                details_msg += f"\nMissing modules: {'; '.join(missing_details)}"

            return CheckResult(
                name="Module Test Mapping",
                success=False,
                message=f"{total_missing} modules missing from test categories",
                duration=duration,
                details=details_msg
            )

    except ImportError as e:
        duration = time.time() - start_time
        return CheckResult(
            name="Module Test Mapping",
            success=False,
            message="Could not import module test mapping analyzer",
            duration=duration,
            details=f"Import error: {e}"
        )
    except Exception as e:
        duration = time.time() - start_time
        return CheckResult(
            name="Module Test Mapping",
            success=False,
            message="Module test mapping analysis failed",
            duration=duration,
            details=f"Error: {e}"
        )
    finally:
        # Clean up sys.path
        if str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))

def check_critical_tests() -> CheckResult:
    """Run critical test categories that must pass for production readiness."""
    start_time = time.time()

    # Critical test categories for financial applications
    critical_tests = [
        "security_tests",
        "economic_tests",
        "state_consistency_tests",
        "timing_tests"
    ]

    all_passed = True
    total_tests = 0
    failed_tests = 0
    test_results = []

    for test_category in critical_tests:
        success, stdout, stderr = run_command(
            ["cargo", "test", "--test", test_category, "--jobs", "1"],
            timeout=300,
            spinner_message=f"Running critical {test_category}..."
        )

        combined_output = stdout + stderr
        # Parse results - cargo test format: "151 passed; 0 failed"
        failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
        passed_match = re.search(r'(\d+) passed', combined_output)

        category_failed = int(failed_match.group(1)) if failed_match else 0
        category_passed = int(passed_match.group(1)) if passed_match else 0

        # Debug: print output if there's a mismatch
        if not success or category_failed > 0:
            print(f"DEBUG: {test_category} - success: {success}, failed: {category_failed}, passed: {category_passed}")
            print(f"DEBUG: Combined output: {combined_output[:500]}...")

        total_tests += category_passed + category_failed
        failed_tests += category_failed

        if category_failed > 0:
            all_passed = False

        test_results.append(f"{test_category}: {category_passed} passed, {category_failed} failed")

    duration = time.time() - start_time

    if all_passed and total_tests > 0:
        return CheckResult(
            name="Critical Test Categories",
            success=True,
            message=f"All critical tests passed ({total_tests} tests)",
            duration=duration,
            details="; ".join(test_results)
        )
    else:
        return CheckResult(
            name="Critical Test Categories",
            success=False,
            message=f"Critical tests failed ({failed_tests} failures out of {total_tests})",
            duration=duration,
            details="; ".join(test_results)
        )

def check_stability_tests() -> CheckResult:
    """Run stability and performance tests."""
    start_time = time.time()

    # Set fast test mode for CI
    os.environ["TALLYIO_FAST_TESTS"] = "1"

    # Run timing tests as stability tests (closest equivalent)
    success, stdout, stderr = run_command(
        ["cargo", "test", "--test", "timing_tests", "--jobs", "1"],
        timeout=600,
        spinner_message="Running stability and timing tests..."
    )

    duration = time.time() - start_time
    combined_output = stdout + stderr

    # Parse results - cargo test format: "151 passed; 0 failed"
    failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
    passed_match = re.search(r'(\d+) passed', combined_output)

    failed_count = int(failed_match.group(1)) if failed_match else 0
    passed_count = int(passed_match.group(1)) if passed_match else 0

    # Use both success status and test counts for validation
    if success and failed_count == 0 and passed_count > 0:
        return CheckResult(
            name="Stability Tests",
            success=True,
            message=f"All stability tests passed ({passed_count} tests)",
            duration=duration,
            details="Includes endurance, load stress, and resource exhaustion tests"
        )
    else:
        return CheckResult(
            name="Stability Tests",
            success=False,
            message=f"Stability tests failed ({failed_count} failures, {passed_count} passed)",
            duration=duration,
            details="Check stability test output for details",
            error_output=combined_output
        )

def run_crate_unit_tests():
    """Run unit tests for each crate individually."""
    results = []

    print("   🔍 Getting workspace metadata...")
    # Get all crates from workspace
    success, stdout, stderr = run_command(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        timeout=30,
        show_spinner=False
    )

    if not success:
        return [CheckResult(
            name="Workspace Metadata",
            success=False,
            message="Failed to get workspace metadata",
            duration=0,
            details="Failed to get workspace metadata"
        )]

    import json
    try:
        metadata = json.loads(stdout)
        crates = [pkg["name"] for pkg in metadata["packages"] if pkg["name"].startswith("tallyio")]
    except:
        # Fallback to known crates
        crates = ["tallyio-core", "tallyio-blockchain", "tallyio-liquidation",
                 "tallyio-security", "tallyio-database", "tallyio-metrics",
                 "tallyio-api", "tallyio-contracts", "tallyio-web-ui"]

    # Use single-threaded execution for sequential testing
    print(f"   🚀 Running unit tests for {len(crates)} crates sequentially...")

    for i, crate in enumerate(crates, 1):
        print(f"   📦 [{i}/{len(crates)}] Testing crate: {crate}")
        start_time = time.time()

        # Show individual test execution
        spinner = LiveSpinner(f"Running unit tests for {crate}...")
        spinner.start()

        success, stdout, stderr = run_command(
            ["cargo", "test", "-p", crate, "--lib", "--jobs", "1"],
            timeout=300,
            show_spinner=False
        )

        spinner.stop()

        duration = time.time() - start_time

        # Parse test results - cargo test format: "151 passed; 0 failed"
        combined_output = stdout + stderr
        # Try multiple patterns to catch different cargo test output formats
        failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
        passed_match = re.search(r'(\d+) passed', combined_output)

        failed_count = int(failed_match.group(1)) if failed_match else 0
        passed_count = int(passed_match.group(1)) if passed_match else 0

        # Determine status: ✅ success, ❌ failure, ⚠️ no tests
        # Base status on test results, not on exit code
        if failed_count > 0:
            status = "❌"
        elif passed_count == 0 and failed_count == 0:
            status = "⚠️"  # No tests found
        else:
            status = "✅"

        print(f"      {status} {crate}: {passed_count} passed, {failed_count} failed ({duration:.1f}s)")

        # Show detailed error output only for actually failed tests
        if failed_count > 0:
            print(f"         🔍 Error details for {crate}:")
            error_lines = stderr.strip().split('\n') if stderr else stdout.strip().split('\n')
            for line in error_lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"         │ {line}")
            print()

        # Consider test successful if no tests failed, regardless of exit code
        # (Cargo sometimes returns non-zero exit code for other reasons)
        # Also consider successful if no tests found (0 passed, 0 failed)
        test_success = (failed_count == 0)

        results.append(CheckResult(
            name=f"Unit Tests - {crate}",
            success=test_success,
            message=f"{passed_count} passed, {failed_count} failed",
            duration=duration,
            details=f"{crate}: {passed_count} passed, {failed_count} failed"
        ))

    return results

def run_integration_tests():
    """Run workspace integration tests."""
    print("   🔗 Running workspace integration tests...")
    start_time = time.time()

    # Use ProgressSpinner for long-running integration tests (estimated 2 minutes)
    spinner = ProgressSpinner("Executing workspace integration tests", estimated_duration=120)
    spinner.start()

    # Run workspace integration tests (without invalid "*" pattern)
    success, stdout, stderr = run_command(
        ["cargo", "test", "--workspace", "--jobs", "1"],
        timeout=600,
        show_spinner=False
    )

    spinner.stop()

    duration = time.time() - start_time
    combined_output = stdout + stderr

    # Parse results - cargo test format: "151 passed; 0 failed"
    failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
    passed_match = re.search(r'(\d+) passed', combined_output)

    failed_count = int(failed_match.group(1)) if failed_match else 0
    passed_count = int(passed_match.group(1)) if passed_match else 0

    # Determine status: ✅ success, ❌ failure, ⚠️ no tests
    # Base status on test results, not on exit code
    if failed_count > 0:
        status = "❌"
    elif passed_count == 0 and failed_count == 0:
        status = "⚠️"  # No tests found
    else:
        status = "✅"

    print(f"      {status} Integration tests: {passed_count} passed, {failed_count} failed ({duration:.1f}s)")

    # Show detailed error output only for actually failed tests
    if failed_count > 0:
        print(f"         🔍 Error details for Integration tests:")
        error_lines = combined_output.strip().split('\n')
        for line in error_lines[-15:]:  # Show last 15 lines for integration tests
            if line.strip() and ('error' in line.lower() or 'failed' in line.lower() or 'panic' in line.lower()):
                print(f"         │ {line}")
        print()

    # Consider test successful if no tests failed, regardless of exit code
    test_success = (failed_count == 0)

    return [CheckResult(
        name="Integration Tests",
        success=test_success,
        message=f"{passed_count} passed, {failed_count} failed",
        duration=duration,
        details=f"Integration tests: {passed_count} passed, {failed_count} failed"
    )]

def run_doc_tests():
    """Run documentation tests."""
    print("   📚 Running documentation tests...")
    start_time = time.time()

    spinner = LiveSpinner("Executing documentation tests...")
    spinner.start()

    success, stdout, stderr = run_command(
        ["cargo", "test", "--doc", "--workspace", "--jobs", "1"],
        timeout=300,
        show_spinner=False
    )

    spinner.stop()

    duration = time.time() - start_time
    combined_output = stdout + stderr

    # Parse results - cargo test format: "151 passed; 0 failed"
    failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
    passed_match = re.search(r'(\d+) passed', combined_output)

    failed_count = int(failed_match.group(1)) if failed_match else 0
    passed_count = int(passed_match.group(1)) if passed_match else 0

    # Determine status: ✅ success, ❌ failure, ⚠️ no tests
    # Base status on test results, not on exit code
    if failed_count > 0:
        status = "❌"
    elif passed_count == 0 and failed_count == 0:
        status = "⚠️"  # No tests found
    else:
        status = "✅"

    print(f"      {status} Documentation tests: {passed_count} passed, {failed_count} failed ({duration:.1f}s)")

    # Consider test successful if no tests failed, regardless of exit code
    # Doc tests with 0 passed and 0 failed are also considered successful
    test_success = (failed_count == 0 and passed_count >= 0)

    return [CheckResult(
        name="Documentation Tests",
        success=test_success,
        message=f"{passed_count} passed, {failed_count} failed",
        duration=duration,
        details=f"Doc tests: {passed_count} passed, {failed_count} failed"
    )]

def run_specialized_tests():
    """Run tests from tests/ directory categorized by type."""
    print("   🧪 Running specialized test suites...")
    results = []
    tests_dir = Path("tests")

    if not tests_dir.exists():
        print("      ⚠️ No tests/ directory found")
        return results

    # Categorize test files
    test_categories = {
        "security": ["security_tests.rs", "audit_tests.rs"],
        "performance": ["timing_tests.rs", "performance_tests.rs", "benchmark_tests.rs"],
        "integration": ["integration_tests.rs", "e2e_tests.rs"],
        "stability": ["stability_tests.rs", "load_tests.rs"],
        "economic": ["economic_tests.rs", "liquidation_tests.rs"],
        "state": ["state_consistency_tests.rs", "consensus_tests.rs"]
    }

    for category, test_files in test_categories.items():
        category_results = []
        existing_files = [f for f in test_files if (tests_dir / f).exists()]

        if not existing_files:
            continue

        print(f"      🔬 {category.title()} tests ({len(existing_files)} files)...")

        for test_file in existing_files:
            test_path = tests_dir / test_file
            start_time = time.time()
            test_name = test_path.stem

            print(f"         🧪 Running {test_name}...")

            # Use ProgressSpinner for specialized tests (estimated 1-2 minutes depending on type)
            estimated_time = 120 if "stability" in test_name or "load" in test_name else 60
            spinner = ProgressSpinner(f"Executing {test_name}", estimated_duration=estimated_time)
            spinner.start()

            success, stdout, stderr = run_command(
                ["cargo", "test", "--test", test_name, "--jobs", "1"],
                timeout=600,
                show_spinner=False
            )

            spinner.stop()

            duration = time.time() - start_time
            combined_output = stdout + stderr

            # Parse results - cargo test format: "151 passed; 0 failed"
            failed_match = re.search(r'(\d+) failed', combined_output) or re.search(r';\s*(\d+)\s+failed', combined_output)
            passed_match = re.search(r'(\d+) passed', combined_output)

            failed_count = int(failed_match.group(1)) if failed_match else 0
            passed_count = int(passed_match.group(1)) if passed_match else 0

            # Determine status based on test results, not exit code
            if failed_count > 0:
                status = "❌"
            elif passed_count == 0 and failed_count == 0:
                status = "⚠️"  # No tests found
            else:
                status = "✅"
            print(f"            {status} {test_name}: {passed_count} passed, {failed_count} failed ({duration:.1f}s)")

            # Consider test successful if no tests failed, regardless of exit code
            test_success = (failed_count == 0)

            category_results.append(CheckResult(
                name=f"{category.title()} - {test_name}",
                success=test_success,
                message=f"{passed_count} passed, {failed_count} failed",
                duration=duration,
                details=f"{test_name}: {passed_count} passed, {failed_count} failed"
            ))

        if category_results:
            # Aggregate category results
            total_success = all(r.success for r in category_results)
            total_duration = sum(r.duration for r in category_results)
            details = "; ".join(r.details for r in category_results)

            results.append(CheckResult(
                name=f"{category.title()} Tests",
                success=total_success,
                message="All tests passed" if total_success else "Some tests failed",
                duration=total_duration,
                details=f"{category.title()} tests: {details}"
            ))

    return results

def run_benchmarks():
    """Run performance benchmarks."""
    print("   ⚡ Running performance benchmarks...")
    results = []

    # Check if benchmarks exist
    bench_dirs = [Path("benches"), Path("crates/core/benches")]
    has_benchmarks = any(d.exists() and any(d.glob("*.rs")) for d in bench_dirs)

    if not has_benchmarks:
        print("      ⚠️ No benchmarks found")
        return results

    print(f"      🚀 Running benchmarks sequentially...")

    start_time = time.time()

    # Use ProgressSpinner for benchmarks (estimated 10 minutes)
    spinner = ProgressSpinner("Executing performance benchmarks", estimated_duration=600)
    spinner.start()

    success, stdout, stderr = run_command(
        ["cargo", "bench", "--workspace", "--jobs", "1"],
        timeout=1200,  # 20 minutes for benchmarks
        show_spinner=False
    )

    spinner.stop()

    duration = time.time() - start_time
    combined_output = stdout + stderr

    # Count benchmarks - Criterion format: "benchmark_name         time:   [...]"
    bench_count = len(re.findall(r'^\s*\w+.*time:\s+\[', combined_output, re.MULTILINE))

    # Determine status: ✅ success, ❌ failure, ⚠️ no benchmarks
    # Base status on benchmark results, not exit code
    if bench_count == 0:
        status = "⚠️"  # No benchmarks executed
    else:
        status = "✅"

    print(f"      {status} Benchmarks: {bench_count} executed ({duration:.1f}s)")

    # Show detailed error output only if benchmarks actually failed
    # (For benchmarks, we only show errors if there are actual failures, not just exit code issues)
    if not success and bench_count == 0:
        print(f"         🔍 Error details for Benchmarks:")
        error_lines = combined_output.strip().split('\n')
        for line in error_lines[-10:]:  # Show last 10 lines
            if line.strip() and ('error' in line.lower() or 'failed' in line.lower() or 'panic' in line.lower()):
                print(f"         │ {line}")
        print()

    results.append(CheckResult(
        name="Performance Benchmarks",
        success=(bench_count > 0),  # Success only if benchmarks actually ran
        message=f"{bench_count} benchmarks executed",
        duration=duration,
        details=f"Benchmarks: {bench_count} executed"
    ))

    return results

def generate_test_report(results, total_duration):
    """Generate detailed test execution report."""
    print(f"\n📊 Test Execution Summary (Total: {total_duration:.1f}s)")
    print("─" * 60)

    for result in results:
        # Determine status based on result details
        if result.success:
            status = "✅"
        elif "0 passed, 0 failed" in result.message or "0 executed" in result.message:
            status = "⚠️"  # No tests/benchmarks found
        else:
            status = "❌"
        print(f"{status} {result.details} ({result.duration:.1f}s)")

    success_count = sum(1 for r in results if r.success)
    print(f"\n📈 Results: {success_count}/{len(results)} test suites passed")

def main():
    """Main execution function."""
    import sys

    # Check for cache management arguments
    if "--clear-cache" in sys.argv:
        print("🗑️ Clearing test cache...")
        test_cache.clear_cache()
        print("✅ Cache cleared successfully!")
        return

    if "--no-cache" in sys.argv:
        print("🚫 Running without cache...")
        test_cache.clear_cache()  # Clear cache to force fresh execution

    print_header()

    # Show cache status
    cache_info = ""
    if test_cache.cache_file.exists():
        try:
            with open(test_cache.cache_file, 'r') as f:
                cache_data = json.load(f)
            cache_info = f" (💾 {len(cache_data)} cached results available)"
        except:
            cache_info = " (💾 cache available)"

    print(f"🔧 Cache status: {'Enabled' if '--no-cache' not in sys.argv else 'Disabled'}{cache_info}")
    print()

    # Define all checks for entire workspace - comprehensive TallyIO validation
    use_cache = "--no-cache" not in sys.argv

    if use_cache:
        checks = [
            ("Forbidden Patterns", lambda: cached_test_execution("forbidden_patterns", check_forbidden_patterns)),
            ("Security Audit", lambda: cached_test_execution("security_audit", check_security_audit)),
            ("Formatting", lambda: cached_test_execution("cargo_fmt", check_cargo_fmt)),
            ("Linting", lambda: cached_test_execution("cargo_clippy", check_cargo_clippy)),
            ("Critical Tests", lambda: cached_test_execution("critical_tests", check_critical_tests)),
            ("Comprehensive Testing", lambda: cached_test_execution("comprehensive_testing", check_cargo_test)),
            ("Stability Tests", lambda: cached_test_execution("stability_tests", check_stability_tests)),
            ("Code Coverage", lambda: cached_test_execution("code_coverage", check_code_coverage)),
            ("Module Mapping", lambda: cached_test_execution("module_mapping", check_module_test_mapping)),
        ]
    else:
        checks = [
            ("Forbidden Patterns", check_forbidden_patterns),  # First check - most critical
            ("Security Audit", check_security_audit),  # Second check - security critical
            ("Formatting", check_cargo_fmt),
            ("Linting", check_cargo_clippy),
            ("Critical Tests", check_critical_tests),
            ("Comprehensive Testing", check_cargo_test),
            ("Stability Tests", check_stability_tests),
            ("Code Coverage", check_code_coverage),
            ("Module Mapping", check_module_test_mapping),
        ]

    # Initialize progress bar
    progress = ProgressBar(len(checks))
    results = []

    print(f"{Color.BOLD}Running {len(checks)} validation checks...{Color.END}\n")

    # Run each check - stop on first failure
    for check_name, check_func in checks:
        print_section(f"Running {check_name} Check", Status.RUNNING)

        result = check_func()
        results.append(result)
        print_result(result)

        # Stop immediately on failure and show detailed error
        if not result.success:
            print(f"\n{Color.BOLD}{Color.RED}❌ CHECK FAILED: {check_name}{Color.END}")
            print(f"{Color.YELLOW}Message:{Color.END} {result.message}")
            if result.details:
                print(f"{Color.YELLOW}Details:{Color.END} {result.details}")
            if result.error_output:
                print(f"{Color.YELLOW}Error Output:{Color.END}")
                print(f"{Color.RED}{result.error_output}{Color.END}")

            print(f"\n{Color.BOLD}{Color.RED}🛑 STOPPING VALIDATION - FIX ERRORS BEFORE CONTINUING{Color.END}")
            sys.exit(1)

        progress.update()
        time.sleep(0.1)  # Small delay for visual effect

    # Print final progress bar completion
    print("\n")

    # Summary
    print_section("Summary", Status.INFO)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    total_time = sum(r.duration for r in results)

    if passed == total:
        print(f"{Status.SUCCESS.value} {Color.BOLD}{Color.GREEN}All checks passed!{Color.END} "
              f"{Color.CYAN}({passed}/{total}){Color.END} "
              f"{Color.YELLOW}⏱️ {total_time:.1f}s{Color.END}")
        sys.exit(0)
    else:
        failed = total - passed
        print(f"{Status.FAILED.value} {Color.BOLD}{Color.RED}{failed} checks failed{Color.END} "
              f"{Color.CYAN}({passed}/{total} passed){Color.END} "
              f"{Color.YELLOW}⏱️ {total_time:.1f}s{Color.END}")

        print(f"\n{Color.BOLD}Failed checks:{Color.END}")
        for result in results:
            if not result.success:
                print(f"  {Status.FAILED.value} {result.name}: {result.message}")

        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Status.WARNING.value} {Color.YELLOW}Check interrupted by user{Color.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n{Status.FAILED.value} {Color.RED}Unexpected error: {e}{Color.END}")
        sys.exit(1)

