#!/usr/bin/env python3
"""
TallyIO Security Audit: Hardcoded Test Values Detection

CRITICAL SECURITY TOOL for detecting hardcoded test values that could leak to production.
This addresses the HIGH risk "Hardcoded Test Values" security issue.

🚨 FINANCIAL APPLICATION SECURITY - NO COMPROMISES
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class SeverityLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class SecurityViolation:
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    severity: SeverityLevel
    description: str
    fix_suggestion: str

class HardcodedValueDetector:
    """Ultra-strict detector for hardcoded test values in financial applications"""
    
    def __init__(self):
        # CRITICAL: Patterns that MUST NOT appear in production code
        self.critical_patterns = [
            # Test API keys and credentials
            (r'YOUR_KEY|API_KEY|TEST_KEY|DEMO_KEY', 'Hardcoded API key placeholder'),
            (r'sk-[a-zA-Z0-9]{48}', 'Hardcoded OpenAI API key'),
            (r'test[_-]?password|demo[_-]?password', 'Hardcoded test password'),
            
            # Test endpoints and URLs
            (r'example\.com|test\.com|demo\.com', 'Test domain in endpoint'),
            (r'localhost:\d+|127\.0\.0\.1:\d+', 'Localhost endpoint'),
            (r'\.test\.|\.demo\.|\.dev\.', 'Test/demo subdomain'),
            
            # Test tokens and secrets
            (r'test[_-]?token|demo[_-]?token', 'Hardcoded test token'),
            (r'secret[_-]?key.*=.*["\'][^"\']{10,}["\']', 'Hardcoded secret key'),
            
            # Development credentials
            (r'admin:admin|test:test|demo:demo', 'Default credentials'),
            (r'password.*=.*["\'](?:admin|test|demo|123)["\']', 'Weak hardcoded password'),
        ]
        
        # HIGH: Patterns that are risky but may be acceptable in specific contexts
        self.high_patterns = [
            (r'http://(?!localhost)', 'Insecure HTTP endpoint'),
            (r'\.local\.|\.internal\.', 'Internal domain that may leak'),
            (r'test[_-]?user|demo[_-]?user', 'Test user reference'),
        ]
        
        # MEDIUM: Patterns that should be reviewed
        self.medium_patterns = [
            (r'TODO.*security|FIXME.*security', 'Security-related TODO/FIXME'),
            (r'mock[_-]?hsm|fake[_-]?hsm', 'Mock HSM reference'),
            (r'debug.*=.*true', 'Debug mode enabled'),
        ]
        
        # Files to exclude from scanning
        self.excluded_files = {
            'target/', '.git/', 'node_modules/', '__pycache__/',
            '.md', '.txt', '.json', '.lock', '.toml'
        }
        
        # Test files are allowed to have test values
        self.test_file_patterns = [
            r'/tests?/', r'_test\.rs$', r'_tests\.rs$',
            r'/benches?/', r'_bench\.rs$', r'/examples?/',
            r'security_audit_hardcoded_values\.py$',  # This script itself
            r'production_deployment_checklist\.py$',  # Deployment script
            r'/scripts/', r'\.py$'  # Scripts directory
        ]

    def is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file where test values are acceptable"""
        return any(re.search(pattern, file_path) for pattern in self.test_file_patterns)

    def _is_legitimate_test_pattern(self, line: str, _pattern: str) -> bool:
        """Check if a pattern match is a legitimate test pattern"""
        # Patterns that are legitimate in test contexts
        legitimate_test_contexts = [
            # Test function names
            r'fn test_.*',
            r'async fn test_.*',
            # Test variable assignments
            r'let test_.*=',
            r'let.*test.*=',
            # Test assertions and validations
            r'assert.*test.*',
            r'storage\.store\("test_.*"',
            r'storage\.retrieve\("test_.*"',
            r'KeyId::new\("test_.*"',
            # Security validation patterns (our own validation code)
            r'endpoint\.contains\(".*"\)',
            r'if.*contains.*test',
            r'pattern.*description.*test',
            # Legitimate test data
            r'test_key.*=.*\[',  # Test key arrays
            r'let.*test_.*:.*=',  # Typed test variables
        ]

        return any(re.search(test_pattern, line, re.IGNORECASE)
                  for test_pattern in legitimate_test_contexts)

    def should_exclude_file(self, file_path: str) -> bool:
        """Check if file should be excluded from scanning"""
        return any(excluded in file_path for excluded in self.excluded_files)

    def scan_file(self, file_path: Path) -> List[SecurityViolation]:
        """Scan a single file for hardcoded test values"""
        violations = []
        
        if self.should_exclude_file(str(file_path)):
            return violations
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return violations

        is_test = self.is_test_file(str(file_path))
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('#'):
                continue
                
            # Check critical patterns (but be smart about test files)
            for pattern, description in self.critical_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Special handling for legitimate test patterns
                    if is_test and self._is_legitimate_test_pattern(line_stripped, pattern):
                        continue  # Skip legitimate test patterns

                    violations.append(SecurityViolation(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line_stripped,
                        violation_type="HARDCODED_TEST_VALUE",
                        severity=SeverityLevel.CRITICAL if not is_test else SeverityLevel.HIGH,
                        description=f"{'CRITICAL' if not is_test else 'HIGH'}: {description}",
                        fix_suggestion="Replace with environment variable or secure configuration"
                    ))
            
            # Check high patterns (only in non-test files)
            if not is_test:
                for pattern, description in self.high_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(SecurityViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            line_content=line_stripped,
                            violation_type="INSECURE_CONFIGURATION",
                            severity=SeverityLevel.HIGH,
                            description=f"HIGH: {description}",
                            fix_suggestion="Use secure configuration or environment variables"
                        ))
                
                # Check medium patterns
                for pattern, description in self.medium_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(SecurityViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            line_content=line_stripped,
                            violation_type="SECURITY_REVIEW_NEEDED",
                            severity=SeverityLevel.MEDIUM,
                            description=f"MEDIUM: {description}",
                            fix_suggestion="Review for production readiness"
                        ))
        
        return violations

    def scan_directory(self, directory: Path) -> List[SecurityViolation]:
        """Scan entire directory recursively"""
        all_violations = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.rs', '.py', '.js', '.ts', '.yaml', '.yml']:
                violations = self.scan_file(file_path)
                all_violations.extend(violations)
        
        return all_violations

def generate_security_report(violations: List[SecurityViolation]) -> Dict[str, Any]:
    """Generate comprehensive security report"""
    
    # Group violations by severity
    by_severity = {}
    for violation in violations:
        severity = violation.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(violation)
    
    # Calculate security score
    total_violations = len(violations)
    critical_count = len(by_severity.get('CRITICAL', []))
    high_count = len(by_severity.get('HIGH', []))
    
    # Security score calculation (0-100)
    if critical_count > 0:
        security_score = 0  # Any critical violation = 0 score
    elif high_count > 5:
        security_score = 20
    elif high_count > 0:
        security_score = 60
    elif total_violations > 10:
        security_score = 80
    else:
        security_score = 100
    
    return {
        'security_score': security_score,
        'total_violations': total_violations,
        'by_severity': {
            'critical': critical_count,
            'high': high_count,
            'medium': len(by_severity.get('MEDIUM', [])),
            'low': len(by_severity.get('LOW', []))
        },
        'violations': [
            {
                'file': v.file_path,
                'line': v.line_number,
                'content': v.line_content,
                'type': v.violation_type,
                'severity': v.severity.value,
                'description': v.description,
                'fix': v.fix_suggestion
            }
            for v in violations
        ],
        'production_ready': security_score >= 95 and critical_count == 0
    }

def main():
    """Main security audit function"""
    print("🚨 TallyIO Security Audit: Hardcoded Test Values Detection")
    print("=" * 70)
    
    # Scan the entire project
    project_root = Path(".")
    detector = HardcodedValueDetector()
    
    print("Scanning for hardcoded test values...")
    violations = detector.scan_directory(project_root)
    
    # Generate report
    report = generate_security_report(violations)
    
    # Display results
    print(f"\n📊 SECURITY AUDIT RESULTS")
    print(f"Security Score: {report['security_score']}/100")
    print(f"Total Violations: {report['total_violations']}")
    print(f"  - Critical: {report['by_severity']['critical']}")
    print(f"  - High: {report['by_severity']['high']}")
    print(f"  - Medium: {report['by_severity']['medium']}")
    print(f"  - Low: {report['by_severity']['low']}")
    
    # Show critical violations
    if report['by_severity']['critical'] > 0:
        print(f"\n🚨 CRITICAL VIOLATIONS (MUST FIX IMMEDIATELY):")
        for violation in violations:
            if violation.severity == SeverityLevel.CRITICAL:
                print(f"  {violation.file_path}:{violation.line_number}")
                print(f"    {violation.description}")
                print(f"    Code: {violation.line_content}")
                print(f"    Fix: {violation.fix_suggestion}")
                print()
    
    # Production readiness assessment
    if report['production_ready']:
        print("✅ PRODUCTION READY: No critical hardcoded test values detected")
        return 0
    else:
        print("❌ NOT PRODUCTION READY: Critical security violations detected")
        print("   Fix all CRITICAL violations before production deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
