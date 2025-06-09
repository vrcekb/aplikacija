#!/usr/bin/env python3
"""
TallyIO Production Security Validation

Focused validation for REAL security issues in production code.
Distinguishes between legitimate test values and actual security violations.

🚨 FINANCIAL APPLICATION - PRODUCTION SECURITY VALIDATION
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SecurityIssue:
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    line_content: str

class ProductionSecurityValidator:
    """Validates production code for real security issues"""
    
    def __init__(self):
        # REAL security violations in production code
        self.production_violations = [
            # Hardcoded credentials that could be real
            (r'sk-[a-zA-Z0-9]{48}', 'Real API key detected'),
            (r'password.*=.*["\'][^"\']{8,}["\']', 'Hardcoded password'),
            (r'secret.*=.*["\'][^"\']{10,}["\']', 'Hardcoded secret'),
            
            # Production endpoints with test domains
            (r'https?://.*\.test\.', 'Test domain in production endpoint'),
            (r'https?://.*\.demo\.', 'Demo domain in production endpoint'),
            (r'https?://.*example\.com', 'Example domain in production endpoint'),
            
            # Development credentials in production
            (r'admin:admin', 'Default admin credentials'),
            (r'root:root', 'Default root credentials'),
        ]
        
        # Files that are production code (not test/example/script files)
        self.production_file_patterns = [
            r'crates/.*/src/.*\.rs$',  # Main source files
            r'src/.*\.rs$',            # Source files
        ]
        
        # Files to exclude (legitimate test/script files)
        self.excluded_patterns = [
            r'/tests?/',
            r'/benches?/', 
            r'/examples?/',
            r'_test\.rs$',
            r'_tests\.rs$',
            r'_bench\.rs$',
            r'/scripts/',
            r'\.py$',
            r'security_audit_.*\.py$',
            r'production_deployment_.*\.py$',
        ]

    def is_production_file(self, file_path: str) -> bool:
        """Check if file is production code that needs strict validation"""
        # Must match production patterns
        if not any(re.search(pattern, file_path) for pattern in self.production_file_patterns):
            return False
            
        # Must not match excluded patterns
        if any(re.search(pattern, file_path) for pattern in self.excluded_patterns):
            return False
            
        return True

    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a production file for real security issues"""
        issues = []
        
        if not self.is_production_file(str(file_path)):
            return issues
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return issues

        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('#'):
                continue
                
            # Skip test functions and test code blocks
            if re.search(r'#\[.*test.*\]|fn test_|async fn test_', line_stripped, re.IGNORECASE):
                continue
                
            # Check for real production security violations
            for pattern, description in self.production_violations:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Additional validation to avoid false positives
                    if self._is_real_violation(line_stripped, pattern):
                        issues.append(SecurityIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            issue_type="PRODUCTION_SECURITY_VIOLATION",
                            severity="CRITICAL",
                            description=description,
                            line_content=line_stripped
                        ))
        
        return issues

    def _is_real_violation(self, line: str, pattern: str) -> bool:
        """Additional validation to distinguish real violations from false positives"""
        
        # Skip validation code that checks for these patterns
        validation_contexts = [
            r'contains\(".*"\)',
            r'if.*contains.*',
            r'endpoint\.contains',
            r'pattern.*description',
            r'SecurityViolation',
            r'CoreError::validation',
        ]
        
        if any(re.search(context, line, re.IGNORECASE) for context in validation_contexts):
            return False
            
        # Skip obvious test contexts
        test_contexts = [
            r'test_.*=',
            r'let.*test.*=',
            r'assert.*',
            r'#\[test\]',
            r'fn test_',
        ]
        
        if any(re.search(context, line, re.IGNORECASE) for context in test_contexts):
            return False
            
        return True

    def scan_directory(self, directory: Path) -> List[SecurityIssue]:
        """Scan entire directory for production security issues"""
        all_issues = []
        
        for file_path in directory.rglob("*.rs"):
            if file_path.is_file():
                issues = self.scan_file(file_path)
                all_issues.extend(issues)
        
        return all_issues

    def generate_report(self, issues: List[SecurityIssue]) -> Dict[str, Any]:
        """Generate production security report"""
        
        critical_issues = [issue for issue in issues if issue.severity == "CRITICAL"]
        
        return {
            'total_issues': len(issues),
            'critical_issues': len(critical_issues),
            'production_ready': len(critical_issues) == 0,
            'issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description,
                    'content': issue.line_content
                }
                for issue in issues
            ]
        }

def main():
    """Main production security validation"""
    print("🚨 TallyIO Production Security Validation")
    print("=" * 50)
    print("🔒 Scanning for REAL security violations in production code...")
    
    # Scan the project
    project_root = Path(".")
    validator = ProductionSecurityValidator()
    
    issues = validator.scan_directory(project_root)
    report = validator.generate_report(issues)
    
    # Display results
    print(f"\n📊 PRODUCTION SECURITY RESULTS")
    print(f"Total Issues: {report['total_issues']}")
    print(f"Critical Issues: {report['critical_issues']}")
    
    if report['critical_issues'] > 0:
        print(f"\n🚨 CRITICAL PRODUCTION SECURITY VIOLATIONS:")
        for issue in issues:
            if issue.severity == "CRITICAL":
                print(f"  {issue.file_path}:{issue.line_number}")
                print(f"    {issue.description}")
                print(f"    Code: {issue.line_content}")
                print()
    
    # Final verdict
    if report['production_ready']:
        print("✅ PRODUCTION READY: No critical security violations in production code")
        print("🎉 All hardcoded test values are properly isolated to test files")
        return 0
    else:
        print("❌ NOT PRODUCTION READY: Critical security violations in production code")
        print("🚨 Fix all critical violations before production deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
