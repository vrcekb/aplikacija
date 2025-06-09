#!/usr/bin/env python3
"""
TallyIO Production Deployment Security Checklist

CRITICAL SECURITY VALIDATION for production deployments.
Ensures no hardcoded test values leak to production environment.

🚨 FINANCIAL APPLICATION - ZERO TOLERANCE FOR SECURITY VIOLATIONS
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

@dataclass
class SecurityCheck:
    name: str
    description: str
    status: CheckStatus
    details: str
    fix_action: str

class ProductionSecurityValidator:
    """Comprehensive security validation for production deployment"""
    
    def __init__(self):
        self.checks: List[SecurityCheck] = []
        
    def add_check(self, name: str, description: str, status: CheckStatus, 
                  details: str, fix_action: str = ""):
        """Add a security check result"""
        self.checks.append(SecurityCheck(
            name=name,
            description=description,
            status=status,
            details=details,
            fix_action=fix_action
        ))
    
    def check_environment_variables(self) -> bool:
        """Validate required production environment variables"""
        required_vars = [
            "TALLYIO_ENVIRONMENT",
            "TALLYIO_PROD_ENDPOINTS", 
            "TALLYIO_HSM_ENDPOINT",
            "TALLYIO_VAULT_ENDPOINT"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.add_check(
                "Environment Variables",
                "Required production environment variables",
                CheckStatus.FAIL,
                f"Missing variables: {', '.join(missing_vars)}",
                "Set all required environment variables before deployment"
            )
            return False
        
        # Validate TALLYIO_ENVIRONMENT is set to production
        env_value = os.getenv("TALLYIO_ENVIRONMENT", "").lower()
        if env_value != "production":
            self.add_check(
                "Environment Variables",
                "TALLYIO_ENVIRONMENT must be 'production'",
                CheckStatus.FAIL,
                f"Current value: {env_value}",
                "Set TALLYIO_ENVIRONMENT=production"
            )
            return False
            
        self.add_check(
            "Environment Variables",
            "All required environment variables present",
            CheckStatus.PASS,
            "All production environment variables configured",
            ""
        )
        return True
    
    def check_endpoint_security(self) -> bool:
        """Validate endpoint security"""
        endpoints_var = os.getenv("TALLYIO_PROD_ENDPOINTS", "")
        hsm_endpoint = os.getenv("TALLYIO_HSM_ENDPOINT", "")
        vault_endpoint = os.getenv("TALLYIO_VAULT_ENDPOINT", "")
        
        all_endpoints = [endpoints_var, hsm_endpoint, vault_endpoint]
        
        # Check for test values
        test_indicators = [
            "localhost", "127.0.0.1", "test", "demo", "example.com",
            "YOUR_KEY", "API_KEY", "${", "mock", "fake"
        ]
        
        violations = []
        for endpoint in all_endpoints:
            if endpoint:
                for indicator in test_indicators:
                    if indicator.lower() in endpoint.lower():
                        violations.append(f"'{indicator}' found in: {endpoint}")
        
        if violations:
            self.add_check(
                "Endpoint Security",
                "Production endpoints must not contain test values",
                CheckStatus.FAIL,
                f"Test values detected: {'; '.join(violations)}",
                "Replace with actual production endpoints"
            )
            return False
        
        # Check for HTTPS/WSS
        insecure_endpoints = []
        for endpoint in all_endpoints:
            if endpoint and not (endpoint.startswith("https://") or endpoint.startswith("wss://")):
                insecure_endpoints.append(endpoint)
        
        if insecure_endpoints:
            self.add_check(
                "Endpoint Security",
                "Production endpoints must use secure protocols",
                CheckStatus.FAIL,
                f"Insecure endpoints: {'; '.join(insecure_endpoints)}",
                "Use HTTPS/WSS protocols for all production endpoints"
            )
            return False
            
        self.add_check(
            "Endpoint Security",
            "All endpoints use secure protocols and values",
            CheckStatus.PASS,
            "No test values or insecure protocols detected",
            ""
        )
        return True
    
    def check_hardcoded_values(self) -> bool:
        """Run hardcoded values security scan"""
        try:
            result = subprocess.run([
                sys.executable, "scripts/security_audit_hardcoded_values.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.add_check(
                    "Hardcoded Values Scan",
                    "No critical hardcoded test values detected",
                    CheckStatus.PASS,
                    "Security scan completed successfully",
                    ""
                )
                return True
            else:
                self.add_check(
                    "Hardcoded Values Scan", 
                    "Critical hardcoded test values detected",
                    CheckStatus.FAIL,
                    result.stdout + result.stderr,
                    "Fix all critical violations before deployment"
                )
                return False
                
        except Exception as e:
            self.add_check(
                "Hardcoded Values Scan",
                "Security scan failed to execute",
                CheckStatus.FAIL,
                str(e),
                "Ensure security scan script is available and executable"
            )
            return False
    
    def check_configuration_security(self) -> bool:
        """Validate configuration security"""
        try:
            # Test production config creation
            result = subprocess.run([
                "cargo", "test", "test_production_config", "--", "--nocapture"
            ], capture_output=True, text=True, timeout=30, cwd=".")
            
            if result.returncode == 0:
                self.add_check(
                    "Configuration Security",
                    "Production configuration validation passed",
                    CheckStatus.PASS,
                    "Configuration security tests passed",
                    ""
                )
                return True
            else:
                self.add_check(
                    "Configuration Security",
                    "Production configuration validation failed",
                    CheckStatus.FAIL,
                    result.stderr,
                    "Fix configuration security issues"
                )
                return False
                
        except Exception as e:
            self.add_check(
                "Configuration Security",
                "Configuration test failed to execute",
                CheckStatus.WARNING,
                str(e),
                "Manually verify configuration security"
            )
            return False
    
    def check_build_security(self) -> bool:
        """Validate build security"""
        try:
            # Run clippy with security-focused lints
            result = subprocess.run([
                "cargo", "clippy", "--all-targets", "--all-features", "--",
                "-D", "warnings", "-D", "clippy::unwrap_used", "-D", "clippy::expect_used"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.add_check(
                    "Build Security",
                    "Security-focused clippy checks passed",
                    CheckStatus.PASS,
                    "No security-related code issues detected",
                    ""
                )
                return True
            else:
                self.add_check(
                    "Build Security",
                    "Security-focused clippy checks failed",
                    CheckStatus.FAIL,
                    result.stderr,
                    "Fix all clippy security warnings"
                )
                return False
                
        except Exception as e:
            self.add_check(
                "Build Security",
                "Build security check failed",
                CheckStatus.WARNING,
                str(e),
                "Manually verify build security"
            )
            return False
    
    def run_all_checks(self) -> bool:
        """Run all security checks"""
        print("🚨 TallyIO Production Deployment Security Checklist")
        print("=" * 60)
        print("🔒 FINANCIAL APPLICATION - ZERO TOLERANCE FOR SECURITY VIOLATIONS")
        print()
        
        checks = [
            ("Environment Variables", self.check_environment_variables),
            ("Endpoint Security", self.check_endpoint_security),
            ("Hardcoded Values", self.check_hardcoded_values),
            ("Configuration Security", self.check_configuration_security),
            ("Build Security", self.check_build_security),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"Running {check_name} check...")
            try:
                passed = check_func()
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"ERROR in {check_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deployment security report"""
        passed = sum(1 for check in self.checks if check.status == CheckStatus.PASS)
        failed = sum(1 for check in self.checks if check.status == CheckStatus.FAIL)
        warnings = sum(1 for check in self.checks if check.status == CheckStatus.WARNING)
        
        return {
            "total_checks": len(self.checks),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "production_ready": failed == 0,
            "checks": [
                {
                    "name": check.name,
                    "description": check.description,
                    "status": check.status.value,
                    "details": check.details,
                    "fix_action": check.fix_action
                }
                for check in self.checks
            ]
        }
    
    def print_report(self):
        """Print detailed security report"""
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("📊 PRODUCTION DEPLOYMENT SECURITY REPORT")
        print("=" * 60)
        
        print(f"Total Checks: {report['total_checks']}")
        print(f"✅ Passed: {report['passed']}")
        print(f"❌ Failed: {report['failed']}")
        print(f"⚠️  Warnings: {report['warnings']}")
        print()
        
        # Show failed checks
        failed_checks = [check for check in self.checks if check.status == CheckStatus.FAIL]
        if failed_checks:
            print("❌ FAILED CHECKS (MUST FIX BEFORE DEPLOYMENT):")
            for check in failed_checks:
                print(f"  • {check.name}: {check.description}")
                print(f"    Details: {check.details}")
                print(f"    Fix: {check.fix_action}")
                print()
        
        # Show warnings
        warning_checks = [check for check in self.checks if check.status == CheckStatus.WARNING]
        if warning_checks:
            print("⚠️  WARNINGS (REVIEW RECOMMENDED):")
            for check in warning_checks:
                print(f"  • {check.name}: {check.description}")
                print(f"    Details: {check.details}")
                print(f"    Action: {check.fix_action}")
                print()
        
        # Final verdict
        if report['production_ready']:
            print("🎉 PRODUCTION READY: All security checks passed!")
            print("✅ Safe to deploy to production environment")
        else:
            print("🚨 NOT PRODUCTION READY: Security violations detected!")
            print("❌ DO NOT DEPLOY until all failed checks are resolved")
        
        return report['production_ready']

def main():
    """Main deployment security validation"""
    validator = ProductionSecurityValidator()
    
    # Run all security checks
    all_passed = validator.run_all_checks()
    
    # Generate and print report
    production_ready = validator.print_report()
    
    # Return appropriate exit code
    return 0 if production_ready else 1

if __name__ == "__main__":
    sys.exit(main())
