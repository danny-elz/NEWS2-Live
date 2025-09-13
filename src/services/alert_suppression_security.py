"""
Security Controls and Input Validation for Alert Suppression System

Implements comprehensive security measures including input validation,
sanitization, authentication, authorization, and audit logging to protect
against security vulnerabilities and ensure data integrity.
"""

import re
import logging
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import html
import json

from prometheus_client import Counter


class SecurityLevel(Enum):
    """Security levels for operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    PRIVILEGED = "privileged"


class ValidationError(Exception):
    """Exception raised for validation failures."""
    pass


class SecurityViolationError(Exception):
    """Exception raised for security violations."""
    pass


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""
    pass


class AuthorizationError(Exception):
    """Exception raised for authorization failures."""
    pass


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    role: str
    permissions: Set[str]
    session_id: str
    ip_address: str
    authenticated_at: datetime
    security_level: SecurityLevel


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    field_name: str
    required: bool
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    custom_validator: Optional[callable] = None


class InputValidator:
    """
    Comprehensive input validation and sanitization.
    
    Protects against injection attacks, XSS, and data integrity issues
    by validating and sanitizing all user inputs.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security metrics
        self.validation_failures = Counter('validation_failures_total', 'Input validation failures', ['field', 'rule'])
        self.security_violations = Counter('security_violations_total', 'Security violations detected', ['violation_type'])
        
        # Common validation patterns
        self.patterns = {
            'patient_id': r'^[A-Z0-9_]{3,20}$',
            'nurse_id': r'^[A-Z0-9_]{3,20}$',
            'ward_id': r'^[A-Z0-9_]{2,10}$',
            'session_id': r'^[a-f0-9]{32,64}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'ip_address': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        }
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',                # XSS
            r'on\w+\s*=',                 # Event handlers
            r';\s*DROP\s+TABLE',          # SQL injection
            r';\s*DELETE\s+FROM',         # SQL injection
            r';\s*UPDATE\s+.*\s+SET',     # SQL injection
            r'UNION\s+SELECT',            # SQL injection
            r'\.\./',                     # Path traversal
            r'\.\.\\',                    # Path traversal (Windows)
            r'etc/passwd',                # File access
            r'cmd\.exe',                  # Command injection
            r'powershell',                # Command injection
        ]
    
    def validate_patient_id(self, patient_id: str) -> str:
        """
        Validate and sanitize patient ID.
        
        Args:
            patient_id: Patient identifier to validate
            
        Returns:
            Validated patient ID
            
        Raises:
            ValidationError: If validation fails
        """
        if not patient_id:
            self.validation_failures.labels(field='patient_id', rule='required').inc()
            raise ValidationError("Patient ID is required")
        
        if not isinstance(patient_id, str):
            self.validation_failures.labels(field='patient_id', rule='type').inc()
            raise ValidationError("Patient ID must be a string")
        
        # Check length
        if len(patient_id) < 3 or len(patient_id) > 20:
            self.validation_failures.labels(field='patient_id', rule='length').inc()
            raise ValidationError("Patient ID must be 3-20 characters")
        
        # Check pattern
        if not re.match(self.patterns['patient_id'], patient_id):
            self.validation_failures.labels(field='patient_id', rule='pattern').inc()
            raise ValidationError("Patient ID contains invalid characters")
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(patient_id, 'patient_id')
        
        return patient_id.upper().strip()
    
    def validate_nurse_id(self, nurse_id: str) -> str:
        """
        Validate and sanitize nurse ID.
        
        Args:
            nurse_id: Nurse identifier to validate
            
        Returns:
            Validated nurse ID
            
        Raises:
            ValidationError: If validation fails
        """
        if not nurse_id:
            self.validation_failures.labels(field='nurse_id', rule='required').inc()
            raise ValidationError("Nurse ID is required")
        
        if not isinstance(nurse_id, str):
            self.validation_failures.labels(field='nurse_id', rule='type').inc()
            raise ValidationError("Nurse ID must be a string")
        
        # Check length
        if len(nurse_id) < 3 or len(nurse_id) > 20:
            self.validation_failures.labels(field='nurse_id', rule='length').inc()
            raise ValidationError("Nurse ID must be 3-20 characters")
        
        # Check pattern
        if not re.match(self.patterns['nurse_id'], nurse_id):
            self.validation_failures.labels(field='nurse_id', rule='pattern').inc()
            raise ValidationError("Nurse ID contains invalid characters")
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(nurse_id, 'nurse_id')
        
        return nurse_id.upper().strip()
    
    def validate_clinical_justification(self, justification: str) -> str:
        """
        Validate and sanitize clinical justification text.
        
        Args:
            justification: Clinical justification text
            
        Returns:
            Sanitized justification
            
        Raises:
            ValidationError: If validation fails
        """
        if not justification:
            self.validation_failures.labels(field='justification', rule='required').inc()
            raise ValidationError("Clinical justification is required")
        
        if not isinstance(justification, str):
            self.validation_failures.labels(field='justification', rule='type').inc()
            raise ValidationError("Justification must be a string")
        
        # Trim whitespace
        justification = justification.strip()
        
        # Check minimum length for meaningful justification
        if len(justification) < 20:
            self.validation_failures.labels(field='justification', rule='min_length').inc()
            raise ValidationError("Clinical justification must be at least 20 characters")
        
        # Check maximum length to prevent DoS
        if len(justification) > 1000:
            self.validation_failures.labels(field='justification', rule='max_length').inc()
            raise ValidationError("Clinical justification must be less than 1000 characters")
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(justification, 'justification')
        
        # Sanitize HTML/script content
        sanitized = self._sanitize_text(justification)
        
        # Ensure justification still meets minimum length after sanitization
        if len(sanitized.strip()) < 20:
            self.validation_failures.labels(field='justification', rule='sanitized_length').inc()
            raise ValidationError("Clinical justification too short after sanitization")
        
        return sanitized
    
    def validate_ward_id(self, ward_id: str) -> str:
        """
        Validate and sanitize ward ID.
        
        Args:
            ward_id: Ward identifier to validate
            
        Returns:
            Validated ward ID
            
        Raises:
            ValidationError: If validation fails
        """
        if not ward_id:
            self.validation_failures.labels(field='ward_id', rule='required').inc()
            raise ValidationError("Ward ID is required")
        
        if not isinstance(ward_id, str):
            self.validation_failures.labels(field='ward_id', rule='type').inc()
            raise ValidationError("Ward ID must be a string")
        
        # Check length
        if len(ward_id) < 2 or len(ward_id) > 10:
            self.validation_failures.labels(field='ward_id', rule='length').inc()
            raise ValidationError("Ward ID must be 2-10 characters")
        
        # Check pattern
        if not re.match(self.patterns['ward_id'], ward_id):
            self.validation_failures.labels(field='ward_id', rule='pattern').inc()
            raise ValidationError("Ward ID contains invalid characters")
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(ward_id, 'ward_id')
        
        return ward_id.upper().strip()
    
    def validate_session_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate session data for authentication.
        
        Args:
            session_data: Session data to validate
            
        Returns:
            Validated session data
            
        Raises:
            ValidationError: If validation fails
        """
        if not session_data:
            self.validation_failures.labels(field='session_data', rule='required').inc()
            raise ValidationError("Session data is required")
        
        required_fields = ['user_id', 'role', 'session_id', 'authenticated_at']
        for field in required_fields:
            if field not in session_data:
                self.validation_failures.labels(field=f'session.{field}', rule='required').inc()
                raise ValidationError(f"Session field {field} is required")
        
        # Validate session ID format
        session_id = session_data.get('session_id', '')
        if not re.match(self.patterns['session_id'], session_id):
            self.validation_failures.labels(field='session.session_id', rule='pattern').inc()
            raise ValidationError("Invalid session ID format")
        
        # Validate user ID
        user_id = session_data.get('user_id', '')
        self.validate_nurse_id(user_id)  # Reuse nurse ID validation
        
        # Validate role
        role = session_data.get('role', '')
        valid_roles = ['nurse', 'charge_nurse', 'doctor', 'admin']
        if role not in valid_roles:
            self.validation_failures.labels(field='session.role', rule='allowed_values').inc()
            raise ValidationError(f"Invalid role: {role}")
        
        return session_data
    
    def _check_dangerous_patterns(self, text: str, field_name: str):
        """
        Check for dangerous patterns that might indicate attacks.
        
        Args:
            text: Text to check
            field_name: Field name for logging
            
        Raises:
            SecurityViolationError: If dangerous pattern found
        """
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.security_violations.labels(violation_type='dangerous_pattern').inc()
                self.logger.warning(f"Dangerous pattern detected in {field_name}: {pattern}")
                raise SecurityViolationError(f"Potentially malicious content detected in {field_name}")
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text content to prevent XSS and injection attacks.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: URLs
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_json_data(self, json_data: str, max_size: int = 10240) -> Dict[str, Any]:
        """
        Validate and parse JSON data safely.
        
        Args:
            json_data: JSON string to validate
            max_size: Maximum JSON size in bytes
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If validation fails
        """
        if not json_data:
            self.validation_failures.labels(field='json_data', rule='required').inc()
            raise ValidationError("JSON data is required")
        
        if not isinstance(json_data, str):
            self.validation_failures.labels(field='json_data', rule='type').inc()
            raise ValidationError("JSON data must be a string")
        
        # Check size to prevent DoS
        if len(json_data.encode('utf-8')) > max_size:
            self.validation_failures.labels(field='json_data', rule='max_size').inc()
            raise ValidationError(f"JSON data exceeds maximum size of {max_size} bytes")
        
        try:
            parsed_data = json.loads(json_data)
            return parsed_data
        except json.JSONDecodeError as e:
            self.validation_failures.labels(field='json_data', rule='format').inc()
            raise ValidationError(f"Invalid JSON format: {str(e)}")


class AuthenticationManager:
    """
    Authentication manager for secure access control.
    
    Handles user authentication, session management, and security controls
    to ensure only authorized users can perform suppression operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = InputValidator()
        
        # Session storage (in production, use Redis or database)
        self.active_sessions = {}
        
        # Security metrics
        self.authentication_attempts = Counter('authentication_attempts_total', 'Authentication attempts', ['result'])
        self.session_events = Counter('session_events_total', 'Session events', ['event_type'])
        
        # Rate limiting (simple in-memory implementation)
        self.rate_limits = {}
        self.max_attempts_per_minute = 5
    
    async def authenticate_user(self, user_id: str, credentials: Dict[str, Any], ip_address: str) -> SecurityContext:
        """
        Authenticate user and create security context.
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials
            ip_address: Client IP address
            
        Returns:
            Security context for authenticated user
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Rate limiting
        if not self._check_rate_limit(ip_address):
            self.authentication_attempts.labels(result='rate_limited').inc()
            raise AuthenticationError("Too many authentication attempts. Please try again later.")
        
        # Validate inputs
        validated_user_id = self.validator.validate_nurse_id(user_id)
        
        # Validate IP address
        if not re.match(self.validator.patterns['ip_address'], ip_address):
            self.authentication_attempts.labels(result='invalid_ip').inc()
            raise AuthenticationError("Invalid IP address format")
        
        try:
            # In production, verify against secure credential store
            user_info = await self._verify_credentials(validated_user_id, credentials)
            
            # Create session
            session_id = self._generate_secure_session_id()
            authenticated_at = datetime.now(timezone.utc)
            
            security_context = SecurityContext(
                user_id=validated_user_id,
                role=user_info['role'],
                permissions=set(user_info.get('permissions', [])),
                session_id=session_id,
                ip_address=ip_address,
                authenticated_at=authenticated_at,
                security_level=SecurityLevel.AUTHENTICATED
            )
            
            # Store session
            self.active_sessions[session_id] = {
                'security_context': security_context,
                'created_at': authenticated_at,
                'last_accessed': authenticated_at,
                'expires_at': authenticated_at + timedelta(hours=8)  # 8-hour session
            }
            
            self.authentication_attempts.labels(result='success').inc()
            self.session_events.labels(event_type='created').inc()
            
            self.logger.info(f"User {validated_user_id} authenticated successfully from {ip_address}")
            
            return security_context
            
        except Exception as e:
            self.authentication_attempts.labels(result='failure').inc()
            self.logger.warning(f"Authentication failed for user {validated_user_id} from {ip_address}: {str(e)}")
            raise AuthenticationError("Authentication failed")
    
    async def validate_session(self, session_id: str, ip_address: str) -> SecurityContext:
        """
        Validate existing session and return security context.
        
        Args:
            session_id: Session identifier
            ip_address: Client IP address
            
        Returns:
            Security context for valid session
            
        Raises:
            AuthenticationError: If session is invalid
        """
        if not session_id or session_id not in self.active_sessions:
            self.session_events.labels(event_type='invalid').inc()
            raise AuthenticationError("Invalid session")
        
        session_data = self.active_sessions[session_id]
        security_context = session_data['security_context']
        
        # Check if session expired
        if datetime.now(timezone.utc) > session_data['expires_at']:
            del self.active_sessions[session_id]
            self.session_events.labels(event_type='expired').inc()
            raise AuthenticationError("Session expired")
        
        # Check IP address consistency (basic security measure)
        if security_context.ip_address != ip_address:
            self.session_events.labels(event_type='ip_mismatch').inc()
            self.logger.warning(f"IP address mismatch for session {session_id}: expected {security_context.ip_address}, got {ip_address}")
            # In production, might want to invalidate session or require re-authentication
        
        # Update last accessed time
        session_data['last_accessed'] = datetime.now(timezone.utc)
        
        return security_context
    
    async def invalidate_session(self, session_id: str):
        """
        Invalidate user session.
        
        Args:
            session_id: Session identifier to invalidate
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.session_events.labels(event_type='invalidated').inc()
            self.logger.info(f"Session {session_id} invalidated")
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """
        Check if IP address is within rate limits.
        
        Args:
            ip_address: Client IP address
            
        Returns:
            True if within limits, False otherwise
        """
        current_time = datetime.now(timezone.utc)
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Clean old attempts (older than 1 minute)
        cutoff_time = current_time - timedelta(minutes=1)
        self.rate_limits[ip_address] = [
            attempt_time for attempt_time in self.rate_limits[ip_address]
            if attempt_time > cutoff_time
        ]
        
        # Check if too many attempts
        if len(self.rate_limits[ip_address]) >= self.max_attempts_per_minute:
            return False
        
        # Record this attempt
        self.rate_limits[ip_address].append(current_time)
        return True
    
    async def _verify_credentials(self, user_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify user credentials against secure store.
        
        In production, this would integrate with:
        - LDAP/Active Directory
        - Database with hashed passwords
        - OAuth/SAML providers
        - Hardware tokens/2FA
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials
            
        Returns:
            User information if valid
            
        Raises:
            AuthenticationError: If credentials invalid
        """
        # Mock implementation for testing
        # In production, use secure credential verification
        mock_users = {
            'NURSE_001': {
                'role': 'nurse',
                'permissions': ['view_alerts', 'acknowledge_alerts']
            },
            'NURSE_002': {
                'role': 'charge_nurse', 
                'permissions': ['view_alerts', 'acknowledge_alerts', 'create_overrides']
            },
            'DOC_001': {
                'role': 'doctor',
                'permissions': ['view_alerts', 'acknowledge_alerts', 'create_overrides', 'modify_thresholds']
            }
        }
        
        if user_id not in mock_users:
            raise AuthenticationError("User not found")
        
        # In production, verify password hash, check 2FA, etc.
        password = credentials.get('password', '')
        if len(password) < 8:  # Basic password requirement
            raise AuthenticationError("Invalid credentials")
        
        return mock_users[user_id]
    
    def _generate_secure_session_id(self) -> str:
        """
        Generate cryptographically secure session ID.
        
        Returns:
            Secure session identifier
        """
        return secrets.token_hex(32)


class AuthorizationManager:
    """
    Authorization manager for permission-based access control.
    
    Ensures users can only perform operations they are authorized for,
    implementing principle of least privilege.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Authorization metrics
        self.authorization_checks = Counter('authorization_checks_total', 'Authorization checks', ['operation', 'result'])
        
        # Permission mappings
        self.operation_permissions = {
            'view_suppression_decisions': ['view_alerts'],
            'create_manual_override': ['create_overrides'],
            'modify_suppression_settings': ['modify_thresholds'],
            'view_system_health': ['view_system_status'],
            'export_suppression_data': ['export_data']
        }
    
    def check_permission(self, security_context: SecurityContext, operation: str) -> bool:
        """
        Check if user has permission for operation.
        
        Args:
            security_context: User security context
            operation: Operation to check permission for
            
        Returns:
            True if authorized, False otherwise
        """
        required_permissions = self.operation_permissions.get(operation, [])
        
        if not required_permissions:
            # Operation not defined - deny by default
            self.authorization_checks.labels(operation=operation, result='undefined').inc()
            self.logger.warning(f"Permission check for undefined operation: {operation}")
            return False
        
        # Check if user has any of the required permissions
        has_permission = bool(security_context.permissions.intersection(required_permissions))
        
        result = 'granted' if has_permission else 'denied'
        self.authorization_checks.labels(operation=operation, result=result).inc()
        
        if not has_permission:
            self.logger.warning(
                f"Authorization denied for user {security_context.user_id} "
                f"(role: {security_context.role}) for operation {operation}. "
                f"Required: {required_permissions}, Has: {list(security_context.permissions)}"
            )
        
        return has_permission
    
    def require_permission(self, security_context: SecurityContext, operation: str):
        """
        Require permission for operation, raising exception if denied.
        
        Args:
            security_context: User security context
            operation: Operation to check permission for
            
        Raises:
            AuthorizationError: If permission denied
        """
        if not self.check_permission(security_context, operation):
            raise AuthorizationError(f"Insufficient permissions for operation: {operation}")


class SecurityAuditLogger:
    """
    Security audit logger for tracking security-relevant events.
    
    Provides comprehensive audit trail for compliance and security monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.security_audit")
        
        # Security event metrics
        self.security_events = Counter('security_events_total', 'Security events', ['event_type', 'severity'])
    
    def log_authentication_event(self, user_id: str, ip_address: str, success: bool, details: Dict[str, Any] = None):
        """Log authentication event."""
        event_type = 'authentication_success' if success else 'authentication_failure'
        severity = 'info' if success else 'warning'
        
        self.security_events.labels(event_type=event_type, severity=severity).inc()
        
        audit_entry = {
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {}
        }
        
        if success:
            self.logger.info(f"Authentication successful: {json.dumps(audit_entry)}")
        else:
            self.logger.warning(f"Authentication failed: {json.dumps(audit_entry)}")
    
    def log_authorization_event(self, user_id: str, operation: str, granted: bool, details: Dict[str, Any] = None):
        """Log authorization event."""
        event_type = 'authorization_granted' if granted else 'authorization_denied'
        severity = 'info' if granted else 'warning'
        
        self.security_events.labels(event_type=event_type, severity=severity).inc()
        
        audit_entry = {
            'event_type': event_type,
            'user_id': user_id,
            'operation': operation,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {}
        }
        
        if granted:
            self.logger.info(f"Authorization granted: {json.dumps(audit_entry)}")
        else:
            self.logger.warning(f"Authorization denied: {json.dumps(audit_entry)}")
    
    def log_security_violation(self, violation_type: str, user_id: str, ip_address: str, details: Dict[str, Any] = None):
        """Log security violation."""
        self.security_events.labels(event_type='security_violation', severity='critical').inc()
        
        audit_entry = {
            'event_type': 'security_violation',
            'violation_type': violation_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {}
        }
        
        self.logger.critical(f"Security violation detected: {json.dumps(audit_entry)}")
    
    def log_suppression_override(self, user_id: str, patient_id: str, justification: str, duration_minutes: int):
        """Log manual suppression override creation."""
        self.security_events.labels(event_type='suppression_override', severity='info').inc()
        
        audit_entry = {
            'event_type': 'suppression_override',
            'user_id': user_id,
            'patient_id': patient_id,
            'justification_length': len(justification),
            'duration_minutes': duration_minutes,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Suppression override created: {json.dumps(audit_entry)}")


class SecureSuppressionWrapper:
    """
    Secure wrapper for suppression operations with comprehensive security controls.
    
    Provides authentication, authorization, input validation, and audit logging
    for all suppression-related operations.
    """
    
    def __init__(self, suppression_engine):
        self.suppression_engine = suppression_engine
        self.validator = InputValidator()
        self.auth_manager = AuthenticationManager()
        self.authz_manager = AuthorizationManager()
        self.audit_logger = SecurityAuditLogger()
        self.logger = logging.getLogger(__name__)
    
    async def create_manual_override_secure(
        self,
        patient_id: str,
        nurse_id: str,
        justification: str,
        duration_minutes: int,
        security_context: SecurityContext
    ):
        """
        Create manual override with security controls.
        
        Args:
            patient_id: Patient identifier
            nurse_id: Nurse identifier
            justification: Clinical justification
            duration_minutes: Override duration
            security_context: Security context
            
        Returns:
            Created override
            
        Raises:
            ValidationError: If input validation fails
            AuthorizationError: If user not authorized
        """
        # Authorization check
        self.authz_manager.require_permission(security_context, 'create_manual_override')
        
        # Input validation
        validated_patient_id = self.validator.validate_patient_id(patient_id)
        validated_nurse_id = self.validator.validate_nurse_id(nurse_id)
        validated_justification = self.validator.validate_clinical_justification(justification)
        
        # Validate duration
        if not isinstance(duration_minutes, int) or duration_minutes < 1 or duration_minutes > 480:  # Max 8 hours
            raise ValidationError("Duration must be between 1 and 480 minutes")
        
        try:
            # Create override using underlying engine
            override = await self.suppression_engine.override_manager.create_override(
                validated_patient_id,
                validated_nurse_id,
                validated_justification,
                duration_minutes
            )
            
            # Audit log
            self.audit_logger.log_suppression_override(
                security_context.user_id,
                validated_patient_id,
                validated_justification,
                duration_minutes
            )
            
            self.logger.info(f"Manual override created by {security_context.user_id} for patient {validated_patient_id}")
            
            return override
            
        except Exception as e:
            self.logger.error(f"Failed to create manual override: {str(e)}")
            raise
    
    async def authenticate_and_execute(self, user_id: str, credentials: Dict[str, Any], ip_address: str, operation: callable, *args, **kwargs):
        """
        Authenticate user and execute operation with security controls.
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials
            ip_address: Client IP address
            operation: Operation to execute
            *args, **kwargs: Operation arguments
            
        Returns:
            Operation result
        """
        try:
            # Authenticate user
            security_context = await self.auth_manager.authenticate_user(user_id, credentials, ip_address)
            
            # Execute operation with security context
            result = await operation(security_context, *args, **kwargs)
            
            return result
            
        except (AuthenticationError, AuthorizationError, ValidationError, SecurityViolationError) as e:
            # Log security event
            self.audit_logger.log_security_violation(
                type(e).__name__,
                user_id,
                ip_address,
                {'error': str(e)}
            )
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in secure operation: {str(e)}")
            raise