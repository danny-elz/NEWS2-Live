"""
Configuration Management System for Alert Suppression

Provides centralized, environment-aware configuration management with
validation, hot-reloading, and audit trails for all suppression settings.
"""

import os
import json
import yaml
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import asyncio

from prometheus_client import Gauge, Counter


class ConfigSource(Enum):
    """Configuration sources in order of precedence."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    DEFAULT = "default"


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    key: str
    value: Any
    source: ConfigSource
    last_updated: datetime
    validation_schema: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    sensitive: bool = False


@dataclass
class SuppressionConfig:
    """Complete suppression system configuration."""
    
    # Core suppression settings
    time_suppression_window_minutes: int = 30
    score_increase_threshold: int = 2
    stability_threshold_hours: int = 2
    score_variance_limit: float = 1.0
    
    # Pattern detection settings
    pattern_learning_enabled: bool = True
    ml_confidence_threshold: float = 0.7
    minimum_history_entries: int = 6
    
    # Manual override settings
    max_override_duration_minutes: int = 480  # 8 hours
    min_justification_length: int = 20
    max_justification_length: int = 1000
    
    # Performance settings
    decision_timeout_seconds: float = 30.0
    max_concurrent_decisions: int = 100
    redis_connection_timeout: int = 5
    redis_operation_timeout: int = 10
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_success_threshold: int = 3
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0
    
    # Health monitoring settings
    health_check_interval_seconds: int = 30
    component_timeout_seconds: float = 10.0
    unhealthy_threshold_count: int = 3
    
    # Security settings
    session_timeout_hours: int = 8
    max_login_attempts_per_minute: int = 5
    require_justification_for_overrides: bool = True
    audit_all_decisions: bool = True
    
    # Metrics and monitoring
    metrics_enabled: bool = True
    detailed_metrics_enabled: bool = True
    metric_retention_days: int = 30
    
    # Data retention
    decision_log_retention_days: int = 90
    acknowledgment_retention_days: int = 30
    override_retention_days: int = 180
    
    # Environment-specific settings
    environment: str = "development"
    debug_enabled: bool = False
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuppressionConfig':
        """Create from dictionary."""
        return cls(**data)


class ConfigValidator:
    """Configuration validation with type checking and business rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.validation_rules = {
            'time_suppression_window_minutes': {
                'type': int,
                'min': 1,
                'max': 120,
                'description': 'Time window for suppression in minutes'
            },
            'score_increase_threshold': {
                'type': int,
                'min': 1,
                'max': 10,
                'description': 'NEWS2 score increase threshold for bypassing suppression'
            },
            'max_override_duration_minutes': {
                'type': int,
                'min': 15,
                'max': 1440,  # 24 hours max
                'description': 'Maximum override duration in minutes'
            },
            'decision_timeout_seconds': {
                'type': float,
                'min': 1.0,
                'max': 120.0,
                'description': 'Maximum time for suppression decision'
            },
            'session_timeout_hours': {
                'type': int,
                'min': 1,
                'max': 24,
                'description': 'User session timeout in hours'
            },
            'environment': {
                'type': str,
                'allowed_values': ['development', 'testing', 'staging', 'production'],
                'description': 'Deployment environment'
            },
            'log_level': {
                'type': str,
                'allowed_values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                'description': 'Logging level'
            }
        }
    
    def validate_config(self, config: SuppressionConfig) -> List[str]:
        """
        Validate configuration against rules.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        config_dict = config.to_dict()
        
        for key, value in config_dict.items():
            if key in self.validation_rules:
                rule_errors = self._validate_field(key, value, self.validation_rules[key])
                errors.extend(rule_errors)
        
        # Business rule validations
        business_rule_errors = self._validate_business_rules(config)
        errors.extend(business_rule_errors)
        
        return errors
    
    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate individual field against rules."""
        errors = []
        
        # Type validation
        expected_type = rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"{field_name}: Expected {expected_type.__name__}, got {type(value).__name__}")
            return errors  # Skip other validations if type is wrong
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            min_val = rules.get('min')
            max_val = rules.get('max')
            
            if min_val is not None and value < min_val:
                errors.append(f"{field_name}: Value {value} below minimum {min_val}")
            
            if max_val is not None and value > max_val:
                errors.append(f"{field_name}: Value {value} above maximum {max_val}")
        
        # Allowed values validation
        allowed_values = rules.get('allowed_values')
        if allowed_values and value not in allowed_values:
            errors.append(f"{field_name}: Value '{value}' not in allowed values: {allowed_values}")
        
        return errors
    
    def _validate_business_rules(self, config: SuppressionConfig) -> List[str]:
        """Validate business logic rules."""
        errors = []
        
        # Timeout consistency
        if config.decision_timeout_seconds > config.circuit_breaker_recovery_timeout:
            errors.append("Decision timeout should not exceed circuit breaker recovery timeout")
        
        # Retry configuration consistency
        if config.retry_max_delay < config.retry_base_delay:
            errors.append("Maximum retry delay must be greater than base delay")
        
        # Health check intervals
        if config.health_check_interval_seconds > 300:  # 5 minutes
            errors.append("Health check interval should not exceed 5 minutes")
        
        # Production environment constraints
        if config.environment == "production":
            if config.debug_enabled:
                errors.append("Debug mode should not be enabled in production")
            
            if config.log_level == "DEBUG":
                errors.append("Debug logging should not be used in production")
            
            if config.session_timeout_hours > 12:
                errors.append("Session timeout should not exceed 12 hours in production")
        
        return errors


class ConfigManager:
    """
    Centralized configuration manager with hot-reloading and persistence.
    
    Manages configuration from multiple sources with precedence ordering
    and provides real-time updates without service restart.
    """
    
    def __init__(self, config_file_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.validator = ConfigValidator()
        
        # Configuration storage
        self.config_values: Dict[str, ConfigValue] = {}
        self.current_config: Optional[SuppressionConfig] = None
        self.config_file_path = config_file_path
        self.config_watchers: List[callable] = []
        
        # Metrics
        self.config_loads = Counter('config_loads_total', 'Configuration loads', ['source', 'result'])
        self.config_validations = Counter('config_validations_total', 'Configuration validations', ['result'])
        self.config_updates = Counter('config_updates_total', 'Configuration updates', ['key'])
        self.config_errors = Counter('config_errors_total', 'Configuration errors', ['error_type'])
        
        # Current config metrics
        self.current_config_gauge = Gauge('current_config_value', 'Current configuration values', ['key'])
        
        # Load initial configuration
        asyncio.create_task(self._initialize_config())
    
    async def _initialize_config(self):
        """Initialize configuration from all sources."""
        try:
            # Load default configuration
            await self._load_default_config()
            
            # Load from file if specified
            if self.config_file_path:
                await self._load_file_config(self.config_file_path)
            
            # Load from environment variables
            await self._load_environment_config()
            
            # Build final configuration
            await self._build_current_config()
            
            self.logger.info("Configuration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration: {str(e)}")
            self.config_errors.labels(error_type='initialization').inc()
            raise
    
    async def _load_default_config(self):
        """Load default configuration values."""
        default_config = SuppressionConfig()
        
        for key, value in default_config.to_dict().items():
            self.config_values[key] = ConfigValue(
                key=key,
                value=value,
                source=ConfigSource.DEFAULT,
                last_updated=datetime.now(timezone.utc),
                description=f"Default value for {key}"
            )
        
        self.config_loads.labels(source='default', result='success').inc()
        self.logger.debug("Default configuration loaded")
    
    async def _load_file_config(self, file_path: str):
        """Load configuration from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"Configuration file {file_path} does not exist")
                return
            
            # Determine file format
            if path.suffix.lower() == '.json':
                format_type = ConfigFormat.JSON
            elif path.suffix.lower() in ['.yml', '.yaml']:
                format_type = ConfigFormat.YAML
            else:
                self.logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            # Load file content
            with open(path, 'r') as f:
                if format_type == ConfigFormat.JSON:
                    file_config = json.load(f)
                elif format_type == ConfigFormat.YAML:
                    file_config = yaml.safe_load(f)
            
            # Update configuration values
            for key, value in file_config.items():
                if key in self.config_values:
                    self.config_values[key] = ConfigValue(
                        key=key,
                        value=value,
                        source=ConfigSource.FILE,
                        last_updated=datetime.now(timezone.utc),
                        description=f"Value from file {file_path}"
                    )
            
            self.config_loads.labels(source='file', result='success').inc()
            self.logger.info(f"Configuration loaded from file: {file_path}")
            
        except Exception as e:
            self.config_loads.labels(source='file', result='error').inc()
            self.logger.error(f"Failed to load configuration from file {file_path}: {str(e)}")
    
    async def _load_environment_config(self):
        """Load configuration from environment variables."""
        env_prefix = "SUPPRESSION_"
        
        for key in self.config_values:
            env_var = f"{env_prefix}{key.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                # Parse environment value based on current type
                current_value = self.config_values[key].value
                parsed_value = self._parse_env_value(env_value, type(current_value))
                
                self.config_values[key] = ConfigValue(
                    key=key,
                    value=parsed_value,
                    source=ConfigSource.ENVIRONMENT,
                    last_updated=datetime.now(timezone.utc),
                    description=f"Value from environment variable {env_var}"
                )
        
        self.config_loads.labels(source='environment', result='success').inc()
        self.logger.debug("Environment configuration loaded")
    
    def _parse_env_value(self, env_value: str, target_type: type) -> Any:
        """Parse environment variable value to target type."""
        if target_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(env_value)
        elif target_type == float:
            return float(env_value)
        else:
            return env_value
    
    async def _build_current_config(self):
        """Build current configuration from all sources."""
        try:
            # Extract values with source precedence
            config_dict = {}
            for key, config_value in self.config_values.items():
                config_dict[key] = config_value.value
            
            # Create configuration object
            new_config = SuppressionConfig.from_dict(config_dict)
            
            # Validate configuration
            validation_errors = self.validator.validate_config(new_config)
            
            if validation_errors:
                self.config_validations.labels(result='error').inc()
                self.logger.error(f"Configuration validation failed: {validation_errors}")
                raise ValueError(f"Configuration validation errors: {validation_errors}")
            
            # Update current configuration
            old_config = self.current_config
            self.current_config = new_config
            
            # Update metrics
            self._update_config_metrics()
            
            # Notify watchers of configuration change
            if old_config != new_config:
                await self._notify_config_watchers(old_config, new_config)
            
            self.config_validations.labels(result='success').inc()
            self.logger.info("Configuration built and validated successfully")
            
        except Exception as e:
            self.config_errors.labels(error_type='build').inc()
            self.logger.error(f"Failed to build configuration: {str(e)}")
            raise
    
    def _update_config_metrics(self):
        """Update Prometheus metrics with current configuration values."""
        if not self.current_config:
            return
        
        config_dict = self.current_config.to_dict()
        
        for key, value in config_dict.items():
            if isinstance(value, (int, float)):
                self.current_config_gauge.labels(key=key).set(value)
            elif isinstance(value, bool):
                self.current_config_gauge.labels(key=key).set(1 if value else 0)
    
    async def _notify_config_watchers(self, old_config: SuppressionConfig, new_config: SuppressionConfig):
        """Notify registered watchers of configuration changes."""
        for watcher in self.config_watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    await watcher(old_config, new_config)
                else:
                    watcher(old_config, new_config)
            except Exception as e:
                self.logger.error(f"Error in configuration watcher: {str(e)}")
    
    def register_watcher(self, watcher: callable):
        """
        Register a configuration change watcher.
        
        Args:
            watcher: Function to call when configuration changes
        """
        self.config_watchers.append(watcher)
        self.logger.debug(f"Configuration watcher registered: {watcher.__name__}")
    
    def get_config(self) -> SuppressionConfig:
        """
        Get current configuration.
        
        Returns:
            Current suppression configuration
        """
        if self.current_config is None:
            raise RuntimeError("Configuration not initialized")
        
        return self.current_config
    
    async def update_config_value(self, key: str, value: Any, source: ConfigSource = ConfigSource.DATABASE):
        """
        Update a single configuration value.
        
        Args:
            key: Configuration key to update
            value: New value
            source: Source of the update
        """
        if key not in self.config_values:
            raise ValueError(f"Unknown configuration key: {key}")
        
        # Update configuration value
        self.config_values[key] = ConfigValue(
            key=key,
            value=value,
            source=source,
            last_updated=datetime.now(timezone.utc),
            description=f"Updated via {source.value}"
        )
        
        # Rebuild configuration
        await self._build_current_config()
        
        self.config_updates.labels(key=key).inc()
        self.logger.info(f"Configuration value updated: {key} = {value} (source: {source.value})")
    
    async def reload_config(self):
        """Reload configuration from all sources."""
        self.logger.info("Reloading configuration from all sources")
        
        try:
            # Reload from file
            if self.config_file_path:
                await self._load_file_config(self.config_file_path)
            
            # Reload from environment
            await self._load_environment_config()
            
            # Rebuild configuration
            await self._build_current_config()
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.config_errors.labels(error_type='reload').inc()
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            raise
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get detailed configuration information.
        
        Returns:
            Configuration information including sources and metadata
        """
        config_info = {
            'current_config': self.current_config.to_dict() if self.current_config else None,
            'config_sources': {},
            'last_updated': None,
            'validation_status': 'valid' if self.current_config else 'invalid'
        }
        
        # Add source information
        latest_update = None
        for key, config_value in self.config_values.items():
            config_info['config_sources'][key] = {
                'value': config_value.value,
                'source': config_value.source.value,
                'last_updated': config_value.last_updated.isoformat(),
                'description': config_value.description
            }
            
            if latest_update is None or config_value.last_updated > latest_update:
                latest_update = config_value.last_updated
        
        if latest_update:
            config_info['last_updated'] = latest_update.isoformat()
        
        return config_info
    
    async def export_config(self, file_path: str, format_type: ConfigFormat = ConfigFormat.JSON):
        """
        Export current configuration to file.
        
        Args:
            file_path: File path to export to
            format_type: Export format
        """
        if not self.current_config:
            raise RuntimeError("No configuration to export")
        
        config_dict = self.current_config.to_dict()
        
        try:
            with open(file_path, 'w') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2, default=str)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False)
            
            self.logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {str(e)}")
            raise


class EnvironmentConfigBuilder:
    """
    Builder for environment-specific configurations.
    
    Provides pre-configured settings for different deployment environments
    with appropriate defaults and security settings.
    """
    
    @staticmethod
    def development_config() -> SuppressionConfig:
        """Development environment configuration."""
        return SuppressionConfig(
            environment="development",
            debug_enabled=True,
            log_level="DEBUG",
            session_timeout_hours=12,
            decision_timeout_seconds=60.0,
            detailed_metrics_enabled=True,
            circuit_breaker_failure_threshold=10,  # More lenient
            max_retry_attempts=5,
            health_check_interval_seconds=60
        )
    
    @staticmethod
    def testing_config() -> SuppressionConfig:
        """Testing environment configuration."""
        return SuppressionConfig(
            environment="testing",
            debug_enabled=False,
            log_level="INFO",
            session_timeout_hours=4,
            decision_timeout_seconds=30.0,
            detailed_metrics_enabled=True,
            circuit_breaker_failure_threshold=5,
            max_retry_attempts=3,
            health_check_interval_seconds=30,
            decision_log_retention_days=7,  # Shorter retention for testing
            acknowledgment_retention_days=7
        )
    
    @staticmethod
    def production_config() -> SuppressionConfig:
        """Production environment configuration."""
        return SuppressionConfig(
            environment="production",
            debug_enabled=False,
            log_level="WARNING",
            session_timeout_hours=8,
            decision_timeout_seconds=10.0,  # Strict timeout
            detailed_metrics_enabled=False,  # Reduce overhead
            circuit_breaker_failure_threshold=3,  # Fail fast
            max_retry_attempts=2,
            health_check_interval_seconds=15,  # Frequent health checks
            audit_all_decisions=True,
            require_justification_for_overrides=True,
            max_login_attempts_per_minute=3,  # Stricter rate limiting
            decision_log_retention_days=365,  # Longer retention for compliance
            override_retention_days=2555  # 7 years for compliance
        )