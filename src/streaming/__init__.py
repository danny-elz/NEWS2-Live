"""
Real-time stream processing package for NEWS2 Live system.

This package contains components for:
- Kafka configuration and management
- Faust stream processing applications
- Exactly-once processing semantics
- Stream monitoring and error handling
"""

# Conditional imports to avoid dependency issues
try:
    from .kafka_config import KafkaConfig, TopicConfig
    from .stream_processor import NEWS2StreamProcessor
    from .idempotency_manager import IdempotencyManager
    from .error_handler import StreamErrorHandler
    from .monitoring import StreamMonitor

    __all__ = [
        'KafkaConfig',
        'TopicConfig',
        'NEWS2StreamProcessor',
        'IdempotencyManager',
        'StreamErrorHandler',
        'StreamMonitor'
    ]
except ImportError:
    # If Kafka/Faust dependencies not available, provide limited functionality
    __all__ = []