"""
Kafka cluster configuration and topic management for NEWS2 streaming.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging

logger = logging.getLogger(__name__)


@dataclass
class TopicConfig:
    """Configuration for a Kafka topic."""
    name: str
    partitions: int
    replication_factor: int
    retention_ms: int
    cleanup_policy: str = "delete"
    compression_type: str = "gzip"
    max_message_bytes: int = 1048576  # 1MB


class KafkaConfig:
    """Kafka cluster configuration and management."""
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = None
        self._producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'key_serializer': lambda x: x.encode('utf-8') if x else None,
            'value_serializer': lambda x: x,
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 5,
            'buffer_memory': 33554432,
            'compression_type': 'gzip'
        }
        
        self._consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
            'value_deserializer': lambda x: x,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,
            'group_id': None,
            'max_poll_records': 500,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 10000
        }
    
    async def initialize(self) -> None:
        """Initialize Kafka admin client and create required topics."""
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers,
            client_id='news2_admin'
        )
        
        # Define required topics
        topics = [
            TopicConfig(
                name='vital_signs_input',
                partitions=10,
                replication_factor=3,
                retention_ms=86400000,  # 24 hours
                cleanup_policy='delete'
            ),
            TopicConfig(
                name='news2_results',
                partitions=10,
                replication_factor=3,
                retention_ms=604800000,  # 7 days
                cleanup_policy='delete'
            ),
            TopicConfig(
                name='news2_alerts',
                partitions=5,
                replication_factor=3,
                retention_ms=2592000000,  # 30 days
                cleanup_policy='delete'
            ),
            TopicConfig(
                name='stream_errors',
                partitions=3,
                replication_factor=3,
                retention_ms=604800000,  # 7 days
                cleanup_policy='delete'
            ),
            TopicConfig(
                name='dead_letter_queue',
                partitions=3,
                replication_factor=3,
                retention_ms=2592000000,  # 30 days
                cleanup_policy='delete'
            )
        ]
        
        await self.create_topics(topics)
        logger.info("Kafka configuration initialized successfully")
    
    async def create_topics(self, topics: List[TopicConfig]) -> None:
        """Create topics if they don't exist."""
        new_topics = []
        
        for topic_config in topics:
            topic_configs = {
                'cleanup.policy': topic_config.cleanup_policy,
                'retention.ms': str(topic_config.retention_ms),
                'compression.type': topic_config.compression_type,
                'max.message.bytes': str(topic_config.max_message_bytes)
            }
            
            new_topic = NewTopic(
                name=topic_config.name,
                num_partitions=topic_config.partitions,
                replication_factor=topic_config.replication_factor,
                topic_configs=topic_configs
            )
            
            new_topics.append(new_topic)
        
        try:
            result = self.admin_client.create_topics(new_topics, validate_only=False)
            
            # Wait for topic creation to complete
            for topic, future in result.items():
                try:
                    future.result()
                    logger.info(f"Successfully created topic: {topic}")
                except TopicAlreadyExistsError:
                    logger.info(f"Topic already exists: {topic}")
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
            raise
    
    def get_producer(self, **kwargs) -> KafkaProducer:
        """Get configured Kafka producer."""
        config = self._producer_config.copy()
        config.update(kwargs)
        return KafkaProducer(**config)
    
    def get_consumer(self, topics: List[str], group_id: str, **kwargs) -> KafkaConsumer:
        """Get configured Kafka consumer."""
        config = self._consumer_config.copy()
        config['group_id'] = group_id
        config.update(kwargs)
        return KafkaConsumer(*topics, **config)
    
    async def get_topic_metadata(self, topic_name: str) -> Dict[str, Any]:
        """Get metadata for a specific topic."""
        try:
            metadata = self.admin_client.describe_topics([topic_name])
            return metadata.get(topic_name, {})
        except Exception as e:
            logger.error(f"Error getting metadata for topic {topic_name}: {e}")
            raise
    
    async def get_consumer_group_info(self, group_id: str) -> Dict[str, Any]:
        """Get information about a consumer group."""
        try:
            groups = self.admin_client.describe_consumer_groups([group_id])
            return groups.get(group_id, {})
        except Exception as e:
            logger.error(f"Error getting consumer group info for {group_id}: {e}")
            raise
    
    async def list_topics(self) -> List[str]:
        """List all available topics."""
        try:
            metadata = self.admin_client.list_topics()
            return list(metadata.topics.keys())
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            raise
    
    async def delete_topics(self, topic_names: List[str]) -> None:
        """Delete specified topics (use with caution)."""
        try:
            result = self.admin_client.delete_topics(topic_names)
            
            for topic, future in result.items():
                try:
                    future.result()
                    logger.info(f"Successfully deleted topic: {topic}")
                except Exception as e:
                    logger.error(f"Failed to delete topic {topic}: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error deleting topics: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Kafka cluster is healthy."""
        try:
            # Try to list topics as a health check
            topics = await self.list_topics()
            return len(topics) >= 0  # Even 0 topics means cluster is reachable
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
            return False
    
    def close(self) -> None:
        """Close admin client connection."""
        if self.admin_client:
            self.admin_client.close()
            logger.info("Kafka admin client closed")