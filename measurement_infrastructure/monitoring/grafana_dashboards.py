#!/usr/bin/env python3
"""
Grafana dashboard configurations for Epic 1 monitoring.

Creates comprehensive dashboards for:
- Stream processing performance
- NEWS2 calculation metrics
- System health monitoring
- Clinical validation tracking
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Configuration for Grafana dashboard."""
    title: str
    description: str
    panels: List[Dict[str, Any]]
    refresh: str = "30s"
    time_range: Dict[str, str] = None
    
    def __post_init__(self):
        if self.time_range is None:
            self.time_range = {"from": "now-1h", "to": "now"}


class GrafanaDashboardGenerator:
    """Generates Grafana dashboards for Epic 1 monitoring."""
    
    def __init__(self):
        self.dashboard_configs = []
    
    def create_stream_processing_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive stream processing performance dashboard."""
        
        panels = [
            # Row 1: Throughput and Latency
            {
                "id": 1,
                "title": "Events Processed Per Second",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                "targets": [{
                    "expr": "rate(stream_events_processed_total[5m])",
                    "legendFormat": "Events/sec"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 800},
                                {"color": "green", "value": 1000}
                            ]
                        },
                        "unit": "reqps"
                    }
                }
            },
            {
                "id": 2,
                "title": "Processing Latency P95",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(stream_processing_duration_seconds_bucket[5m])) * 1000",
                    "legendFormat": "P95 Latency"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 50},
                                {"color": "red", "value": 100}
                            ]
                        },
                        "unit": "ms"
                    }
                }
            },
            {
                "id": 3,
                "title": "Error Rate",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                "targets": [{
                    "expr": "rate(stream_events_failed_total[5m]) / rate(stream_events_processed_total[5m]) * 100",
                    "legendFormat": "Error Rate"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 1},
                                {"color": "red", "value": 5}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            },
            {
                "id": 4,
                "title": "Kafka Consumer Lag",
                "type": "stat",
                "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                "targets": [{
                    "expr": "kafka_consumer_lag_seconds",
                    "legendFormat": "Lag"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 5},
                                {"color": "red", "value": 30}
                            ]
                        },
                        "unit": "s"
                    }
                }
            },
            
            # Row 2: Time series charts
            {
                "id": 5,
                "title": "Throughput Over Time",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "rate(stream_events_processed_total[5m])",
                        "legendFormat": "Events Processed"
                    },
                    {
                        "expr": "rate(stream_events_failed_total[5m])",
                        "legendFormat": "Events Failed"
                    }
                ],
                "yAxes": [
                    {"label": "Events/second", "min": 0},
                    {"show": False}
                ],
                "legend": {"show": True, "values": True, "current": True}
            },
            {
                "id": 6,
                "title": "Latency Percentiles",
                "type": "graph", 
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.50, rate(stream_processing_duration_seconds_bucket[5m])) * 1000",
                        "legendFormat": "P50"
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(stream_processing_duration_seconds_bucket[5m])) * 1000",
                        "legendFormat": "P95"
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(stream_processing_duration_seconds_bucket[5m])) * 1000",
                        "legendFormat": "P99"
                    }
                ],
                "yAxes": [
                    {"label": "Latency (ms)", "min": 0},
                    {"show": False}
                ]
            },
            
            # Row 3: System Resources
            {
                "id": 7,
                "title": "CPU Usage",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16},
                "targets": [{
                    "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                    "legendFormat": "CPU Usage"
                }],
                "yAxes": [
                    {"label": "Percent", "min": 0, "max": 100},
                    {"show": False}
                ]
            },
            {
                "id": 8,
                "title": "Memory Usage",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16},
                "targets": [{
                    "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                    "legendFormat": "Memory Usage"
                }],
                "yAxes": [
                    {"label": "Percent", "min": 0, "max": 100},
                    {"show": False}
                ]
            },
            {
                "id": 9,
                "title": "Network I/O",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16},
                "targets": [
                    {
                        "expr": "rate(node_network_receive_bytes_total[5m])",
                        "legendFormat": "RX"
                    },
                    {
                        "expr": "rate(node_network_transmit_bytes_total[5m])",
                        "legendFormat": "TX"
                    }
                ],
                "yAxes": [
                    {"label": "Bytes/sec", "min": 0},
                    {"show": False}
                ]
            }
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Epic 1 - Stream Processing Performance",
                "description": "Real-time monitoring of NEWS2 stream processing performance",
                "tags": ["epic1", "stream-processing", "performance"],
                "timezone": "browser",
                "panels": panels,
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_news2_clinical_dashboard(self) -> Dict[str, Any]:
        """Create NEWS2 clinical metrics dashboard."""
        
        panels = [
            # Row 1: Clinical Metrics
            {
                "id": 1,
                "title": "NEWS2 Score Distribution",
                "type": "histogram",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [{
                    "expr": "news2_scores_calculated",
                    "legendFormat": "NEWS2 Scores"
                }],
                "options": {
                    "bucketSize": 1,
                    "bucketOffset": 0
                }
            },
            {
                "id": 2,
                "title": "Risk Category Distribution",
                "type": "piechart",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [{
                    "expr": "increase(news2_scores_calculated[5m])",
                    "legendFormat": "{{risk_category}}"
                }]
            },
            
            # Row 2: COPD Patient Metrics
            {
                "id": 3,
                "title": "COPD vs Standard Patients",
                "type": "stat",
                "gridPos": {"h": 6, "w": 6, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "sum(rate(copd_patients_processed_total[5m]))",
                        "legendFormat": "COPD Patients"
                    },
                    {
                        "expr": "sum(rate(stream_events_processed_total{patient_type=\"standard\"}[5m]))",
                        "legendFormat": "Standard Patients"
                    }
                ]
            },
            {
                "id": 4,
                "title": "Scale Usage",
                "type": "stat",
                "gridPos": {"h": 6, "w": 6, "x": 6, "y": 8},
                "targets": [
                    {
                        "expr": "sum(rate(news2_scores_calculated{scale_used=\"1\"}[5m]))",
                        "legendFormat": "Scale 1 (Standard)"
                    },
                    {
                        "expr": "sum(rate(news2_scores_calculated{scale_used=\"2\"}[5m]))",
                        "legendFormat": "Scale 2 (COPD)"
                    }
                ]
            },
            
            # Row 3: Alerts and Red Flags
            {
                "id": 5,
                "title": "High Risk Alerts",
                "type": "table",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [{
                    "expr": "increase(news2_scores_calculated{risk_category=\"HIGH\"}[5m])",
                    "format": "table",
                    "instant": True
                }],
                "transformations": [
                    {
                        "id": "organize",
                        "options": {
                            "excludeByName": {},
                            "indexByName": {},
                            "renameByName": {
                                "Value": "Count",
                                "risk_category": "Risk Category",
                                "scale_used": "Scale Used"
                            }
                        }
                    }
                ]
            },
            
            # Row 4: Trending Analysis
            {
                "id": 6,
                "title": "Average NEWS2 Score Trends",
                "type": "graph",
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                "targets": [
                    {
                        "expr": "avg(news2_scores_calculated{scale_used=\"1\"})",
                        "legendFormat": "Average Score (Scale 1)"
                    },
                    {
                        "expr": "avg(news2_scores_calculated{scale_used=\"2\"})",
                        "legendFormat": "Average Score (Scale 2)"
                    }
                ],
                "yAxes": [
                    {"label": "NEWS2 Score", "min": 0, "max": 20},
                    {"show": False}
                ]
            }
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Epic 1 - NEWS2 Clinical Metrics",
                "description": "Clinical validation and NEWS2 calculation metrics",
                "tags": ["epic1", "news2", "clinical"],
                "timezone": "browser",
                "panels": panels,
                "time": {"from": "now-4h", "to": "now"},
                "refresh": "1m",
                "schemaVersion": 27,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_system_health_dashboard(self) -> Dict[str, Any]:
        """Create system health monitoring dashboard."""
        
        panels = [
            # Row 1: Service Health
            {
                "id": 1,
                "title": "Service Health Status",
                "type": "stat",
                "gridPos": {"h": 6, "w": 4, "x": 0, "y": 0},
                "targets": [
                    {"expr": "up{job=\"kafka\"}", "legendFormat": "Kafka"},
                    {"expr": "up{job=\"redis\"}", "legendFormat": "Redis"},
                    {"expr": "up{job=\"stream-processor\"}", "legendFormat": "Stream Processor"},
                    {"expr": "up{job=\"postgres\"}", "legendFormat": "PostgreSQL"}
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [
                            {"options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}, "type": "value"}
                        ],
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "green", "value": 1}
                            ]
                        }
                    }
                }
            },
            
            # Circuit Breaker States
            {
                "id": 2,
                "title": "Circuit Breaker States",
                "type": "stat",
                "gridPos": {"h": 6, "w": 8, "x": 4, "y": 0},
                "targets": [{
                    "expr": "circuit_breaker_state",
                    "legendFormat": "{{service_name}}"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [
                            {"options": {"0": {"text": "CLOSED"}, "1": {"text": "HALF-OPEN"}, "2": {"text": "OPEN"}}, "type": "value"}
                        ],
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 1},
                                {"color": "red", "value": 2}
                            ]
                        }
                    }
                }
            },
            
            # Redis Connections
            {
                "id": 3,
                "title": "Redis Active Connections",
                "type": "stat",
                "gridPos": {"h": 6, "w": 4, "x": 12, "y": 0},
                "targets": [{
                    "expr": "redis_connections_active",
                    "legendFormat": "Active Connections"
                }]
            },
            
            # Duplicate Detection
            {
                "id": 4,
                "title": "Duplicate Events Detected",
                "type": "stat",
                "gridPos": {"h": 6, "w": 8, "x": 16, "y": 0},
                "targets": [{
                    "expr": "increase(duplicate_events_detected_total[5m])",
                    "legendFormat": "Duplicates"
                }],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 10},
                                {"color": "red", "value": 100}
                            ]
                        }
                    }
                }
            },
            
            # Row 2: Error Analysis
            {
                "id": 5,
                "title": "Error Types Over Time",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                "targets": [
                    {
                        "expr": "rate(stream_events_failed_total{error_type=\"validation_error\"}[5m])",
                        "legendFormat": "Validation Errors"
                    },
                    {
                        "expr": "rate(stream_events_failed_total{error_type=\"calculation_error\"}[5m])",
                        "legendFormat": "Calculation Errors"
                    },
                    {
                        "expr": "rate(stream_events_failed_total{error_type=\"database_error\"}[5m])",
                        "legendFormat": "Database Errors"
                    },
                    {
                        "expr": "rate(stream_events_failed_total{error_type=\"network_error\"}[5m])",
                        "legendFormat": "Network Errors"
                    }
                ],
                "yAxes": [
                    {"label": "Errors/second", "min": 0},
                    {"show": False}
                ]
            },
            {
                "id": 6,
                "title": "Retry Attempts",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                "targets": [
                    {
                        "expr": "rate(stream_events_failed_total{retry_attempt=\"1\"}[5m])",
                        "legendFormat": "First Attempt"
                    },
                    {
                        "expr": "rate(stream_events_failed_total{retry_attempt=\"2\"}[5m])",
                        "legendFormat": "Second Attempt"
                    },
                    {
                        "expr": "rate(stream_events_failed_total{retry_attempt=\"3\"}[5m])",
                        "legendFormat": "Third Attempt"
                    }
                ],
                "yAxes": [
                    {"label": "Retries/second", "min": 0},
                    {"show": False}
                ]
            },
            
            # Row 3: Performance Impact
            {
                "id": 7,
                "title": "JVM Heap Usage",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 14},
                "targets": [{
                    "expr": "jvm_memory_bytes_used{area=\"heap\"} / jvm_memory_bytes_max{area=\"heap\"} * 100",
                    "legendFormat": "Heap Usage %"
                }],
                "yAxes": [
                    {"label": "Percent", "min": 0, "max": 100},
                    {"show": False}
                ]
            },
            {
                "id": 8,
                "title": "Garbage Collection",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 14},
                "targets": [
                    {
                        "expr": "rate(jvm_gc_collection_seconds_sum[5m])",
                        "legendFormat": "GC Time"
                    },
                    {
                        "expr": "rate(jvm_gc_collection_seconds_count[5m])",
                        "legendFormat": "GC Collections"
                    }
                ]
            },
            {
                "id": 9,
                "title": "Disk I/O",
                "type": "graph",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 14},
                "targets": [
                    {
                        "expr": "rate(node_disk_read_bytes_total[5m])",
                        "legendFormat": "Read"
                    },
                    {
                        "expr": "rate(node_disk_written_bytes_total[5m])",
                        "legendFormat": "Write"
                    }
                ],
                "yAxes": [
                    {"label": "Bytes/sec", "min": 0},
                    {"show": False}
                ]
            }
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Epic 1 - System Health",
                "description": "System health monitoring and error tracking",
                "tags": ["epic1", "system", "health"],
                "timezone": "browser",
                "panels": panels,
                "time": {"from": "now-2h", "to": "now"},
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def generate_all_dashboards(self) -> Dict[str, Any]:
        """Generate all Epic 1 monitoring dashboards."""
        dashboards = {
            "stream_processing": self.create_stream_processing_dashboard(),
            "news2_clinical": self.create_news2_clinical_dashboard(),
            "system_health": self.create_system_health_dashboard()
        }
        
        return dashboards
    
    def save_dashboards_to_files(self, output_dir: str = "measurement_infrastructure/monitoring"):
        """Save dashboard configurations to JSON files."""
        dashboards = self.generate_all_dashboards()
        
        for name, dashboard in dashboards.items():
            filename = f"{output_dir}/dashboard_{name}.json"
            with open(filename, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            print(f"Dashboard saved: {filename}")
        
        # Create docker-compose for monitoring stack
        self.create_monitoring_stack_compose(output_dir)
    
    def create_monitoring_stack_compose(self, output_dir: str):
        """Create docker-compose file for monitoring stack."""
        compose_content = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
      - ./datasources:/etc/grafana/provisioning/datasources
    networks:
      - monitoring
    depends_on:
      - prometheus

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    networks:
      - monitoring

  kafka-exporter:
    image: danielqsj/kafka-exporter:latest
    container_name: kafka-exporter
    ports:
      - "9308:9308"
    command:
      - '--kafka.server=kafka:9092'
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
"""
        
        with open(f"{output_dir}/docker-compose-monitoring.yml", 'w') as f:
            f.write(compose_content.strip())
        
        # Create Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'kafka-exporter'
    static_configs:
      - targets: ['kafka-exporter:9308']

  - job_name: 'stream-processor'
    static_configs:
      - targets: ['stream-processor:8000']
    metrics_path: '/metrics'

  - job_name: 'news2-api'
    static_configs:
      - targets: ['news2-api:8080']
    metrics_path: '/metrics'
"""
        
        with open(f"{output_dir}/prometheus.yml", 'w') as f:
            f.write(prometheus_config.strip())
        
        print(f"Monitoring stack configuration saved to: {output_dir}/docker-compose-monitoring.yml")


if __name__ == "__main__":
    generator = GrafanaDashboardGenerator()
    generator.save_dashboards_to_files()
    
    print("\n" + "="*60)
    print("GRAFANA DASHBOARDS GENERATED")
    print("="*60)
    print("\nTo start monitoring stack:")
    print("cd measurement_infrastructure/monitoring")
    print("docker-compose -f docker-compose-monitoring.yml up -d")
    print("\nAccess Grafana at: http://localhost:3000")
    print("Username: admin, Password: admin")
    print("\nDashboards available:")
    print("- Epic 1 - Stream Processing Performance")
    print("- Epic 1 - NEWS2 Clinical Metrics") 
    print("- Epic 1 - System Health")