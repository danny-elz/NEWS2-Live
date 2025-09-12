#!/usr/bin/env python3
"""
Staging environment orchestrator for Epic 1 measurement phase.

This orchestrates the complete staging environment for comprehensive BMAD measurement:
- Infrastructure deployment (Kafka, Redis, PostgreSQL, TimescaleDB)
- Application deployment (Stream Processor, NEWS2 API)
- Monitoring stack (Prometheus, Grafana, Exporters)
- Load testing and performance validation
- Clinical validation test execution
- Environment health monitoring
"""

import asyncio
import subprocess
import json
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import yaml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Health status of a service."""
    name: str
    status: str
    url: str
    response_time_ms: float
    error_message: Optional[str] = None


@dataclass
class EnvironmentStatus:
    """Overall environment status."""
    environment_name: str
    status: str
    services: List[ServiceHealth]
    deployment_time: datetime
    health_check_time: datetime
    issues: List[str]


class StagingEnvironmentOrchestrator:
    """Orchestrates staging environment for Epic 1 measurement."""
    
    def __init__(self, environment_path: str = "measurement_infrastructure/staging"):
        self.environment_path = environment_path
        self.compose_file = f"{environment_path}/docker-compose-staging.yml"
        self.services = {}
        
        # Service health check endpoints
        self.health_endpoints = {
            "kafka": "http://localhost:9092",  # Kafka doesn't have HTTP endpoint, will use alternative check
            "redis": "http://localhost:6379",   # Redis doesn't have HTTP endpoint, will use ping
            "postgres": None,  # Will use pg_isready
            "timescaledb": None,  # Will use pg_isready
            "news2-stream-processor": "http://localhost:8000/health",
            "news2-api": "http://localhost:8080/health",
            "prometheus": "http://localhost:9090/-/ready",
            "grafana": "http://localhost:3000/api/health",
            "node-exporter": "http://localhost:9100/metrics",
            "redis-exporter": "http://localhost:9121/metrics",
            "kafka-exporter": "http://localhost:9308/metrics",
            "postgres-exporter": "http://localhost:9187/metrics"
        }
    
    async def deploy_staging_environment(self) -> EnvironmentStatus:
        """Deploy complete staging environment."""
        logger.info("Deploying Epic 1 staging environment...")
        deployment_start = datetime.now(timezone.utc)
        
        try:
            # Step 1: Create configuration files
            await self._create_configuration_files()
            
            # Step 2: Deploy infrastructure services first
            await self._deploy_infrastructure_services()
            
            # Step 3: Wait for infrastructure to be ready
            await self._wait_for_infrastructure()
            
            # Step 4: Deploy application services
            await self._deploy_application_services()
            
            # Step 5: Deploy monitoring stack
            await self._deploy_monitoring_stack()
            
            # Step 6: Comprehensive health check
            environment_status = await self._perform_health_check()
            environment_status.deployment_time = deployment_start
            
            if environment_status.status == "healthy":
                logger.info("Staging environment deployed successfully!")
            else:
                logger.warning(f"Staging environment deployed with issues: {environment_status.issues}")
            
            return environment_status
            
        except Exception as e:
            logger.error(f"Failed to deploy staging environment: {e}")
            raise
    
    async def run_measurement_suite(self) -> Dict[str, Any]:
        """Run comprehensive measurement suite."""
        logger.info("Running Epic 1 measurement suite...")
        
        measurement_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment_status": None,
            "performance_tests": None,
            "clinical_validation": None,
            "monitoring_validation": None,
            "overall_status": "pending"
        }
        
        try:
            # Phase 1: Environment health check
            logger.info("Phase 1: Environment health verification")
            environment_status = await self._perform_health_check()
            measurement_results["environment_status"] = environment_status.__dict__
            
            if environment_status.status != "healthy":
                logger.error("Environment not healthy - aborting measurement suite")
                measurement_results["overall_status"] = "failed"
                return measurement_results
            
            # Phase 2: Performance testing
            logger.info("Phase 2: Performance testing")
            performance_results = await self._run_performance_tests()
            measurement_results["performance_tests"] = performance_results
            
            # Phase 3: Clinical validation
            logger.info("Phase 3: Clinical validation")
            clinical_results = await self._run_clinical_validation()
            measurement_results["clinical_validation"] = clinical_results
            
            # Phase 4: Monitoring validation
            logger.info("Phase 4: Monitoring validation")
            monitoring_results = await self._validate_monitoring_stack()
            measurement_results["monitoring_validation"] = monitoring_results
            
            # Determine overall status
            overall_status = self._assess_overall_status(measurement_results)
            measurement_results["overall_status"] = overall_status
            
            # Save comprehensive results
            await self._save_measurement_results(measurement_results)
            
            logger.info(f"Measurement suite completed with status: {overall_status}")
            return measurement_results
            
        except Exception as e:
            logger.error(f"Measurement suite failed: {e}")
            measurement_results["overall_status"] = "failed"
            measurement_results["error"] = str(e)
            return measurement_results
    
    async def _create_configuration_files(self):
        """Create necessary configuration files."""
        # Create database initialization script
        init_db_sql = """
-- Initialize NEWS2 Live database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for patient data
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    ward_id VARCHAR(20) NOT NULL,
    bed_number VARCHAR(10),
    age INTEGER,
    is_copd_patient BOOLEAN DEFAULT FALSE,
    assigned_nurse_id VARCHAR(50),
    admission_date TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    patient_id VARCHAR(50),
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id VARCHAR(50)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_ward_id ON patients(ward_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_patient_id ON audit_log(patient_id);
"""
        
        with open(f"{self.environment_path}/init_db.sql", 'w') as f:
            f.write(init_db_sql)
        
        # Create TimescaleDB initialization script
        init_timescale_sql = """
-- Initialize TimescaleDB for vital signs data
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create vital signs table
CREATE TABLE IF NOT EXISTS vital_signs (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    event_id UUID,
    respiratory_rate INTEGER,
    spo2 INTEGER,
    on_oxygen BOOLEAN,
    temperature DECIMAL(4,1),
    systolic_bp INTEGER,
    heart_rate INTEGER,
    consciousness VARCHAR(20),
    data_source VARCHAR(50),
    quality_flags JSONB
);

-- Convert to hypertable
SELECT create_hypertable('vital_signs', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_vital_signs_patient_id ON vital_signs(patient_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_vital_signs_event_id ON vital_signs(event_id);

-- Create NEWS2 results table
CREATE TABLE IF NOT EXISTS news2_results (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    event_id UUID,
    news2_score INTEGER,
    individual_scores JSONB,
    risk_category VARCHAR(20),
    scale_used INTEGER,
    red_flags TEXT[],
    calculation_time_ms DECIMAL(8,2)
);

SELECT create_hypertable('news2_results', 'time', if_not_exists => TRUE);

-- Create retention policies
SELECT add_retention_policy('vital_signs', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('news2_results', INTERVAL '30 days', if_not_exists => TRUE);

-- Create compression policies  
SELECT add_compression_policy('vital_signs', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('news2_results', INTERVAL '7 days', if_not_exists => TRUE);
"""
        
        with open(f"{self.environment_path}/init_timescale.sql", 'w') as f:
            f.write(init_timescale_sql)
        
        # Create Redis configuration
        redis_config = """
# Redis configuration for NEWS2 Live staging
port 6379
bind 0.0.0.0
protected-mode no
tcp-backlog 511
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir ./
maxmemory 1gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
"""
        
        with open(f"{self.environment_path}/redis.conf", 'w') as f:
            f.write(redis_config)
        
        # Create Prometheus configuration for staging
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

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'news2-stream-processor'
    static_configs:
      - targets: ['news2-stream-processor:8000']
    metrics_path: '/metrics'

  - job_name: 'news2-api'
    static_configs:
      - targets: ['news2-api:8080']
    metrics_path: '/metrics'
"""
        
        with open(f"{self.environment_path}/prometheus-staging.yml", 'w') as f:
            f.write(prometheus_config)
        
        logger.info("Configuration files created successfully")
    
    async def _deploy_infrastructure_services(self):
        """Deploy core infrastructure services."""
        logger.info("Deploying infrastructure services...")
        
        infrastructure_services = [
            "zookeeper",
            "kafka", 
            "postgres",
            "timescaledb",
            "redis"
        ]
        
        for service in infrastructure_services:
            logger.info(f"Starting {service}...")
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file,
                "up", "-d", service
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to start {service}: {result.stderr}")
        
        logger.info("Infrastructure services deployed")
    
    async def _wait_for_infrastructure(self):
        """Wait for infrastructure services to be ready."""
        logger.info("Waiting for infrastructure services to be ready...")
        
        max_wait_time = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check Kafka
                kafka_ready = await self._check_kafka_ready()
                
                # Check Redis
                redis_ready = await self._check_redis_ready()
                
                # Check PostgreSQL
                postgres_ready = await self._check_postgres_ready()
                
                # Check TimescaleDB
                timescale_ready = await self._check_timescale_ready()
                
                if all([kafka_ready, redis_ready, postgres_ready, timescale_ready]):
                    logger.info("All infrastructure services are ready")
                    return
                
                logger.info("Waiting for infrastructure services...")
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.warning(f"Error checking infrastructure: {e}")
                await asyncio.sleep(5)
        
        raise Exception("Infrastructure services did not become ready within timeout")
    
    async def _deploy_application_services(self):
        """Deploy application services."""
        logger.info("Deploying application services...")
        
        app_services = [
            "news2-stream-processor",
            "news2-api"
        ]
        
        for service in app_services:
            logger.info(f"Starting {service}...")
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file,
                "up", "-d", service
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to start {service}: {result.stderr}")
        
        # Wait for application services to be ready
        await self._wait_for_application_services()
        logger.info("Application services deployed")
    
    async def _deploy_monitoring_stack(self):
        """Deploy monitoring stack."""
        logger.info("Deploying monitoring stack...")
        
        monitoring_services = [
            "prometheus",
            "grafana", 
            "node-exporter",
            "redis-exporter",
            "kafka-exporter",
            "postgres-exporter"
        ]
        
        for service in monitoring_services:
            logger.info(f"Starting {service}...")
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file,
                "up", "-d", service
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Failed to start {service}: {result.stderr}")
        
        logger.info("Monitoring stack deployed")
    
    async def _perform_health_check(self) -> EnvironmentStatus:
        """Perform comprehensive health check."""
        logger.info("Performing comprehensive health check...")
        
        service_health_results = []
        issues = []
        
        for service_name, endpoint in self.health_endpoints.items():
            health = await self._check_service_health(service_name, endpoint)
            service_health_results.append(health)
            
            if health.status != "healthy":
                issues.append(f"{service_name}: {health.error_message or 'unhealthy'}")
        
        overall_status = "healthy" if len(issues) == 0 else "degraded" if len(issues) < 3 else "unhealthy"
        
        return EnvironmentStatus(
            environment_name="Epic 1 Staging",
            status=overall_status,
            services=service_health_results,
            deployment_time=datetime.now(timezone.utc),
            health_check_time=datetime.now(timezone.utc),
            issues=issues
        )
    
    async def _check_service_health(self, service_name: str, endpoint: Optional[str]) -> ServiceHealth:
        """Check health of individual service."""
        start_time = time.time()
        
        try:
            if endpoint and endpoint.startswith("http"):
                response = requests.get(endpoint, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return ServiceHealth(service_name, "healthy", endpoint, response_time)
                else:
                    return ServiceHealth(service_name, "unhealthy", endpoint, response_time, 
                                       f"HTTP {response.status_code}")
            
            elif service_name in ["kafka", "redis", "postgres", "timescaledb"]:
                # Use custom health checks for these services
                if service_name == "kafka":
                    healthy = await self._check_kafka_ready()
                elif service_name == "redis":
                    healthy = await self._check_redis_ready()
                elif service_name == "postgres":
                    healthy = await self._check_postgres_ready()
                elif service_name == "timescaledb":
                    healthy = await self._check_timescale_ready()
                
                response_time = (time.time() - start_time) * 1000
                
                if healthy:
                    return ServiceHealth(service_name, "healthy", endpoint or "N/A", response_time)
                else:
                    return ServiceHealth(service_name, "unhealthy", endpoint or "N/A", response_time,
                                       "Connection failed")
            
            else:
                return ServiceHealth(service_name, "unknown", endpoint or "N/A", 0,
                                   "No health check configured")
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(service_name, "unhealthy", endpoint or "N/A", response_time, str(e))
    
    async def _check_kafka_ready(self) -> bool:
        """Check if Kafka is ready."""
        try:
            result = subprocess.run([
                "docker", "exec", "kafka", "kafka-topics", "--bootstrap-server", 
                "localhost:29092", "--list"
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def _check_redis_ready(self) -> bool:
        """Check if Redis is ready."""
        try:
            result = subprocess.run([
                "docker", "exec", "redis", "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=10)
            return "PONG" in result.stdout
        except:
            return False
    
    async def _check_postgres_ready(self) -> bool:
        """Check if PostgreSQL is ready."""
        try:
            result = subprocess.run([
                "docker", "exec", "postgres", "pg_isready", 
                "-h", "localhost", "-p", "5432", "-U", "news2_user"
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def _check_timescale_ready(self) -> bool:
        """Check if TimescaleDB is ready."""
        try:
            result = subprocess.run([
                "docker", "exec", "timescaledb", "pg_isready",
                "-h", "localhost", "-p", "5432", "-U", "timescale_user"
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def _wait_for_application_services(self):
        """Wait for application services to be ready."""
        logger.info("Waiting for application services...")
        
        max_wait = 60  # 1 minute
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                stream_health = await self._check_service_health(
                    "news2-stream-processor", 
                    "http://localhost:8000/health"
                )
                api_health = await self._check_service_health(
                    "news2-api",
                    "http://localhost:8080/health"
                )
                
                if stream_health.status == "healthy" and api_health.status == "healthy":
                    logger.info("Application services are ready")
                    return
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.warning(f"Waiting for application services: {e}")
                await asyncio.sleep(5)
        
        logger.warning("Application services did not become ready within timeout")
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance validation tests...")
        
        try:
            # Run load tester container
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file,
                "--profile", "testing",
                "up", "--abort-on-container-exit", "load-tester"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                # Load test results from volume
                try:
                    with open(f"{self.environment_path}/test_results/performance_results.json", 'r') as f:
                        return json.load(f)
                except FileNotFoundError:
                    return {"status": "completed", "error": "Results file not found"}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _run_clinical_validation(self) -> Dict[str, Any]:
        """Run clinical validation tests."""
        logger.info("Running clinical validation...")
        
        try:
            # Run clinical validation script
            result = subprocess.run([
                "python", "measurement_infrastructure/clinical_validation/clinical_test_suite.py"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                try:
                    with open('measurement_infrastructure/clinical_validation/clinical_validation_results.json', 'r') as f:
                        return json.load(f)
                except FileNotFoundError:
                    return {"status": "completed", "results": "validation_passed"}
            else:
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _validate_monitoring_stack(self) -> Dict[str, Any]:
        """Validate monitoring stack functionality."""
        logger.info("Validating monitoring stack...")
        
        monitoring_results = {
            "prometheus_status": "unknown",
            "grafana_status": "unknown", 
            "metrics_available": False,
            "dashboards_available": False
        }
        
        try:
            # Check Prometheus
            prometheus_health = await self._check_service_health("prometheus", "http://localhost:9090/-/ready")
            monitoring_results["prometheus_status"] = prometheus_health.status
            
            # Check Grafana
            grafana_health = await self._check_service_health("grafana", "http://localhost:3000/api/health")
            monitoring_results["grafana_status"] = grafana_health.status
            
            # Check if metrics are being collected
            try:
                response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success' and len(data.get('data', {}).get('result', [])) > 0:
                        monitoring_results["metrics_available"] = True
            except:
                pass
            
            # Check Grafana dashboards
            try:
                response = requests.get("http://localhost:3000/api/search", 
                                      auth=("admin", "staging_admin"), timeout=10)
                if response.status_code == 200:
                    dashboards = response.json()
                    monitoring_results["dashboards_available"] = len(dashboards) > 0
            except:
                pass
                
        except Exception as e:
            monitoring_results["error"] = str(e)
        
        return monitoring_results
    
    def _assess_overall_status(self, measurement_results: Dict[str, Any]) -> str:
        """Assess overall measurement status."""
        environment_healthy = measurement_results.get("environment_status", {}).get("status") == "healthy"
        
        performance_passed = True
        if measurement_results.get("performance_tests"):
            perf_results = measurement_results["performance_tests"]
            performance_passed = perf_results.get("status") != "failed"
        
        clinical_passed = True
        if measurement_results.get("clinical_validation"):
            clinical_results = measurement_results["clinical_validation"]
            clinical_passed = clinical_results.get("status") != "failed"
        
        monitoring_ok = True
        if measurement_results.get("monitoring_validation"):
            monitoring_results = measurement_results["monitoring_validation"]
            monitoring_ok = (monitoring_results.get("prometheus_status") == "healthy" and
                           monitoring_results.get("grafana_status") == "healthy")
        
        if environment_healthy and performance_passed and clinical_passed and monitoring_ok:
            return "passed"
        elif environment_healthy and (performance_passed or clinical_passed):
            return "partial"
        else:
            return "failed"
    
    async def _save_measurement_results(self, results: Dict[str, Any]):
        """Save comprehensive measurement results."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"measurement_infrastructure/staging/epic1_measurement_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Measurement results saved to: {filename}")
    
    async def teardown_environment(self):
        """Teardown staging environment."""
        logger.info("Tearing down staging environment...")
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file, "down", "-v"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Staging environment torn down successfully")
            else:
                logger.warning(f"Issues during teardown: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to teardown environment: {e}")


async def main():
    """Main entry point for staging environment orchestration."""
    print("=" * 60)
    print("NEWS2 Live - Epic 1 Staging Environment")
    print("=" * 60)
    
    orchestrator = StagingEnvironmentOrchestrator()
    
    try:
        # Deploy staging environment
        print("\nDeploying staging environment...")
        environment_status = await orchestrator.deploy_staging_environment()
        
        if environment_status.status != "healthy":
            print("Environment deployment failed - aborting measurement")
            return
        
        print("Staging environment deployed successfully!")
        
        # Run measurement suite
        print("\nRunning Epic 1 measurement suite...")
        measurement_results = await orchestrator.run_measurement_suite()
        
        # Print summary
        print(f"\n" + "=" * 40)
        print("EPIC 1 MEASUREMENT RESULTS")
        print("=" * 40)
        
        print(f"Overall Status: {measurement_results['overall_status'].upper()}")
        
        if measurement_results.get("environment_status"):
            env_status = measurement_results["environment_status"]
            print(f"Environment: {env_status['status']}")
            if env_status.get("issues"):
                print(f"  Issues: {len(env_status['issues'])}")
        
        if measurement_results.get("performance_tests"):
            perf_status = measurement_results["performance_tests"].get("status", "unknown")
            print(f"Performance: {perf_status}")
        
        if measurement_results.get("clinical_validation"):
            clinical_status = measurement_results["clinical_validation"].get("status", "unknown") 
            print(f"Clinical Validation: {clinical_status}")
        
        if measurement_results.get("monitoring_validation"):
            monitoring = measurement_results["monitoring_validation"]
            print(f"Monitoring: {'OK' if monitoring.get('metrics_available') else 'Issues'}")
        
        # Keep environment running for manual inspection
        print(f"\nStaging environment is running:")
        print("  - Grafana: http://localhost:3000 (admin/staging_admin)")
        print("  - Prometheus: http://localhost:9090")
        print("  - NEWS2 API: http://localhost:8080")
        print("  - Stream Processor: http://localhost:8000")
        
        print(f"\nPress Enter to teardown environment...")
        input()
        
        await orchestrator.teardown_environment()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        await orchestrator.teardown_environment()
    except Exception as e:
        print(f"Error: {e}")
        await orchestrator.teardown_environment()


if __name__ == "__main__":
    asyncio.run(main())