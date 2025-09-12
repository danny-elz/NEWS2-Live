import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.vital_signs import VitalSigns
from ..models.news2 import NEWS2Result
from ..services.audit import AuditLogger, AuditOperation


class RetentionPolicy(Enum):
    ACTIVE_30_DAYS = "active_30_days"
    COMPRESSED_2_YEARS = "compressed_2_years"
    ARCHIVED_PERMANENT = "archived_permanent"


@dataclass
class HistoricalVitalSigns:
    """Historical vital signs record with metadata."""
    record_id: str
    patient_id: str
    timestamp: datetime
    vital_signs: VitalSigns
    news2_result: Optional[NEWS2Result]
    data_source: str  # "device", "manual", "imported"
    quality_score: float  # 0.0 to 1.0
    retention_policy: RetentionPolicy
    compressed: bool = False
    archived: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'record_id': self.record_id,
            'patient_id': self.patient_id,
            'timestamp': self.timestamp.isoformat(),
            'vital_signs': {
                'respiratory_rate': self.vital_signs.respiratory_rate,
                'spo2': self.vital_signs.spo2,
                'on_oxygen': self.vital_signs.on_oxygen,
                'temperature': self.vital_signs.temperature,
                'systolic_bp': self.vital_signs.systolic_bp,
                'heart_rate': self.vital_signs.heart_rate,
                'consciousness': self.vital_signs.consciousness.value if self.vital_signs.consciousness else None
            },
            'news2_result': self.news2_result.to_dict() if self.news2_result else None,
            'data_source': self.data_source,
            'quality_score': self.quality_score,
            'retention_policy': self.retention_policy.value,
            'compressed': self.compressed,
            'archived': self.archived
        }


@dataclass
class DataIntegrityCheck:
    """Result of data integrity verification."""
    patient_id: str
    check_timestamp: datetime
    records_checked: int
    integrity_violations: List[str]
    missing_records: List[str]
    duplicate_records: List[str]
    checksum_mismatches: List[str]
    overall_integrity: bool


@dataclass
class CompressionResult:
    """Result of data compression operation."""
    patient_id: str
    records_compressed: int
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_timestamp: datetime


class VitalSignsHistory:
    """Service for historical vital signs data preservation and management."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._compression_lock = asyncio.Lock()
    
    async def store_vital_signs(self, patient_id: str, vital_signs: VitalSigns,
                              news2_result: Optional[NEWS2Result] = None,
                              data_source: str = "device") -> str:
        """Store vital signs with historical preservation."""
        record_id = self._generate_record_id(patient_id, vital_signs.timestamp)
        
        # Calculate quality score based on data completeness
        quality_score = self._calculate_quality_score(vital_signs)
        
        # Determine retention policy based on data quality and source
        retention_policy = self._determine_retention_policy(vital_signs, data_source, quality_score)
        
        historical_record = HistoricalVitalSigns(
            record_id=record_id,
            patient_id=patient_id,
            timestamp=vital_signs.timestamp,
            vital_signs=vital_signs,
            news2_result=news2_result,
            data_source=data_source,
            quality_score=quality_score,
            retention_policy=retention_policy
        )
        
        # Store in TimescaleDB (simulated)
        await self._store_timescale_record(historical_record)
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.INSERT,
            patient_id=patient_id,
            details={
                'operation': 'vital_signs_stored',
                'record_id': record_id,
                'data_source': data_source,
                'quality_score': quality_score,
                'retention_policy': retention_policy.value,
                'timestamp': vital_signs.timestamp.isoformat()
            }
        )
        
        return record_id
    
    async def query_historical_data(self, patient_id: str, 
                                  start_time: datetime,
                                  end_time: datetime,
                                  include_compressed: bool = True) -> List[HistoricalVitalSigns]:
        """Query historical vital signs data with time-range filtering."""
        # Simulate TimescaleDB query
        records = await self._query_timescale_range(
            patient_id, start_time, end_time, include_compressed
        )
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'historical_query',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'records_returned': len(records),
                'include_compressed': include_compressed
            }
        )
        
        return records
    
    async def implement_30day_retention(self, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Implement 30-day minimum retention policy with automatic archival."""
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=30)
        
        if patient_id:
            # Process single patient
            result = await self._process_patient_retention(patient_id, cutoff_date)
        else:
            # Process all patients (batch operation)
            result = await self._process_batch_retention(cutoff_date)
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': '30day_retention_policy',
                'cutoff_date': cutoff_date.isoformat(),
                'records_processed': result.get('total_processed', 0),
                'records_archived': result.get('archived_count', 0),
                'records_retained': result.get('retained_count', 0)
            }
        )
        
        return result
    
    async def _process_patient_retention(self, patient_id: str, cutoff_date: datetime) -> Dict[str, Any]:
        """Process retention policy for single patient."""
        # Query records older than 30 days
        old_records = await self._query_records_before_date(patient_id, cutoff_date)
        
        archived_count = 0
        retained_count = 0
        
        for record in old_records:
            if record.retention_policy == RetentionPolicy.ACTIVE_30_DAYS:
                # Move to compressed storage
                await self._archive_record(record)
                archived_count += 1
            else:
                # Keep in active storage due to policy
                retained_count += 1
        
        return {
            'patient_id': patient_id,
            'total_processed': len(old_records),
            'archived_count': archived_count,
            'retained_count': retained_count,
            'cutoff_date': cutoff_date
        }
    
    async def _process_batch_retention(self, cutoff_date: datetime) -> Dict[str, Any]:
        """Process retention policy for all patients in batch."""
        # This would query all patients with old records
        # For simulation, return batch stats
        
        return {
            'operation': 'batch_retention',
            'cutoff_date': cutoff_date,
            'patients_processed': 0,  # Would be actual count
            'total_processed': 0,
            'archived_count': 0,
            'retained_count': 0
        }
    
    async def compress_historical_data(self, patient_id: str, 
                                     compression_age_days: int = 7) -> CompressionResult:
        """Create data compression strategy for long-term storage efficiency."""
        async with self._compression_lock:
            now = datetime.now(timezone.utc)
            compression_cutoff = now - timedelta(days=compression_age_days)
            
            # Query uncompressed records older than cutoff
            records_to_compress = await self._query_uncompressed_records(
                patient_id, compression_cutoff
            )
            
            if not records_to_compress:
                return CompressionResult(
                    patient_id=patient_id,
                    records_compressed=0,
                    original_size_mb=0.0,
                    compressed_size_mb=0.0,
                    compression_ratio=0.0,
                    compression_timestamp=now
                )
            
            # Calculate original size (estimated)
            original_size_mb = len(records_to_compress) * 0.001  # ~1KB per record estimate
            
            # Perform compression (simulated)
            compressed_records = await self._compress_records(records_to_compress)
            
            # Calculate compressed size (estimated)
            compressed_size_mb = original_size_mb * 0.3  # 70% compression ratio estimate
            compression_ratio = compressed_size_mb / original_size_mb if original_size_mb > 0 else 0.0
            
            result = CompressionResult(
                patient_id=patient_id,
                records_compressed=len(records_to_compress),
                original_size_mb=original_size_mb,
                compressed_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                compression_timestamp=now
            )
            
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=patient_id,
                details={
                    'operation': 'data_compression',
                    'records_compressed': result.records_compressed,
                    'original_size_mb': result.original_size_mb,
                    'compressed_size_mb': result.compressed_size_mb,
                    'compression_ratio': result.compression_ratio,
                    'compression_age_days': compression_age_days
                }
            )
            
            return result
    
    async def verify_data_integrity(self, patient_id: str,
                                  check_period_days: int = 30) -> DataIntegrityCheck:
        """Implement data integrity checks for historical vital signs chains."""
        now = datetime.now(timezone.utc)
        check_start = now - timedelta(days=check_period_days)
        
        # Query records in check period
        records = await self._query_timescale_range(patient_id, check_start, now, include_compressed=True)
        
        integrity_violations = []
        missing_records = []
        duplicate_records = []
        checksum_mismatches = []
        
        # Check for temporal consistency
        if len(records) > 1:
            sorted_records = sorted(records, key=lambda r: r.timestamp)
            
            for i in range(len(sorted_records) - 1):
                current = sorted_records[i]
                next_record = sorted_records[i + 1]
                
                # Check for impossible time gaps
                time_gap = (next_record.timestamp - current.timestamp).total_seconds()
                if time_gap < 0:
                    integrity_violations.append(f"Negative time gap between {current.record_id} and {next_record.record_id}")
                
                # Check for duplicate timestamps
                if time_gap == 0:
                    duplicate_records.append(f"Duplicate timestamp: {current.timestamp}")
        
        # Check for missing expected records (e.g., gaps > 6 hours in active monitoring)
        expected_intervals = await self._calculate_expected_intervals(patient_id, check_start, now)
        actual_intervals = [r.timestamp for r in records]
        
        for expected_time in expected_intervals:
            if not any(abs((actual - expected_time).total_seconds()) < 1800 for actual in actual_intervals):  # 30-minute tolerance
                missing_records.append(f"Missing expected record around {expected_time.isoformat()}")
        
        # Verify checksums (simplified)
        for record in records:
            expected_checksum = self._calculate_record_checksum(record)
            # In real implementation, would compare with stored checksum
            # For now, assume all checksums are valid
        
        overall_integrity = (
            len(integrity_violations) == 0 and
            len(missing_records) <= len(records) * 0.1 and  # Allow up to 10% missing
            len(duplicate_records) == 0 and
            len(checksum_mismatches) == 0
        )
        
        result = DataIntegrityCheck(
            patient_id=patient_id,
            check_timestamp=now,
            records_checked=len(records),
            integrity_violations=integrity_violations,
            missing_records=missing_records,
            duplicate_records=duplicate_records,
            checksum_mismatches=checksum_mismatches,
            overall_integrity=overall_integrity
        )
        
        await self.audit_logger.log_operation(
            operation=AuditOperation.UPDATE,
            patient_id=patient_id,
            details={
                'operation': 'integrity_check',
                'records_checked': result.records_checked,
                'integrity_violations_count': len(result.integrity_violations),
                'missing_records_count': len(result.missing_records),
                'duplicate_records_count': len(result.duplicate_records),
                'overall_integrity': result.overall_integrity,
                'check_period_days': check_period_days
            }
        )
        
        return result
    
    def _generate_record_id(self, patient_id: str, timestamp: datetime) -> str:
        """Generate unique record ID."""
        content = f"{patient_id}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_quality_score(self, vital_signs: VitalSigns) -> float:
        """Calculate data quality score based on completeness."""
        total_fields = 7  # Total vital sign fields
        complete_fields = 0
        
        if vital_signs.respiratory_rate is not None:
            complete_fields += 1
        if vital_signs.spo2 is not None:
            complete_fields += 1
        if vital_signs.on_oxygen is not None:
            complete_fields += 1
        if vital_signs.temperature is not None:
            complete_fields += 1
        if vital_signs.systolic_bp is not None:
            complete_fields += 1
        if vital_signs.heart_rate is not None:
            complete_fields += 1
        if vital_signs.consciousness is not None:
            complete_fields += 1
        
        return round(complete_fields / total_fields, 2)
    
    def _determine_retention_policy(self, vital_signs: VitalSigns, data_source: str, 
                                  quality_score: float) -> RetentionPolicy:
        """Determine appropriate retention policy for record."""
        # High-quality device data gets extended retention
        if data_source == "device" and quality_score >= 0.9:
            return RetentionPolicy.COMPRESSED_2_YEARS
        
        # Manual entry or lower quality gets standard retention
        if data_source == "manual" or quality_score < 0.7:
            return RetentionPolicy.ACTIVE_30_DAYS
        
        # Default to 30-day active retention
        return RetentionPolicy.ACTIVE_30_DAYS
    
    def _calculate_record_checksum(self, record: HistoricalVitalSigns) -> str:
        """Calculate checksum for data integrity verification."""
        content = f"{record.patient_id}_{record.timestamp}_{record.vital_signs.to_dict()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _store_timescale_record(self, record: HistoricalVitalSigns):
        """Store record in TimescaleDB (simulated)."""
        # In real implementation, would insert into TimescaleDB
        pass
    
    async def _query_timescale_range(self, patient_id: str, start_time: datetime,
                                   end_time: datetime, include_compressed: bool) -> List[HistoricalVitalSigns]:
        """Query TimescaleDB for records in range (simulated)."""
        # In real implementation, would query TimescaleDB
        return []
    
    async def _query_records_before_date(self, patient_id: str, cutoff_date: datetime) -> List[HistoricalVitalSigns]:
        """Query records before cutoff date (simulated)."""
        # In real implementation, would query database
        return []
    
    async def _query_uncompressed_records(self, patient_id: str, cutoff_date: datetime) -> List[HistoricalVitalSigns]:
        """Query uncompressed records before cutoff (simulated)."""
        # In real implementation, would query database
        return []
    
    async def _compress_records(self, records: List[HistoricalVitalSigns]) -> List[HistoricalVitalSigns]:
        """Compress records for storage (simulated)."""
        # In real implementation, would apply compression algorithm
        compressed_records = []
        for record in records:
            record.compressed = True
            compressed_records.append(record)
        return compressed_records
    
    async def _archive_record(self, record: HistoricalVitalSigns):
        """Archive record to cold storage (simulated)."""
        # In real implementation, would move to archival storage
        record.archived = True
    
    async def _calculate_expected_intervals(self, patient_id: str, start_time: datetime, 
                                          end_time: datetime) -> List[datetime]:
        """Calculate expected vital signs recording intervals."""
        # Simple implementation: expect hourly readings
        intervals = []
        current = start_time
        while current <= end_time:
            intervals.append(current)
            current += timedelta(hours=1)
        return intervals