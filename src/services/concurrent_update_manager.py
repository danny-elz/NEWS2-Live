import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Callable, Awaitable, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
from uuid import uuid4

from ..models.patient_state import PatientState, ConcurrentUpdateError, PatientStateError
from ..services.audit import AuditLogger, AuditOperation


@dataclass
class LockInfo:
    """Information about distributed lock."""
    lock_id: str
    patient_id: str
    operation_type: str
    acquired_at: datetime
    expires_at: datetime
    holder_id: str


@dataclass
class TransactionContext:
    """Context for managing database transactions."""
    transaction_id: str
    patient_id: str
    operations: List[str]
    started_at: datetime
    timeout_seconds: int = 30


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 2.0
    exponential_base: float = 2.0
    jitter: bool = True


class ConcurrentUpdateManager:
    """Manager for handling concurrent patient state updates safely."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self._locks: Dict[str, LockInfo] = {}
        self._lock_mutex = asyncio.Lock()
        self._transactions: Dict[str, TransactionContext] = {}
        self._retry_semaphore = asyncio.Semaphore(100)  # Limit concurrent retries
        
    @asynccontextmanager
    async def distributed_lock(self, patient_id: str, operation_type: str, 
                              timeout_seconds: int = 30, holder_id: Optional[str] = None):
        """Acquire distributed lock for critical patient operations."""
        lock_id = str(uuid4())
        holder_id = holder_id or f"process_{id(self)}"
        
        acquired = False
        try:
            acquired = await self._acquire_lock(patient_id, operation_type, lock_id, 
                                              timeout_seconds, holder_id)
            if not acquired:
                raise ConcurrentUpdateError(
                    f"Failed to acquire lock for patient {patient_id} operation {operation_type}"
                )
            
            yield lock_id
            
        finally:
            if acquired:
                await self._release_lock(lock_id, patient_id)
    
    async def _acquire_lock(self, patient_id: str, operation_type: str, lock_id: str,
                           timeout_seconds: int, holder_id: str) -> bool:
        """Acquire lock with timeout."""
        async with self._lock_mutex:
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(seconds=timeout_seconds)
            
            # Check if there's already a valid lock for this patient
            existing_lock = self._locks.get(patient_id)
            if existing_lock and existing_lock.expires_at > now:
                # Lock already held
                return False
            
            # Clean up expired lock if it exists
            if existing_lock and existing_lock.expires_at <= now:
                del self._locks[patient_id]
            
            # Create new lock
            lock_info = LockInfo(
                lock_id=lock_id,
                patient_id=patient_id,
                operation_type=operation_type,
                acquired_at=now,
                expires_at=expires_at,
                holder_id=holder_id
            )
            
            self._locks[patient_id] = lock_info
            
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=patient_id,
                details={
                    'operation': 'lock_acquired',
                    'lock_id': lock_id,
                    'operation_type': operation_type,
                    'holder_id': holder_id,
                    'timeout_seconds': timeout_seconds
                }
            )
            
            return True
    
    async def _release_lock(self, lock_id: str, patient_id: str):
        """Release distributed lock."""
        async with self._lock_mutex:
            existing_lock = self._locks.get(patient_id)
            if existing_lock and existing_lock.lock_id == lock_id:
                del self._locks[patient_id]
                
                await self.audit_logger.log_operation(
                    operation=AuditOperation.UPDATE,
                    patient_id=patient_id,
                    details={
                        'operation': 'lock_released',
                        'lock_id': lock_id,
                        'holder_id': existing_lock.holder_id
                    }
                )
    
    @asynccontextmanager
    async def transaction(self, patient_id: str, operation_types: List[str], 
                         timeout_seconds: int = 30):
        """Manage database transaction for multiple operations."""
        transaction_id = str(uuid4())
        
        # Create transaction context
        tx_context = TransactionContext(
            transaction_id=transaction_id,
            patient_id=patient_id,
            operations=operation_types,
            started_at=datetime.now(timezone.utc),
            timeout_seconds=timeout_seconds
        )
        
        self._transactions[transaction_id] = tx_context
        
        try:
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=patient_id,
                details={
                    'operation': 'transaction_started',
                    'transaction_id': transaction_id,
                    'operations': operation_types,
                    'timeout_seconds': timeout_seconds
                }
            )
            
            yield tx_context
            
            # Commit transaction (would interact with actual database)
            await self._commit_transaction(transaction_id)
            
        except Exception as e:
            # Rollback transaction
            await self._rollback_transaction(transaction_id, str(e))
            raise
        finally:
            # Clean up transaction
            self._transactions.pop(transaction_id, None)
    
    async def _commit_transaction(self, transaction_id: str):
        """Commit database transaction."""
        tx_context = self._transactions.get(transaction_id)
        if tx_context:
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=tx_context.patient_id,
                details={
                    'operation': 'transaction_committed',
                    'transaction_id': transaction_id,
                    'duration_seconds': (datetime.now(timezone.utc) - tx_context.started_at).total_seconds()
                }
            )
    
    async def _rollback_transaction(self, transaction_id: str, error_reason: str):
        """Rollback database transaction."""
        tx_context = self._transactions.get(transaction_id)
        if tx_context:
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=tx_context.patient_id,
                details={
                    'operation': 'transaction_rolled_back',
                    'transaction_id': transaction_id,
                    'error_reason': error_reason,
                    'duration_seconds': (datetime.now(timezone.utc) - tx_context.started_at).total_seconds()
                }
            )
    
    async def retry_with_backoff(self, operation: Callable[[], Awaitable[Any]], 
                                patient_id: str, operation_name: str,
                                retry_config: Optional[RetryConfig] = None) -> Any:
        """Execute operation with exponential backoff retry mechanism."""
        if retry_config is None:
            retry_config = RetryConfig()
        
        async with self._retry_semaphore:
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    result = await operation()
                    
                    if attempt > 0:
                        await self.audit_logger.log_operation(
                            operation=AuditOperation.UPDATE,
                            patient_id=patient_id,
                            details={
                                'operation': 'retry_succeeded',
                                'operation_name': operation_name,
                                'attempt_number': attempt + 1,
                                'total_attempts': retry_config.max_attempts
                            }
                        )
                    
                    return result
                    
                except (ConcurrentUpdateError, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    
                    if attempt < retry_config.max_attempts - 1:
                        # Calculate delay with exponential backoff
                        delay = min(
                            retry_config.base_delay * (retry_config.exponential_base ** attempt),
                            retry_config.max_delay
                        )
                        
                        # Add jitter to avoid thundering herd
                        if retry_config.jitter:
                            import random
                            delay *= (0.5 + 0.5 * random.random())
                        
                        await self.audit_logger.log_operation(
                            operation=AuditOperation.UPDATE,
                            patient_id=patient_id,
                            details={
                                'operation': 'retry_attempt',
                                'operation_name': operation_name,
                                'attempt_number': attempt + 1,
                                'error_type': type(e).__name__,
                                'error_message': str(e),
                                'retry_delay_seconds': delay
                            }
                        )
                        
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed
                        await self.audit_logger.log_operation(
                            operation=AuditOperation.UPDATE,
                            patient_id=patient_id,
                            details={
                                'operation': 'retry_exhausted',
                                'operation_name': operation_name,
                                'total_attempts': retry_config.max_attempts,
                                'final_error_type': type(e).__name__,
                                'final_error_message': str(e)
                            }
                        )
                except Exception as e:
                    # Non-retryable exception
                    last_exception = e
                    break
            
            # All retries exhausted or non-retryable error
            raise last_exception
    
    async def safe_state_update(self, patient_id: str, 
                              update_operation: Callable[[PatientState], Awaitable[PatientState]],
                              operation_name: str,
                              max_retries: int = 3) -> PatientState:
        """Safely update patient state with optimistic locking and retry mechanism."""
        
        async def update_with_lock():
            async with self.distributed_lock(patient_id, operation_name):
                # Get current state (would be from database)
                current_state = await self._get_current_state(patient_id)
                if not current_state:
                    raise PatientStateError(f"Patient {patient_id} not found")
                
                # Execute update operation
                updated_state = await update_operation(current_state)
                
                # Validate version hasn't changed (optimistic locking)
                if updated_state.state_version <= current_state.state_version:
                    updated_state.state_version = current_state.state_version + 1
                
                # Save updated state (would be to database)
                await self._save_state(updated_state)
                
                return updated_state
        
        retry_config = RetryConfig(max_attempts=max_retries)
        return await self.retry_with_backoff(
            update_with_lock,
            patient_id,
            operation_name,
            retry_config
        )
    
    async def resolve_conflict(self, patient_id: str, 
                             conflicting_updates: List[Dict[str, Any]]) -> PatientState:
        """Resolve conflicts from simultaneous updates."""
        async with self.distributed_lock(patient_id, "conflict_resolution"):
            # Get current state
            current_state = await self._get_current_state(patient_id)
            if not current_state:
                raise PatientStateError(f"Patient {patient_id} not found")
            
            # Apply conflict resolution strategy
            resolved_state = current_state
            
            for update in conflicting_updates:
                # Simple last-writer-wins strategy
                # In production, would use more sophisticated merge logic
                if update.get('timestamp', datetime.min) > resolved_state.last_updated:
                    # Apply the update
                    for key, value in update.get('changes', {}).items():
                        if hasattr(resolved_state, key):
                            setattr(resolved_state, key, value)
            
            # Increment version and update timestamp
            resolved_state.state_version += 1
            resolved_state.last_updated = datetime.now(timezone.utc)
            
            await self.audit_logger.log_operation(
                operation=AuditOperation.UPDATE,
                patient_id=patient_id,
                details={
                    'operation': 'conflict_resolved',
                    'conflicting_updates_count': len(conflicting_updates),
                    'resolution_strategy': 'last_writer_wins',
                    'final_version': resolved_state.state_version
                }
            )
            
            await self._save_state(resolved_state)
            return resolved_state
    
    async def _get_current_state(self, patient_id: str) -> Optional[PatientState]:
        """Get current patient state (placeholder for database query)."""
        # In real implementation, would query database
        # For now, return None to simulate not found
        return None
    
    async def _save_state(self, state: PatientState) -> None:
        """Save patient state (placeholder for database operation)."""
        # In real implementation, would save to database
        pass
    
    def get_lock_status(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get current lock status for patient."""
        lock_info = self._locks.get(patient_id)
        if not lock_info:
            return None
        
        now = datetime.now(timezone.utc)
        return {
            'lock_id': lock_info.lock_id,
            'operation_type': lock_info.operation_type,
            'holder_id': lock_info.holder_id,
            'acquired_at': lock_info.acquired_at.isoformat(),
            'expires_at': lock_info.expires_at.isoformat(),
            'is_expired': lock_info.expires_at <= now,
            'remaining_seconds': max(0, (lock_info.expires_at - now).total_seconds())
        }
    
    async def cleanup_expired_locks(self):
        """Clean up expired locks (should be called periodically)."""
        async with self._lock_mutex:
            now = datetime.now(timezone.utc)
            expired_locks = [
                (patient_id, lock_info) 
                for patient_id, lock_info in self._locks.items()
                if lock_info.expires_at <= now
            ]
            
            for patient_id, lock_info in expired_locks:
                del self._locks[patient_id]
                await self.audit_logger.log_operation(
                    operation=AuditOperation.UPDATE,
                    patient_id=patient_id,
                    details={
                        'operation': 'lock_expired_cleanup',
                        'lock_id': lock_info.lock_id,
                        'holder_id': lock_info.holder_id,
                        'operation_type': lock_info.operation_type
                    }
                )