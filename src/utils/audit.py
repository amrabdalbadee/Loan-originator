import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, UUID4, Field

class EventType(str, Enum):
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    DECISION_MADE = "DECISION_MADE"
    OVERRIDE = "OVERRIDE"
    PII_ACCESS = "PII_ACCESS"
    CONFIG_CHANGE = "CONFIG_CHANGE"

class AuditLogEntry(BaseModel):
    """
    6.2 Immutable Audit Log
    AuditLogEntry is used for keeping a tamper-evident record of all system events.
    Each entry includes a SHA-256 hash of the payload and a previous_hash for chaining.
    """
    audit_id: UUID4 = Field(default_factory=uuid.uuid4)
    event_type: EventType
    application_id: UUID4
    actor_id: UUID4
    actor_role: str
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    payload_hash: str
    previous_hash: Optional[str] = None

class Auditor:
    def __init__(self):
        self.log: List[AuditLogEntry] = []
        self._last_hash: Optional[str] = None

    def log_event(self, event_type: EventType, application_id: UUID4, actor_id: UUID4, actor_role: str, payload: Dict[str, Any]):
        """
        Creates an audit log entry for the event and ensures its integrity via chaining.
        """
        payload_str = str(payload).encode('utf-8')
        payload_hash = hashlib.sha256(payload_str).hexdigest()
        
        entry = AuditLogEntry(
            event_type=event_type,
            application_id=application_id,
            actor_id=actor_id,
            actor_role=actor_role,
            payload_hash=payload_hash,
            previous_hash=self._last_hash
        )
        
        self.log.append(entry)
        self._last_hash = payload_hash
        return entry

auditor = Auditor()
