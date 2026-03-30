from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, UUID4, Field

class ApplicantType(str, Enum):
    SALARIED = "SALARIED"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    FREELANCE = "FREELANCE"

class LoanStatus(str, Enum):
    APPROVED = "APPROVED"
    CONDITIONAL = "CONDITIONAL"
    DECLINED = "DECLINED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REJECTED = "REJECTED"
    INCOMPLETE = "INCOMPLETE"

class ApplicationBase(BaseModel):
    applicant_nid: str
    applicant_type: ApplicantType
    loan_amount_requested: float = Field(gt=0)
    loan_tenor_months: int = Field(ge=6, le=360)
    annual_interest_rate: float
    status: LoanStatus = LoanStatus.PENDING_REVIEW

class ApplicationCreate(ApplicationBase):
    pass

class ApplicationInDB(ApplicationBase):
    application_id: UUID4
    submission_timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_timestamp: Optional[datetime] = None
    decision_reason_code: Optional[str] = None

    class Config:
        from_attributes = True

class StageVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    INCOMPLETE = "INCOMPLETE"

class StageResult(BaseModel):
    stage_number: int
    verdict: StageVerdict
    confidence: float
    extracted_data: Dict[str, Any]
    flags: List[str] = []
    model_versions: Dict[str, str]
    processing_duration_ms: int
