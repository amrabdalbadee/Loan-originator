from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENV: str = "development"

    # OCR Thresholds (from SDD section 8)
    OCR_AUTO_ACCEPT_THRESHOLD: float = 0.80
    NAME_FUZZY_MATCH_THRESHOLD: float = 0.90
    SALARY_VS_BANK_TOLERANCE: float = 0.05
    DECLARED_VS_VERIFIED_INCOME_TOLERANCE: float = 0.10
    SIGNATURE_PASS_THRESHOLD: float = 0.85
    SIGNATURE_REVIEW_THRESHOLD: float = 0.65
    DTI_CAP_PCT: float = 50.0

    # DB Config
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/loan_originator"

    # AI Model Endpoints
    QWEN_VL_ENDPOINT: Optional[str] = None
    SIGNET_ENDPOINT: Optional[str] = None

    # Storage
    S3_BUCKET: str = "loan-originator-docs"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
