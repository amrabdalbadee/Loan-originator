from typing import Dict, Any, List
from src.core.config import settings

class NLPClassifier:
    def __init__(self, model_name: str = "GPT-4o"):
        self.model_name = model_name

    async def classify_loan_purpose(self, loan_purpose_text: str) -> Dict[str, Any]:
        """
        Classify free-text loan purpose into predefined categories.
        (Personal, Auto, Home Improvement, Business).
        Uses fine-tuned BERT or GPT-4o.
        """
        # Load model and processor (NLP Classifier)
        # category = await self.model.classify(loan_purpose_text)
        
        # Placeholder result for now
        category = "Personal"
        confidence = 0.94
        
        return {
          "category": category,
          "confidence": confidence,
          "model_version": "v2.0.1"
        }

    async def cross_validate_income(self, declared_income: float, verified_income: float) -> Dict[str, Any]:
        """
        Discrepancy > 10% → REVIEW (Section 3.3 SDD)
        """
        discrepancy_pct = abs(declared_income - verified_income) / max(declared_income, verified_income, 1.0)
        
        status = "PASS" if discrepancy_pct <= settings.DECLARED_VS_VERIFIED_INCOME_TOLERANCE else "REVIEW"
        
        return {
          "discrepancy_pct": discrepancy_pct,
          "status": status,
          "action": "Proceed" if status == "PASS" else "Flag for human review"
        }

nlp_classifier = NLPClassifier()
