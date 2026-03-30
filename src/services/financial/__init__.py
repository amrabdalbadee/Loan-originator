from typing import Dict, Any
from src.core.config import settings

class FinancialDecisionService:
    @staticmethod
    def calculate_emi(principal: float, annual_rate: float, tenor_months: int) -> float:
        """
        EMI = [P x r x (1+r)^n] / [(1+r)^n – 1]
        """
        if principal <= 0 or annual_rate <= 0 or tenor_months <= 0:
            return 0.0
            
        r = (annual_rate / 100) / 12  # monthly rate
        n = tenor_months              # number of installments
        
        emi = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)
        return emi

    @staticmethod
    def calculate_dti(existing_debt: float, proposed_emi: float, gross_monthly_income: float) -> float:
        """
        Post-Loan DTI = (existing_monthly_debt + proposed_emi) / gross_monthly_income * 100 (%)
        """
        if gross_monthly_income <= 0:
            return 100.0
            
        total_monthly_obligations = existing_debt + proposed_emi
        dti_pct = (total_monthly_obligations / gross_monthly_income) * 100
        return dti_pct

    @staticmethod
    def provide_verdict(post_loan_dti: float) -> Dict[str, Any]:
        """
        DTI_CAP: 50%
        Conditional approval band above cap: +10% (Section 8 SDD)
        """
        if post_loan_dti <= settings.DTI_CAP_PCT:
            status = "APPROVED"
            reason = "Within DTI cap"
        elif post_loan_dti <= (settings.DTI_CAP_PCT + 10.0):
            status = "CONDITIONAL"
            reason = "Within conditional DTI band (+10%)"
        else:
            status = "DECLINED"
            reason = "DEBT_BURDEN" # Post-loan DTI > cap + 10% (Section 10 SDD)
            
        return {
          "status": status,
          "reason_code": reason if status != "APPROVED" else None,
          "post_loan_dti": post_loan_dti
        }

financial_service = FinancialDecisionService()
