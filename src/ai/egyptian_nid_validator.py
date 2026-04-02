"""
Egyptian National ID (NID) Validator
=====================================
Validates the 14-digit Egyptian NID.

Egyptian NID Structure (14 digits):
  Pos  1    : Century code  — 2 = 1900s, 3 = 2000s
  Pos  2–3  : Year (YY)
  Pos  4–5  : Month (MM)
  Pos  6–7  : Day (DD)
  Pos  8–9  : Governorate code (01–27, 88 for abroad)
  Pos 10–13 : Individual sequence number (0001–9999)
  Pos 14    : Gender digit — odd = Male, even = Female

Check-digit algorithm:
  The Egyptian government's internal issuance systems apply a Luhn-10
  check over all 14 digits. A Mod-97 weighted variant is also provided
  as an alternative via CheckAlgorithm.MOD97.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_GOVERNORATE_CODES: set[int] = set(range(1, 28)) | {88}

GOVERNORATE_NAMES: dict[int, str] = {
    1: "Cairo", 2: "Alexandria", 3: "Port Said", 4: "Suez",
    11: "Damietta", 12: "Dakahlia", 13: "Ash Sharqia", 14: "Kaliobeya",
    15: "Kafr El-Sheikh", 16: "Gharbia", 17: "Monoufia", 18: "Beheira",
    19: "Ismailia", 21: "Giza", 22: "Beni Suef", 23: "Fayoum",
    24: "El Menia", 25: "Assiut", 26: "Sohag", 27: "Qena",
    28: "Aswan", 29: "Luxor", 31: "Red Sea", 32: "New Valley",
    33: "Matrouh", 34: "North Sinai", 35: "South Sinai",
    88: "Born Abroad",
}

MOD97_WEIGHTS: list[int] = [pow(2, i, 97) for i in range(14)]


# ---------------------------------------------------------------------------
# Check-digit algorithms
# ---------------------------------------------------------------------------

class CheckAlgorithm(str, Enum):
    LUHN  = "luhn"
    MOD97 = "mod97"


def _luhn_valid(digits: list[int]) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _luhn_expected(body_digits: list[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(body_digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - (total % 10)) % 10


def _mod97_valid(digits: list[int]) -> bool:
    return sum(digits[i] * MOD97_WEIGHTS[i] for i in range(14)) % 97 == 1


def _mod97_expected(body_digits: list[int]) -> Optional[int]:
    w13 = MOD97_WEIGHTS[13]
    inv_w13 = pow(w13, -1, 97)
    s_body = sum(body_digits[i] * MOD97_WEIGHTS[i] for i in range(13))
    cd = ((1 - s_body) * inv_w13) % 97
    return cd if cd <= 9 else None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class NIDBirthInfo:
    year: int
    month: int
    day: int
    dob: date


@dataclass
class NIDComponents:
    century_code:     int
    birth_info:       NIDBirthInfo
    governorate_code: int
    governorate_name: str
    sequence:         str
    gender_digit:     int
    gender:           str          # "M" or "F"


@dataclass
class NIDValidationResult:
    status:          ValidationStatus
    nid:             str
    algorithm:       CheckAlgorithm
    components:      Optional[NIDComponents] = None
    fraud_alert:     bool = False
    failure_reasons: list[str] = field(default_factory=list)
    stage:           str = "Stage 1"

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASS

    def to_dict(self) -> dict:
        d: dict = {
            "stage":           self.stage,
            "status":          self.status.value,
            "nid":             self.nid,
            "check_algorithm": self.algorithm.value,
            "fraud_alert":     self.fraud_alert,
            "failure_reasons": self.failure_reasons,
        }
        if self.components:
            c = self.components
            d["parsed"] = {
                "date_of_birth":    c.birth_info.dob.isoformat(),
                "governorate_code": c.governorate_code,
                "governorate_name": c.governorate_name,
                "sequence":         c.sequence,
                "gender_digit":     c.gender_digit,
                "gender":           c.gender,
            }
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_egyptian_nid(
    nid: str,
    algorithm: CheckAlgorithm = CheckAlgorithm.LUHN,
    reference_date: Optional[date] = None,
) -> NIDValidationResult:
    """
    Full Stage-1 Egyptian NID validation.

    Steps
    -----
    1. Format: exactly 14 numeric digits
    2. Century code: 2 or 3
    3. Date of birth: valid Gregorian, not future, not >150 years ago
    4. Governorate code: 01–27 or 88
    5. Sequence: 0001–9999
    6. Check digit: Luhn-10 or Mod-97

    Any failure -> FAIL + fraud_alert=True.
    """
    today = reference_date or date.today()
    raw = str(nid).strip()
    failures: list[str] = []

    # 1. Format
    if not re.fullmatch(r"\d{14}", raw):
        reason = (
            "non-numeric characters present"
            if not re.fullmatch(r"\d+", raw)
            else f"length is {len(raw)}, expected 14"
        )
        return NIDValidationResult(
            status=ValidationStatus.FAIL,
            nid=raw,
            algorithm=algorithm,
            fraud_alert=True,
            failure_reasons=[f"Format check FAILED: {reason}."],
        )

    digits = [int(c) for c in raw]

    # 2. Century code
    century_code = digits[0]
    if century_code == 2:
        century = 1900
    elif century_code == 3:
        century = 2000
    else:
        failures.append(
            f"Century code '{century_code}' is invalid (must be 2 for 19xx or 3 for 20xx)."
        )
        century = None

    # 3. Date of birth
    yy = int(raw[1:3])
    mm = int(raw[3:5])
    dd = int(raw[5:7])
    birth_info: Optional[NIDBirthInfo] = None

    if century is not None:
        yyyy = century + yy
        try:
            dob = date(yyyy, mm, dd)
            if dob > today:
                failures.append(f"Date of birth {dob.isoformat()} is in the future.")
            elif (today.year - dob.year) > 150:
                failures.append(f"Date of birth {dob.isoformat()} implies age > 150 years.")
            else:
                birth_info = NIDBirthInfo(year=yyyy, month=mm, day=dd, dob=dob)
        except ValueError:
            failures.append(
                f"Date of birth {yyyy}-{mm:02d}-{dd:02d} is not a valid calendar date."
            )

    # 4. Governorate code
    gov_code = int(raw[7:9])
    if gov_code not in VALID_GOVERNORATE_CODES:
        failures.append(
            f"Governorate code '{gov_code:02d}' is not a recognised Egyptian "
            f"governorate (valid: 01-27, 88)."
        )
    gov_name = GOVERNORATE_NAMES.get(gov_code, f"Unknown ({gov_code:02d})")

    # 5. Sequence
    seq_str = raw[9:13]
    if int(seq_str) == 0:
        failures.append(f"Sequence '{seq_str}' is invalid (must be 0001-9999).")

    # 6. Check digit
    gender_digit = digits[13]

    if algorithm == CheckAlgorithm.LUHN:
        if not _luhn_valid(digits):
            expected = _luhn_expected(digits[:13])
            failures.append(
                f"Check-digit FAILED (Luhn-10): position 14 is '{gender_digit}', "
                f"expected '{expected}'."
            )
    else:
        if not _mod97_valid(digits):
            expected_cd = _mod97_expected(digits[:13])
            if expected_cd is not None:
                failures.append(
                    f"Check-digit FAILED (Mod-97): position 14 is '{gender_digit}', "
                    f"expected '{expected_cd}'."
                )
            else:
                failures.append(
                    "Check-digit FAILED (Mod-97): NID body is internally inconsistent "
                    "(no single-digit solution exists)."
                )

    # Assemble
    if failures:
        return NIDValidationResult(
            status=ValidationStatus.FAIL,
            nid=raw,
            algorithm=algorithm,
            fraud_alert=True,
            failure_reasons=failures,
        )

    gender = "M" if gender_digit % 2 == 1 else "F"
    components = NIDComponents(
        century_code=century_code,
        birth_info=birth_info,
        governorate_code=gov_code,
        governorate_name=gov_name,
        sequence=seq_str,
        gender_digit=gender_digit,
        gender=gender,
    )

    return NIDValidationResult(
        status=ValidationStatus.PASS,
        nid=raw,
        algorithm=algorithm,
        components=components,
    )


# ---------------------------------------------------------------------------
# Stage 1 pipeline wrapper
# ---------------------------------------------------------------------------

def stage1_nid_check(
    nid: str,
    algorithm: CheckAlgorithm = CheckAlgorithm.LUHN,
) -> dict:
    """
    Entry point for Stage 1 of the Loan Originator pipeline.
    Returns a serialisable dict for merging into pipeline state.
    """
    return validate_egyptian_nid(nid, algorithm=algorithm).to_dict()


# ---------------------------------------------------------------------------
# Helpers for test construction
# ---------------------------------------------------------------------------

def build_valid_nid(
    century: int = 2, yy: int = 90, mm: int = 3, dd: int = 15,
    gov: int = 1, seq: int = 101,
    algorithm: CheckAlgorithm = CheckAlgorithm.LUHN,
) -> str:
    body = f"{century}{yy:02d}{mm:02d}{dd:02d}{gov:02d}{seq:04d}"
    body_digits = [int(c) for c in body]
    if algorithm == CheckAlgorithm.LUHN:
        cd = _luhn_expected(body_digits)
    else:
        cd = _mod97_expected(body_digits)
        if cd is None:
            raise ValueError("Cannot produce single-digit Mod-97 check for this body.")
    return body + str(cd)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    alg = CheckAlgorithm.LUHN
    valid = build_valid_nid(algorithm=alg)
    wrong_cd = valid[:-1] + str((int(valid[-1]) + 1) % 10)

    test_cases = [
        ("PASS  | Valid NID",                        valid),
        ("FAIL  | Wrong check digit",                wrong_cd),
        ("FAIL  | Bad century code (1)",             "1" + valid[1:]),
        ("FAIL  | Invalid governorate (99)",         valid[:7] + "99" + valid[9:]),
        ("FAIL  | Impossible date Feb-30",           "2" + "900230" + valid[7:]),
        ("FAIL  | Future birth year",                "3" + "500315" + valid[7:]),
        ("FAIL  | Sequence 0000",                    valid[:9] + "0000" + valid[13]),
        ("FAIL  | Too short (13 digits)",            valid[:13]),
        ("FAIL  | Contains letters",                 valid[:5] + "AB" + valid[7:]),
    ]

    for label, nid_val in test_cases:
        res = stage1_nid_check(nid_val, algorithm=alg)
        print(f"[{label}]")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print()
