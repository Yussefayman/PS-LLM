import os
import re
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

DENYLIST = [
    "password", "pin", "cvv", "hack", "exploit",
    "override", "ignore instructions",
]

PII_PATTERNS = {
    "national_id": r"\b[12]\d{9}\b",
    "phone":       r"\b05\d{8}\b",
    "iban":        r"\bSA\d{22}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}


def check_denylist(query: str) -> None:
    lowered = query.lower()
    for term in DENYLIST:
        if term in lowered:
            raise HTTPException(
                status_code=400,
                detail=f"Blocked term detected: '{term}'"
            )


def check_pii(query: str) -> None:
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, query):
            raise HTTPException(
                status_code=400,
                detail=f"Query contains sensitive information ({label}). Please do not share personal details."
            )


def check_confidence(top_score: float) -> None:
    if top_score < CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant snippet found (score: {top_score:.3f}, threshold: {CONFIDENCE_THRESHOLD})"
        )