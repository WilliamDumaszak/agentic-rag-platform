"""
Input/output guardrails for the agentic RAG pipeline.

Two layers of protection:
  1. Input validation  — blocks injection attempts, oversized queries, empty content.
  2. Output validation — detects PII in LLM responses before returning to the client.

Design decision: guardrails are kept lightweight (regex-based) to add <1ms latency.
For production, replace with NeMo Guardrails or Azure AI Content Safety.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── prompt injection patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)",
    r"you\s+are\s+now\s+(?!helpful|an?\s+assistant)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"\bjailbreak\b",
    r"(override|bypass)\s+(your\s+)?(safety|guidelines|rules|restrictions)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(?!a\s+helpful)",
    r"forget\s+(everything|all)\s+(you\s+)?(know|were\s+told)",
    r"new\s+instructions?:",
    r"system\s+prompt\s*:",
]

# ── PII patterns ───────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b",              # US SSN
    r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b",        # Credit card number
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # Email
    r"\b(\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",  # Phone (US/intl)
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_COMPILED_PII = [re.compile(p) for p in _PII_PATTERNS]

# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


# ── public API ────────────────────────────────────────────────────────────────

def validate_input(query: str, max_length: int = 1_000) -> ValidationResult:
    """
    Validate a user query before processing.

    Checks: non-empty, within length limit, no prompt injection patterns.
    """
    if not query or not query.strip():
        return ValidationResult(valid=False, reason="Query cannot be empty.")

    if len(query) > max_length:
        return ValidationResult(
            valid=False,
            reason=f"Query exceeds maximum length of {max_length} characters.",
        )

    for pattern in _COMPILED_INJECTION:
        if pattern.search(query):
            logger.warning(f"Injection attempt blocked: '{query[:100]}'")
            return ValidationResult(
                valid=False, reason="Query contains disallowed content."
            )

    return ValidationResult(valid=True)


def validate_output(response: str) -> ValidationResult:
    """
    Validate a generated LLM response before returning to the client.

    Checks: non-empty, minimum length, no PII.
    """
    if not response or len(response.strip()) < 5:
        return ValidationResult(valid=False, reason="Response is too short or empty.")

    for pattern in _COMPILED_PII:
        if pattern.search(response):
            logger.warning("PII detected in LLM output — response blocked.")
            return ValidationResult(
                valid=False, reason="Response contains sensitive information and was blocked."
            )

    return ValidationResult(valid=True)


def sanitize_output(response: str) -> str:
    """Replace detected PII in output with [REDACTED]. Use when blocking is too aggressive."""
    sanitized = response
    for pattern in _COMPILED_PII:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized
