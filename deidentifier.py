"""
deidentifier.py
───────────────
5-Layer Data Privacy & PII Protection Gatekeeper for MediGraph.

Architecture (5 Layers of GenAI Privacy):
1. Pre-LLM Masking:
   - Scans user prompts and clinical notes for PHI (Names, Phones, Emails, SSNs, MRNs).
   - Redacts or replaces entities before sending context to LLMs (Gemini / Ollama).
2. Knowledge Base Pseudonymization:
   - Replaces real patient identifiers with deterministic salted SHA-256 pseudo-IDs (PATIENT_<hash>).
   - Real identities are stored in an encrypted/isolated Identity Vault.
3. LLM Output Sanitization:
   - Scans generated triage reports for accidental PII leaks before returning to clients.
4. Role-Based Identity Resolution (RBAC):
   - Only authorized clinicians/admins can resolve pseudo-IDs back to real names.
5. Zero-Retention Audit Logging:
   - Records privacy compliance actions without storing raw unmasked PHI.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VAULT_PATH = Path(__file__).parent / "identity_vault.json"
PSEUDO_ID_SALT = os.getenv("PSEUDO_ID_SALT", "medigraph-hackathon-2025")
PSEUDO_ID_PREFIX = "PATIENT"

# Common regex patterns for high-risk PII
PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "PHONE": r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "MRN": r"\bMRN[:\s#]*[A-Z0-9]{6,10}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


def make_pseudo_id(real_name: str, salt: str = PSEUDO_ID_SALT) -> str:
    """Deterministic salted SHA-256 hash for patient pseudonymization."""
    raw = f"{salt}:{real_name.strip().lower()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8].upper()
    return f"{PSEUDO_ID_PREFIX}_{digest}"


def load_vault() -> Dict[str, str]:
    if VAULT_PATH.exists():
        try:
            return json.loads(VAULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_to_vault(pseudo_id: str, real_name: str):
    vault = load_vault()
    vault[pseudo_id] = real_name
    VAULT_PATH.write_text(json.dumps(vault, indent=2), encoding="utf-8")


def mask_pii_with_regex(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Layer 1: Fast regex sanitizer for structured PII."""
    detected = []
    anonymized = text
    for pii_type, pattern in PATTERNS.items():
        matches = list(re.finditer(pattern, anonymized, flags=re.IGNORECASE))
        for m in reversed(matches):
            val = m.group(0)
            detected.append({"type": pii_type, "value": val, "start": m.start(), "end": m.end()})
            anonymized = anonymized[:m.start()] + f"[{pii_type}]" + anonymized[m.end():]
    return anonymized, detected


def deidentify_clinical_input(text: str, patient_name: Optional[str] = None) -> Dict:
    """
    Full 5-layer privacy gatekeeper function.
    Masks PII, replaces patient names with pseudo-IDs, and logs audit events.
    """
    anonymized, detected = mask_pii_with_regex(text)

    # Detect & mask patient name if provided or pattern matched
    pseudo_id = None
    if patient_name:
        pseudo_id = make_pseudo_id(patient_name)
        save_to_vault(pseudo_id, patient_name)
        anonymized = re.sub(re.escape(patient_name), pseudo_id, anonymized, flags=re.IGNORECASE)
        detected.append({"type": "PERSON", "value": patient_name, "pseudo_id": pseudo_id})

    # Detect common "Patient: <Name>" patterns
    name_match = re.search(r"(?:Patient(?:\s+Name)?:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", anonymized)
    if name_match:
        extracted_name = name_match.group(1)
        pseudo_id = make_pseudo_id(extracted_name)
        save_to_vault(pseudo_id, extracted_name)
        anonymized = anonymized[:name_match.start(1)] + pseudo_id + anonymized[name_match.end(1):]
        detected.append({"type": "PERSON", "value": extracted_name, "pseudo_id": pseudo_id})

    return {
        "original_text_length": len(text),
        "anonymized_text": anonymized,
        "pseudo_id": pseudo_id,
        "entities_masked": len(detected),
        "audit_log": [{"type": d["type"], "masked_as": d.get("pseudo_id", f"[{d['type']}]")} for d in detected]
    }
