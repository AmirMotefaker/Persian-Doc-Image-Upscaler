from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealESRGANCandidate:
    key: str = "realesrgan-doc"
    name: str = "Real-ESRGAN Document Candidate"
    status: str = "benchmark-only"
    default_allowed: bool = False
    risk: str = "may hallucinate or alter Persian glyphs and digits"


def describe() -> dict[str, str | bool]:
    candidate = RealESRGANCandidate()
    return {
        "key": candidate.key,
        "name": candidate.name,
        "status": candidate.status,
        "default_allowed": candidate.default_allowed,
        "risk": candidate.risk,
    }
