from __future__ import annotations


def describe() -> dict[str, str | bool]:
    return {
        "key": "swinir-doc",
        "name": "SwinIR Document Candidate",
        "status": "benchmark-only",
        "default_allowed": False,
        "note": "requires Persian text fidelity benchmark before product use",
    }
