from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSpec:
    key: str
    display_name: str
    status: str
    target_use: str
    can_be_default: bool


AVAILABLE_ENGINES = (
    EngineSpec(
        key="persian-document-hdv2",
        display_name="PersianDocumentHDv2",
        status="active-baseline",
        target_use="deterministic Persian document baseline",
        can_be_default=True,
    ),
    EngineSpec(
        key="realesrgan-doc",
        display_name="Real-ESRGAN Document Candidate",
        status="benchmark-only",
        target_use="learned 4x restoration candidate; never default without fidelity gate",
        can_be_default=False,
    ),
    EngineSpec(
        key="swinir-doc",
        display_name="SwinIR Document Candidate",
        status="benchmark-only",
        target_use="learned text restoration candidate; never default without fidelity gate",
        can_be_default=False,
    ),
)


def get_engine_names() -> list[str]:
    return [engine.key for engine in AVAILABLE_ENGINES]
