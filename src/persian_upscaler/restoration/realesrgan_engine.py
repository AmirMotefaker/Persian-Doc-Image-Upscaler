from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def build_command(
    executable: Path,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    model_name: str = "realesrgan-x4plus",
    scale: int = 4,
) -> list[str]:
    return [
        str(executable),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model_name,
        "-s",
        str(scale),
        "-m",
        str(model_dir),
        "-f",
        "png",
    ]


def restore_file(
    input_path: Path,
    output_path: Path,
    executable: Path,
    model_dir: Path,
    model_name: str = "realesrgan-x4plus",
    scale: int = 4,
    timeout_seconds: int = 240,
) -> Path:
    executable = executable.resolve()
    model_dir = model_dir.resolve()
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not executable.is_file():
        raise RuntimeError(f"Real-ESRGAN executable not found: {executable}")
    if not input_path.is_file():
        raise RuntimeError(f"Input image not found: {input_path}")

    param = model_dir / f"{model_name}.param"
    weights = model_dir / f"{model_name}.bin"
    if not param.is_file() or not weights.is_file():
        raise RuntimeError(
            f"Real-ESRGAN model files are missing for {model_name}: {model_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    command = build_command(
        executable,
        input_path,
        output_path,
        model_dir,
        model_name=model_name,
        scale=scale,
    )
    print(
        "[P2] Real-ESRGAN benchmark candidate start "
        f"model={model_name} scale={scale}x",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=executable.parent,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown Real-ESRGAN error"
        raise RuntimeError(
            f"Real-ESRGAN candidate failed with exit {completed.returncode}: {stderr}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Real-ESRGAN candidate produced no output image")

    print(f"[P2] Real-ESRGAN candidate done output={output_path}", flush=True)
    return output_path
