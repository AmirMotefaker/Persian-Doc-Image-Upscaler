from pathlib import Path

from persian_upscaler.restoration.realesrgan_engine import build_command, describe
from persian_upscaler.restoration.registry import AVAILABLE_ENGINES, get_engine_names


def test_restoration_registry_contains_expected_engines():
    names = get_engine_names()

    assert "persian-document-hdv2" in names
    assert "realesrgan-doc" in names
    assert "swinir-doc" in names
    assert len(AVAILABLE_ENGINES) == 3


def test_learned_candidates_are_not_default():
    learned = [engine for engine in AVAILABLE_ENGINES if engine.key != "persian-document-hdv2"]

    assert learned
    assert all(engine.status == "benchmark-only" for engine in learned)
    assert all(engine.can_be_default is False for engine in learned)


def test_realesrgan_candidate_remains_benchmark_only():
    metadata = describe()

    assert metadata["status"] == "benchmark-only"
    assert metadata["default_allowed"] is False
    assert "glyphs" in str(metadata["risk"])


def test_realesrgan_command_is_explicit_and_isolated():
    command = build_command(
        Path("tool/realesrgan-ncnn-vulkan.exe"),
        Path("fixture.png"),
        Path("candidate.png"),
        Path("tool/models"),
    )

    assert command[0].endswith("realesrgan-ncnn-vulkan.exe")
    assert command[command.index("-n") + 1] == "realesrgan-x4plus"
    assert command[command.index("-s") + 1] == "4"
    assert command[command.index("-f") + 1] == "png"
