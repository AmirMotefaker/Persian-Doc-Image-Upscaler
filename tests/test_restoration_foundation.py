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
