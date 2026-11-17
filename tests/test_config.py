"""settings.yaml overrides written by the web UI, and value validation."""

import pytest
import yaml

from field_mapper.config import load_config, overrides_path_for, save_overrides, validate_override


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("team_number: 0\ncamera:\n  index: 0\n  width: 1280\ndetector:\n  confidence: 0.5\n")
    return p


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.web.port == 5805 and cfg.source.type == "camera"


def test_overrides_merge_on_top(cfg_file):
    save_overrides(cfg_file, validate_override(cfg_file, {"team_number": 1234, "camera": {"index": 2}}))
    cfg = load_config(cfg_file)
    assert cfg.team_number == 1234
    assert cfg.camera.index == 2
    assert cfg.camera.width == 1280  # untouched keys come from config.yaml
    # A second save merges with the first rather than replacing it.
    save_overrides(cfg_file, validate_override(cfg_file, {"detector": {"confidence": 0.3}}))
    cfg = load_config(cfg_file)
    assert (cfg.team_number, cfg.camera.index, cfg.detector.confidence) == (1234, 2, 0.3)
    saved = yaml.safe_load(overrides_path_for(cfg_file).read_text())
    assert saved == {"team_number": 1234, "camera": {"index": 2}, "detector": {"confidence": 0.3}}


def test_overrides_can_be_ignored(cfg_file):
    save_overrides(cfg_file, {"team_number": 99})
    assert load_config(cfg_file, overrides=False).team_number == 0


def test_values_from_the_browser_are_coerced(cfg_file):
    merged = validate_override(cfg_file, {"team_number": "254", "detector": {"confidence": "0.4"}, "networktables": {"enabled": "false"}})
    save_overrides(cfg_file, merged)
    cfg = load_config(cfg_file)
    assert cfg.team_number == 254 and cfg.detector.confidence == 0.4 and cfg.networktables.enabled is False


@pytest.mark.parametrize(
    "bad, message",
    [({"team_number": "abc"}, "team_number"), ({"camera": {"zoom": 3}}, "zoom"), ({"camera": 5}, "camera")],
)
def test_bad_values_are_rejected_and_not_saved(cfg_file, bad, message):
    with pytest.raises(ValueError, match=message):
        validate_override(cfg_file, bad)
    assert not overrides_path_for(cfg_file).exists()
