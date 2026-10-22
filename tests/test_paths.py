"""Spec-relative resolution of tracks and renders (engine/paths.py)."""

import os

import pytest

from dj_render.paths import default_render_path, resolve_tracks_dir


@pytest.fixture
def library(tmp_path):
    """<lib>/a.mp3, <lib>/analysis/, <lib>/CyborgProjects/P/{specs,tracks}/."""
    (tmp_path / "analysis").mkdir()
    (tmp_path / "a.mp3").write_bytes(b"")
    proj = tmp_path / "CyborgProjects" / "P"
    (proj / "specs").mkdir(parents=True)
    (proj / "tracks").mkdir()
    (proj / "tracks" / "local.wav").write_bytes(b"")
    return tmp_path, proj / "specs" / "s.json"


def test_library_root_wins_from_any_working_directory(library, tmp_path_factory, monkeypatch):
    lib, spec = library
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert resolve_tracks_dir(spec, "../music/", ["a.mp3"]) == str(lib)


def test_project_tracks_folder(library):
    _, spec = library
    assert resolve_tracks_dir(spec, "old/tracks/", ["local.wav"]).endswith(os.path.join("P", "tracks"))


def test_declared_relative_to_spec_first(library):
    lib, spec = library
    (spec.parent / "here").mkdir()
    (spec.parent / "here" / "a.mp3").write_bytes(b"")
    assert resolve_tracks_dir(spec, "here", ["a.mp3"]) == str(spec.parent / "here")


def test_all_tracks_must_be_in_one_folder(library):
    _, spec = library
    with pytest.raises(FileNotFoundError):
        resolve_tracks_dir(spec, None, ["a.mp3", "local.wav"])


def test_renders_go_to_the_project(library):
    _, spec = library
    assert default_render_path(spec, "x.wav") == str(spec.parent.parent / "renders" / "x.wav")
    assert default_render_path(None, "x.wav") == os.path.join("output", "x.wav")
