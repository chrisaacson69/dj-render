"""Where a spec's tracks and renders live, found from the spec file itself.

Specs live in the shared music library: `<library>/CyborgProjects/<project>/specs/x.json`,
with originals at `<library>/` and renders in `<project>/renders/`. A library is a folder
with an `analysis/` subfolder (the layout dj-score's `adapters/library` owns; repeated
here so the engine does not depend on dj-score).
"""

from __future__ import annotations

import os
from pathlib import Path


def library_root(path: Path) -> Path | None:
    for d in Path(path).resolve().parents:
        if (d / "analysis").is_dir():
            return d
    return None


def project_dir(spec_path: Path) -> Path | None:
    """`<project>/` for a spec at `<project>/specs/x.json`."""
    spec_dir = Path(spec_path).resolve().parent
    return spec_dir.parent if spec_dir.name == "specs" else None


def resolve_tracks_dir(spec_path: Path, declared: str | None, files: list[str]) -> str:
    """The first folder holding every track the spec names: the declared `tracks_dir`
    relative to the spec, the library root, the project's `tracks/`, then the declared
    path relative to the working directory (how older specs were written)."""
    spec_dir = Path(spec_path).resolve().parent
    candidates = []
    if declared:
        candidates.append(spec_dir / declared)
    candidates += [library_root(spec_path), project_dir(spec_path) and project_dir(spec_path) / "tracks"]
    if declared:
        candidates.append(Path(os.getcwd()) / declared)
    for d in candidates:
        if d is not None and all((d / f).exists() for f in files):
            return str(d)
    raise FileNotFoundError(f"no folder holds all tracks of {spec_path} (tried {[str(c) for c in candidates if c]})")


def default_render_path(spec_path: Path | None, name: str) -> str:
    """`<project>/renders/<name>` for a spec in a project, else `output/<name>`."""
    project = project_dir(spec_path) if spec_path else None
    return str(project / "renders" / name) if project else os.path.join("output", name)
