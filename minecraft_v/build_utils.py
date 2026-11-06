import copy
import json
import os
from pathlib import Path
from typing import Any

from litemapy import Region
from minecraft_v.placement_engine.ir import ComponentList

# Override per build (e.g. synth.sh --artifacts-dir) so parallel builds don't share files.
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("MINECRAFT_V_ARTIFACTS_DIR", "build/artifacts"))
DEFAULT_SNAPSHOT_DIR = DEFAULT_ARTIFACTS_DIR / "snapshots"


def save_schematic_artifact(filename: str, region: Region, snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out = snapshot_dir / filename
    # Saving renumbers the palette in place; a live workspace's writers and indexes would keep the old ids.
    region = copy.copy(region)
    region._Region__blocks = region._Region__blocks.copy()
    region._Region__palette = list(region._Region__palette)
    schematic = region.as_schematic(name=filename, author="MineRTL")
    schematic.save(str(out))
    return out


def save_artifact(filename: str, data: Any, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, indent: int | None = 2) -> Path:
    """indent=None writes one line, several times faster for big data: json only indents in Python."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / filename
    out.write_text(json.dumps(data, indent=indent))
    return out


def save_build_artifacts(
        component_list: ComponentList,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> str:
    """Writes component_list.json; returns its text."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "component_list.json"
    text = component_list.model_dump_json(indent=2)
    out.write_text(text)
    return text
