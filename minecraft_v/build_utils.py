import json
from pathlib import Path
from typing import Any

from litemapy import Region
from minecraft_v.placement_engine.ir import ComponentList

DEFAULT_ARTIFACTS_DIR = Path("build/artifacts")
DEFAULT_SNAPSHOT_DIR = DEFAULT_ARTIFACTS_DIR / "snapshots"


def save_schematic_artifact(filename: str, region: Region, snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out = snapshot_dir / filename
    schematic = region.as_schematic(name=filename, author="MineRTL")
    schematic.save(str(out))
    return out


def save_artifact(filename: str, data: Any, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / filename
    out.write_text(json.dumps(data, indent=2))
    return out


def save_build_artifacts(
        component_list: ComponentList,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "component_list.json"
    out.write_text(component_list.model_dump_json(indent=2))
    return out
