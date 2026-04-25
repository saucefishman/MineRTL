import json
from pathlib import Path
from typing import Any

from minecraft_v.placement_ir import ComponentList

DEFAULT_ARTIFACTS_DIR = Path("build/artifacts")


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
