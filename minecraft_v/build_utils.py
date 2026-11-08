import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import IO, Any

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


_GZIP_LEVEL = 1  # JSON gzips about tenfold even at level 1, and it is several times faster than the default


def artifact_path(path: str | Path) -> Path:
    """path, or path + ".gz" when only that exists: scripts name artifacts as they were, compressed or not."""
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    return gz if not path.exists() and gz.exists() else path


def artifact_exists(path: str | Path) -> bool:
    return artifact_path(path).exists()


def open_artifact(path: str | Path, mode: str = "r") -> IO:
    """open() for a saved artifact; reads gzip transparently, by the name or by its .gz twin. Text mode by default."""
    path = artifact_path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode if "b" in mode else mode + "t", encoding=None if "b" in mode else "utf-8")
    return open(path, mode, encoding=None if "b" in mode else "utf-8")


def load_artifact_json(path: str | Path) -> Any:
    with open_artifact(path) as f:
        return json.load(f)


def save_artifact(filename: str, data: Any, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, indent: int | None = 2,
                  compress: bool = False) -> Path:
    """indent=None writes one line, several times faster for big data: json only indents in Python.
    compress: writes <filename>.gz instead (read it with open_artifact/load_artifact_json under either name)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / filename
    if compress:
        plain, out = out, out.with_name(out.name + ".gz")
        plain.unlink(missing_ok=True)  # an older build's, which a reader would take first
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=_GZIP_LEVEL) as f:
            json.dump(data, f, indent=indent)
        return out
    out.write_text(json.dumps(data, indent=indent))
    return out


def save_build_artifacts(
        component_list: ComponentList,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> str:
    """Writes component_list.json.gz, one line; returns the SHA-256 of the uncompressed JSON. Each part is dumped on its
    own: the whole text of a big list is gigabytes, and nothing here needs it in memory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "component_list.json").unlink(missing_ok=True)  # an older build's, which a reader would take first
    digest = hashlib.sha256()
    with gzip.open(artifacts_dir / "component_list.json.gz", "wb", compresslevel=_GZIP_LEVEL) as f:
        def put(data: bytes) -> None:
            f.write(data)
            digest.update(data)

        put(b'{"schema_version":%s,"components":[' % json.dumps(component_list.schema_version).encode())
        def put_all(items) -> None:
            for i, item in enumerate(items):
                put(item.model_dump_json().encode() if i == 0 else b"," + item.model_dump_json().encode())

        put_all(component_list.components)
        put(b'],"nets":[')
        put_all(component_list.nets)
        put(b"]}")
    return digest.hexdigest()
