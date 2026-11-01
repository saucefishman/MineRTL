"""$write/$display as probes: read driver blocks at run time, no wire added.

Yosys keeps each call as a $print cell. Its arguments stay in the design and the cell itself is dropped;
the "probes" of <out>.server.json say which blocks to sample and how to print them, for `mchprs_headless --probes`.
"""
from __future__ import annotations

import re

from litemapy import BlockState, Region

from minecraft_v.cell_library import CELL_TYPE_MAP, SCHEMATIC_MAP
from minecraft_v.models import Module
from minecraft_v.placement_engine.ir import Direction
from minecraft_v.progress import log

DEBUG_CELLS = ("$print", "$check")
# Solid and a full cube in MCHPRS as in vanilla, and not used by wire (wool) or gates (stone, quartz).
PROBE_MARKER = BlockState("minecraft:lime_concrete")

_FIELD = re.compile(r"([<>=])(.)(\d*)([bodhc])[-+ ]?#?_?([us]?)$")
_ATTACHED = {"north": (0, 0, 1), "south": (0, 0, -1), "east": (-1, 0, 0), "west": (1, 0, 0)}


def parse_format(fmt: str) -> list[dict]:
    """Yosys's FORMAT into tokens; a field's "bits" index the probe's data, LSB first. ValueError if unsupported."""
    tokens: list[dict] = []
    text = ""
    next_bit = i = 0
    while i < len(fmt):
        if fmt[i] != "{":
            text += fmt[i]
            i += 2 if fmt[i] == "}" else 1
        elif fmt[i + 1] == "{":
            text += "{"
            i += 2
        else:
            end = fmt.index("}", i)
            width, spec = fmt[i + 1:end].split(":", 1)
            field = _FIELD.match(spec)
            if field is None:
                raise ValueError(f"unsupported field {{{width}:{spec}}}")
            if text:
                tokens.append({"text": text})
                text = ""
            justify, pad, min_width, radix, sign = field.groups()
            bits = list(range(next_bit, next_bit + int(width)))
            if radix == "c" and len(bits) > 8:  # %s: one field per byte, first character in the top byte
                tokens += [{"bits": bits[low:low + 8], "radix": "c"} for low in reversed(range(0, len(bits), 8))]
            else:
                tokens.append({"bits": bits, "radix": "x" if radix == "h" else radix, "signed": sign == "s",
                               "width": int(min_width or 0), "pad": pad, "justify": justify})
            next_bit += int(width)
            i = end + 1
    if text:
        tokens.append({"text": text})
    return tokens


def _bit_drivers(module: Module) -> dict[int, tuple[str, str]]:
    drivers: dict[int, tuple[str, str]] = {}
    for port_name, port in module.ports.items():
        if port.direction == "input":
            for i, bit in enumerate(port.bits):
                if isinstance(bit, int):
                    pin = port_name if len(port.bits) == 1 else f"{port_name}[{i}]"
                    drivers[bit] = (pin, pin)  # the builder gives each bit its own pin component
    for cell_name, cell in module.cells.items():
        if cell.type not in CELL_TYPE_MAP:
            continue
        for pin in SCHEMATIC_MAP[CELL_TYPE_MAP[cell.type]].pins:
            if pin.direction == Direction.OUT:
                for bit in cell.connections.get(pin.name, []):
                    if isinstance(bit, int):
                        drivers[bit] = (cell_name, pin.name)
    return drivers


def extract_probes(module: Module) -> list[dict]:
    """Probe specs for the module's $print cells in print order, with driver endpoints instead of positions."""
    drivers = _bit_drivers(module)

    def source(bit: int | str, where: str) -> dict:
        if isinstance(bit, str):
            return {"const": 1 if bit == "1" else 0}
        if bit not in drivers:
            log(f"[probes] WARNING: {where}: undriven bit reads as 0")
            return {"const": 0}
        return {"driver": drivers[bit]}

    probes = []
    # Lower PRIORITY runs later: source order within a clocked block.
    for name, cell in sorted(module.cells.items(), key=lambda kv: -int(kv[1].parameters.get("PRIORITY", "0"), 2)):
        if cell.type != "$print":
            continue
        where = str(cell.attributes.get("src", name)).rsplit("/", 1)[-1]
        triggers = cell.connections["TRG"]
        if len(triggers) != 1:
            log(f"[probes] WARNING: {where}: skipped, needs one clock edge ({len(triggers)} triggers)")
            continue
        try:
            tokens = parse_format(cell.parameters["FORMAT"])
        except ValueError as e:
            log(f"[probes] WARNING: {where}: skipped, {e}")
            continue
        rising = int(cell.parameters["TRG_POLARITY"], 2) & 1
        probes.append({
            "name": where,
            "format": tokens,
            "trigger": source(triggers[0], where),
            "edge": "rising" if rising else "falling",
            "enable": source(cell.connections["EN"][0], where),
            "data": [source(bit, where) for bit in cell.connections["ARGS"]],
        })
    return probes


def probe_components(probes: list[dict]) -> set[str]:
    """Components whose outputs get sampled: their logic must settle within the clock period."""
    return {entry["driver"][0] for p in probes for entry in (p["enable"], *p["data"]) if "driver" in entry}


def _mark(workspace: Region, pos: tuple[int, int, int]) -> None:
    """Swap the block the tapped torch or repeater stands on or hangs from for the marker."""
    block = workspace[pos]
    offset = _ATTACHED[block["facing"]] if block.id == "minecraft:redstone_wall_torch" else (0, -1, 0)
    support = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
    if workspace[support].id in ("minecraft:air", "minecraft:redstone_wire"):
        log(f"[probes] WARNING: no block to mark under tap {pos}")
        return
    workspace[support] = PROBE_MARKER


def resolve_probes(probes: list[dict], positions: dict[tuple[str, str], tuple[int, int, int]],
                   workspace: Region, low: list[int]) -> tuple[list[dict], int]:
    """Resolve drivers to blocks relative to `low` and mark them. Returns the probes and the number of blocks tapped."""
    tapped: set[tuple[int, int, int]] = set()

    def resolve(entry: dict) -> dict:
        if "const" in entry:
            return entry
        pos = positions[tuple(entry["driver"])]
        kind = workspace[pos].id
        if kind not in ("minecraft:repeater", "minecraft:redstone_torch", "minecraft:redstone_wall_torch"):
            raise RuntimeError(f"probe tap {entry['driver']} at {pos} is {kind}, not a torch or repeater")
        tapped.add(pos)
        return {"pos": [c - lo for c, lo in zip(pos, low)]}

    out = [{**p, "trigger": {**resolve(p["trigger"]), "edge": p["edge"]}, "enable": resolve(p["enable"]),
            "data": [resolve(d) for d in p["data"]]} for p in probes]
    for p in out:
        del p["edge"]
    for pos in sorted(tapped):
        _mark(workspace, pos)
    return out, len(tapped)
