import json
from enum import Enum
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path

CURRENT_SCHEMA_VERSION = "0.1"


class ComponentType(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    DFF = "DFF"
    DFFE = "DFFE"
    DLATCH = "DLATCH"
    FULL_ADDER = "FULLADDER"
    MUX = "MUX"
    INPUT_PIN = "INPUT_PIN"
    OUTPUT_PIN = "OUTPUT_PIN"
    CUSTOM = "CUSTOM"


class Direction(str, Enum):
    IN = "IN"
    OUT = "OUT"


class CardinalDirection(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


def _dump(value, as_json: bool):
    """A model as plain data: dicts, lists and (unless as_json) tuples and enum members, as pydantic's model_dump gives."""
    if isinstance(value, _Model):
        return {f.name: _dump(getattr(value, f.name), as_json) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value if as_json else value
    if isinstance(value, (list, tuple)):
        items = [_dump(v, as_json) for v in value]
        return items if as_json or isinstance(value, list) else tuple(items)
    if isinstance(value, dict):
        return {k: _dump(v, as_json) for k, v in value.items()}
    return value


class _Model:
    """Plain slotted dataclasses stand for the IR: a netlist of millions of gates holds millions of these, and a
    pydantic model costs ~480 bytes an object against ~60-280 here. The few pydantic calls the code makes remain."""
    __slots__ = ()

    def model_copy(self, update: dict | None = None):
        return replace(self, **update) if update else replace(self)

    def model_dump(self, mode: str = "python") -> dict:
        return _dump(self, mode == "json")

    def model_dump_json(self) -> str:
        return json.dumps(_dump(self, True), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def model_construct(cls, **values):
        """An instance of cls with exactly these fields, unvalidated (the rest at their defaults)."""
        obj = cls.__new__(cls)
        for f in fields(cls):
            if f.name in values:
                value = values[f.name]
            elif f.default_factory is not MISSING:
                value = f.default_factory()
            else:
                value = f.default
            setattr(obj, f.name, value)
        return obj


@dataclass(slots=True)
class PinRef(_Model):
    name: str
    direction: Direction
    side: CardinalDirection | None = None
    offset: tuple[int, int, int] = (0, 0, 0)
    const_value: str | None = None

    @classmethod
    def model_validate(cls, p: dict) -> "PinRef":
        """From model_dump(mode="json")'s form: enum values as strings, the offset as a list."""
        return cls(name=p["name"], direction=Direction(p["direction"]),
                   side=None if p.get("side") is None else CardinalDirection(p["side"]),
                   offset=tuple(p.get("offset", (0, 0, 0))), const_value=p.get("const_value"))


@dataclass(slots=True)
class Footprint(_Model):
    width: int = 1
    height: int = 1
    depth: int = 1
    layer: str | None = None


@dataclass(slots=True)
class PlacementHints(_Model):
    preferred_pos: tuple[int, int, int] | None = None
    orientation: str | None = None
    fixed: bool = False
    region: str | None = None
    keepout: int = 0


@dataclass(slots=True)
class Component(_Model):
    id: str
    type: ComponentType
    pins: list[PinRef]
    params: dict = field(default_factory=dict)
    footprint: Footprint = field(default_factory=Footprint)
    hints: PlacementHints = field(default_factory=PlacementHints)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len({pin.name for pin in self.pins}) != len(self.pins):
            raise ValueError(f"dup pin names in {self.id}")


@dataclass(slots=True)
class NetEndpoint(_Model):
    component_id: str
    pin_name: str


@dataclass(slots=True)
class NetConnection(_Model):
    net_id: str
    source: NetEndpoint
    sinks: list[NetEndpoint]


@dataclass(slots=True)
class ComponentList(_Model):
    schema_version: str
    components: list[Component]
    nets: list[NetConnection]

    def __post_init__(self):
        self.validate_graph()

    def validate_graph(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"bad schema {self.schema_version}, need {CURRENT_SCHEMA_VERSION}")
        by_id = {component.id: component for component in self.components}
        if len(by_id) != len(self.components):
            raise ValueError("duplicate component id")
        if len({net.net_id for net in self.nets}) != len(self.nets):
            raise ValueError("duplicate net id")
        wide: dict[str, dict[str, Direction]] = {}  # name -> direction for the components with many pins, made once
        for net in self.nets:
            if len(net.sinks) == 0:
                raise ValueError(f"net {net.net_id} has no sinks")
            if _endpoint_direction(by_id, wide, net.source, net.net_id) != Direction.OUT:
                raise ValueError(
                    f"Net '{net.net_id}' source '{net.source.component_id}:{net.source.pin_name}' source not OUT")
            for sink in net.sinks:
                if _endpoint_direction(by_id, wide, sink, net.net_id) != Direction.IN:
                    raise ValueError(f"Net '{net.net_id}' sink '{sink.component_id}:{sink.pin_name}' sink not IN")

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("a component list is a JSON object")
        unknown = set(payload) - {"schema_version", "components", "nets"}
        if unknown:
            raise ValueError(f"unknown component list fields {sorted(unknown)}")
        components = [Component(
            id=c["id"], type=ComponentType(c["type"]),
            pins=[PinRef.model_validate(p) for p in c["pins"]],
            params=c.get("params", {}), footprint=Footprint(**c.get("footprint", {})),
            hints=_hints(c.get("hints") or {}),
            metadata=c.get("metadata", {})) for c in payload["components"]]
        nets = [NetConnection(net_id=n["net_id"], source=NetEndpoint(**n["source"]), sinks=[NetEndpoint(**e) for e in n["sinks"]])
                for n in payload["nets"]]
        return cls(schema_version=payload.get("schema_version", CURRENT_SCHEMA_VERSION), components=components, nets=nets)

    model_validate = from_dict
    model_validate_json = from_json


def _hints(payload: dict) -> PlacementHints:
    preferred = payload.get("preferred_pos")
    return PlacementHints(**{**payload, "preferred_pos": None if preferred is None else tuple(preferred)})


def load_component_list(path_or_dict):
    if isinstance(path_or_dict, dict):
        payload = path_or_dict
    else:
        path = Path(path_or_dict)
        from minecraft_v.build_utils import load_artifact_json  # a function-level import: build_utils imports this module
        payload = load_artifact_json(path)
    return ComponentList.from_dict(payload)


def _endpoint_direction(by_id, wide, endpoint, net_id):
    component = by_id.get(endpoint.component_id)
    if component is None:
        raise ValueError(f"net {net_id} bad component {endpoint.component_id}")
    if len(component.pins) > 8:  # a macro's hundreds of pins: looked up by name, not scanned
        pins = wide.get(component.id)
        if pins is None:
            pins = wide[component.id] = {pin.name: pin.direction for pin in component.pins}
        direction = pins.get(endpoint.pin_name)
    else:
        direction = next((pin.direction for pin in component.pins if pin.name == endpoint.pin_name), None)
    if direction is None:
        raise ValueError(f"net {net_id} bad pin {endpoint.component_id}:{endpoint.pin_name}")
    return direction
