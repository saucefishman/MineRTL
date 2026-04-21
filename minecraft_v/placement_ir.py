from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

CURRENT_SCHEMA_VERSION = "0.1"

class ComponentType(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"
    DFF = "DFF"
    INPUT_PIN = "INPUT_PIN"
    OUTPUT_PIN = "OUTPUT_PIN"
    CUSTOM = "CUSTOM"

class Direction(str, Enum):
    IN = "IN"
    OUT = "OUT"

@dataclass
class PinRef:
    name: str
    direction: Direction
    side: str = None
    offset: tuple = (0, 0, 0)

@dataclass
class Footprint:
    width: int
    height: int
    depth: int
    layer: str = None

@dataclass
class PlacementHints:
    preferred_pos: tuple = None
    orientation: str = None
    fixed: bool = False
    region: str = None
    keepout: int = 0

@dataclass
class Component:
    id: str
    type: ComponentType
    pins: list[PinRef]
    params: dict = field(default_factory=dict)
    footprint: Footprint = field(default_factory=lambda: Footprint(1, 1, 1))
    hints: PlacementHints = field(default_factory=PlacementHints)
    metadata: dict = field(default_factory=dict)

@dataclass
class NetEndpoint:
    component_id: str
    pin_name: str

@dataclass
class NetConnection:
    net_id: str
    source: NetEndpoint
    sinks: list[NetEndpoint]

@dataclass
class ComponentList:
    schema_version: str
    components: list[Component]
    nets: list[NetConnection]

    @classmethod
    def from_dict(cls, payload):
        schema_version = str(payload.get("schema_version", CURRENT_SCHEMA_VERSION))
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"bad schema {schema_version}, need {CURRENT_SCHEMA_VERSION}"
            )
        raw_components = payload.get("components")
        raw_nets = payload.get("nets")
        if not isinstance(raw_components, list):
            raise ValueError("components list needed")
        if not isinstance(raw_nets, list):
            raise ValueError("nets list needed")
        components = [component_from_dict(c) for c in raw_components]
        nets = [net_from_dict(n) for n in raw_nets]
        component_ids = [component.id for component in components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("duplicate component id")
        net_ids = [net.net_id for net in nets]
        if len(net_ids) != len(set(net_ids)):
            raise ValueError("duplicate net id")
        pin_index = {}
        for component in components:
            pin_names = [pin.name for pin in component.pins]
            if len(pin_names) != len(set(pin_names)):
                raise ValueError(f"dup pin names in {component.id}")
            pin_index[component.id] = {pin.name: pin.direction for pin in component.pins}
        for net in nets:
            source_direction = _resolve_endpoint_direction(pin_index, net.source, net.net_id)
            if source_direction != Direction.OUT:
                raise ValueError(
                    f"Net '{net.net_id}' source '{net.source.component_id}:{net.source.pin_name}' "
                    "source not OUT"
                )
            if len(net.sinks) == 0:
                raise ValueError(f"net {net.net_id} has no sinks")
            for sink in net.sinks:
                sink_direction = _resolve_endpoint_direction(pin_index, sink, net.net_id)
                if sink_direction != Direction.IN:
                    raise ValueError(
                        f"Net '{net.net_id}' sink '{sink.component_id}:{sink.pin_name}' "
                        "sink not IN"
                    )
        return cls(
            schema_version=schema_version,
            components=components,
            nets=nets,
        )

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))

def component_from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError("component must be object")
    component_id = _require_str(raw, "id", "component")
    type_raw = _require_str(raw, "type", f"component '{component_id}'")
    pins_raw = raw.get("pins")
    if not isinstance(pins_raw, list):
        raise ValueError(f"{component_id} pins should be list")
    pins = [pin_from_dict(pin, component_id) for pin in pins_raw]
    footprint = footprint_from_dict(raw.get("footprint", {}), component_id)
    hints = hints_from_dict(raw.get("hints", {}), component_id)
    params = _optional_dict(raw.get("params"), "params", component_id)
    metadata = _optional_dict(raw.get("metadata"), "metadata", component_id)
    return Component(
        id=component_id,
        type=ComponentType(type_raw),
        pins=pins,
        params=params,
        footprint=footprint,
        hints=hints,
        metadata=metadata,
    )

def pin_from_dict(raw, component_id):
    if not isinstance(raw, dict):
        raise ValueError(f"pin for {component_id} must be object")
    name = _require_str(raw, "name", f"component '{component_id}' pin")
    direction_raw = _require_str(raw, "direction", f"component '{component_id}' pin '{name}'")
    side = raw.get("side")
    if side is not None and not isinstance(side, str):
        raise ValueError(f"{component_id}.{name} side should be string")
    offset = _parse_xyz(raw.get("offset", (0, 0, 0)), f"component '{component_id}' pin '{name}'")
    return PinRef(
        name=name,
        direction=Direction(direction_raw),
        side=side,
        offset=offset,
    )

def footprint_from_dict(raw, component_id):
    if not isinstance(raw, dict):
        raise ValueError(f"{component_id} footprint should be object")
    width = _optional_int(raw.get("width"), default=1, field_name="width", context=component_id)
    height = _optional_int(raw.get("height"), default=1, field_name="height", context=component_id)
    depth = _optional_int(raw.get("depth"), default=1, field_name="depth", context=component_id)
    layer = raw.get("layer")
    if layer is not None and not isinstance(layer, str):
        raise ValueError(f"{component_id} footprint layer should be string")
    return Footprint(width=width, height=height, depth=depth, layer=layer)

def hints_from_dict(raw, component_id):
    if not isinstance(raw, dict):
        raise ValueError(f"{component_id} hints should be object")
    preferred_pos = raw.get("preferred_pos")
    if preferred_pos is not None:
        preferred_pos = _parse_xyz(preferred_pos, f"component '{component_id}' hints preferred_pos")
    orientation = raw.get("orientation")
    if orientation is not None and not isinstance(orientation, str):
        raise ValueError(f"{component_id} orientation should be string")
    fixed = raw.get("fixed", False)
    if not isinstance(fixed, bool):
        raise ValueError(f"{component_id} fixed should be bool")
    region = raw.get("region")
    if region is not None and not isinstance(region, str):
        raise ValueError(f"{component_id} region should be string")
    keepout = _optional_int(raw.get("keepout"), default=0, field_name="keepout", context=component_id)
    return PlacementHints(
        preferred_pos=preferred_pos,
        orientation=orientation,
        fixed=fixed,
        region=region,
        keepout=keepout,
    )

def net_from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError("net must be object")
    net_id = _require_str(raw, "net_id", "net")
    source_raw = raw.get("source")
    sinks_raw = raw.get("sinks")
    if not isinstance(source_raw, dict):
        raise ValueError(f"net {net_id} source should be object")
    if not isinstance(sinks_raw, list):
        raise ValueError(f"net {net_id} sinks should be list")
    source = endpoint_from_dict(source_raw, f"net '{net_id}' source")
    sinks = [endpoint_from_dict(sink, f"net '{net_id}' sinks") for sink in sinks_raw]
    return NetConnection(net_id=net_id, source=source, sinks=sinks)

def endpoint_from_dict(raw, context):
    if not isinstance(raw, dict):
        raise ValueError(f"{context} should be object")
    component_id = _require_str(raw, "component_id", context)
    pin_name = _require_str(raw, "pin_name", context)
    return NetEndpoint(component_id=component_id, pin_name=pin_name)

def load_component_list(path_or_dict):
    if isinstance(path_or_dict, dict):
        payload = path_or_dict
    else:
        path = Path(path_or_dict)
        payload = json.loads(path.read_text(encoding="utf-8"))
    return ComponentList.from_dict(payload)

def _resolve_endpoint_direction(pin_index, endpoint, net_id):
    component_pins = pin_index.get(endpoint.component_id)
    if component_pins is None:
        raise ValueError(
            f"net {net_id} bad component {endpoint.component_id}"
        )
    direction = component_pins.get(endpoint.pin_name)
    if direction is None:
        raise ValueError(
            f"net {net_id} bad pin {endpoint.component_id}:{endpoint.pin_name}")
    return direction

def _require_str(raw, field_name, context):
    value = raw.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{context} missing {field_name}")
    return value

def _optional_int(value, default, field_name, context):
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{context} {field_name} should be int")
    return value

def _optional_dict(value, field_name, component_id):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{component_id} {field_name} should be object")
    return value

def _parse_xyz(value, context):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{context} should have 3 coords")
    if not all(isinstance(x, int) for x in value):
        raise ValueError(f"{context} coords should be ints")
    return (value[0], value[1], value[2])
