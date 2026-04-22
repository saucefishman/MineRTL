from enum import Enum
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

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

class PinRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    direction: Direction
    side: CardinalDirection | None = None
    offset: tuple[int, int, int] = (0, 0, 0)

class Footprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = 1
    height: int = 1
    depth: int = 1
    layer: str | None = None

class PlacementHints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preferred_pos: tuple[int, int, int] | None = None
    orientation: str | None = None
    fixed: bool = False
    region: str | None = None
    keepout: int = 0

class Component(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: ComponentType
    pins: list[PinRef]
    params: dict = Field(default_factory=dict)
    footprint: Footprint = Field(default_factory=Footprint)
    hints: PlacementHints = Field(default_factory=PlacementHints)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_unique_pin_names(self):
        pin_names = [pin.name for pin in self.pins]
        if len(pin_names) != len(set(pin_names)):
            raise ValueError(f"dup pin names in {self.id}")
        return self

class NetEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component_id: str
    pin_name: str

class NetConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    net_id: str
    source: NetEndpoint
    sinks: list[NetEndpoint]

class ComponentList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    components: list[Component]
    nets: list[NetConnection]

    @model_validator(mode="before")
    @classmethod
    def default_schema_version(cls, data):
        if isinstance(data, dict) and "schema_version" not in data:
            updated = dict(data)
            updated["schema_version"] = CURRENT_SCHEMA_VERSION
            return updated
        return data

    @model_validator(mode="after")
    def validate_graph(self):
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"bad schema {self.schema_version}, need {CURRENT_SCHEMA_VERSION}")
        components = self.components
        nets = self.nets
        component_ids = [component.id for component in components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("duplicate component id")
        net_ids = [net.net_id for net in nets]
        if len(net_ids) != len(set(net_ids)):
            raise ValueError("duplicate net id")
        pin_index = {}
        for component in components:
            pin_index[component.id] = {pin.name: pin.direction for pin in component.pins}
        for net in nets:
            if len(net.sinks) == 0:
                raise ValueError(f"net {net.net_id} has no sinks")
            source_direction = _resolve_endpoint_direction(pin_index, net.source, net.net_id)
            if source_direction != Direction.OUT:
                raise ValueError(f"Net '{net.net_id}' source '{net.source.component_id}:{net.source.pin_name}' source not OUT")
            for sink in net.sinks:
                sink_direction = _resolve_endpoint_direction(pin_index, sink, net.net_id)
                if sink_direction != Direction.IN:
                    raise ValueError(f"Net '{net.net_id}' sink '{sink.component_id}:{sink.pin_name}' sink not IN")
        return self

    @classmethod
    def from_json(cls, payload):
        return cls.model_validate_json(payload)

    @classmethod
    def from_dict(cls, payload):
        return cls.model_validate(payload)

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
        raise ValueError(f"net {net_id} bad component {endpoint.component_id}")
    direction = component_pins.get(endpoint.pin_name)
    if direction is None:
        raise ValueError(f"net {net_id} bad pin {endpoint.component_id}:{endpoint.pin_name}")
    return direction
