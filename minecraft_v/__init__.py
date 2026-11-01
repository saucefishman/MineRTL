# Importing the placement engine takes over a second: not until something asks, so `python -m minecraft_v`
# can show progress first.
__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Component",
    "ComponentList",
    "ComponentType",
    "Direction",
    "Footprint",
    "NetConnection",
    "NetEndpoint",
    "PinRef",
    "PlacementHints",
    "load_component_list",
]


def __getattr__(name: str):
    if name in __all__:
        from minecraft_v.placement_engine import ir
        return getattr(ir, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
