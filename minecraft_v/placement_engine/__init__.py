__all__ = ["build_litematic_from_component_list"]


def __getattr__(name: str):  # lazy: the builder imports build_utils, which imports .ir from this package
    if name in __all__:
        from .builder import build_litematic_from_component_list
        return build_litematic_from_component_list
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
