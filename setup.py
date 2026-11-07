"""Builds the Cython kernels (minecraft_v/placement_engine/_astar.pyx, _signals.pyx, _place.pyx); metadata is in pyproject.toml."""
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [Extension(
            "minecraft_v.placement_engine._astar",
            ["minecraft_v/placement_engine/_astar.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        ), Extension(
            "minecraft_v.placement_engine._signals",
            ["minecraft_v/placement_engine/_signals.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
            extra_link_args=["-pthread"],
        ), Extension(
            "minecraft_v.placement_engine._place",
            ["minecraft_v/placement_engine/_place.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off", "-pthread"],  # rounds as numpy does, step by step
            extra_link_args=["-pthread"],
        )],
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False,
                             "cdivision": True, "initializedcheck": False},
    ),
)
