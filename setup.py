"""Builds the Cython A* kernel (minecraft_v/placement_engine/_astar.pyx); metadata is in pyproject.toml."""
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [Extension(
            "minecraft_v.placement_engine._astar",
            ["minecraft_v/placement_engine/_astar.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        )],
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False,
                             "cdivision": True, "initializedcheck": False},
    ),
)
