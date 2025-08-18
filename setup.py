"""setup.py: Builds Cython (.pyx) files into importable Python modules"""

from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include

pyx_files = [
    "gosp_pipeline.pyx",
    "bgp.pyx",
    "tgp.pyx",
    "tcp.pyx",
    "rastio.pyx",
    "parallel.pyx",
    "skip_bgp.pyx",
]

cython_directives = {
    "boundscheck": False,
    "wraparound":  False,
    "nonecheck":   False,
    "language_level": 3
}

extensions = cythonize(
    pyx_files,
    compiler_directives=cython_directives,
    include_path=[get_include()],  # NumPy headers
)



setup(
    name="gosp",
    version="1.1",
    packages=["gosp"],
    package_dir={"gosp": "src/python_scripts/gosp"},
    ext_modules=extensions
)