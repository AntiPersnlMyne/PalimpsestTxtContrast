"""setup.py: Builds Cython (.pyx) files into importable Python modules"""

from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include
from os.path import join

PYX_DIR = "src/python_scripts/gosp"

# Build a full list of file paths
pyx_files = [
  join(PYX_DIR, fn) for fn in [
      "bgp.pyx",
      "tgp.pyx",
      "tcp.pyx",
      "rastio.pyx",
      "parallel.pyx",
      "skip_bgp.pyx",
      "file_utils.pyx",
      "math_utils.pyx",
  ]
]

cython_directives = {
    "language_level": 3,  # python3
    "boundscheck": False, # array bounds check
    "wraparound":  False, # disable negative indices
    "cdivision": True,     # C division semantics
    # "initializedcheck": False
    # "nonecheck":   False, # something, idk
}

extensions = cythonize(
    pyx_files,
    compiler_directives=cython_directives,
    include_path=[get_include()],  # NumPy headers
)



setup(
    name="gosp",
    version="1.3",
    packages=["gosp"],
    package_dir={"gosp": PYX_DIR},
    ext_modules=extensions
)