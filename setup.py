import sys
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from numpy import get_include
from os.path import join


cython_directives = {
    "language_level": 3,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
}

# Platform-specific OpenMP flags
extra_compile_args = []
extra_link_args = []
if sys.platform == "win32":
    # MSVC: use /openmp
    extra_compile_args = ["/openmp"]
    extra_link_args = []
else:
    # GCC/Clang: use -fopenmp
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "gosp." + fn[:-4],  # gosp.parallel, gosp.bgp, etc.
        [join("gosp", fn)],
        include_dirs=[get_include()], 
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    for fn in [
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

setup(
    name="gosp",
    version="2.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=cython_directives),
    zip_safe=False,
)
