from sys import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from numpy import get_include
from os.path import join, dirname, abspath, exists
from os import makedirs


# =================
# Cython optimizers
# =================
cython_directives = {
    "language_level": 3,    # python3
    "boundscheck": False,   # skip array bounds check
    "wraparound": False,    # prevent negative indexing
    "cdivision": True,      # use c-style division
}


# =========================================
# OpenMP Compiler Flags (platform-specific)
# =========================================
extra_compile_args = []
extra_link_args = []
if platform == "win32":
    extra_compile_args = ["/openmp"]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]


# =================================
# 'build_ext' To Redict Build Files
# =================================
class build_ext_build_dir(build_ext):
    def build_extensions(self):
        # Ensure build directory exists
        build_dir = join(dirname(abspath(__file__)), "gosp", "build")
        if not exists(build_dir):
            makedirs(build_dir)

        # Redirect each extension's build output
        for ext in self.extensions:
            ext.name = "gosp.build." + ext.name.split(".")[-1]
            ext.sources = [join("gosp", src.split(".")[-1] + ".pyx") for src in ext.sources]

        # Proceed with normal build_ext
        super().build_extensions()


# =========================
# Extensions List (Sources)
# =========================
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


# ===============================
# Setup run by 'pip install -e .'
# ===============================
setup(
    name="gosp",
    version="3.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=cython_directives),
    zip_safe=False,
)
