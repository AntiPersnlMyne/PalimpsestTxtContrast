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


# ============
# Source Files
# ============
pyx_files = [
    "bgp.pyx",
    "tgp.pyx",
    "tcp.pyx",
    "rastio.pyx",
    "parallel.pyx",
    "skip_bgp.pyx",
    "file_utils.pyx",
    "math_utils.pyx",
]


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
class BuildExt(build_ext):
    def build_extensions(self):
        # Ensure build directory exists
        build_dir = join(dirname(abspath(__file__)), "gosp", "build")
        if not exists(build_dir):
            makedirs(build_dir)

        # Redirect each extension's build output
        for ext in self.extensions:
            ext.name = "gosp.build." + ext.name.split(".")[-1]
            
        # Makes class call itself creating build_ext
        super().build_extensions()


# =========================
# Extensions List (Sources)
# =========================
extensions = [
    Extension(
        "gosp." + fn[:-4],  # module name ; e.g. gosp.bgp
        [join("gosp", "src", fn)],  # source location ; e.g. gosp.src.bgp
        include_dirs=[get_include()], 
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    for fn in pyx_files
]


# ===============================
# Setup run by 'pip install -e .'
# ===============================
setup(
    name="gosp",
    version="3.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=cython_directives),
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
