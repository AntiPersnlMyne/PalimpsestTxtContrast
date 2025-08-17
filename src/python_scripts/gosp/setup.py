from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs=[np.get_include()]),
]
# Build Cython
setup(
    name="bgp.py",
    ext_modules=cythonize(extensions),
    compiler_directives={'boundscheck': False, 'wraparound': False}
)

'''Run the following:

python src/python_scripts/gosp/setup.py build_ext --inplace

'''
