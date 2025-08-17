from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("bgp_cython.pyx", compiler_directives={'boundscheck': False, 'wraparound': False}),
    include_dirs=[np.get_include()]
)

'''Run the following:

python setup_cython.py build_ext --inplace

'''
