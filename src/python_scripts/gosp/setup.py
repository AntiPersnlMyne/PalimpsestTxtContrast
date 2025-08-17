from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(
        "bgp_cython.pyx", # name of cython file
        compiler_directives={'boundscheck': False, 'wraparound': False}),
    include_dirs=[np.get_include()]
)

'''Run the following:

python setup.py build_ext --inplace

'''
