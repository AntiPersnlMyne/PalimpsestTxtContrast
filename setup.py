"""setup.py: Builds Cython (.pyx) files into importable Python modules"""

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="gosp",
    version="1.0",
    packages=["gosp"],
    package_dir={"gosp": "src/python_scripts/gosp"},
    ext_modules=cythonize(
        [
            Extension(
                "gosp.test", # module name in Python
                ["src/python_scripts/gosp/test.pyx"],
                language="c",
            )
        ]
    ),
)
