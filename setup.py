"""Build the optional Cython-generated accelerator from its portable C source."""

from __future__ import annotations

import os

from setuptools import Extension, setup

source = "src/rdsframe/_cython_core.c"
extensions = [Extension("rdsframe._cython_core", [source], optional=True)]

# Maintainers can regenerate the C source explicitly after editing the .pyx;
# ordinary source installs do not require Cython and compile failure is optional.
if os.environ.get("RDSFRAME_USE_CYTHON") == "1":
    from Cython.Build import cythonize

    extensions[0].sources = ["src/rdsframe/_cython_core.pyx"]
    extensions = cythonize(extensions, compiler_directives={"language_level": 3})

setup(ext_modules=extensions)
