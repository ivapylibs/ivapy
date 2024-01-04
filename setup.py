#!/usr/bin/env/python
"""!
@defgroup   IVAPy IVAPy

@brief  Utility functions and glue code for IVALab python libraries. 

Contains generic class instances and useful utility functions that support the
greater IVALab codebase.  Using it simplifies implementation.
"""

from setuptools import setup, find_packages

setup(
    name="ivapy",
    version="1.0.1",
    description="General purpose modules that are useful.",
    author="IVALab",
    packages=find_packages(),
    install_requires=[
        "dataclasses",
        "numpy",
        "scipy",
        "opencv-contrib-python",
        "matplotlib",
        "roipoly"
    ],
)
