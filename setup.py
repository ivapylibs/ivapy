#!/usr/bin/env/python
"""!
@defgroup   IVAPy IVAPy

@brief  Utility functions and glue code for IVALab python libraries. 

Contains generic class instances and useful utility functions that support the
greater IVALab codebase.  Using it simplifies implementation.


@addtogroup IVAPy
@{
@defgroup   ivapy_testing IVAPy Testing Utilities

@brief  Utility functions and glue code for IVALab python test scripts. 

Contains helper functions that make test creation and coding easier (less lines)
to help focus on what is done more so than how to do it.


@defgroup   Display_CV  Display via OpenCV
@brief      Display-related utility functions built on opencv (cv2) libraries.



@}
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
