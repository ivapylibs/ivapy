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
