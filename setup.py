#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="thermotar",
    version="0.0.1",
    author="Aidan C",
    packages=find_packages(),
    scripts=[
        "thermotar/bin/pot_profile",
        "thermotar/bin/check_chunk",
        "thermotar/bin/ave_thermo",
        "thermotar/bin/thermopl",
        "thermotar/bin/thermo_stats",
    ],
    url="",
    description="A package for analysing the output of simulations.",
    long_description=open("README.md").read(),
    # parquet is added to install pyarrow, so no warning about future.
    install_requires=[
        "pandas[performance,parquet]",
        "numpy",
        "scipy",
        "matplotlib",
        "typing-extensions",
    ],
)
