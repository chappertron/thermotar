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
    description="An awesome package that does something",
    long_description=open("README.md").read(),
    install_requires=["pandas", "numpy", "matplotlib"],
)
