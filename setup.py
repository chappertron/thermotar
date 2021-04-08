#!/usr/bin/env python

from setuptools import setup

setup(
    name='thermotar',
    version='0.0.1',
    author='Aidan C',
    packages=['thermotar'],
    scripts=['thermotar/bin/pot_profile','thermotar/bin/check_chunk'],
    url='',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[
       "pandas",
       "numpy",
   ],
)
