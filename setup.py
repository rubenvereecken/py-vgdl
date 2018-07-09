import os
from setuptools import setup, find_packages

setup(
    name='vgdl',
    version="1.0.1",
    description='A video game description language (VGDL) built on top pf pygame',
    author='Tom Schaul',
    url='https://github.com/schaul/py-vgdl',
    packages=find_packages(),
    install_requires=['pygame']
)

