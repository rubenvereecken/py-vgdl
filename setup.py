import os
from setuptools import setup, find_packages

# Feature-specific dependencies
extras = {
    'gym': ['gym'],
    'pybrain': ['pybrain'],
    'human': ['gym', 'gym_recording'],
}

# Also add an 'all' group which just install everything
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(
    name='vgdl',
    version="1.0.1",
    description='A video game description language (VGDL) built on top pf pygame',
    author='Tom Schaul',
    url='https://github.com/schaul/py-vgdl',
    packages=find_packages(),
    install_requires=['pygame'],
    package_data={
        # setuptools forces us to have data files within the respective package
        'vgdl': ['games/', 'sprites/']
    },
    extras_require=extras
)
