from setuptools import find_packages
from setuptools import setup

setup(
    name='panda_motion_generator_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('panda_motion_generator_msgs', 'panda_motion_generator_msgs.*')),
)
