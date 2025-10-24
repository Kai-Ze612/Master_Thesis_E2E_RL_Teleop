from setuptools import find_packages
from setuptools import setup

setup(
    name='multi_mode_control_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('multi_mode_control_msgs', 'multi_mode_control_msgs.*')),
)
