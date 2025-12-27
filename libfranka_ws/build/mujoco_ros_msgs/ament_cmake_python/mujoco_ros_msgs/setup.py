from setuptools import find_packages
from setuptools import setup

setup(
    name='mujoco_ros_msgs',
    version='0.9.0',
    packages=find_packages(
        include=('mujoco_ros_msgs', 'mujoco_ros_msgs.*')),
)
