from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'E2E_Teleoperation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Optional: Add config files if you have them
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Optional: Add additional resources
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kai',
    maintainer_email='ge62meq@mytum.de',
    description='End-to-End Teleoperation Framework',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'remote_node = E2E_Teleoperation.nodes.remote_robot:main',
            'local_node = E2E_Teleoperation.nodes.local_robot_sim:main',
            'agent_node = E2E_Teleoperation.nodes.agent:main',
            'remote_sim = E2E_Teleoperation.nodes.remote_sim:main',
        ],
    },
)