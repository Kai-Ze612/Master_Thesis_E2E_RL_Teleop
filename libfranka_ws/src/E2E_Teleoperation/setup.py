from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'E2E_Teleoperation'

setup(
    name=package_name,
    version='0.0.0',
    # This finds 'E2E_Teleoperation' and 'E2E_Teleoperation.nodes'
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Optional: Configs
        (os.path.join('share', package_name, 'config'), glob('E2E_Teleoperation/config/*.py')),
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
            'local_robot = E2E_Teleoperation.nodes.local_robot_sim:main',
            'agent_node = E2E_Teleoperation.nodes.agent:main',
        ],
    },
)