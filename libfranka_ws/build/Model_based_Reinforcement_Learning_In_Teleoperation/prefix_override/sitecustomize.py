import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kaize/Downloads/Master_Study_Master_Thesis/libfranka_ws/install/Model_based_Reinforcement_Learning_In_Teleoperation'
