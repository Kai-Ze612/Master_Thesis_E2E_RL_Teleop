import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/kai/Kai_Backup/Master_Study/Master_Thesis/Implementation/libfranka_ws/install/Model_based_Reinforcement_Learning_In_Teleoperation'
