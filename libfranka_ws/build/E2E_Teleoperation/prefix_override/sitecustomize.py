import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/kai/NewDisk/Kai_thesis/Master_Thesis_E2E_RL_Teleop/libfranka_ws/install/E2E_Teleoperation'
