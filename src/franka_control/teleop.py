import numpy as np
from scipy.spatial.transform import Rotation as R
from move_group_custom import MoveGroup

import sys, select, tty, termios

settings = termios.tcgetattr(sys.stdin)
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

TRA_ACTION = 0.1
ROT_ACTION = 10
READY = [0, -45, 0, -135, 0, 90, 45]
READY_rad = [r * np.pi / 180 for r in READY]

move_group = MoveGroup()
gripper_action = 0

while True:
    key = getKey()

    ee_action = np.zeros([6])

    # Position
    if key == "i":  # +x
        ee_action[0] = TRA_ACTION
    elif key == "k":  # -x
        ee_action[0] = -TRA_ACTION
    elif key == "j":  # +y
        ee_action[1] = TRA_ACTION
    elif key == "l":  # -y
        ee_action[1] = -TRA_ACTION
    elif key == "u":  # +z
        ee_action[2] = TRA_ACTION
    elif key == "o":  # -z
        ee_action[2] = -TRA_ACTION

    # Rotation (axis-angle)
    if key == "1":
        ee_action[3:6] = (ROT_ACTION, 0, 0)
    elif key == "2":
        ee_action[3:6] = (-ROT_ACTION, 0, 0)
    elif key == "3":
        ee_action[3:6] = (0, ROT_ACTION, 0)
    elif key == "4":
        ee_action[3:6] = (0, -ROT_ACTION, 0)
    elif key == "5":
        ee_action[3:6] = (0, 0, ROT_ACTION)
    elif key == "6":
        ee_action[3:6] = (0, 0, -ROT_ACTION)

    # Gripper
    if key == "f":  # open gripper
        gripper_action = 0
    elif key == "g":  # close gripper
        gripper_action = 1

    if key == 'r':
        move_group.plan_and_execute_joints(READY_rad)
    elif key == 'q':
        break

    current_pose = move_group.get_current_pose_in_list()
    target_pose = current_pose.copy()
    target_pose[:3] += ee_action[:3]
    r0 = R.from_quat([current_pose[4], current_pose[5], current_pose[6], current_pose[3]])
    r = r0.as_euler('xyz', degrees=True) + ee_action[3:]
    temp = R.from_euler('xyz', r, degrees=True).as_quat()
    target_pose[3:] = temp[[3, 0, 1, 2]]

    move_group.combine_interface(pose=target_pose, gripper=gripper_action, sync=False)