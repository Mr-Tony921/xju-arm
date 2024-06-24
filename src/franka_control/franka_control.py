import numpy as np
from move_group_custom import MoveGroup

# param
TRAJECTORY_FILE = f"../../data/droid_episode_1_trajectory.npz"
STRIDE = 5
READY = [0, -45, 0, -135, 0, 90, 45]
READY_rad = [r * np.pi / 180 for r in READY]

POSITION_TOLERANCE = 0.0001
JOINT_TOLERANCE = 0.00001
DEFAULT_JOINT_IMPEDANCE = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]
HARD_JOINT_IMPEDANCE = [6000.0, 12000.0, 6000.0, 10000.0, 5000.0, 8000.0, 2000.0]
SOFT_JOINT_IMPEDANCE = [100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 10.0]

def main():
    with np.load(TRAJECTORY_FILE) as trj:
        poses = trj['poses']
        grippers = trj['grippers']

    move_group = MoveGroup()
    # move_group.set_tolerance(joint=JOINT_TOLERANCE, position=POSITION_TOLERANCE)
    # move_group.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)
    print("GOTO READY JOINTS")
    move_group.plan_and_execute_joints(READY_rad)

    # for i, (pose, gripper) in enumerate(zip(poses, grippers)):
    #     if i == len(poses) - 1 or i % STRIDE == 0:
    #         move_group.combine_interface(pose=pose, gripper=gripper)

    move_group.trajectory_interface(poses)

if __name__ == "__main__":
    main()
