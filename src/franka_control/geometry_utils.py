import numpy as np

def interpolate_trajectory(waypoints, steps):
    """
    Interpolates a trajectory between given waypoints.

    Args:
    waypoints (list): List of lists representing the waypoints. Each inner list should contain
                      the position (x, y, z) and orientation (w, x, y, z) in quaternion format.
                      Format: [x, y, z, w, i, j, k]
                      At least two waypoints are required. The first and last points of the
                      output trajectory will be the same as the first and last points of the
                      input trajectory, respectively.
    steps (int): Total number of points in the output trajectory, including the first and last
                 waypoints.

    Returns:
    list: A list of interpolated positions and orientations along the trajectory.
          Each element in the list is a tuple representing a point on the trajectory,
          with the format (x, y, z, w, i, j, k).

    Note:
    This function uses cubic spline interpolation for positions and spherical linear
    interpolation (Slerp) for orientations between waypoints.
    """
    assert len(waypoints) >= 2 and steps >= 2, "waypoints size and output steps must be at least 2"

    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    positions = np.array([wp[:3] for wp in waypoints])
    key_times = np.zeros(len(positions))
    for i in range(1, len(positions)):
        key_times[i] = key_times[i-1] + np.linalg.norm(positions[i]-positions[i-1])

    orientations = np.array([[wp[i] for i in [4,5,6,3]] for wp in waypoints])
    key_rots = R.from_quat(orientations)

    spline = CubicSpline(key_times, positions, axis=0)
    slerp = Slerp(key_times, key_rots)

    new_positions = spline(np.linspace(0, key_times[-1], steps))

    new_positions[0] = positions[0]
    new_positions[-1] = positions[-1]

    times = np.zeros(len(new_positions))
    for i in range(1, len(new_positions)):
        times[i] = times[i-1] + np.linalg.norm(new_positions[i]-new_positions[i-1])
    times *= key_times[-1] / times[-1]
    times[-1] = key_times[-1]
    interp_rots = slerp(times)
    new_orientations = interp_rots.as_quat()

    return np.array(
        [
            (
                new_positions[i][0],
                new_positions[i][1],
                new_positions[i][2],
                new_orientations[i][3],
                new_orientations[i][0],
                new_orientations[i][1],
                new_orientations[i][2],
            )
            for i in range(len(new_positions))
        ]
    )

def plot_frame(ax, poses, color, length=0.01):
    from scipy.spatial.transform import Rotation as R
    for pose in poses:
        mat = R.from_quat(pose[[4,5,6,3]]).as_dcm() # as_matrix() if scipy.__version__ >= 1.4.0
        for i in range(3):
            end = pose[:3] + length * mat[:,i]
            ax.plot(
                [pose[0], end[0]],
                [pose[1], end[1]],
                [pose[2], end[2]],
                color=color
            )

def compare_trajectory(ori_poses, new_poses):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ori_poses[0,0], ori_poses[0,1], ori_poses[0,2], color='r', s=25)
    ax.scatter(new_poses[0,0], new_poses[0,1], new_poses[0,2], color='g', s=25)
    plot_frame(ax, ori_poses, 'r')
    plot_frame(ax, new_poses, 'g')
    for i in range(len(ori_poses)-1):
        j = i + 1
        ax.plot(
            [ori_poses[i,0], ori_poses[j,0]],
            [ori_poses[i,1], ori_poses[j,1]],
            [ori_poses[i,2], ori_poses[j,2]],
            color="r", label='Origin trajectory' if i == 0 else ''
        )
        ax.plot(
            [new_poses[i,0], new_poses[j,0]],
            [new_poses[i,1], new_poses[j,1]],
            [new_poses[i,2], new_poses[j,2]],
            color="g", label='New trajectory' if i == 0 else ''
        )
    ax.set_title("Base Field")
    # ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    # plt.savefig(f'{DATA_FILE}.png')

def joints_to_pose(joints):
    """
    Apply forward kinematics method for franka panda robot.

    Args:
    joints (list): List of target joints value, must orderd from 0 to 6

    Returns:
    list: List of target eef pose / panda_link8 with the format (x, y, z, w, i, j, k).
    """
    assert len(joints) == 7, "joints length must be 7"

    def dh_params(joints):
        M_PI = np.pi

        # Create DH parameters (data given by maker franka-emika)
        dh = np.array(
            [
                [0, 0, 0.333, joints[0]],
                [-M_PI / 2, 0, 0, joints[1]],
                [M_PI / 2, 0, 0.316, joints[2]],
                [M_PI / 2, 0.0825, 0, joints[3]],
                [-M_PI / 2, -0.0825, 0.384, joints[4]],
                [M_PI / 2, 0, 0, joints[5]],
                [M_PI / 2, 0.088, 0.107, joints[6]],
            ]
        )
        return dh

    def TF_matrix(i, dh):
        # Define Transformation matrix based on DH params
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        q = dh[i][3]

        TF = np.array(
            [
                [np.cos(q), -np.sin(q), 0, a],
                [
                    np.sin(q) * np.cos(alpha),
                    np.cos(q) * np.cos(alpha),
                    -np.sin(alpha),
                    -np.sin(alpha) * d,
                ],
                [
                    np.sin(q) * np.sin(alpha),
                    np.cos(q) * np.sin(alpha),
                    np.cos(alpha),
                    np.cos(alpha) * d,
                ],
                [0, 0, 0, 1],
            ]
        )
        return TF

    dh_parameters = dh_params(joints)

    T_01 = TF_matrix(0,dh_parameters)
    T_12 = TF_matrix(1,dh_parameters)
    T_23 = TF_matrix(2,dh_parameters)
    T_34 = TF_matrix(3,dh_parameters)
    T_45 = TF_matrix(4,dh_parameters)
    T_56 = TF_matrix(5,dh_parameters)
    T_67 = TF_matrix(6,dh_parameters)
    T_07 = T_01@T_12@T_23@T_34@T_45@T_56@T_67 


    translation = T_07[:3,3]
    from transforms3d.quaternions import mat2quat
    quaternion = mat2quat(T_07[:3,:3])

    return np.concatenate((translation, quaternion))

def joints_to_pose_batch_tf(joints_batch):
    """
    Apply forward kinematics method for franka panda robot.

    Args:
    joints_batch (tf.Tensor): Tensor of shape (batch_size, 7) with target joints values

    Returns:
    tf.Tensor: Tensor of shape (batch_size, 7) with target eef poses in the format (x, y, z, w, i, j, k).
    """
    assert joints_batch.shape[1] == 7, "Each element in the batch must have 7 joint values"
    import tensorflow as tf
    def mat2euler(matrix):
        """
        Convert rotation matrices to Euler angles (XYZ order).

        Args:
        matrix (tf.Tensor): Tensor of shape (batch_size, 3, 3) representing rotation matrices.

        Returns:
        tf.Tensor: Tensor of shape (batch_size, 3) representing Euler angles (yaw, pitch, roll).
        """        
        # Calculate yaw (Z)
        yaw = tf.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
        
        # Calculate pitch (Y)
        pitch = tf.atan2(-matrix[:, 2, 0], tf.sqrt(tf.square(matrix[:, 2, 1]) + tf.square(matrix[:, 2, 2])))
        
        # Calculate roll (X)
        roll = tf.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
        
        return tf.stack([roll, pitch, yaw], axis=1)

    
    
    def dh_params(joints_batch):
        M_PI = np.pi

        batch_size = tf.shape(joints_batch)[0]
        zeros = tf.zeros([batch_size], dtype=tf.float32)
        ones = tf.ones([batch_size], dtype=tf.float32)

        dh = tf.stack(
            [
                tf.stack([zeros, zeros, 0.333 * ones, joints_batch[:, 0]], axis=1),
                tf.stack([-M_PI / 2 * ones, zeros, zeros, joints_batch[:, 1]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, 0.316 * ones, joints_batch[:, 2]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.0825 * ones, zeros, joints_batch[:, 3]], axis=1),
                tf.stack([-M_PI / 2 * ones, -0.0825 * ones, 0.384 * ones, joints_batch[:, 4]], axis=1),
                tf.stack([M_PI / 2 * ones, zeros, zeros, joints_batch[:, 5]], axis=1),
                tf.stack([M_PI / 2 * ones, 0.088 * ones, 0.107 * ones, joints_batch[:, 6]], axis=1),
            ], axis=1
        )
        return dh

    def TF_matrix(i, dh):
        # Define Transformation matrix based on DH params
        alpha = dh[:, i, 0]
        a = dh[:, i, 1]
        d = dh[:, i, 2]
        q = dh[:, i, 3]

        TF = tf.stack(
            [
                tf.stack([tf.cos(q), -tf.sin(q), tf.zeros_like(q), a], axis=1),
                tf.stack([tf.sin(q) * tf.cos(alpha), tf.cos(q) * tf.cos(alpha), -tf.sin(alpha), -tf.sin(alpha) * d], axis=1),
                tf.stack([tf.sin(q) * tf.sin(alpha), tf.cos(q) * tf.sin(alpha), tf.cos(alpha), tf.cos(alpha) * d], axis=1),
                tf.stack([tf.zeros_like(q), tf.zeros_like(q), tf.zeros_like(q), tf.ones_like(q)], axis=1)
            ], axis=1
        )
        return TF
    dh_parameters = dh_params(joints_batch)

    T_01 = TF_matrix(0, dh_parameters)
    T_12 = TF_matrix(1, dh_parameters)
    T_23 = TF_matrix(2, dh_parameters)
    T_34 = TF_matrix(3, dh_parameters)
    T_45 = TF_matrix(4, dh_parameters)
    T_56 = TF_matrix(5, dh_parameters)
    T_67 = TF_matrix(6, dh_parameters)

    T_07 = tf.linalg.matmul(T_01, T_12)
    T_07 = tf.linalg.matmul(T_07, T_23)
    T_07 = tf.linalg.matmul(T_07, T_34)
    T_07 = tf.linalg.matmul(T_07, T_45)
    T_07 = tf.linalg.matmul(T_07, T_56)
    T_07 = tf.linalg.matmul(T_07, T_67)

    translation = T_07[:, :3, 3]
    
    euler = mat2euler(T_07[:, :3, :3])
    return tf.concat([translation, euler], axis=1)