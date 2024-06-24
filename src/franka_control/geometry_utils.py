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

    return [(new_positions[i][0], new_positions[i][1], new_positions[i][2],
             new_orientations[i][3], new_orientations[i][0], new_orientations[i][1],
             new_orientations[i][2]) for i in range(len(new_positions))]