import numpy as np

def get_acceleration_of(v0: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """
    Calculate the acceleration required to change velocity from v0 to v.

    Parameters:
    - v0: Initial velocity vector (3D numpy array).
    - v: Target velocity vector (3D numpy array).
    - m: Mass of the interceptor.
    - dt: Time step.

    Returns:
    - Acceleration vector (3D numpy array).
    """
    dv = v - v0 # change in velocity
    return dv / dt


def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    # Unpack Euler angles
    rx, ry, rz = euler_angles
    
    # Rotation matrix around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation matrix around Y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation matrix around Z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Total rotation matrix
    return Rz @ Ry @ Rx