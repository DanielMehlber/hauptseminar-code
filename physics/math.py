import numpy as np

def mach_to_ms(mach: float) -> float:
    """
    Convert Mach number to meters per second.

    Parameters:
    - mach: Mach number.

    Returns:
    - Speed in meters per second.
    """
    return mach * 343.2  # Speed of sound at sea level in m/s

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


def rotate_x_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix for a rotation around the X-axis.

    Parameters:
    - angle: Angle in radians.

    Returns:
    - Rotation matrix (3x3 numpy array).
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotate_y_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix for a rotation around the Y-axis.

    Parameters:
    - angle: Angle in radians.

    Returns:
    - Rotation matrix (3x3 numpy array).
    """
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotate_z_matrix(angle: float) -> np.ndarray:
    """
    Create a rotation matrix for a rotation around the Z-axis.

    Parameters:
    - angle: Angle in radians.

    Returns:
    - Rotation matrix (3x3 numpy array).
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

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


def gramm_schmidt_ortho(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Perform Gram-Schmidt orthogonalization to find an orthonormal basis.
    
    Inputs:
    - v: The vector to be orthogonalized (3D numpy array).
    - reference: The reference vector (3D numpy array).

    Returns:
    - v1: The normalized version of the input vector.
    - v2: The orthogonalized vector to the reference vector.
    - v3: The cross product of v1 and v2, which is orthogonal to both.
    
    """
    v1 = v / np.linalg.norm(v)

    v2 = reference - np.dot(reference, v) * v
    v2 = v2 / np.linalg.norm(v2)
    
    v3 = np.cross(v1, v2)
    v3 = v3 / np.linalg.norm(v3)

    return v1, v2, v3


def project_on_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Project a vector onto a plane defined by its normal vector.

    Parameters:
    - v: The vector to be projected (3D numpy array).
    - plane_normal: The normal vector of the plane (3D numpy array).

    Returns:
    - The projected vector (3D numpy array).
    """
    return v - np.dot(v, plane_normal) * plane_normal