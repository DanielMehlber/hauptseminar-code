import numpy as np
import physics.math as physics
import math

class PhysicalMissleModel:
    """
    A missile's model is based on rigid body motion and other physical principles. It is purely physical
    and does not emit and actions or observations. It sole purpose is to keep the missile's physical state
    and to apply accelerations to it.
    """
    def __init__(self, velocity=np.ndarray([0, 0, 100]), max_acc=50 * 9.81, pos=np.zeros(3)):
        """
        Initializes the missile's rigit body motion model in 3D space.

        Args:
            velocity (np.ndarray): The initial velocity vector of the missile in m/s.
            max_acc (float): The maximum lateral acceleration of the missile in m/s^2.
            pos (np.ndarray): The initial position of the missile in m.
        """
        # required for resetting the missile to its initial state
        self.world_init_pos = pos.copy()
        self.world_init_velocity = velocity.copy()
        self.max_speed = np.linalg.norm(velocity)

        # limits
        self.max_lat_acc = max_acc
        self.max_axes_acc = math.sqrt(self.max_lat_acc**2 / 2.0)
        
        # represents orientation of missile in 3D space
        self.body_to_world_rot_mat: np.ndarray = None

        self.reset()

    def reset(self, uncertainty: float = 0.0):
        self.world_pos = self.world_init_pos.copy()
        self.steps = 0  

        # displace the missile's start position based on uncertainty
        uncertainty_pos_dispacement = np.random.normal(0, uncertainty * 1000, 3)
        self.world_pos += uncertainty_pos_dispacement

        # rotate the missile's start velocity vector based on uncertainty
        uncertainty_angle_displacement = np.random.normal(0, uncertainty * math.pi/4, 3)
        velocity = physics.euler_to_rotation_matrix(uncertainty_angle_displacement) @ self.world_init_velocity
        
        self._init_orientation_matrix(velocity)

    def _build_orthonormal_body_frame(self, velocity_vec):
        """
        Builds a orthonormal basis in the body frame of the missile. The basis is built
        from the velocity vector of the missile and the world up vector.

        If the velocity vector is aligned with the world up vector, the horizontal axis
        is used as the reference vector.

        The basis is built using the Gram-Schmidt process to ensure orthogonality.
        The basis is returned as three orthonormal vectors: longitude_axis, right_axis, up_axis.
        """

        # return default orientation if velocity = 0
        if np.linalg.norm(velocity_vec) < 1e-6:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

        longitude_axis = velocity_vec / np.linalg.norm(velocity_vec)
        right_axis = np.zeros(3)
        up_axis = np.zeros(3)

        world_up = np.array([0, 0, 1])
        world_horizontal = np.array([1, 0, 0])

        # check if the velocity vector is the up vector
        if not np.allclose(longitude_axis, world_up):
            # if so, we can use the horizontal axis as the reference
            longitude_axis, up_axis, right_axis = physics.gramm_schmidt_ortho(longitude_axis, world_up)

        else:
            # if not, we can use the horizontal axis as the reference
            longitude_axis, right_axis, up_axis = physics.gramm_schmidt_ortho(longitude_axis, world_horizontal)

        return longitude_axis, right_axis, up_axis

    def _build_orientation_matrix(self, roll_axis, yaw_axis, pitch_axis):
        """
        Builds the orientation matrix from the roll, yaw, and pitch axes.

        Args:
            roll_axis (np.ndarray): The roll axis of the missile.
            yaw_axis (np.ndarray): The yaw axis of the missile.
            pitch_axis (np.ndarray): The pitch axis of the missile.

        Returns:
            np.ndarray: The orientation matrix of the missile.
        """
        return np.array([
            roll_axis,
            yaw_axis,
            pitch_axis
        ]).T

    def _init_orientation_matrix(self, velocity_vec):
        # Rebuild the orientation matrix based on the current velocity vector
        longitude_axis, right_axis, up_axis = self._build_orthonormal_body_frame(velocity_vec)
        self.body_to_world_rot_mat = self._build_orientation_matrix(longitude_axis, right_axis, up_axis)

    def calculate_local_angles_to(self, missile_space_vector: np.ndarray) -> np.ndarray:
        """
        Calculates the yaw and pitch angles the missile would need to turn towards the given
        vector in the missile's local space.
        The yaw angle is the angle in the horizontal plane, and the pitch angle is the angle in the vertical plane.

        Args:
            missile_space_vector (np.ndarray): The vector in the missile's local space.

        Returns:
            tuple: The yaw and pitch angles in radians.
        """
        missile_space_roll_axis = np.array([1.0, 0.0, 0.0]) # x is forward (roll axis)
        missile_space_vector /= np.linalg.norm(missile_space_vector) # normalize the vector

        # project onto 2D horizontal plane to get horizontal angle of change (yaw angle)
        missile_space_new_velocity_h = np.array([missile_space_vector[0], missile_space_vector[1]])
        missile_space_new_velocity_h /= np.linalg.norm(missile_space_new_velocity_h)
        missile_space_roll_axis_h = np.array([missile_space_roll_axis[0], missile_space_roll_axis[1]])

        yaw_angle = np.arccos(np.dot(missile_space_new_velocity_h, missile_space_roll_axis_h))
        yaw_angle *= np.sign(missile_space_vector[1]) # left/right of nose

        # project onto 2D vertical plane to get vertical angle of change (pitch angle)
        missile_space_new_velocity_v = np.array([missile_space_vector[0], missile_space_vector[2]])
        missile_space_new_velocity_v /= np.linalg.norm(missile_space_new_velocity_v)
        missile_space_roll_axis_v = np.array([missile_space_roll_axis[0], missile_space_roll_axis[2]])

        pitch_angle = np.arccos(np.dot(missile_space_new_velocity_v, missile_space_roll_axis_v))
        pitch_angle *= np.sign(missile_space_vector[2]) # above/below nose

        return yaw_angle, pitch_angle
    
    def _apply_acceleration(self, lat_acc: np.ndarray, dt: float):
        """
        We want to take the lateral 2D acceleration vector in missile space and apply it to the current velocity vector
        and world position. We also want to determine the new orientation of the missile based on its new velocity vector.

        Approach:
        1. Transform the lateral acceleration vector into missile space.
        2. Apply the acceleration command to the current velocity vector to get the new velocity vector (in missile space).
        3. Calculate the yaw and pitch angles the missile would need to turn towards the new velocity vector (in missile space).
        4. Create a rotation matrix from the yaw and pitch angles.
        5. Calculate the new roll, yaw, and pitch axes of the missile (in missile space).
        6. Build a new orientation matrix from the new roll, yaw, and pitch axes (in world space).
        7. Apply the velocity vector to the current position of the missile (in world space).

        Note that no roll is performed and all transformations are only in pitch and yaw. Automatic rolling could confuse
        the reinforcement learning agent.

        TODO: There might be a more efficient way to do this, but this is a good start.
        """

        if self.max_speed < 1e-6:
            return

        missile_space_roll_axis = np.array([1.0, 0.0, 0.0]) # x is forward (roll axis)
        missile_space_velocity_vec = missile_space_roll_axis * self.max_speed

        # apply acceleration command to current velocity vector
        missile_space_acc = np.array([0.0, lat_acc[0], lat_acc[1]]) # y is left (pitch axis), z is up (yaw axis)
        missile_space_new_velocity_vec = missile_space_velocity_vec + missile_space_acc * dt
        missile_space_new_velocity_vec *= self.max_speed / np.linalg.norm(missile_space_new_velocity_vec) # normalize to constant speed
        missile_space_new_roll_axis = missile_space_new_velocity_vec / np.linalg.norm(missile_space_new_velocity_vec)
        
        # how much the missile has to turn to get to the new velocity vector
        yaw_angle, pitch_angle = self.calculate_local_angles_to(missile_space_new_velocity_vec.copy())

        # create rotation matrix from angles
        yaw_rotation_matrix = physics.rotate_z_matrix(yaw_angle)
        pitch_rotation_matrix = physics.rotate_y_matrix(pitch_angle).T

        # calculate new axes of orientation in missile space
        missile_space_new_roll_axis = yaw_rotation_matrix @ pitch_rotation_matrix @ missile_space_roll_axis
        missile_space_new_yaw_axis = yaw_rotation_matrix @ pitch_rotation_matrix @ np.array([0.0, 1.0, 0.0])

        missile_space_new_pitch_axis = np.cross(missile_space_new_roll_axis, missile_space_new_yaw_axis)
        missile_space_new_pitch_axis /= np.linalg.norm(missile_space_new_pitch_axis)

        # transform new axes into world space
        world_new_roll_axis = self.body_to_world_rot_mat @ missile_space_new_roll_axis
        world_new_yaw_axis = self.body_to_world_rot_mat @ missile_space_new_yaw_axis
        world_new_pitch_axis = self.body_to_world_rot_mat @ missile_space_new_pitch_axis

        # build the new orientation matrix
        self.body_to_world_rot_mat = self._build_orientation_matrix(world_new_roll_axis, world_new_yaw_axis, world_new_pitch_axis)

        # update world position of missile
        world_space_new_velocity_vec = self.get_velocity()
        self.world_pos += world_space_new_velocity_vec * dt

        # assert that no components are NaN or Inf
        assert np.any(np.isfinite(self.world_pos)), "Position contains NaN or Inf values."
        assert np.any(np.isfinite(self.body_to_world_rot_mat)), "Orientation matrix contains NaN or Inf values."
        assert np.any(np.isfinite(self.max_speed)), "Speed contains NaN or Inf values."

    def _clamp_accleration(self, lat_acc: np.ndarray) -> np.ndarray:
        magnitude = np.linalg.norm(lat_acc)
        if magnitude > self.max_lat_acc:
            lat_acc = lat_acc / magnitude * self.max_lat_acc
        return lat_acc

    def accelerate(self, lat_acc: np.ndarray, dt=0.1, t=0.0):
        """
        Tries to execute a guidance acceleration command. If the physical model
        cannot fully execute this command it is clamped to an executable one, respecting
        accelration and turn limits.

        Args:
            acc (np.ndarray): 2D lateral acceleration vector in the plane of the missile's velocity vector in m/s^2.
            dt (float): the delta time in which the command should be executed in seconds
            t (float): the total time of the simulation in seconds
        """
        lat_acc = self._clamp_accleration(lat_acc)
        acc_command = lat_acc * self.max_axes_acc # convert percentage to m/s^2
        self._apply_acceleration(acc_command, dt)
    
    def get_velocity(self) -> np.ndarray:
        """
        Returns the current velocity of the missile in m/s.

        Returns:
            np.ndarray: The current velocity vector of the missile in m/s.
        """
        return self.body_to_world_rot_mat @ np.array([self.max_speed, 0.0, 0.0])