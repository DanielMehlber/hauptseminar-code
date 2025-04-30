import numpy as np
import models.physics as physics
import math

class MissileModel:
    def __init__(self, velocity=np.ndarray([0, 0, 100]), max_acc=50 * 9.81, pos=np.zeros(3)):
        """
        Initializes the missile's rigit body motion model in 3D space.

        Args:
            speed (float): The speed of the missile in m/s.
            max_acc (float): The maximum possible lateral acceleration of the missile in m/s^2.
            pos (np.ndarray): The initial position of the missile in 3D space.
        """
        # required for resetting the missile to its initial state
        self.init_pos = pos.copy()
        self.init_velocity = velocity.copy()
        self.speed = np.linalg.norm(velocity)

        # limits
        self.max_lat_acc = max_acc
        self.max_axes_acc = math.sqrt(self.max_lat_acc**2 / 2.0)
        
        # represents orientation of missile in 3D space
        self.orientation_matrix: np.ndarray = None

        self.reset()

    def reset(self):
        self.pos = self.init_pos.copy()
        self.steps = 0  

        self._init_orientation_matrix(self.init_velocity)

    def _calc_body_frame(self, velocity_vec):
        """
        Builds a orthonormal basis in the body frame of the missile. The basis is built
        from the velocity vector of the missile and the world up vector.

        If the velocity vector is aligned with the world up vector, the horizontal axis
        is used as the reference vector.

        The basis is built using the Gram-Schmidt process to ensure orthogonality.
        The basis is returned as three orthonormal vectors: longitude_axis, right_axis, up_axis.
        """
        longitude_axis = velocity_vec / np.linalg.norm(velocity_vec)
        right_axis = np.zeros(3)
        up_axis = np.zeros(3)

        world_up = np.array([0, 0, 1])
        world_horizontal = np.array([1, 0, 0])

        # check if the velocity vector is the up vector
        if not np.allclose(longitude_axis, world_up):
            # if so, we can use the horizontal axis as the reference
            longitude_axis, up_axis, right_axis = physics.gramm_schmidt(longitude_axis, world_up)

        else:
            # if not, we can use the horizontal axis as the reference
            longitude_axis, right_axis, up_axis = physics.gramm_schmidt(longitude_axis, world_horizontal)

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
        longitude_axis, right_axis, up_axis = self._calc_body_frame(velocity_vec)
        self.orientation_matrix = self._build_orientation_matrix(longitude_axis, right_axis, up_axis)

    
    def _apply_acceleration_2(self, lat_acc: np.ndarray, dt: float):
        missile_space_roll_axis = np.array([1.0, 0.0, 0.0]) # x is forward (roll axis)
        missile_space_velocity_vec = missile_space_roll_axis * self.speed

        missile_space_acc = np.array([0.0, lat_acc[0], lat_acc[1]]) # y is left (pitch axis), z is up (yaw axis)
        missile_space_new_velocity_vec = missile_space_velocity_vec + missile_space_acc * dt
        missile_space_new_velocity_vec *= self.speed / np.linalg.norm(missile_space_new_velocity_vec) # normalize to constant speed
        missile_space_new_roll_axis = missile_space_new_velocity_vec / np.linalg.norm(missile_space_new_velocity_vec)
        
        # project onto horizontal plane to get horizontal angle of change (yaw angle)
        missile_space_new_velocity_h = np.array([missile_space_new_roll_axis[0], missile_space_new_roll_axis[1], 0.0])
        missile_space_new_velocity_h /= np.linalg.norm(missile_space_new_velocity_h)

        yaw_angle = np.arccos(np.dot(missile_space_new_velocity_h, missile_space_roll_axis))
        yaw_angle *= np.sign(lat_acc[0]) # left/right of nose

        # project onto vertical plane to get vertical angle of change (pitch angle)
        missile_space_new_velocity_v = np.array([missile_space_new_roll_axis[0], 0.0, missile_space_new_roll_axis[2]])
        missile_space_new_velocity_v /= np.linalg.norm(missile_space_new_velocity_v)

        pitch_angle = np.arccos(np.dot(missile_space_new_velocity_v, missile_space_roll_axis))
        pitch_angle *= np.sign(lat_acc[1]) # above/below nose

        # create rotation matrix from angles
        yaw_rotation_matrix = physics.rotate_z_matrix(yaw_angle)
        pitch_rotation_matrix = physics.rotate_y_matrix(pitch_angle).T

        # calculate new axes of orientation in missile space
        missile_space_new_roll_axis = yaw_rotation_matrix @ pitch_rotation_matrix @ missile_space_roll_axis
        missile_space_new_yaw_axis = yaw_rotation_matrix @ pitch_rotation_matrix @ np.array([0.0, 1.0, 0.0])

        missile_space_new_pitch_axis = np.cross(missile_space_new_roll_axis, missile_space_new_yaw_axis)
        missile_space_new_pitch_axis /= np.linalg.norm(missile_space_new_pitch_axis)

        # transform new axes into world space
        world_new_roll_axis = self.orientation_matrix @ missile_space_new_roll_axis
        world_new_yaw_axis = self.orientation_matrix @ missile_space_new_yaw_axis
        world_new_pitch_axis = self.orientation_matrix @ missile_space_new_pitch_axis


        # build the new orientation matrix
        self.orientation_matrix = self._build_orientation_matrix(world_new_roll_axis, world_new_yaw_axis, world_new_pitch_axis)

        world_space_new_velocity_vec = self.velocity()

        # update position of missile
        self.pos += world_space_new_velocity_vec * dt

        # TODO: check if the new velocity vector is correct
        expected_world_space_velocity_vec = self.orientation_matrix @ missile_space_new_velocity_vec
        current_world_space_velocity_vec = self.velocity()
        # print (f"MissileModel - New velocity vector: {expected_world_space_velocity_vec} vs. {current_world_space_velocity_vec} with error {np.linalg.norm(expected_world_space_velocity_vec - current_world_space_velocity_vec)}")

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
        acc_command = lat_acc * self.max_axes_acc # convert percentage to m/s^2
        self._apply_acceleration_2(acc_command, dt)
    
    def velocity(self) -> np.ndarray:
        """
        Returns the current velocity of the missile in m/s.

        Returns:
            np.ndarray: The current velocity vector of the missile in m/s.
        """
        return self.orientation_matrix @ np.array([self.speed, 0.0, 0.0])