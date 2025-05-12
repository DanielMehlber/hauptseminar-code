import numpy as np
from gym.observations import InterceptorObservations 
import models.physics as physics

class PlanarProportionalNavPilot:
    def __init__(self, max_acc: float, max_speed: float, n: float):
        self.n = n
        self.max_speed = max_speed # required to convert norm closing rate to closing rate in m/s
        self.max_acc = max_acc # max acceleration in m/s^2
        pass

    def step(self, observations: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate the acceleration command based on the observations and time step.

        This implementation uses the 2D proportional navigation algorithm to determine the
        acceleration command for the interceptor. The acceleration commands are on a 2D plane
        placed in 3D space defined by the line of sight (LOS) vector and the interceptor velocity vector.
        
        Args:
            observations (np.ndarray): The observations from the environment.
            dt (float): The time step for the simulation.
        
        Returns:
            np.ndarray: The lateral acceleration command of the interceptor.
        """

        # are all in missile space (pov of the seeker)
        observations: InterceptorObservations = InterceptorObservations(observations)

        # 2D plane is defined by two vectors: the LOS vector and the interceptor velocity vector
        # The algorithm only thinks on this 2D plane, so we need to project our observations onto it.
        missile_space_velocity_vec = np.array([1.0, 0.0, 0.0])
        missile_space_los_vector = observations.distance_vec / np.linalg.norm(observations.distance_vec)
        missile_space_plane_normal = np.cross(missile_space_los_vector, missile_space_velocity_vec)
        missile_space_plane_normal /= np.linalg.norm(missile_space_plane_normal)

        # project the closing rate onto the plane
        closing_rate_vec_on_plane = physics.project_on_plane(observations.closing_rate_vec, missile_space_plane_normal)

        # given the pitch and yaw angles (for 3D space), we need to get the single angle
        # between previous and current LOS vector on the plane. We therefore reconstruct the
        # previous LOS vector by rotating the current LOS vector about the angular rate
        yaw_angle, pitch_angle = observations.los_angle_rates_vec * dt
        yaw_rot_matrix = physics.rotate_z_matrix(-yaw_angle)
        pitch_rot_matrix = physics.rotate_y_matrix(-pitch_angle)

        previous_los_vector = yaw_rot_matrix @ pitch_rot_matrix  @ missile_space_los_vector

        # project the previous LOS vector to the plane
        previous_los_vector_on_plane = physics.project_on_plane(previous_los_vector, missile_space_plane_normal)
        previous_los_vector_on_plane /= np.linalg.norm(previous_los_vector_on_plane)

        # angle between the two LOS vectors on the plane
        unsigned_angle = np.arccos(np.dot(previous_los_vector_on_plane, missile_space_los_vector))

        # we need the vector perpendicular to the LOS vector on the plane for two reasons:
        # 1. to determine the direction of the angular rate
        # 2. to calculate the acceleration command perpendicular to the LOS vector
        missile_space_los_perpendicular_vec = np.cross(missile_space_plane_normal, missile_space_los_vector)
        missile_space_los_perpendicular_vec /= np.linalg.norm(missile_space_los_perpendicular_vec)

        # determine the sign of the angle
        signed_angle = 0.0
        if np.dot(missile_space_los_perpendicular_vec, previous_los_vector_on_plane) < 0:
            signed_angle = unsigned_angle
        else:
            signed_angle = -unsigned_angle

        angular_rate = signed_angle / dt

        # calculate the acceleration command perpendicular to the LOS vector
        los_acc_command_magnitude = self.n * np.linalg.norm(closing_rate_vec_on_plane) * angular_rate
        los_acc_command_vec = los_acc_command_magnitude * missile_space_los_perpendicular_vec

        # project the acceleration command onto the lateral plane of the interceptor (perpendicular to the velocity vector)
        lateral_acc_command_vec = physics.project_on_plane(los_acc_command_vec, missile_space_velocity_vec)

        # normalize the acceleration command to the max acceleration
        norm_lateral_acc_command = lateral_acc_command_vec / self.max_acc
        norm_lateral_acc_command = np.clip(norm_lateral_acc_command, -1, 1)

        # drop the x component (acceleration in the direction of the velocity vector)
        return np.array([norm_lateral_acc_command[1], norm_lateral_acc_command[2]])

    
    def _get_observations(self, observations: np.ndarray) -> tuple:        
        norm_distance = observations[0:3]
        norm_closing_rate = observations[3:6]
        norm_los_angle = observations[6:8]
        norm_los_angle_rate = observations[8:10]
        world_space_interceptor_orientation = observations[10:13]
        missile_space_turn_rate = observations[13:16]

        return norm_distance, norm_closing_rate, norm_los_angle, norm_los_angle_rate, world_space_interceptor_orientation, missile_space_turn_rate
        
        

class SpaceProportionalNavPilot:
    def __init__(self, max_acc: float, max_speed: float, n: float):
        self.n = n
        self.max_speed = max_speed # required to convert norm closing rate to closing rate in m/s
        self.max_acc = max_acc # max acceleration in m/s^2
        pass

    def step(self, observations: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate the acceleration command based on the observations and time step.
        
        Args:
            observations (np.ndarray): The observations from the environment.
            dt (float): The time step for the simulation.
        
        Returns:
            np.ndarray: The acceleration command for the interceptor.
        """
        observations = InterceptorObservations(observations)

        # is a 3D vector, but we need 2D (we can ignore the closing rate along the longitudinal axis x)
        seeker_closing_rate_vec = np.array([observations.closing_rate_vec[1], observations.closing_rate_vec[2]])

        # proportional guidance law
        real_acc_command = self.n * observations.los_angle_rates_vec * seeker_closing_rate_vec

        # clamp the acceleration command to the max acceleration
        norm_acc_command = real_acc_command / self.max_acc
        norm_acc_command = np.clip(norm_acc_command, -1, 1)
        return norm_acc_command
    
    def _get_observations(self, observations: np.ndarray) -> tuple:        
        norm_distance = observations[0:3]
        norm_closing_rate = observations[3:6]
        norm_los_angle = observations[6:8]
        norm_los_angle_rate = observations[8:10]
        world_space_interceptor_orientation = observations[10:13]
        missile_space_turn_rate = observations[13:16]

        return norm_distance, norm_closing_rate, norm_los_angle, norm_los_angle_rate, world_space_interceptor_orientation, missile_space_turn_rate
        
        