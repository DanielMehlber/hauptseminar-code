import numpy as np
from environment.observations import InterceptorObservations 
from models.missile import PhysicalMissleModel
import models.physics as physics

class PlanarProportionalNavPilot:
    def __init__(self, max_acc: float, n: float):
        self.n = n
        self.max_acc = max_acc # max acceleration in m/s^2

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
        missile_space_los_vector = observations.los_distance_vec / np.linalg.norm(observations.los_distance_vec)
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
        acc_command_maginude = np.linalg.norm(lateral_acc_command_vec)
        if acc_command_maginude > self.max_acc:
            lateral_acc_command_vec *= self.max_acc / acc_command_maginude
        norm_lateral_acc_command = lateral_acc_command_vec / self.max_acc

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
        
        
class ZemProportionalNavPilot:
    def __init__(self, max_acc: float, n: float):
        self.n = n
        self.max_acc = max_acc # max acceleration in m/s^2

        # to calculate the target relative velocity vector form outside observations
        self.world_space_last_target_pos = None
        self.world_space_last_interceptor_pos = None

        # to calculate the target relative velocity vector from missile observations
        self.missile_space_last_target_position = None

    def _get_target_velocity_from_outside_measurements(self, interceptor: PhysicalMissleModel, target: PhysicalMissleModel, dt: float):
        """
        Get the target velocity vector (in the missile's frame of reference) by looking at outside measurements from a ground station.
        This calculation uses world position coordinates of the target to calculate its velocity and then transforms it into the
        interceptor's frame of reference.
        """
        assert dt > 1e-6, "delta time is too small for stable calculations"

        # get interceptor world velocity because we need relative velocity
        world_space_interceptor_vel_vec = np.zeros(3)
        if self.world_space_last_interceptor_pos is not None:
            # differntiate them to get velocity
            world_space_interceptor_vel_vec = (interceptor.world_pos - self.world_space_last_interceptor_pos) / dt
        
        missile_space_target_relative_vel_vec= np.zeros(3)
        if self.world_space_last_target_pos is not None:
            # differntiate them to get velocity
            world_space_target_velocity_vec = (target.world_pos - self.world_space_last_target_pos) / dt

            # make relative to interceptor velocity
            world_space_relative_target_velocity_vec = world_space_target_velocity_vec - world_space_interceptor_vel_vec

            # transform into missile space
            missile_space_target_relative_vel_vec = interceptor.body_to_world_rot_mat.T @ world_space_relative_target_velocity_vec
     
        assert np.all(np.isfinite(missile_space_target_relative_vel_vec)), "velocity overflow encountered"

        # store for the next time step
        self.world_space_last_target_pos = target.world_pos.copy()
        self.world_space_last_interceptor_pos = interceptor.world_pos.copy()

        return missile_space_target_relative_vel_vec

    def _get_target_velocity_from_missile_observations(self, observations: InterceptorObservations, dt: float):
        """
        Get the target velocity vector (in the missile's frame of reference) by taking observations of the seeker and compensate
        for disturbances by the intereptor itself, e.g. its own turn rate. This is more complex than using outside measurements, but 
        necessary if outside measurements are not given.
        """
        assert dt > 1e-6, "delta time is too small for stable calculations"

        missile_space_target_velocity_vec = np.zeros(3)
        if self.missile_space_last_target_position is not None:
            # build a rotation matrix that undoes the missile's last turn
            missile_space_turn_angles = observations.missile_space_turn_rate * dt
            missile_space_yaw_angle, missile_space_pitch_angle = missile_space_turn_angles[0], missile_space_turn_angles[1]
            undo_turn_rot_matrix = physics.rotate_z_matrix(-missile_space_yaw_angle) @ physics.rotate_y_matrix(-missile_space_pitch_angle)

            # cancel own rotation from the target position vector
            missile_space_last_target_adjusted_pos_vec = undo_turn_rot_matrix @ self.missile_space_last_target_position
            missile_space_target_velocity_vec = (observations.los_distance_vec - missile_space_last_target_adjusted_pos_vec) / dt

        self.missile_space_last_target_position = observations.los_distance_vec.copy()
        return missile_space_target_velocity_vec

    def _normalize_command(self, lateral_acc_command_vec: np.ndarray) -> np.ndarray:
        lateral_acc_command_magnitude = np.linalg.norm(lateral_acc_command_vec)
        if lateral_acc_command_magnitude > self.max_acc:
            lateral_acc_command_vec *= self.max_acc / lateral_acc_command_magnitude
        norm_lateral_acc_command = lateral_acc_command_vec / self.max_acc

        return norm_lateral_acc_command

    def _using_world_coordinates(self, observations: InterceptorObservations, dt: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel) -> np.ndarray:
        world_target_velocity = target.get_velocity()
        world_interceptor_velocity = interceptor.get_velocity()

        # relative to interceptor
        world_relative_target_velocity = world_target_velocity - world_interceptor_velocity
        world_relative_target_position = target.world_pos - interceptor.world_pos

        # calcute zero-effort miss
        distance = np.linalg.norm(world_relative_target_position)
        closing_velocity = -np.dot(world_relative_target_position, world_relative_target_velocity) / distance
        time_to_go = distance / closing_velocity
        zero_effort_miss_vec = world_relative_target_position + world_relative_target_velocity * time_to_go


        # make zero-effort-miss perpendicular to line-of-sight vector
        los_vector = world_relative_target_position / np.linalg.norm(world_relative_target_position)
        zem_los_parallel_vec = np.dot(los_vector, zero_effort_miss_vec)
        zem_los_perpendicular_vec = zero_effort_miss_vec - zem_los_parallel_vec

        # crate acceleration command and project onto missile's lateral plane
        world_acc_command = self.n * zem_los_perpendicular_vec / (time_to_go**2)
        world_interceptor_longitude_vec = world_interceptor_velocity / np.linalg.norm(world_interceptor_velocity)
        world_lateral_acc_command = physics.project_on_plane(world_acc_command, world_interceptor_longitude_vec)

        # rotate into missile's frame of reference
        missile_space_lateral_acc_command = interceptor.body_to_world_rot_mat.T @ world_lateral_acc_command
        norm_missile_space_lateral_acc_command = self._normalize_command(missile_space_lateral_acc_command)
        return norm_missile_space_lateral_acc_command
    
    def _using_missile_observations(self, observations: InterceptorObservations, dt: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel) -> np.ndarray:
        los_unit_vec = observations.los_distance_vec / np.linalg.norm(observations.los_distance_vec)
        
        # the zem vector is basically the predicted intercept point of the missile and target
        distance_to_target = np.linalg.norm(observations.los_distance_vec)
        closing_rate_to_target = np.dot(observations.closing_rate_vec, los_unit_vec)
        time_to_go = distance_to_target / closing_rate_to_target

        if abs(time_to_go) < 1e-6 or not np.isfinite(time_to_go):
            print("time to go is not finite or too small")
            return np.zeros(2), None # continue course

        # we need target velocity to infer the predicted intercept point
        missile_space_target_relative_vel_ve = self._get_target_velocity_from_outside_measurements(interceptor, target, dt)
        # missile_space_target_velocity_vec = self._get_target_velocity_from_missile_observations(observations, dt)
        missile_space_zem_vec = observations.los_distance_vec + missile_space_target_relative_vel_ve * time_to_go
        
        # get the component which is perpendicular to the line of sight vector
        missile_space_zem_los_parallel_magnitude = np.dot(missile_space_zem_vec, los_unit_vec)
        missile_space_zem_los_perpendicular_vec = missile_space_zem_vec - (missile_space_zem_los_parallel_magnitude * los_unit_vec)

        # perpendicular to the line of sight vector
        # missile_space_acc_command_vec = self.n * missile_space_zem_los_perpendicular_vec * 1 / time_to_go**2
        missile_space_acc_command_vec = self.n * missile_space_zem_vec * (1 / time_to_go**2)

        # project the acceleration command onto the lateral plane of the interceptor (perpendicular to the velocity vector)
        lateral_acc_command_vec = physics.project_on_plane(missile_space_acc_command_vec, np.array([1.0, 0.0, 0.0]))

        # limit the acceleration command and normalize it
        norm_lateral_acc_command = self._normalize_command(lateral_acc_command_vec)

        # drop the x component (acceleration in the direction of the velocity vector)
        return np.array([norm_lateral_acc_command[1], norm_lateral_acc_command[2]])


    def step(self, observations: np.ndarray, dt: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel) -> np.ndarray:
        """
        Calculate the acceleration command based on the observations and time step.
        
        Args:
            observations (np.ndarray): The observations from the environment.
            dt (float): The time step for the simulation.
        
        Returns:
            np.ndarray: The acceleration command for the interceptor.
        """
        observations: InterceptorObservations = InterceptorObservations(observations)
        command = self._using_missile_observations(observations, dt, interceptor, target)
        return command
        
    
    def _get_observations(self, observations: np.ndarray) -> tuple:        
        norm_distance = observations[0:3]
        norm_closing_rate = observations[3:6]
        norm_los_angle = observations[6:8]
        norm_los_angle_rate = observations[8:10]
        world_space_interceptor_orientation = observations[10:13]
        missile_space_turn_rate = observations[13:16]

        return norm_distance, norm_closing_rate, norm_los_angle, norm_los_angle_rate, world_space_interceptor_orientation, missile_space_turn_rate
        
        
class SpaceProportionalNavPilot:
    def __init__(self, max_acc: float, n: float):
        self.n = n
        self.max_acc = max_acc # max acceleration in m/s^2
        pass

    def step(self, observations: np.ndarray, dt: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel) -> np.ndarray:
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