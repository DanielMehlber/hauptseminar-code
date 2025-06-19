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
        

def estimate_time_to_go(distance, closing_rate):
    time_to_go = distance / closing_rate
    return time_to_go

def calculate_zero_effort_miss_vector(target_pos: np.ndarray, rel_target_vel: np.ndarray, time_to_go: float):
    return target_pos + rel_target_vel * time_to_go

def project_los_perpendicular(zem_vector: np.ndarray, unit_los_vector: np.ndarray) -> np.ndarray:
    zem_parallel = np.dot(zem_vector, unit_los_vector) * unit_los_vector
    zem_perpendicular = zem_vector - zem_parallel

    return zem_perpendicular

def calculate_acceleration_command(zem_perpendicular: np.ndarray, n: float, time_to_go: float) -> np.ndarray:
    return n * zem_perpendicular / (time_to_go ** 2)

def zem_proportional_guidance(n: float, los_vec: np.ndarray, closing_rate: float, rel_target_vel: np.ndarray) -> np.ndarray:
    """
    Produces a acceleration vector perpendicular to the line-of-sight to reach the predicted
    intercept point. This is the zero-effort-miss proportional guidance law for three-dimensional
    engagement scenarios.

    Args:
        n (float): Proportional navigation constant, i.e. gain of the command
        los_vec (np.ndarray): Line-of-sight vector from interceptor to target, i.e. the relative position of target or distance vector
        closing_rate (float): Closing rate of the interceptor to the target, i.e. the relative velocity along the line-of-sight vector
        rel_target_vel (np.ndarray): Relative velocity of the target to the interceptor's velocity

    Returns:
        np.ndarray: Acceleration vector perpendicular to the line-of-sight vector
    """
    distance = np.linalg.norm(los_vec)
    time_to_go = estimate_time_to_go(distance, closing_rate)
    
    # target has been lost or we missed
    if time_to_go < 1e-6 or not np.isfinite(time_to_go):
        return np.zeros(3)  # continue course

    # zero-effort-miss vector: points towards the predicted intercept point
    zem_vec = calculate_zero_effort_miss_vector(los_vec, rel_target_vel, time_to_go)
    
    # get perpendicular component of zero-effort-miss to line-of-sight vector
    unit_los_vector = los_vec / np.linalg.norm(los_vec)
    zem_perpendicular = project_los_perpendicular(zem_vec, unit_los_vector)

    # convert the distance vector into acceleration
    acc_command = calculate_acceleration_command(zem_perpendicular, n, time_to_go)
    return acc_command
        
class ZemProportionalNavPilot:
    def __init__(self, max_acc: float, n: float):
        self.n = n
        self.max_acc = max_acc # max acceleration in m/s^2

        # to calculate the target relative velocity vector form outside observations
        self.world_space_last_target_pos = None
        self.world_space_last_interceptor_pos = None

        # to calculate the target relative velocity vector from missile observations
        self.missile_space_last_target_position = None

    def measure_relative_target_velocity_from_ground(self, interceptor: PhysicalMissleModel, target: PhysicalMissleModel, dt: float) -> np.ndarray:
        """
        Get the target velocity by looking at outside measurements from a ground station. In constrast to seeker data, 
        the ground station can track both the interceptor and the target, so we can calculate the relative velocity vector.

        Args:
            interceptor (PhysicalMissleModel): The interceptor model.
            target (PhysicalMissleModel): The target model.
            dt (float): The time step for the simulation.

        Returns:
            np.ndarray: The relative target velocity vector in the world space.
        """
        assert dt > 1e-6, "delta time is too small for stable calculations"

        # get interceptor world velocity because we need relative velocity
        world_space_interceptor_vel_vec = np.zeros(3)
        if self.world_space_last_interceptor_pos is not None:
            # differntiate them to get velocity
            world_space_interceptor_vel_vec = (interceptor.world_pos - self.world_space_last_interceptor_pos) / dt
        
        world_space_relative_target_velocity_vec = np.zeros(3)
        if self.world_space_last_target_pos is not None:
            # differntiate them to get velocity
            world_space_target_velocity_vec = (target.world_pos - self.world_space_last_target_pos) / dt

            # make relative to interceptor velocity
            world_space_relative_target_velocity_vec = world_space_target_velocity_vec - world_space_interceptor_vel_vec
     
        assert np.all(np.isfinite(world_space_relative_target_velocity_vec)), "velocity overflow encountered"

        # store for the next time step
        self.world_space_last_target_pos = target.world_pos.copy()
        self.world_space_last_interceptor_pos = interceptor.world_pos.copy()

        return world_space_relative_target_velocity_vec

    def measure_relative_target_velocity_from_seeker(self, observations: InterceptorObservations, own_speed: float, dt: float) -> np.ndarray:
        """
        Get the target velocity by looking at seeker data. Seeker data is already in missile space, but it is biased
        by the interceptor's own angular velocity and movement. We need to cancel these effects to get the 
        relative target velocity vector.

        Args:
            observations (InterceptorObservations): The observations from the interceptor's seeker (biased by own dynamics)
            own_speed (float): The speed of the interceptor in m/s. Needed to subtract the own movement from the target's last position.
            dt (float): The time step for the simulation.

        Returns:
            np.ndarray: The relative target velocity vector in the missile space.
        """
        assert dt > 1e-6, "delta time is too small for stable calculations"

        missile_space_target_velocity_vec = np.zeros(3)
        if self.missile_space_last_target_position is not None:
            # build a rotation matrix that undoes the missile's last turn
            missile_space_turn_angles = observations.missile_space_turn_rate * dt
            missile_space_yaw_angle, missile_space_pitch_angle = missile_space_turn_angles[0], missile_space_turn_angles[1]
            undo_turn_rot_matrix = physics.rotate_z_matrix(-missile_space_yaw_angle) @ physics.rotate_y_matrix(-missile_space_pitch_angle)

            # cancel own rotation from the target's last position vector
            missile_space_last_target_adjusted_pos_vec = undo_turn_rot_matrix @ self.missile_space_last_target_position

            # cancel own movement from the target's last position vector
            missile_space_last_target_adjusted_pos_vec += np.array([-own_speed * dt, 0.0, 0.0]) 

            missile_space_target_velocity_vec = (observations.los_distance_vec - missile_space_last_target_adjusted_pos_vec) / dt

        self.missile_space_last_target_position = observations.los_distance_vec.copy()
        return missile_space_target_velocity_vec
    
    def _normalize_command(self, lateral_acc_command_vec: np.ndarray) -> np.ndarray:
        lateral_acc_command_magnitude = np.linalg.norm(lateral_acc_command_vec)
        if lateral_acc_command_magnitude > self.max_acc:
            lateral_acc_command_vec *= self.max_acc / lateral_acc_command_magnitude
        norm_lateral_acc_command = lateral_acc_command_vec / self.max_acc

        return norm_lateral_acc_command

    def _calc_command_on_ground_station(self, n: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel, dt: float) -> np.ndarray:
        """
        Command is calculated by a ground station that can track and observe absolute positions and velocities of 
        both the interceptor and the target. This makes the task of calculating relative velocities much easier, but
        the command has to be sent to the interceptor, which is a delay that can cause problems in the engagement.
        """
        world_space_los_vec = target.world_pos - interceptor.world_pos
        world_space_rel_target_vel_vec = self.measure_relative_target_velocity_from_ground(interceptor, target, dt)

        # project relative speed onto los vector
        closing_rate = -np.dot(world_space_los_vec, world_space_rel_target_vel_vec) / np.linalg.norm(world_space_los_vec)

        # in the world coordinate system
        world_space_acc_command_vec = zem_proportional_guidance(n, world_space_los_vec, closing_rate, world_space_rel_target_vel_vec)

        # move into missile space and project onto its lateral plane
        missile_space_acc_command_vec = interceptor.body_to_world_rot_mat.T @ world_space_acc_command_vec
        missile_space_lat_acc_command = physics.project_on_plane(missile_space_acc_command_vec, np.array([1.0, 0.0, 0.0]))

        # drop the longitudinal component (x): we can only accelerate in the lateral plane
        missile_space_lat_acc_command = np.array([missile_space_lat_acc_command[1], missile_space_lat_acc_command[2]])

        return missile_space_lat_acc_command

    def _calc_command_onboard(self, n: float, observations: InterceptorObservations, dt: float, interceptor: PhysicalMissleModel) -> np.ndarray:
        """
        Command is calculated onboard the interceptor using seeker data directly. This avoid the transmission delay of
        commands from a ground station, but requires the interceptor to use biased observations, like relative positions
        which are affected by the interceptor's own angular velocity.
        """
        # get the relative target velocity vector from the seeker data
        own_speed = np.linalg.norm(interceptor.get_velocity())
        missile_space_rel_target_vel_vec = self.measure_relative_target_velocity_from_seeker(observations, own_speed, dt)

        # project the relative velocity onto the LOS vector
        closing_rate = -np.dot(observations.los_distance_vec, missile_space_rel_target_vel_vec) / np.linalg.norm(observations.los_distance_vec)

        # in the missile coordinate system
        missile_space_acc_command_vec = zem_proportional_guidance(n, observations.los_distance_vec, closing_rate, missile_space_rel_target_vel_vec)

        # project onto the lateral plane of the interceptor
        missile_space_lat_acc_command = physics.project_on_plane(missile_space_acc_command_vec, np.array([1.0, 0.0, 0.0]))

        # drop the longitudinal component (x): we can only accelerate in the lateral plane
        missile_space_lat_acc_command = np.array([missile_space_lat_acc_command[1], missile_space_lat_acc_command[2]])

        return missile_space_lat_acc_command

    def step(self, observations: np.ndarray, dt: float, interceptor: PhysicalMissleModel, target: PhysicalMissleModel) -> np.ndarray:
        """
        Calculate the acceleration command based on the observations and time step.
        
        Args:
            observations (np.ndarray): The observations from the environment.
            dt (float): The time step for the simulation.
        
        Returns:
            np.ndarray: The acceleration command for the interceptor.
        """
        # missile_space_lateral_acc_vector = self._command_from_ground_station(self.n, interceptor, target, dt)
        missile_space_lateral_acc_vector = self._calc_command_onboard(self.n, InterceptorObservations(observations), dt, interceptor)

        # clamp command to physical limits
        missile_space_norm_lateral_acc_command = self._normalize_command(missile_space_lateral_acc_vector)

        return missile_space_norm_lateral_acc_command
    

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