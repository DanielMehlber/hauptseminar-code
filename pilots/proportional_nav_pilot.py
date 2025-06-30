import numpy as np
from environment.observations import GroundBaseObservations, InterceptorObservations 
from physics.missile import PhysicalMissleModel
import physics.math as math
from physics.noise import LinearDistanceNoise
from pilots.pilot import Pilot

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
        
class ZemProportionalNavPilot(Pilot):
    def __init__(self, max_acc: float, n: float):
        super().__init__("Zero-Effort Miss Prop. Nav.")
        self.n = n
        self.max_acc = max_acc # max acceleration in m/s^2

        # to calculate the target relative velocity vector form outside observations
        self.world_space_last_target_pos = None
        self.world_space_last_interceptor_pos = None

        # to calculate the target relative velocity vector from missile observations
        self.missile_space_last_target_position = None

        self.world_space_last_interceptor_orientation = None

    def reset(self):
        super().reset()
        
        self.world_space_last_target_pos = None
        self.world_space_last_interceptor_pos = None
        self.missile_space_last_target_position = None
        self.world_space_last_interceptor_orientation = None

    def measure_relative_target_velocity_from_ground(self, observations: GroundBaseObservations, dt: float) -> np.ndarray:
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
            # differentiate the last and current position to get the velocity
            world_space_interceptor_vel_vec = (observations.world_space_interceptor_pos - self.world_space_last_interceptor_pos) / dt
        
        world_space_target_velocity_vec = np.zeros(3)
        if self.world_space_last_target_pos is not None:
            # differntiate them to get velocity
            world_space_target_velocity_vec = (observations.world_space_target_pos - self.world_space_last_target_pos) / dt

        # make relative to interceptor velocity
        world_space_relative_target_velocity_vec = world_space_target_velocity_vec - world_space_interceptor_vel_vec
        assert np.all(np.isfinite(world_space_relative_target_velocity_vec)), "velocity overflow encountered"

        # store for the next time step
        self.world_space_last_target_pos = observations.world_space_target_pos.copy()
        self.world_space_last_interceptor_pos = observations.world_space_interceptor_pos.copy()
        return world_space_relative_target_velocity_vec

    def measure_relative_target_velocity_from_seeker(self, obs: InterceptorObservations, interceptor: PhysicalMissleModel, own_speed: float, dt: float) -> np.ndarray:
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
        missile_space_current_target_pos = obs.current_frame.seeker.los_unit_vec * obs.current_frame.seeker.distance_to_target

        missile_space_relative_target_velocity = np.zeros(3)
        if self.missile_space_last_target_position is not None:
            # build a rotation matrix that undoes the missile's last turn
            missile_space_turn_angles = obs.current_frame.imu.missile_space_turn_rate * dt
            missile_space_yaw_angle, missile_space_pitch_angle = missile_space_turn_angles[0], missile_space_turn_angles[1]
            undo_turn_rot_matrix = math.rotate_z_matrix(missile_space_yaw_angle) @ math.rotate_y_matrix(missile_space_pitch_angle)

            # cancel own rotation from the target's last position vector
            missile_space_last_los_vector_adjusted = undo_turn_rot_matrix.T @ self.missile_space_last_target_position

            missile_space_relative_target_velocity = (missile_space_current_target_pos - missile_space_last_los_vector_adjusted) / dt

        self.missile_space_last_target_position = missile_space_current_target_pos.copy()
        return missile_space_relative_target_velocity
    
    def _normalize_command(self, lateral_acc_command_vec: np.ndarray) -> np.ndarray:
        lateral_acc_command_magnitude = np.linalg.norm(lateral_acc_command_vec)
        if lateral_acc_command_magnitude > self.max_acc:
            lateral_acc_command_vec *= self.max_acc / lateral_acc_command_magnitude
        norm_lateral_acc_command = lateral_acc_command_vec / self.max_acc

        return norm_lateral_acc_command

    def _calc_command_on_ground_station(self, n: float, observations: GroundBaseObservations, interceptor: PhysicalMissleModel, dt: float) -> np.ndarray:
        """
        Command is calculated by a ground station that can track and observe absolute positions and velocities of 
        both the interceptor and the target. This makes the task of calculating relative velocities much easier, but
        the command has to be sent to the interceptor, which is a delay that can cause problems in the engagement.
        """
        world_space_los_vec = observations.world_space_target_pos - observations.world_space_interceptor_pos
        world_space_rel_target_vel_vec = self.measure_relative_target_velocity_from_ground(observations, dt)

        # project relative speed onto los vector
        closing_rate = -np.dot(world_space_los_vec, world_space_rel_target_vel_vec) / np.linalg.norm(world_space_los_vec)

        # in the world coordinate system
        world_space_acc_command_vec = zem_proportional_guidance(n, world_space_los_vec, closing_rate, world_space_rel_target_vel_vec)

        # move into missile space and project onto its lateral plane
        missile_space_acc_command_vec = interceptor.body_to_world_rot_mat.T @ world_space_acc_command_vec
        missile_space_lat_acc_command = math.project_on_plane(missile_space_acc_command_vec, np.array([1.0, 0.0, 0.0]))

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
        missile_space_rel_target_vel_vec = self.measure_relative_target_velocity_from_seeker(observations, interceptor, own_speed, dt)

        # project the relative velocity onto the LOS vector
        closing_rate = -np.dot(observations.current_frame.seeker.los_unit_vec, missile_space_rel_target_vel_vec)

        # in the missile coordinate system
        missile_space_los_vec = observations.current_frame.seeker.los_unit_vec * observations.current_frame.seeker.distance_to_target
        missile_space_acc_command_vec = zem_proportional_guidance(n, missile_space_los_vec, closing_rate, missile_space_rel_target_vel_vec)

        # project onto the lateral plane of the interceptor
        missile_space_lat_acc_command = math.project_on_plane(missile_space_acc_command_vec, np.array([1.0, 0.0, 0.0]))

        # drop the longitudinal component (x): we can only accelerate in the lateral plane
        missile_space_lat_acc_command = np.array([missile_space_lat_acc_command[1], missile_space_lat_acc_command[2]])

        return missile_space_lat_acc_command

    def step(self, observations: np.ndarray, interceptor: PhysicalMissleModel, dt: float, on_board: bool = False) -> np.ndarray:
        """
        Calculate the acceleration command based on the observations and time step.
        
        Args:
            observations (np.ndarray): The observations from the environment. Can be either InterceptorObservations or GroundBaseObservations.
            dt (float): The time step for the simulation.
        
        Returns:
            np.ndarray: The acceleration command for the interceptor.
        """

        missile_space_lateral_acc_vector = np.zeros(2)

        if on_board:
            # calculate command onboard the interceptor using seeker 
            interceptor_obs = InterceptorObservations(observations)
            missile_space_lateral_acc_vector = self._calc_command_onboard(self.n, interceptor_obs, dt, interceptor)
        else:
            ground_base_obs = GroundBaseObservations(observations)
            missile_space_lateral_acc_vector = self._calc_command_on_ground_station(self.n, ground_base_obs, interceptor, dt)

        # clamp command to physical limits
        missile_space_norm_lateral_acc_command = self._normalize_command(missile_space_lateral_acc_vector)

        # return np.array([0.0, 0.0])
        return missile_space_norm_lateral_acc_command