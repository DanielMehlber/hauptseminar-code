import numpy as np 

class ProportionalNavigationAgent:
    def __init__(self, max_acc: float, max_speed: float ,n: float):
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
        (
            norm_distance_vec,          # distance to target (relative to initial distance)
            norm_closing_rate_vec,      # closing rate (relative to max speed)
            norm_los_angles_vec,        # line-of-sight angles (from radians to [-1, 1])
            norm_los_angle_rates_vec,   # line-of-sight angle rates (from radians to [-1, 1])
            _,
            _,
        ) = self._get_observations(observations)

        # de-normlize to real world space (was relative to max-speed)
        real_closing_rate_vec = norm_closing_rate_vec * self.max_speed
        real_los_angle_rates_vec = norm_los_angle_rates_vec * np.pi

        # is a 3D vector, but we need 2D (we can ignore the closing rate along the longitudinal axis x)
        real_closing_rate_vec = np.array([real_closing_rate_vec[1], real_closing_rate_vec[2]])

        # proportional guidance law
        real_acc_command = self.n * real_los_angle_rates_vec * real_closing_rate_vec

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
        
        

        