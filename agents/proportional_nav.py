import numpy as np 

class ProportionalNavigationAgent:
    def __init__(self, max_acc: float, init_distance: float ,n: float):
        self.n = n
        self.init_distance = init_distance # required to convert norm closing rate to closing rate in m/s
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
            norm_distance,
            norm_closing_rate,
            norm_los_angle_vec,
            norm_los_rate_vec,
            _,
            _,
        ) = self._get_observations(observations)

        # de-normlize to real world space
        real_closing_rate = norm_closing_rate * self.init_distance
        max_change_velocity = self.max_acc * dt

        # re-normalize to max acceleration
        norm_closing_rate = -real_closing_rate / max_change_velocity

        # command must be between -1 and 1 respectively
        # therefore, every input is normed to the max acceleration
        acc_command = self.n * norm_los_rate_vec * norm_closing_rate
        return acc_command
    
    def _get_observations(self, observations: np.ndarray) -> tuple:        
        norm_distance = observations[0]
        norm_closing_rate = observations[1]
        norm_los_angle = observations[2:4]
        norm_los_angle_rate = observations[4:6]
        world_space_interceptor_orientation = observations[6:9]
        missile_space_turn_rate = observations[9:11]

        return norm_distance, norm_closing_rate, norm_los_angle, norm_los_angle_rate, world_space_interceptor_orientation, missile_space_turn_rate
        
        

        