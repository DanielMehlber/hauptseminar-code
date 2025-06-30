from physics.missile import PhysicalMissleModel
from pilots.pilot import Pilot
from physics import math as pmath
import numpy as np

class RandomEvasionPilot(Pilot):
    def __init__(self, missile: PhysicalMissleModel = None, safe_height: float = 1000.0):
        """
        This pilot generates random course diversions for the target, while still trying to 
        maintain a smooth trajectory and its original course.

        Depending on the uncertainty level, it will adjust the aggressiveness and evasion magnitude 
        of the target.
        """
        self._last_command = None
        self.deviation = np.zeros(2)  # Initialize deviation to zero
        self.missile = missile
        self.safe_height = safe_height

    def step(self, dt: float, t: float) -> np.ndarray:
        # Generate a smooth random acceleration command
        if self._last_command is None:
            self._last_command = np.zeros(2)  # Initialize with a zero command

        # Determine aggression and trajectory maintainance based on uncertainty level
        aggression, trajectory_maintainance = self._determine_parameters(self.uncertainty)

        mean_command = -self.deviation
        
        # if missile model is available, check if it is below the safe height and 
        # pull up if necessary.
        if self.missile is not None:
            # If a missile is provided, use its altitude to adjust the parameters
            altitude = self.missile.world_pos[2]
            if altitude < self.safe_height:
                # get direction on the lateral plane to pull up
                missile_space_up = self.missile.body_to_world_rot_mat.T @ np.array([0.0, 0.0, 1.0])
                missile_space_up_lateral = pmath.project_on_plane(missile_space_up, np.array([1.0, 0.0, 0.0]))
                missile_space_up_lateral = np.array([missile_space_up_lateral[1], missile_space_up_lateral[2]])

                # pull up
                pull_up_factor = 1.0 - (altitude / self.safe_height)
                mean_command += pull_up_factor * missile_space_up_lateral * aggression

        # Add a small random change to the last command
        delta = np.random.normal(loc=mean_command, scale=aggression, size=2)
        new_command = self._last_command + delta * dt

        self.deviation += new_command * trajectory_maintainance * dt

        # Clip the command to ensure it stays within [-1, 1]
        new_command = np.clip(new_command, -1, 1)

        # Update the last command for smoothness
        self._last_command = new_command

        return new_command
    
    def reset(self):
        self._last_command = None
        self.deviation = np.zeros(2)

    def _determine_parameters(self, uncertainty):
        min_aggression = 0.01
        max_aggression = 0.5
        min_trajectory_maintainance = 0.4
        max_trajectory_maintainance = 0.01
        
        # interpolate parameters based on uncertainty
        aggression = min_aggression + (max_aggression - min_aggression) * uncertainty
        trajectory_maintainance = min_trajectory_maintainance + (max_trajectory_maintainance - min_trajectory_maintainance) * uncertainty

        return aggression, trajectory_maintainance