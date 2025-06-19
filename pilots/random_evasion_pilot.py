from pilots.pilot import Pilot
import numpy as np

class RandomEvasionPilot(Pilot):
    def __init__(self):
        """
        This pilot generates random course diversions for the target, while still trying to 
        maintain a smooth trajectory and its original course.

        Depending on the uncertainty level, it will adjust the aggressiveness and evasion magnitude 
        of the target.
        """
        self._last_command = None
        self.deviation = np.zeros(2)  # Initialize deviation to zero

    def step(self, dt: float, t: float, uncerctainty=0.0) -> np.ndarray:
        # Generate a smooth random acceleration command
        if self._last_command is None:
            self._last_command = np.zeros(2)  # Initialize with a zero command

        # Determine aggression and trajectory maintainance based on uncertainty level
        aggression, trajectory_maintainance = self._determine_parameters(uncerctainty)

        # Add a small random change to the last command
        delta = np.random.normal(loc=-self.deviation, scale=aggression, size=2)
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
        min_trajectory_maintainance = 0.001
        max_trajectory_maintainance = 0.1
        
        # interpolate parameters based on uncertainty
        aggression = min_aggression + (max_aggression - min_aggression) * uncertainty
        trajectory_maintainance = min_trajectory_maintainance + (max_trajectory_maintainance - min_trajectory_maintainance) * uncertainty

        return aggression, trajectory_maintainance