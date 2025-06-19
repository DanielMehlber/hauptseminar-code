from pilots.pilot import Pilot
import numpy as np

class RandomEvasionPilot(Pilot):
    def __init__(self, aggression: float = 0.1, trajectory_maintainance: float = 0.01):
        """
        This pilot generates random course diversions for the target, while still trying to 
        maintain a smooth trajectory and its original course.

        Args:
            aggression (float): A factor to control the smoothness of the command. 
                                Higher values lead to more aggressive evasions.
            trajectory_maintainance (float): A factor to control how much the pilot tries to maintain the original trajectory.
        """
        self._last_command = None
        self.aggression = aggression  # Factor to control the smoothness of the command
        self.deviation = np.zeros(2)  # Initialize deviation to zero
        self.trajectory_maintainance = trajectory_maintainance

    def step(self, dt: float, t: float) -> np.ndarray:
        # Generate a smooth random acceleration command
        if self._last_command is None:
            self._last_command = np.zeros(2)  # Initialize with a zero command

        # Add a small random change to the last command
        delta = np.random.normal(loc=-self.deviation, scale=self.aggression, size=2)
        new_command = self._last_command + delta * dt

        self.deviation += new_command * self.trajectory_maintainance * dt

        # Clip the command to ensure it stays within [-1, 1]
        new_command = np.clip(new_command, -1, 1)

        # Update the last command for smoothness
        self._last_command = new_command


        return new_command
    
    def reset(self):
        self._last_command = None
        self.deviation = np.zeros(2)