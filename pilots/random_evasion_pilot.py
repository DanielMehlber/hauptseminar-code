from pilots.pilot import Pilot
import numpy as np

class RandomEvasionPilot(Pilot):
    def __init__(self, factor: float = 0.1):
        self._last_command = None
        self.factor = factor  # Factor to control the smoothness of the command
        self.deviation = np.zeros(2)  # Initialize deviation to zero

    def step(self, dt: float, t: float) -> np.ndarray:
        # Generate a smooth random acceleration command
        if self._last_command is None:
            self._last_command = np.zeros(2)  # Initialize with a zero command

        # Add a small random change to the last command
        delta = np.random.normal(loc=-self.deviation, scale=self.factor, size=2)
        new_command = self._last_command + delta * dt

        self.deviation += new_command * 0.01 * dt

        # Clip the command to ensure it stays within [0, 1]
        new_command = np.clip(new_command, 0, 1)

        # Update the last command for smoothness
        self._last_command = new_command


        return new_command
    
    def reset(self):
        self._last_command = None
        self.deviation = np.zeros(2)