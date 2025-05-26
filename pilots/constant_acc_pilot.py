from pilots.pilot import Pilot
import numpy as np

class ConstantAccelerationPilot(Pilot):
    def __init__(self, acc: np.ndarray):
        self.acc = acc

    def step(self, dt: float, t: float) -> np.ndarray:
        return self.acc
    
    def reset(self):
        pass