import numpy as np

class LinearDistanceNoise:
    def __init__(self, spread_per_meter: float = 0.01):
        """
        Initialize the linear distance noise model.

        :param spread_per_meter: The noise spread per meter of distance.
        """
        self.spread_per_meter = spread_per_meter

    def apply(self, point: np.ndarray, distance: float, intensity=1.0) -> np.ndarray:
        """
        Add noise to a point based on the distance.

        :param point: The original point as a numpy array.
        :param distance: The distance from the origin.
        :return: The noisy point as a numpy array.
        """
        noise = np.random.normal(0, distance * self.spread_per_meter * intensity, size=point.shape)
        return point + noise