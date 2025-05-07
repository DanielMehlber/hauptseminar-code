import abc
import numpy as np


class Pilot(abc.ABC):
    """
    A pilot is capable of controlling a missile (target or interceptor) in the
    simulation environment. It issues acceleration commands which are then executed
    by the missile's physical model. This is a strict separation of physics and behavior.
    """
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def step(self, dt: float, t: float) -> np.ndarray:
        """
        Perform a step in the agent's logic.
        This method should be overridden by subclasses to implement specific behavior.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def reset(self):
        """
        Reset the agent's state.
        This method should be overridden by subclasses to implement specific behavior.
        """

        raise NotImplementedError("Subclasses must implement this method.")
