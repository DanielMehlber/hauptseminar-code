import abc

@abc.ABC
class Agent(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def step(self, dt: float, t: float):
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
