import abc
from typing import Generic, TypeVar

def interpolate_float(first: float, second: float, alpha: float) -> float:
    """
    Interpolate between two floats.
    
    Args:
        first (float): The first float.
        second (float): The second float.
        alpha (float): The interpolation factor (0 <= alpha <= 1).
    
    Returns:
        float: The interpolated float.
    """
    return (1 - alpha) * first + alpha * second

class Interpolatable(abc.ABC):
    """
    An interface for classes that can be interpolated.
    
    Methods:
        interpolate: Interpolate between two instances of the class.
    """

    @abc.abstractmethod
    def interpolate(self, other: 'Interpolatable', alpha: float) -> 'Interpolatable':
        """
        Interpolate between two instances of the class.
        
        Args:
            other (Interpolatable): The other instance to interpolate with.
            alpha (float): The interpolation factor (0 <= alpha <= 1).
        
        Returns:
            Interpolatable: The interpolated instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")

T = TypeVar('T', bound=Interpolatable)

class TimeSeries(Generic[T]):
    """
    A time series class to hold data at specific time points.
    
    Attributes:
        data (dict): A dictionary mapping time points to data values.
    """
    def __init__(self):
        self.all: dict[float, T] = {}

    def add(self, time: float, value: T):
        """
        Add a value at a specific time point.
        
        Args:
            time (float): The time point.
            value (T): The value to add.
        """
        self.all[time] = value

    def get(self, time: float) -> T:
        """
        Get the value at a specific time point.
        If there is not exact match, it will be interpolated between the two closest points.
        
        Args:
            time (float): The time point.
        
        Returns:
            T: The value at the specified time point.
        """
        if time in self.all:
            return self.all[time]

        # Find the two closest time points
        times = sorted(self.all.keys())
        for i in range(len(times) - 1):
            if times[i] <= time <= times[i + 1]:
                t1, t2 = times[i], times[i + 1]
                v1, v2 = self.all[t1], self.all[t2]
                alpha = (time - t1) / (t2 - t1)
                return v1.interpolate(v2, alpha)

        return None  # No data available for the given time
    
    def get_all_until(self, time: float) -> dict[float, T]:
        """
        Get all values up to a specific time point.
        
        Args:
            time (float): The time point.
        
        Returns:
            Dict[time, T]: A list of values up to the specified time point.
        """
        min_time = min(self.all.keys()) if self.all else None

        # no data available
        if min_time is None:
            return {}

        # return first time point for any time before
        if time < min_time:
            return {min_time: self.all[min_time]}

        # collect all data points up to the specified time
        series = {t: v for t, v in self.all.items() if t <= time}
        
        # append interpolated if needed
        if time not in series:
            series[time] = self.get(time)

        return series
        
        