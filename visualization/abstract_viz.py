import abc
import data.episode as ep

class AbstractVisualizer(abc.ABC):

    @abc.abstractmethod
    def set_episode_data(self, data: ep.Episode):
        raise NotImplementedError("set_episode_data method not implemented.")
    
    @abc.abstractmethod
    def render(self, time: float):
        raise NotImplementedError("render method not implemented.")
    
    @abc.abstractmethod
    def playback(self, time: float, speed: float = 1.0, fps: int = 10):
        """
        Play back the episode data at a given speed.
        
        Args:
            time (float): The time to play back to.
            speed (float): The playback speed.
        """
        raise NotImplementedError("playback method not implemented.")
    
    @abc.abstractmethod
    def save_playback(self, filename: str, time: float, speed: float = 1.0, fps: int = 10):
        """
        Save the visualization to a file.
        
        Args:
            filename (str): The name of the file to save to.
        """
        raise NotImplementedError("save method not implemented.")
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError("close method not implemented.")
    
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError("reset method not implemented.")
