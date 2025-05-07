from dataclasses import dataclass
import numpy as np

@dataclass
class InterceptorObservations:
    """
    Observation space for a missile aiming for a target in the simulation environment.
    """
    # seeker observations
    distance_vec: np.ndarray                        # distance to target (relative to initial distance)
    closing_rate_vec: np.ndarray                    # closing rate (relative to max speed)
    los_angles_vec: np.ndarray                      # line-of-sight angles (from radians to [-1, 1])
    los_angle_rates_vec: np.ndarray                 # line-of-sight angle rates (from radians to [-1, 1])
    
    # gyroscope observations
    world_space_interceptor_orientation: np.ndarray # interceptor orientation in world space (x, y, z)
    missile_space_turn_rate: np.ndarray             # turn rate in missile space (x, y, z)

    def __init__(self, space: np.ndarray = None):
        if space is not None:
            self.unpack(space)

    def pack(self) -> np.ndarray:
        """
        Pack the observations into a single numpy array.
        
        Returns:
            np.ndarray: The packed observations.
        """
        return np.concatenate([
            self.distance_vec,
            self.closing_rate_vec,
            self.los_angles_vec,
            self.los_angle_rates_vec,
            self.world_space_interceptor_orientation,
            self.missile_space_turn_rate
        ])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        """
        Unpack the packed observations into the individual components.
        """
        self.distance_vec = packed_observations[0:3]
        self.closing_rate_vec = packed_observations[3:6]
        self.los_angles_vec = packed_observations[6:8]
        self.los_angle_rates_vec = packed_observations[8:10]
        self.world_space_interceptor_orientation = packed_observations[10:13]
        self.missile_space_turn_rate = packed_observations[13:16]