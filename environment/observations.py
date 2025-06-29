from dataclasses import dataclass
import numpy as np

@dataclass
class InterceptorObservations:
    """
    Observation space for a missile aiming for a target in the simulation environment.
    """
    # seeker observations
    los_distance_vec: np.ndarray                    # distance to target (or relative position)
    previous_los_distance_vec: np.ndarray           # previous distance to target (or relative position)
    closing_rate_vec: np.ndarray                    # closing rate
    los_angles_vec: np.ndarray                      # line-of-sight angles (from radians to [-1, 1])
    los_angle_rates_vec: np.ndarray                 # line-of-sight angle rates (from radians to [-1, 1])
    
    # gyroscope observations
    world_space_interceptor_orientation: np.ndarray # interceptor orientation in world space (x, y, z)
    missile_space_turn_rate: np.ndarray             # turn rate in missile space (x, y, z) measured by IMU
    missile_space_acceleration: np.ndarray          # acceleration in missile space (x, y, z) measured by IMU

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
            self.los_distance_vec,
            self.previous_los_distance_vec,
            self.closing_rate_vec,
            self.los_angles_vec,
            self.los_angle_rates_vec,
            self.world_space_interceptor_orientation,
            self.missile_space_turn_rate,
            self.missile_space_acceleration
        ])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        """
        Unpack the packed observations into the individual components.
        """
        self.los_distance_vec = packed_observations[0:3]
        self.previous_los_distance_vec = packed_observations[3:6]
        self.closing_rate_vec = packed_observations[6:9]
        self.los_angles_vec = packed_observations[9:11]
        self.los_angle_rates_vec = packed_observations[11:13]
        self.world_space_interceptor_orientation = packed_observations[13:16]
        self.missile_space_turn_rate = packed_observations[16:19]
        self.missile_space_acceleration = packed_observations[19:22]


@dataclass
class GroundBaseObservations:
    """
    Observation space for a ground base in the simulation environment. The ground base tracks
    the interceptor and the target using radar and can issue commands to the interceptor.
    """
    # ground base observations
    world_space_interceptor_pos: np.ndarray                # position of the interceptor in world space (x, y, z)
    world_space_target_pos: np.ndarray                    # position of the target in world space (x, y, z)
    
    def __init__(self, space: np.ndarray = None):
        if space is not None:
            self.unpack(space)

    def pack(self) -> np.ndarray:
        """
        Pack the observations into a single numpy array.
        
        Returns:
            np.ndarray: The packed observations.
        """
        return np.concatenate([self.world_space_interceptor_pos, self.world_space_target_pos])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        """
        Unpack the packed observations into the individual components.
        """
        self.world_space_interceptor_pos = packed_observations[0:3]
        self.world_space_target_pos = packed_observations[3:6]