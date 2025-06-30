from dataclasses import dataclass
import numpy as np

@dataclass
class SeekerObservations:
    """
    Observation space for a missile seeker in the simulation environment.
    This class contains the observations related to the seeker, which is responsible
    for tracking the target and providing guidance information to the interceptor.
    """
    los_unit_vec: np.ndarray                        # line-of-sight unit vector to target (x, y, z), i.e. direction to target
    distance_to_target: float                       # distance to target (scalar)
    closing_rate_vec: np.ndarray                    # closing rate vector (x, y, z) to target
    los_angles_vec: np.ndarray                      # line-of-sight angles (azimuth, elevation)
    los_angle_rates_vec: np.ndarray                 # line-of-sight angle rates (azimuth rate, elevation rate)

    def __init__(self, buffer: np.ndarray = None):
        if buffer is not None:
            self.unpack(buffer)

    def pack(self) -> np.ndarray:
        return np.concatenate([
            self.los_unit_vec,
            np.array([self.distance_to_target]),
            self.closing_rate_vec,
            self.los_angles_vec,
            self.los_angle_rates_vec
        ])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        assert packed_observations is not None, "Packed observations cannot be None"
        self.los_unit_vec = packed_observations[0:3]
        self.distance_to_target = packed_observations[3]
        self.closing_rate_vec = packed_observations[4:7]
        self.los_angles_vec = packed_observations[7:9]
        self.los_angle_rates_vec = packed_observations[9:11]

@dataclass
class ImuObservations:
    """
    Observation space for a missile's Inertial Measurement Unit (IMU) in the simulation environment.
    This class contains the observations related to the IMU, which measures the interceptor's orientation
    and motion in space.
    """
    world_space_interceptor_orientation: np.ndarray  # unit vector representing the orientation of the interceptor in world space (x, y, z)
    missile_space_turn_rate: np.ndarray              # turn rate in missile space (yaw, pitch, roll) measured by IMU
    missile_space_acceleration: np.ndarray           # acceleration in missile space (x, y, z) measured by IMU

    def __init__(self, buffer: np.ndarray = None):
        if buffer is not None:
            self.unpack(buffer)

    def pack(self) -> np.ndarray:
        return np.concatenate([
            self.world_space_interceptor_orientation,
            self.missile_space_turn_rate,
            self.missile_space_acceleration
        ])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        assert packed_observations is not None, "Packed observations cannot be None"
        self.world_space_interceptor_orientation = packed_observations[0:3]
        self.missile_space_turn_rate = packed_observations[3:6]
        self.missile_space_acceleration = packed_observations[6:9]

@dataclass
class InterceptorFrameObservations:
    """
    Observation space for a missile interceptor in the simulation environment
    for a single point in time.
    """
    seeker: SeekerObservations          # observations from the seeker
    imu: ImuObservations                # observations from the IMU (Inertial Measurement

    def __init__(self, buffer: np.ndarray = None):
        if buffer is not None:
            self.unpack(buffer)

    def pack(self) -> np.ndarray:
        return np.concatenate([self.seeker.pack(), self.imu.pack()])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        assert packed_observations is not None, "Packed observations cannot be None"
        self.seeker = SeekerObservations(packed_observations[0:11])
        self.imu = ImuObservations(packed_observations[11:20])

@dataclass
class InterceptorObservations:
    current_frame: InterceptorFrameObservations  # current frame observations of the interceptor
    previous_frame: InterceptorFrameObservations  # previous frame observations of the interceptor

    def __init__(self, buffer: np.ndarray = None):
        if buffer is not None:
            self.unpack(buffer)

    def pack(self) -> np.ndarray:
        return np.concatenate([self.current_frame.pack(), self.previous_frame.pack()])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        assert packed_observations is not None, "Packed observations cannot be None"
        self.current_frame = InterceptorFrameObservations(packed_observations[0:20])
        self.previous_frame = InterceptorFrameObservations(packed_observations[20:40])


@dataclass
class GroundBaseObservations:
    """
    Observation space for a ground base in the simulation environment. The ground base tracks
    the interceptor and the target using radar and can issue commands to the interceptor.
    """
    # ground base observations
    world_space_interceptor_pos: np.ndarray                # position of the interceptor in world space (x, y, z)
    world_space_target_pos: np.ndarray                    # position of the target in world space (x, y, z)
    
    def __init__(self, buffer: np.ndarray = None):
        if buffer is not None:
            self.unpack(buffer)

    def pack(self) -> np.ndarray:
        return np.concatenate([self.world_space_interceptor_pos, self.world_space_target_pos])
    
    def unpack(self, packed_observations: np.ndarray) -> None:
        assert packed_observations is not None, "Packed observations cannot be None"
        self.world_space_interceptor_pos = packed_observations[0:3]
        self.world_space_target_pos = packed_observations[3:6]