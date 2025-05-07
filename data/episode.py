import numpy as np
from dataclasses import dataclass, field
from data.time_series import *

@dataclass
class TargetState(Interpolatable):
    position: np.ndarray
    velocity: np.ndarray

    def interpolate(self, other: 'TargetState', alpha: float) -> 'TargetState':
        interpolated_position = interpolate_float(self.position, other.position, alpha)
        interpolated_velocity = interpolate_float(self.velocity, other.velocity, alpha)
        return TargetState(interpolated_position, interpolated_velocity)

@dataclass
class InterceptorState(Interpolatable):
    position: np.ndarray    # current position of the interceptor
    velocity: np.ndarray    # current velocity of the interceptor
    command: np.ndarray     # last issued acceleration command
    los_angle: np.ndarray   # line of sight angle to the target
    distance: float         # distance to the target

    def interpolate(self, other: 'InterceptorState', alpha: float) -> 'InterceptorState':
        interpolated_position = interpolate_float(self.position, other.position, alpha)
        interpolated_velocity = interpolate_float(self.velocity, other.velocity, alpha)
        interpolated_command = interpolate_float(self.command, other.command, alpha)
        interpolated_los_angle = interpolate_float(self.los_angle, other.los_angle, alpha)
        interpolated_distance = interpolate_float(self.distance, other.distance, alpha)
        return InterceptorState(interpolated_position, interpolated_velocity, interpolated_command, interpolated_los_angle, interpolated_distance)

@dataclass
class Interceptor:
    states: TimeSeries[InterceptorState]

    def __init__(self):
        self.states = TimeSeries[InterceptorState]()

@dataclass
class Episode:
    target_states: TimeSeries[TargetState] # time -> TargetState
    interceptors: dict[str, Interceptor] # id -> Interceptor

    def __init__(self):
        self.target_states = TimeSeries[TargetState]()
        self.interceptors = {}

    def get_interceptor(self, interceptor_id: str) -> Interceptor:
        """
        Get the interceptor data for a given interceptor ID.
        
        Args:
            interceptor_id (str): The ID of the interceptor.
        
        Returns:
            Interceptor: The interceptor data.
        """
        if interceptor_id not in self.interceptors:
            self.interceptors[interceptor_id] = Interceptor()
        return self.interceptors[interceptor_id]