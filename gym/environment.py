import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import data.episode as ep
import time
from models.missile import MissileModel
import math


class MissileEnvSettings:
    def __init__(self, realtime=False, time_step=1.0, min_dt=0.1):
        self.time_step = time_step    # Speed multiplier for simulation (e.g., 1.0 = real-time)
        self.min_dt = min_dt          # Minimum delta time for simulation steps
        self.realtime = realtime      # If True, the simulation will run in real-time


class MissileEnv(gym.Env):
    """
    A custom environment for simulating missile interception scenarios. 
    It employs a target and an interceptor missile model, allowing for reinforcement learning
    training and evaluation.
    """

    def __init__(self, target: MissileModel, interceptor: MissileModel, settings=MissileEnvSettings()):
        super().__init__()
        self.settings = settings
        self.last_step_time = None
        self.sim_time = 0.0  # Tracks total simulation time

        self.target = target
        self.interceptor = interceptor

        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # required as anchor point to normalize distance measurements
        self.missile_space_start_distance_vec = self.interceptor.orientation_matrix.T @ (self.target.pos - self.interceptor.pos)

        # required to calculate observations (which as basically changes in position and velocity)
        self.missile_space_last_los_vec: np.ndarray = None # required to calculate closing rate
        self.missile_space_last_los_angle: np.ndarray = None # required to calculate line-of-sight angle rate

        self.current_episode: ep.Episode = None # current episode object

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.interceptor.reset()
        self.target.reset()
        self.sim_time = 0.0

        # data for display in visualizer
        self.last_step_time = time.time()
        self.last_acc_command = np.zeros(2, dtype=np.float32)
        self.last_missile_orientation_matrix = np.eye(3, dtype=np.float32)

        # init sensor state for differential measurements
        self._update_sensor_data()
        self.current_episode = ep.Episode()

        return self._get_obs(self.settings.min_dt)

    def _line_of_sight_angle(self):
        # Calculate the angle between the interceptor and target positions
        world_space_los_vector = self.target.pos - self.interceptor.pos
        
        # transform to interceptor space: we want to calculate the LOS angle 
        # from the interceptor's perspective
        missile_reference = self.interceptor.orientation_matrix.T
        missile_space_los_vector = missile_reference @ world_space_los_vector
        missile_space_los_vector /= np.linalg.norm(missile_space_los_vector)

        x, y, z = missile_space_los_vector
        norm_xy = np.sqrt(x**2 + y**2)

        # azimuth (horizontal) angle in body-frame: −π … +π
        h_angle = np.arctan2(y, x)                                  # left/right of nose

        # elevation (vertical) angle: −π/2 … +π/2
        v_angle = np.arctan2(z, norm_xy)                            # above/below nose

        
        # Note: a single argtan(x) and arctan(y) would not consider the sign of the
        # x and y components, which is important for determining the correct quadrant. 
        return np.array([h_angle, v_angle], dtype=np.float32)
    

    def _update_sensor_data(self):
        # required for delta-distance in observations (closing rate)
        world_space_los_vec = self.target.pos - self.interceptor.pos # in world space, but we need it from missile's pov
        self.missile_space_last_los_vec = self.interceptor.orientation_matrix.T @ world_space_los_vec

        # required for turning rate of interceptor
        world_space_interceptor_velocity = self.interceptor.get_velocity()
        self.world_space_last_interceptor_orientation = world_space_interceptor_velocity / np.linalg.norm(world_space_interceptor_velocity)

        # required for line-of-sight angle rate (from seeker's point of view)
        self.missile_space_last_los_angle = self._line_of_sight_angle()

    def _update_episode_data(self):
        # Update the episode data with the current state of the interceptor and target
        interceptor_state = ep.InterceptorState(
            position=self.interceptor.pos.copy(),
            velocity=self.interceptor.get_velocity().copy(),
            command=self.last_acc_command.copy(),
            los_angle=self.missile_space_last_los_angle.copy(),
            distance=np.linalg.norm(self.missile_space_last_los_vec), # distance to target in missile space
        )

        target_state = ep.TargetState(
            position=self.target.pos.copy(),
            velocity=self.target.get_velocity().copy()
        )

        # Add the states to the episode
        self.current_episode.target_states.add(self.sim_time, target_state)
        self.current_episode.get_interceptor("Agent").states.add(self.sim_time, interceptor_state)

    def step(self, action):
        dt = 0.0
        if self.settings.realtime:
            step_time = time.time()
            real_dt = step_time - self.last_step_time

            # if dt gets to small, equations explode
            wait_time = self.settings.min_dt - real_dt
            if wait_time > 0:
                time.sleep(wait_time)

            real_dt = time.time() - self.last_step_time

            self.last_step_time = step_time

            # Apply simulation time scaling
            dt = real_dt
            self.sim_time += dt
        else:
            # We use a fixed time step
            dt = self.settings.time_step
            self.sim_time += dt

        # Update entities with scaled delta time
        self.target.accelerate(np.array([0.0, 0.0]), dt=dt, t=self.sim_time)
        self.interceptor.accelerate(action, dt=dt, t=self.sim_time)

        # Update values for visualization
        self.last_acc_command = action.copy()
        self.last_missile_orientation_matrix = self.interceptor.orientation_matrix.copy()

        obs = self._get_obs(dt)
        status = self._check_status()
        done = self._check_done(status)
        reward = self._get_reward(action, status, dt)
        
        # Update after reward calculation because it needs the last sensor data
        self._update_sensor_data()        
        self._update_episode_data()

        return obs, reward, done, False, {}

    def render(self):
        pass

    def _get_obs(self, dt):

        # line-of-sight from interceptor to target (from seeker's point of view) - length is distance to target
        missile_space_los_vec = self.interceptor.orientation_matrix.T @ (self.target.pos - self.interceptor.pos) # target position in missile space

        # distance and closing rate to the target (from seekers point of view)
        missile_space_distance_before_vec = np.abs(self.missile_space_last_los_vec) # we care about component-wise distance to target
        missile_space_distance_vec = np.abs(missile_space_los_vec) # same here
        missile_space_closing_rate_vec = (missile_space_distance_before_vec - missile_space_distance_vec) / dt # positive if closing in on target

        # seeker line-of-sight angle rate
        missile_space_los_angles_vec = self._line_of_sight_angle()
        missile_space_los_angles_rate_vec = (self.missile_space_last_los_angle - missile_space_los_angles_vec) / dt

        # interceptor orientation (e.g. by gyroscopes)
        world_space_interceptor_velocity_vec = self.interceptor.get_velocity()
        world_space_interceptor_orientation_vec = world_space_interceptor_velocity_vec / np.linalg.norm(world_space_interceptor_velocity_vec)
        
        # interceptor turn angles in missile space (e.g. by gyroscopes)
        missile_space_last_interceptor_orientation_vec = self.interceptor.orientation_matrix.T @ self.world_space_last_interceptor_orientation
        missile_space_yaw_angle, missile_space_pitch_angle = self.interceptor.calculate_local_angles_to(missile_space_last_interceptor_orientation_vec)
        missile_space_pitch_angle *= -1.0 # invert pitch angle to match the interceptor's coordinate system
        missile_space_yaw_angle *= -1.0 # invert yaw angle to match the interceptor's coordinate system

        # interceptor turn rate in missile space (e.g. by gyroscopes)
        missile_space_turn_angles_vec = np.array([missile_space_yaw_angle, missile_space_pitch_angle])
        norm_missile_space_turn_angles_vec = missile_space_turn_angles_vec / np.pi # normalize to [-1, 1]
        missile_space_turn_rate_vec = norm_missile_space_turn_angles_vec / dt

        # norm measurements to avoid flooding the network
        start_distance = np.linalg.norm(self.missile_space_start_distance_vec) # to clamp vector values relative to the initial distance

        # TODO: think of better ways to normalize these values
        norm_distance_vec = missile_space_los_vec / start_distance                              # relative to initial distance
        norm_closing_rate_vec = missile_space_closing_rate_vec / self.interceptor.max_speed     # relative to max speed
        norm_los_angles_vec = missile_space_los_angles_vec / np.pi                              # converted to interval [-1, 1] 
        norm_los_angle_rates_vec = missile_space_los_angles_rate_vec / np.pi                    # converted to interval [-1, 1]

        # assert that no component is NaN of Inf
        assert np.all(np.isfinite(norm_distance_vec)), "norm_distance_vec contains NaN or Inf values"
        assert np.all(np.isfinite(norm_closing_rate_vec)), "norm_closing_rate_vec contains NaN or Inf values"
        assert np.all(np.isfinite(norm_los_angles_vec)), "norm_los_angles_vec contains NaN or Inf values"
        assert np.all(np.isfinite(norm_los_angle_rates_vec)), "norm_los_angle_rates_vec contains NaN or Inf values"
        assert np.all(np.isfinite(world_space_interceptor_orientation_vec)), "world_space_interceptor_orientation_vec contains NaN or Inf values"
        assert np.all(np.isfinite(missile_space_turn_rate_vec)), "missile_space_turn_rate_vec contains NaN or Inf values"

        # pack all observations into a single vector
        return np.concatenate([
            norm_distance_vec, norm_closing_rate_vec, norm_los_angles_vec, norm_los_angle_rates_vec, # seeker data
            world_space_interceptor_orientation_vec, missile_space_turn_rate_vec, # internal sensor data (gyroscopes, etc)
        ])

    def _get_reward(self, action, status, dt):
        # The less the distance, the higher the reward
        dist = np.linalg.norm(self.interceptor.pos - self.target.pos)
        dist_reward = -dist / np.linalg.norm(self.missile_space_start_distance_vec)

        # We want to reward the interceptor for closing in on the target
        closing_rate_reward = (self.missile_space_last_los_vec - dist) * dt

        # We want to reward/punish the interceptor for certain events
        event_reward = 0.0
        if status == "hit":
            event_reward = +2
        elif status == "crashed":
            event_reward = -5
        elif status == "expired":
            event_reward = -1
            
        
        # We want to keep the interceptor energy efficient (less commands = better)
        action_punishment = -np.linalg.norm(action) / self.interceptor.max_lat_acc

        # We want the interceptor to avoid the ground (z < 0)
        interceptor_altidude = self.interceptor.pos[2]
        safe_altitude = 1000

        # Exponential penalty for altitude below safe level
        ground_penalty = -np.exp(-interceptor_altidude / safe_altitude) if status != "crashed" else -1.0

        # Weighting the rewards
        dist_reward *= 5.0
        closing_rate_reward *= 3.0
        event_reward *= 10.0
        action_punishment *= 5.0
        ground_penalty *= 10.0 

        # Combine all rewards
        reward = dist_reward + closing_rate_reward + event_reward + action_punishment + ground_penalty
        return reward

    def _check_done(self, status):
        return status != "ongoing"
    
    def _check_status(self):
        if self.interceptor.pos[2] < 0:
            print ("Interceptor crashed into the ground")
            return "crashed"
        elif np.linalg.norm(self.interceptor.pos - self.target.pos) < 50:
            print ("Interceptor hit the target")
            return "hit"
        elif self.sim_time > 50:
            print ("Simulation expired")
            return "expired"
        else:
            return "ongoing"
    

