import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from gym.visualisation import MissileVisualizer
import time
from models.missile import MissileModel


class MissileEnvSettings:
    def __init__(self, realtime=False, time_speed=1.0, min_dt=0.1):
        self.realtime = realtime        # If True, slows simulation to real-time
        self.time_speed = time_speed    # Speed multiplier for simulation (e.g., 1.0 = real-time)
        self.min_dt = min_dt                # Minimum delta time for simulation steps


class MissileEnv(gym.Env):

    def __init__(self, target: MissileModel, interceptor: MissileModel, settings=MissileEnvSettings()):
        super().__init__()
        self.settings = settings
        self.last_step_time = None
        self.sim_time = 0.0  # Tracks total simulation time

        self.target = target
        self.interceptor = interceptor

        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-500, high=500, shape=(2,), dtype=np.float32)

        self.visualizer = MissileVisualizer(fit_view=True)
        self.start_distance = np.linalg.norm(self.interceptor.pos - self.target.pos)

        # required to calculate observations (which as basically changes in position and velocity)
        self.last_distance = None # required to calculate closing rate
        self.last_los_angle = None # required to calculate line-of-sight angle rate

        self.reset()

    def _line_of_sight_angle(self):
        # Calculate the angle between the interceptor and target positions
        world_los_vector = self.target.pos - self.interceptor.pos
        
        # transform to interceptor space: we want to calculate the LOS angle 
        # from the interceptor's perspective
        reference = self.interceptor.get_orientation_matrix()
        missile_los_vector = reference @ world_los_vector
        missile_los_vector /= np.linalg.norm(missile_los_vector)

        x, y, z = missile_los_vector
        norm_xy = np.sqrt(x**2 + y**2)

        # azimuth (horizontal) angle in body-frame: −π … +π
        h_angle = np.arctan2(y, x)                                  # left/right of nose

        # elevation (vertical) angle: −π/2 … +π/2
        v_angle = np.arctan2(z, norm_xy)                            # above/below nose

        
        # Note: a single argtan(x) and arctan(y) would not consider the sign of the
        # x and y components, which is important for determining the correct quadrant. 
        return np.array([h_angle, v_angle], dtype=np.float32)
    


    def _update_sensor_data(self):
        self.last_distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        self.last_los_angle = self._line_of_sight_angle()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.interceptor.reset()
        self.target.reset()
        self.trajectory = []
        self.sim_time = 0.0
        self.last_step_time = time.time()
        self.visualizer.reset()
    
        self.last_distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        self.last_los_angle = self._line_of_sight_angle()

        return self._get_obs(self.settings.min_dt), {}

    def step(self, action):
        step_time = time.time()
        real_dt = step_time - self.last_step_time

        # if dt gets to small, equations explode
        wait_time = self.settings.min_dt - real_dt
        if wait_time > 0:
            time.sleep(wait_time)

        real_dt = time.time() - self.last_step_time
        self.last_step_time = step_time

        # Apply simulation time scaling
        dt = real_dt * self.settings.time_speed
        self.sim_time += dt

        # Update entities with scaled delta time
        self.target.accelerate(np.zeros(2), dt=dt, t=self.sim_time)
        oversteered = self.interceptor.accelerate(action, dt=dt, t=self.sim_time)

        self.trajectory.append((self.interceptor.pos.copy(), self.target.pos.copy()))

        obs = self._get_obs(dt)
        status = self._check_status()
        done = self._check_done(status)
        reward = self._get_reward(action, oversteered, status, dt)
        
        # Update after reward calculation because it needs the last sensor data
        self._update_sensor_data()

        distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        print(f"Reward: {reward} (Distance: {distance}), Done: {done}, Time: {self.sim_time:.2f}s")

        self.render()
        
        return obs, reward, done, False, {}

    def render(self):
        if len(self.trajectory) > 3:
            self.visualizer.update(self.interceptor.pos, self.target.pos, sim_time=self.sim_time)

    def _get_obs(self, dt):
        # closing rate to the target
        distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        closing_rate = (self.last_distance - distance) / dt

        # line-of-sight angle rate
        los_angle = self._line_of_sight_angle()
        los_angle_rate = (los_angle - self.last_los_angle) / dt

        return np.concatenate([
            distance, closing_rate, los_angle, los_angle_rate, # sensor data
            self.interceptor.pos, self.interceptor.vel, # reflecting interceptor state
        ])

    def _get_reward(self, action, oversteered, status, dt):
        # The less the distance, the higher the reward
        dist = np.linalg.norm(self.interceptor.pos - self.target.pos)
        dist_reward = -dist / self.start_distance

        # We want to reward the interceptor for closing in on the target
        closing_rate = (self.last_distance - dist) / dt
        dist_reward += closing_rate

        # We want to reward/punish the interceptor for certain events
        event_reward = 0.0
        if status == "hit":
            event_reward = +10
        elif status == "crashed":
            event_reward = -10
        elif status == "expired":
            event_reward = -5

        # We want to punish the interceptor for oversteering
        if oversteered:
            event_reward -= 10
        
        # We want to keep the interceptor energy efficient (less commands = better)
        action_punishment = -np.linalg.norm(action) / self.interceptor.max_acc_magnitude

        # We want the interceptor to avoid the ground (z < 0)
        ground_penalty = -np.exp(-self.interceptor.pos[2] / 50) if self.interceptor.pos[2] < 50 else 0

        reward = dist_reward + event_reward + action_punishment + ground_penalty

        print(f"Reward: {reward} = Dist: {dist_reward:.2f} + Event: {event_reward:.2f} + Action: {action_punishment:.2f} + Ground: {ground_penalty:.2f}")

        return reward
        

    def _check_done(self, status):
        return status != "ongoing"
    
    def _check_status(self):
        if self.interceptor.pos[2] < 0:
            return "crashed"
        elif np.linalg.norm(self.interceptor.pos - self.target.pos) < 50:
            return "hit"
        elif self.sim_time > 50:
            return "expired"
        else:
            return "ongoing"

