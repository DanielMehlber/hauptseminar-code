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
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

        self.visualizer = MissileVisualizer()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.interceptor.reset()
        self.target.reset()
        self.trajectory = []
        self.sim_time = 0.0
        self.last_step_time = time.time()
        self.visualizer.reset()

        return self._get_obs(), {}

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
        self.interceptor.accelerate(np.array([5.0, 0.0]), dt=dt, t=self.sim_time)

        self.trajectory.append((self.interceptor.pos.copy(), self.target.pos.copy()))

        obs = self._get_obs()
        done = self._check_done()
        reward = self._get_reward(done)

        distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        print(f"Reward: {reward} (Distance: {distance}), Done: {done}, Time: {self.sim_time:.2f}s")

        self.render()

        return obs, reward, done, False, {}

    def render(self):
        if len(self.trajectory) > 3:
            self.visualizer.update(self.interceptor.pos, self.target.pos, sim_time=self.sim_time)

    def _get_obs(self):
        return np.concatenate([
            self.interceptor.pos, self.interceptor.velocity,
            self.target.pos, self.target.velocity
        ])

    def _get_reward(self, done):
        dist = np.linalg.norm(self.interceptor.pos - self.target.pos)

        if self.interceptor.pos[2] < 0:
            return -1000

        return 1000 if dist < 50 else -(dist ** 2) / 1000

    def _check_done(self):
        distance = np.linalg.norm(self.interceptor.pos - self.target.pos)
        return distance < 20 or self.sim_time > 50

