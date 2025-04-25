from gym.environment import MissileEnv, MissileEnvSettings
import numpy as np
import math

settings = MissileEnvSettings()
settings.time_speed = 1.0

env = MissileEnv(settings=settings)
obs, _ = env.reset()

while True:
    action = np.random.uniform(math.pi / 2, -math.pi / 2, size=3)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

print("Simulation finished.")