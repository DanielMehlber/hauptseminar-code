from gym.environment import MissileEnv, MissileEnvSettings
import numpy as np
from models.missile import MissileModel
from stable_baselines3 import SAC

settings = MissileEnvSettings()
settings.time_step = 0.5
settings.realtime = False

target = MissileModel(velocity=np.array([0.0, 50.0, 0.0]), max_acc=50 * 9.81, pos=np.array([0.0, -1000.0, 5000.0]))
interceptor = MissileModel(velocity=np.array([0.0, 0.0, 200.0]), max_acc=50 * 9.81, pos=np.array([0.0, 0.0, 300.0]))

env = MissileEnv(settings=settings, interceptor=interceptor, target=target)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./.logs/sac")
model.learn(total_timesteps=10000)

print("Simulation finished.")