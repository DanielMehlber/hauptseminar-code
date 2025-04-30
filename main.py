from gym.environment import MissileEnv, MissileEnvSettings
import numpy as np
from models.missile import MissileModel
from stable_baselines3 import SAC
import models.physics as physics

settings = MissileEnvSettings()
settings.time_step = 0.1
settings.realtime = False

target = MissileModel(velocity=np.array([0.0, physics.mach_to_ms(5), 0.0]), max_acc=50 * 9.81, pos=np.array([0.0, -30_000.0, 80_000.0]))
interceptor = MissileModel(velocity=np.array([0.0, 0.0, physics.mach_to_ms(8.0)]), max_acc=50 * 9.81, pos=np.array([0.0, 0.0, 100.0]))

env = MissileEnv(settings=settings, interceptor=interceptor, target=target)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./.logs/sac")
model.learn(total_timesteps=100_000)

# just run the agent after
# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

print("Simulation finished.")