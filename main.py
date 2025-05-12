from gym.environment import MissileEnv, MissileEnvSettings
import gym.visualisation.interactive as viz
from pilots.proportional_nav_pilot import PlanarProportionalNavPilot
from models.missile import PhysicalMissleModel
import models.physics as physics
import numpy as np
import time

if __name__ == "__main__":
    settings = MissileEnvSettings()

    visualizer = viz.RealtimeVisualizer()

    target = PhysicalMissleModel(velocity=np.array([0.0, physics.mach_to_ms(0.2), 0.0]), max_acc=400 * 9.81, pos=np.array([0.0, -30_000.0, 60_000.0]))
    interceptor = PhysicalMissleModel(velocity=np.array([0.0, 0.0, physics.mach_to_ms(5.0)]), max_acc=100 * 9.81, pos=np.array([0.0, 0.0, 100.0]))

    env = MissileEnv(settings=settings, target=target, interceptor=interceptor, visualizer=visualizer)
    agent = PlanarProportionalNavPilot(speed=interceptor.max_speed, n=5.0)

    done = False
    obs = env.reset()
    while not done:
        start_time = time.time()

        action = agent.step(obs, settings.time_step)  # Get the acceleration command from the agent
        obs, reward, done, _, _ = env.step(action)  # Take a step in the environment
        
        delta = time.time() - start_time
        if delta < settings.time_step:
            time.sleep(settings.time_step - delta)

    env.close()