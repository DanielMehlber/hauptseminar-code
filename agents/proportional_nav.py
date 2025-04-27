from models.missile import MissileModel
import numpy as np
from agent import Agent


class ProportionalNavAgent(MissileModel, Agent):
    def __init__(self, speed=100, max_acc=50 * 9.81, pos=np.zeros(3)):
        MissileModel.__init__(speed=speed, max_acc=max_acc, pos=pos)
        Agent.__init__(self, name="ProportionalNavAgent")

    def step(self, dt: float, t: float):
        command = np.zeros(3)
        MissileModel.execute_rotation_command(self, command, dt=dt, t=t)

    def reset(self):
        MissileModel.reset(self)
        

        