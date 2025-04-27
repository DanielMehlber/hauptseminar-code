import numpy as np
from models.physics import euler_to_rotation_matrix

class MissileModel:
    def __init__(self, velocity=np.ndarray([0, 0, 100]), max_acc=50 * 9.81, pos=np.zeros(3)):
        """
        Initializes the missile's rigit body motion model in 3D space.

        Args:
            speed (float): The speed of the missile in m/s.
            max_acc (float): The maximum possible lateral acceleration of the missile in m/s^2.
            pos (np.ndarray): The initial position of the missile in 3D space.
        """
        self.init_pos = pos.copy()
        self.init_velocity = velocity.copy()
        self.speed = np.linalg.norm(velocity)

        self.velocity = velocity.copy()
        self.max_acc_magnitude = max_acc

        self.reset()

    def __clamp_lateral_acc(self, lat_acc):
        command_acc = lat_acc
        acc_magnitude = np.linalg.norm(lat_acc)
        if (acc_magnitude > self.max_acc_magnitude):
            command_acc = lat_acc * (self.max_acc_magnitude / acc_magnitude)
            print (f"MissileModel - Acceleration clamped: {command_acc} ({acc_magnitude / 9.81:.2f}g)")

        return command_acc
    
    def __lateral_to_world_acc(self, lat_acc: np.ndarray):
        world_acc = np.array([lat_acc[0], lat_acc[1], 0])
        rotation_matrix = euler_to_rotation_matrix(self.velocity / np.linalg.norm(self.velocity))
        world_acc = rotation_matrix @ world_acc

        print (f"MissileModel - World acceleration: {world_acc} ({np.linalg.norm(world_acc) / 9.81:.2f}g)")

        return world_acc
    
    def __apply_acceleration(self, world_acc: np.ndarray, dt: float):
        self.velocity += world_acc * dt
        self.velocity *= self.speed / np.linalg.norm(self.velocity)  # normalize to constant speed
        self.pos += self.velocity * dt

    def accelerate(self, lat_acc: np.ndarray, dt=0.1, t=0.0):
        """
        Tries to execute a guidance acceleration command. If the physical model
        cannot fully execute this command it is clamped to an executable one, respecting
        accelration and turn limits.

        Args:
            acc (np.ndarray): 2D lateral acceleration vector in the plane of the missile's velocity vector in m/s^2.
            dt (float): the delta time in which the command should be executed in seconds
            t (float): the total time of the simulation in seconds
        """
        print (f"MissileModel - Acceleration command: {lat_acc}")
        command_acc = self.__clamp_lateral_acc(lat_acc) # prevent exceeding max acceleration

        world_acc = self.__lateral_to_world_acc(command_acc) # from 2D lateral to 3D world coordinates
        self.__apply_acceleration(world_acc, dt)
        

    def reset(self):
        self.pos = self.init_pos.copy()
        self.velocity = self.init_velocity.copy()
        self.last_acc_vec = None
        self.steps = 0  