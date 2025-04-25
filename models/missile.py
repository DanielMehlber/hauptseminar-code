import numpy as np
from .physics import euler_to_rotation_matrix, get_acceleration_of

class MissileModel:
    def __init__(self, speed=100, max_acc=50 * 9.81, pos=np.zeros(3)):
        """
        Initializes the missile's rigit body motion model in 3D space.

        Args:
            speed (float): The speed of the missile in m/s.
            max_acc (float): The maximum possible lateral acceleration of the missile in m/s^2.
            pos (np.ndarray): The initial position of the missile in 3D space.
        """
        self.init_pos = pos.copy()
        self.init_vector = np.array([0, 0, speed])
        self.speed = speed
        self.max_acc_magnitude = max_acc

        self.reset()

    def execute_command(self, rotation_commend, dt=0.1, t=0.0):
        """
        Executes a guidance command in the physical model by applying a rotation command 
        to the entity's velocity vector, calculating the resulting acceleration, and updating 
        the entity's position and velocity accordingly.

        If the rotation command exceeeds the maximum acceleration, it is clamped.

        Args:
            rotation_commend (np.ndarray): A 3D vector representing the rotation command 
                to be applied to the entity's velocity.
            dt (float, optional): The time step for the simulation. Defaults to 0.1.
            t (float, optional): The current simulation time. Defaults to 0.0.
        """
        # create velocity vector after applying action (rotation)
        action_rotation_mat = euler_to_rotation_matrix(rotation_commend)
        new_velocity_vec = action_rotation_mat @ self.velocity

        # calculate necessary acceleration to achieve the action velocity from the current velocity
        self.last_acc_vec = get_acceleration_of(self.velocity, new_velocity_vec, dt)

        # clamp acceleration to max acceleration (to avoid unrealistic maneuvers)
        acceleration_magnitude = np.linalg.norm(self.last_acc_vec)
        if acceleration_magnitude > self.max_acc_magnitude:
            self.last_acc_vec = self.last_acc_vec / acceleration_magnitude * self.max_acc_magnitude
            print ("Clamping acceleration to max acceleration")

        # calculate new velocity vector
        self.velocity = self.velocity + self.last_acc_vec * dt
        self.velocity *= self.speed / np.linalg.norm(self.velocity)  # normalize to speed

        self.pos += self.velocity * dt

        print (f"Entity position: {self.pos}, velocity: {np.linalg.norm(self.velocity)}, theoretical acceleration: {acceleration_magnitude}")

        self.steps += 1

    def reset(self):
        self.pos = self.init_pos.copy()
        self.velocity = np.array([0, 0, self.speed])
        self.last_acc_vec = None
        self.steps = 0