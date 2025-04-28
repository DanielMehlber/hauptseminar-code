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

        self.vel = velocity.copy()
        self.max_acc_magnitude = max_acc

        self.reset()

    def __clamp_lateral_acc(self, lat_acc):
        command_acc = lat_acc
        clamped = False
        acc_magnitude = np.linalg.norm(lat_acc)
        if (acc_magnitude > self.max_acc_magnitude):
            command_acc = lat_acc * (self.max_acc_magnitude / acc_magnitude)
            print (f"MissileModel - Acceleration clamped: {command_acc} ({acc_magnitude / 9.81:.2f}g)")
            clamped = True

        return command_acc, clamped
    
    def get_orientation_matrix(self):
        # basically the gram schmidt process to build an orthonormal basis
        z_up = np.array([0, 0, 1])
        horizontal_axis = np.array([1, 0, 0])

        # longitude axis is the missile velocity vector
        longitude_axis = self.vel / np.linalg.norm(self.vel)
        latitude_axis_up = np.zeros(3)
        latitude_axis_right = np.zeros(3)

        # we must compute the axis which is not parallel to the missile velocity vector first
        # otherwise if, for example, z_up || longitude_axis, projection will not work.
        if not np.allclose(longitude_axis, z_up):
            # missile up vector is on the plane of the velocity vector and the z_up vector
            # we calculate it using projection (like in the Gramm-Schmidt process)
            latitude_axis_up = z_up - (np.dot(longitude_axis, z_up) * longitude_axis)
            latitude_axis_up /= np.linalg.norm(latitude_axis_up)

            # missile right vector is the cross product of the longitude and latitude axis
            latitude_axis_right = np.cross(longitude_axis, latitude_axis_up)
            latitude_axis_right /= np.linalg.norm(latitude_axis_right)

        else:
            # missile up vector is the cross product of the velocity vector and the horizontal axis
            latitude_axis_up = np.cross(longitude_axis, horizontal_axis)
            latitude_axis_up /= np.linalg.norm(latitude_axis_up)

            # missile right vector is the cross product of the longitude and up axis
            latitude_axis_right = np.cross(longitude_axis, latitude_axis_up)
            latitude_axis_right /= np.linalg.norm(latitude_axis_right)

        matrix = np.array([
            latitude_axis_right,
            latitude_axis_up,
            longitude_axis,
        ]).T

        return matrix
    
    def __lateral_to_world_acc(self, lat_acc: np.ndarray):
        world_acc = np.array([lat_acc[0], lat_acc[1], 0])
        rotation_matrix = self.get_orientation_matrix()
        world_acc = rotation_matrix @ world_acc

        return world_acc
    
    def __apply_acceleration(self, world_acc: np.ndarray, dt: float):
        self.vel += world_acc * dt
        self.vel *= self.speed / np.linalg.norm(self.vel)  # normalize to constant speed
        self.pos += self.vel * dt

    def accelerate(self, lat_acc: np.ndarray, dt=0.1, t=0.0):
        """
        Tries to execute a guidance acceleration command. If the physical model
        cannot fully execute this command it is clamped to an executable one, respecting
        accelration and turn limits.

        Args:
            acc (np.ndarray): 2D lateral acceleration vector in the plane of the missile's velocity vector in m/s^2.
            dt (float): the delta time in which the command should be executed in seconds
            t (float): the total time of the simulation in seconds

        Returns:
            bool: True if the command was executed successfully, False otherwise.
        """
        command_acc, oversteered = self.__clamp_lateral_acc(lat_acc) # prevent exceeding max acceleration (ingnored for now)
        world_acc = self.__lateral_to_world_acc(command_acc) # from 2D lateral to 3D world coordinates
        self.__apply_acceleration(world_acc, dt)
        
        return not oversteered

    def reset(self):
        self.pos = self.init_pos.copy()
        self.vel = self.init_velocity.copy()
        self.last_acc_vec = None
        self.steps = 0  