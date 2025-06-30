from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import data.episode as ep
import time
from physics.missile import PhysicalMissleModel
from physics.noise import LinearDistanceNoise
from pilots.pilot import Pilot
from environment.observations import GroundBaseObservations, ImuObservations, InterceptorFrameObservations, InterceptorObservations, SeekerObservations
import math
from copy import deepcopy

@dataclass
class MissileEnvSettings:
    time_step: float = 0.1          # Speed multiplier for simulation (e.g., 1.0 = real-time)
    min_dt: float = 0.01            # Minimum delta time for simulation steps
    realtime: bool = False          # If True, the simulation will run in real-time
    time_limit: float = 60.0        # Time limit for the simulation in seconds
    hit_distance: float = 100       # Distance at which the interceptor is considered to have hit the target


class MissileEnv(gym.Env):
    """
    A custom environment for simulating missile interception scenarios. The environment variance 
    is controlled by the uncertainty setting, which influences the noise and uncertainty in the simulation
    environment.
    """

    def __init__(self, target: PhysicalMissleModel, 
                 interceptor: PhysicalMissleModel, 
                 target_pilot: Pilot = None,
                 uncertainty: float = 0.0, 
                 settings=MissileEnvSettings()):
        
        super().__init__()
        self.settings = settings
        self.last_step_time = None
        self.sim_time = 0.0  # Tracks total simulation time

        self.target = target
        self.interceptor = interceptor
        self.target_pilot = target_pilot

        # apply uncertainty to interceptor and target
        self.interceptor.reset()
        self.target.reset()
        self.interceptor_state = "midcourse"

        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # required as anchor point to normalize distance measurements
        self.missile_space_start_distance_vec = self.interceptor.body_to_world_rot_mat.T @ (self.target.world_pos - self.interceptor.world_pos)
        self.world_space_last_interceptor_velocity: np.ndarray = self.interceptor.get_velocity() # required to calculate the interceptor's acceleration

        # required to calculate observations (which as basically changes in position and velocity)
        self.interceptor_previous_frame_observations: InterceptorFrameObservations = None
        self.interceptor_current_frame_observations: InterceptorFrameObservations = None
        self.world_space_last_interceptor_position: np.ndarray = None # required to calculate interceptor acceleration

        self.current_episode: ep.Episode = None # current episode object

        # controls the uncertainty during the simulation
        self.set_uncertainty(uncertainty)

        self.current_agent_name = "Agent"

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.interceptor.reset()
        self.interceptor.world_pos[2] = 100.0 # ensure interceptor is above ground
        self.target.reset()
        self.interceptor_state = "midcourse"

        if self.target_pilot is not None:
            self.target_pilot.reset()

        self.sim_time = 0.0

        # data for display in visualizer
        self.last_step_time = time.time()
        self.last_acc_command = np.zeros(2, dtype=np.float32)
        self.last_missile_orientation_matrix = np.eye(3, dtype=np.float32)
        self.world_space_last_interceptor_velocity = self.interceptor.get_velocity()

        # reset observations of sensors
        self.interceptor_current_frame_observations = None
        self.interceptor_previous_frame_observations = None

        # init sensor state for differential measurements
        self._take_measurements(self.settings.min_dt)
        self.current_episode = ep.Episode()

        info = {}
        return self.get_interceptor_observations(norm=True).pack(), info

    def set_uncertainty(self, uncertainty: float):
        """
        Sets the uncertainty for the environment, which influences the noise in the simulation.
        """
        self._uncertainty = uncertainty
        self.interceptor.set_uncertainty(uncertainty)
        self.target.set_uncertainty(uncertainty)
        self.target_pilot.set_uncertainty(uncertainty) if self.target_pilot is not None else None

    def _line_of_sight_angle(self, world_target_pos, world_interceptor_pos):
        # Calculate the angle between the interceptor and target positions
        world_space_los_vector = world_target_pos - world_interceptor_pos
        
        # transform to interceptor space: we want to calculate the LOS angle 
        # from the interceptor's perspective
        missile_reference = self.interceptor.body_to_world_rot_mat.T
        missile_space_los_vector = missile_reference @ world_space_los_vector
        missile_space_los_vector /= np.linalg.norm(missile_space_los_vector)

        x, y, z = missile_space_los_vector
        norm_xy = np.sqrt(x**2 + y**2)

        # azimuth (horizontal) angle in body-frame: −π … +π
        h_angle = np.arctan2(y, x)                                  # left/right of nose

        # elevation (vertical) angle: −π/2 … +π/2
        v_angle = np.arctan2(z, norm_xy)                            # above/below nose

        
        # Note: a single argtan(x) and arctan(y) would not consider the sign of the
        # x and y components, which is important for determining the correct quadrant. 
        return np.array([h_angle, v_angle], dtype=np.float32)
    
    def _take_measurements(self, dt):
        self.interceptor_previous_frame_observations = self.interceptor_current_frame_observations
        self.interceptor_current_frame_observations = self._get_current_frame_interceptor_obs(dt)

    def _update_episode_data(self):
        # Update the episode data with the current state of the interceptor and target
        interceptor_state = ep.InterceptorState(
            position=self.interceptor.world_pos.copy(),
            velocity=self.interceptor.get_velocity().copy(),
            command=self.last_acc_command.copy(),
            distance=np.linalg.norm(self.interceptor.world_pos - self.target.world_pos), # distance to target in missile space
            predicted_intercept_point=None # can be set with other means
        )

        target_state = ep.TargetState(
            position=self.target.world_pos.copy(),
            velocity=self.target.get_velocity().copy()
        )

        # Add the states to the episode
        self.current_episode.target_states.add(self.sim_time, target_state)
        self.current_episode.get_interceptor(self.current_agent_name).states.add(self.sim_time, interceptor_state)

    def set_current_agent_name(self, name: str):
        """
        Sets the name of the current agent for the episode.
        This is used for visualization and logging purposes.
        """
        self.current_agent_name = name

    def step(self, action: np.ndarray, norm_observations=True):
        dt = 0.0
        if self.settings.realtime:
            step_time = time.time()
            real_dt = step_time - self.last_step_time

            # if dt gets to small, equations explode
            wait_time = self.settings.min_dt - real_dt
            if wait_time > 0:
                time.sleep(wait_time)

            real_dt = time.time() - self.last_step_time

            self.last_step_time = step_time

            # Apply simulation time scaling
            dt = real_dt
            self.sim_time += dt
        else:
            # We use a fixed time step
            dt = self.settings.time_step
            self.sim_time += dt

        # build observations by taking measurements
        self._take_measurements(dt)

        # get target action from pilot (if available)
        target_action = np.zeros(2, dtype=np.float32)
        if self.target_pilot is not None:
            target_action = self.target_pilot.step(dt, self.sim_time, self._uncertainty)

        # for calculating the interceptor's acceleration in the observations
        self.world_space_last_interceptor_velocity = self.interceptor.get_velocity()

        # Update entities with scaled delta time
        self.target.accelerate(target_action, dt=dt, t=self.sim_time)
        self.interceptor.accelerate(action, dt=dt, t=self.sim_time)

        # Update values for visualization
        self.last_acc_command = action.copy()
        self.last_missile_orientation_matrix = self.interceptor.body_to_world_rot_mat.copy()

        # depending on rl agent or pilot, we can either normalize the observation space or not
        obs = self.get_interceptor_observations(norm=norm_observations)
        status = self._check_status()
        done = self._check_done(status)
        reward, info = self._get_reward(self.get_interceptor_observations(norm=False), action, status, dt)
        
        # Update after reward calculation because it needs the last sensor data
        self._update_episode_data()

        return obs.pack(), reward, done, False, info

    def render(self):
        pass

    def get_ground_base_observations(self) -> GroundBaseObservations:
        """
        Returns the observations for the ground base, which includes the interceptor and target positions.
        """
        # get radar measurements and add noise
        radar_to_interceptor_distance = np.linalg.norm(self.interceptor.world_pos)
        world_space_noisy_interceptor_pos = LinearDistanceNoise().apply(self.interceptor.world_pos, radar_to_interceptor_distance, intensity=self._uncertainty)

        radar_to_target_distance = np.linalg.norm(self.target.world_pos)
        world_space_noisy_target_pos = LinearDistanceNoise().apply(self.target.world_pos, radar_to_target_distance, intensity=self._uncertainty)

        observations = GroundBaseObservations()
        observations.world_space_interceptor_pos = world_space_noisy_interceptor_pos
        observations.world_space_target_pos = world_space_noisy_target_pos

        return observations

    def _get_current_frame_interceptor_obs(self, dt) -> InterceptorFrameObservations:
        """
        Returns the unnormalized observations for the interceptor. This includes data from
        multiple sensors, like seekers and gyroscopes.
        """
        # simulates distance based noise on the radar measurements
        seeker_to_target_distance = np.linalg.norm(self.target.world_pos - self.interceptor.world_pos)
        world_space_noisy_target_pos = LinearDistanceNoise().apply(self.target.world_pos,
                                                                    seeker_to_target_distance,
                                                                    intensity=self._uncertainty)


        # line-of-sight from interceptor to target (from seeker's point of view) - length is distance to target
        missile_space_los_vec = self.interceptor.body_to_world_rot_mat.T @ (world_space_noisy_target_pos - self.interceptor.world_pos) # target position in missile space
        seeker_distance_to_target = np.linalg.norm(missile_space_los_vec)
        seeker_los_unit_vec = missile_space_los_vec / seeker_distance_to_target if seeker_distance_to_target > 0 else np.zeros(3, dtype=np.float32)

        # closing rate to the target in missile space
        previous_frame_obs = self.interceptor_previous_frame_observations

        # in case the previous frame observations are not available, i.e. at the start of the simulation
        missile_space_previous_distance = seeker_distance_to_target
        missile_space_previous_los_unit_vec = seeker_los_unit_vec

        if previous_frame_obs:
            missile_space_previous_distance = previous_frame_obs.seeker.distance_to_target
            missile_space_previous_los_unit_vec = previous_frame_obs.seeker.los_unit_vec

        missile_space_previous_distance_vec = np.abs(missile_space_previous_los_unit_vec) * missile_space_previous_distance
        missile_space_current_distance_vec = np.abs(missile_space_los_vec) # same here
        seeker_closing_rate_vec = (missile_space_previous_distance_vec - missile_space_current_distance_vec) / dt # positive if closing in on target

        # seeker line-of-sight angle rate
        seeker_los_angles_vec = self._line_of_sight_angle(world_space_noisy_target_pos, self.interceptor.world_pos)
        seeker_los_angles_rates_vec = np.zeros(2, dtype=np.float32) # default value if no previous frame is available
        if previous_frame_obs is not None:
            seeker_los_angles_rates_vec = (seeker_los_angles_vec - previous_frame_obs.seeker.los_angles_vec) / dt

        seeker_obs = SeekerObservations()
        seeker_obs.los_unit_vec = seeker_los_unit_vec
        seeker_obs.distance_to_target = seeker_distance_to_target
        seeker_obs.closing_rate_vec = seeker_closing_rate_vec
        seeker_obs.los_angles_vec = seeker_los_angles_vec
        seeker_obs.los_angle_rates_vec = seeker_los_angles_rates_vec

        # IMU observations
        world_space_interceptor_velocity_vec = self.interceptor.get_velocity()
        imu_world_space_interceptor_orientation = world_space_interceptor_velocity_vec / np.linalg.norm(world_space_interceptor_velocity_vec)
        
        # interceptor turn angles in missile space (e.g. by gyroscopes)
        imu_missile_space_turn_rate = np.zeros(3, dtype=np.float32) # default value if no previous frame is available
        if previous_frame_obs is not None:
            missile_space_last_interceptor_orientation_vec = self.interceptor.body_to_world_rot_mat.T @ previous_frame_obs.imu.world_space_interceptor_orientation
            missile_space_yaw_angle, missile_space_pitch_angle = self.interceptor.calculate_local_angles_to(missile_space_last_interceptor_orientation_vec)
            missile_space_pitch_angle *= -1.0 # invert pitch angle to match the interceptor's coordinate system
            missile_space_yaw_angle *= -1.0 # invert yaw angle to match the interceptor's coordinate system
            missile_space_pitch_angle_rate = missile_space_pitch_angle / dt
            missile_space_yaw_angle_rate = missile_space_yaw_angle / dt
            imu_missile_space_turn_rate = np.array([missile_space_yaw_angle_rate, missile_space_pitch_angle_rate, 0.0])

        # acceleration measured by the inertial measurement unit (IMU) in missile space
        world_space_interceptor_acceleration = (self.interceptor.get_velocity() - self.world_space_last_interceptor_velocity) / dt
        imu_missile_space_acceleration = self.interceptor.body_to_world_rot_mat.T @ world_space_interceptor_acceleration
        
        imu_obs = ImuObservations()
        imu_obs.world_space_interceptor_orientation = imu_world_space_interceptor_orientation
        imu_obs.missile_space_turn_rate = imu_missile_space_turn_rate
        imu_obs.missile_space_acceleration = imu_missile_space_acceleration

        # create the observation object
        current_frame_obs = InterceptorFrameObservations()
        current_frame_obs.seeker = seeker_obs
        current_frame_obs.imu = imu_obs

        return current_frame_obs

    def get_interceptor_observations(self, norm=True) -> InterceptorObservations:

        assert self.interceptor_current_frame_observations is not None, \
            "Interceptor observations are not available. Make sure to call step() or reset() before accessing observations."

        if self.interceptor_previous_frame_observations is None:
            # if no previous frame observations are available, we use the current frame observations
            # as both current and previous frame observations
            self.interceptor_previous_frame_observations = deepcopy(self.interceptor_current_frame_observations)

        # attach previous frame observations to capture deltas
        obs = InterceptorObservations()
        obs.current_frame = deepcopy(self.interceptor_current_frame_observations)
        obs.previous_frame = deepcopy(self.interceptor_previous_frame_observations)

        if norm:
            # normalize the observations to a range suitable for RL agents
            obs.current_frame = self._normalize_interceptor_frame_observations(obs.current_frame)
            obs.previous_frame = self._normalize_interceptor_frame_observations(obs.previous_frame)

        # pack all observations into a single vector
        return obs
    
    def set_current_predicted_intercept_point(self, interceptor: str, point: np.ndarray):
        """
        Sets predicted intercept point for a specific interceptor at the current time of simulation.
        """
        # get states over time of interceptor
        series = self.current_episode.get_interceptor(interceptor).states
        state = series.get(self.sim_time)

        # update of insert state with intercept point
        state.predicted_intercept_point = point.copy()
        series.add(self.sim_time, state)

    
    def _normalize_interceptor_frame_observations(self, frame: InterceptorFrameObservations) -> InterceptorFrameObservations:
        # anchor points for normalization
        start_distance = np.linalg.norm(self.missile_space_start_distance_vec)
        max_speed = self.interceptor.max_speed + self.target.max_speed
        max_acc = self.interceptor.max_lat_acc

        # apply logarithmic scaling to the distance vector to avoid too small values near the target
        frame.seeker.distance_to_target = np.log(np.abs(frame.seeker.distance_to_target) + 1.0) / np.log(start_distance + 1)

        # linear normalizations
        frame.seeker.closing_rate_vec /= max_speed                   # relative to max speed
        frame.seeker.los_angles_vec /= np.pi                         # converted to interval [-1, 1]
        frame.seeker.los_angle_rates_vec /= np.pi                    # converted to interval [-1, 1]
        frame.imu.missile_space_turn_rate /= np.pi                   # converted to interval [-1, 1]
        frame.imu.missile_space_acceleration /= max_acc              # relative acceleration limit of airframe

        return frame


    def _get_terminal_phase_reward(self, obs: InterceptorObservations, 
                                   action: np.ndarray, 
                                   status: str, 
                                   relative_terminal_distance: float, dt):
        """
        The terminal phase purely focuses on hitting the target, no matter what. Energy
        efficiency is not a goal anymore.
        """

        # We include the distance reward of the midcourse phase, to avoid a drop when 
        # switching to terminal phase, which could discourage the agent.
        start_distance = np.linalg.norm(self.missile_space_start_distance_vec)
        dist = obs.current_frame.seeker.distance_to_target
        base_dist_penalty = -dist / start_distance # relative to initial distance

        # additional reward for decreasing the terminal distance on top
        relative_terminal_distance = dist / relative_terminal_distance # relative to terminal distance [0, 1]
        terminal_distance_reward = math.pow(3, (1.0 - relative_terminal_distance)) - 1.0 # exponential reward for distance [0, 2]

        # overlap the two rewards to avoid too low values
        distance_reward = base_dist_penalty + terminal_distance_reward

        # We also want to exponentially reward high closing rates
        distance_before = obs.previous_frame.seeker.distance_to_target
        closing_rate = (distance_before - dist) / dt
        closing_reward = closing_rate / self.interceptor.max_speed # relative to max speed

        # add an additional non-linear quadratic reward for high closing rates
        closing_reward += math.pow(3, closing_reward) - 1
        
        # A slight action punishment should be employed to avoid unnecessary maneuvers
        action_punishment = -np.linalg.norm(action) # [0, 1]

        # We want to reward/punish the interceptor for certain events
        event_reward = 0.0
        if status == "hit":
            event_reward = +1
        elif status == "crashed":
            event_reward = -1
        elif status == "expired":
            event_reward = -1

        distance_reward *= 0.5
        closing_reward *= 1.0 # because it is non-linear
        action_punishment *= 1.0
        event_reward *= 10.0

        info = {}
        info["dist-reward"] = distance_reward
        info["closing-rate-reward"] = closing_reward
        info["event-reward"] = event_reward
        info["action-punishment"] = action_punishment
        info["ground-penality"] = 0.0

        terminal_reward = distance_reward + closing_reward + action_punishment + event_reward
        info["reward"] = terminal_reward

        return terminal_reward, info

    def _get_midcourse_reward(self, obs: InterceptorObservations, 
                              action: np.ndarray, 
                              status: str, dt: float):
        """
        The midcourse phase focuses on brining the interceptor into a close vicinity to the 
        target in an energy-efficient manner. Its energy must be saved for more aggressive 
        maneuvers in the terminal phase.
        """
        # The less the distance, the higher the reward
        start_distance = np.linalg.norm(self.missile_space_start_distance_vec)
        current_distance = obs.current_frame.seeker.distance_to_target
        dist_penalty = -current_distance / start_distance # relative to initial distance

        # We want to reward the interceptor for closing in on the target
        previous_distance = obs.previous_frame.seeker.distance_to_target
        closing_rate_reward = (previous_distance - current_distance) / dt # positive if closing in on target
        closing_rate_reward /= np.linalg.norm(self.interceptor.get_velocity()) # relative to max speed

        # We want to reward/punish the interceptor for certain events
        event_reward = 0.0
        if status == "hit":
            event_reward = +1
        elif status == "crashed":
            event_reward = -1
        elif status == "expired":
            event_reward = -1
            
        
        # We want to keep the interceptor energy efficient (less commands = better)
        # small acceleration commands are better than large ones
        action_punishment = -np.linalg.norm(action)

        # We want the interceptor to avoid the ground (z < 0)
        interceptor_altidude = self.interceptor.world_pos[2]
        safe_altitude = 1000.0
        ground_penalty = -(1 - (interceptor_altidude / safe_altitude)) if interceptor_altidude <= safe_altitude else 0.0

        # Weighting the rewards
        dist_penalty *= 0.5
        closing_rate_reward *= 1.0 # 3.0
        event_reward *= 4.0
        action_punishment *= 2.0
        ground_penalty *= 4.0

        info = {}
        info["dist-reward"] = dist_penalty
        info["closing-rate-reward"] = closing_rate_reward
        info["event-reward"] = event_reward
        info["action-punishment"] = action_punishment
        info["ground-penality"] = ground_penalty

        # Combine all rewards
        reward = dist_penalty + closing_rate_reward + event_reward + action_punishment + ground_penalty
        info["reward"] = reward

        return reward, info

    def _get_reward(self, obs: InterceptorObservations, 
                    action: np.ndarray, 
                    status: str, dt: float) -> tuple[float, dict]:
        """
        We need different rewards for the interceptor's phases: 
        1) The midcourse phase focues on bringing it in a close vicinity to the target in an e
        nergy efficient manner. We need this energy in the terminal phase.
        2) The terminal phase tries to hit the target, no matter the energy consumption.

        Furthermore, this split is also necessary because the start distance is very large (e.g. >10km)
        and the requried hit distance very small (e.g. <50m or even hit-to-kill).
        """

        # In closer vicinity to the target, we start the terminal phase. This value is proportional
        # to the velocity of the interceptor. As estimate, we take the distance the interceptor
        # travels over a short durtation of time (e.g. 2 seconds).
        terminal_distance = self.interceptor.max_speed * 3 # distance = speed * time (1s)
        if obs.current_frame.seeker.distance_to_target > terminal_distance:
            if self.interceptor_state != "midcourse":
                self.interceptor_state = "midcourse"
            return self._get_midcourse_reward(obs, action, status, dt)
        else:
            if self.interceptor_state != "terminal":
                self.interceptor_state = "terminal"
            return self._get_terminal_phase_reward(obs, action, status, terminal_distance, dt)

    def _check_done(self, status):
        return status != "ongoing"
    
    def _check_status(self):
        if self.interceptor.world_pos[2] < 0:
            print ("Interceptor crashed into the ground")
            return "crashed"
        elif np.linalg.norm(self.interceptor.world_pos - self.target.world_pos) < self.settings.hit_distance:
            print ("Interceptor hit the target")
            return "hit"
        elif self.sim_time > self.settings.time_limit:
            return "expired"
        else:
            return "ongoing"
    

