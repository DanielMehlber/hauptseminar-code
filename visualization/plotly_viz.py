import numpy as np
import plotly.graph_objects as go
from gym.episode import Episode
from visualization.abstract_viz import AbstractVisualizer
from IPython.display import display

import plotly.io as pio
pio.templates.default = "plotly_white"

import time

class PlotlyVisualizer(AbstractVisualizer):
    """
    A 3D trajectory visualizer using Plotly for Jupyter notebooks or any interactive environment.
    """
    def __init__(self, episode: Episode = None):
        # Use FigureWidget for interactive updating
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'  # Ensures equal aspect ratio between axes
            ),
            title='Interceptor vs Target Trajectories',
        )

        self.interceptor_traces = {}  # {id: (line_trace, point_trace)}
        self.target_trace = None
        self.target_point = None
        self.episode = episode

        self.displaying = False  # Flag to control display in Jupyter

    def set_episode_data(self, data: Episode):
        """
        Store the episode data. We'll use TimeSeries.get_all_until(time) during rendering.
        """
        self.episode = data

    def _get_episode_limits(self, sim_time):
        """
        Compute the limits of the episode data for the given simulation time.
        """
        all_states = [s.position for s in self.episode.target_states.get_all_until(sim_time).values()]
        for interceptor in self.episode.interceptors.values():
            all_states.extend([s.position for s in interceptor.states.get_all_until(sim_time).values()])

        all_states = np.array(all_states)
        
        if all_states.size == 0:
            return (0, 0), (0, 0), (0, 0)  # Default limits if no data is available

        # get min and max for each axis
        min_x, max_x = np.min(all_states[:, 0]), np.max(all_states[:, 0])
        min_y, max_y = np.min(all_states[:, 1]), np.max(all_states[:, 1])
        min_z, max_z = np.min(all_states[:, 2]), np.max(all_states[:, 2])

        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        max_range = max(x_range, y_range, z_range)

        mid_x = (max_x + min_x) / 2
        mid_y = (max_y + min_y) / 2

        x_limits = (mid_x - max_range / 2, mid_x + max_range / 2)
        y_limits = (mid_y - max_range / 2, mid_y + max_range / 2)
        z_limits = (0, max_range)

        # add 10% padding to the limits
        x_limits = (x_limits[0] - 0.1 * max_range, x_limits[1] + 0.1 * max_range)
        y_limits = (y_limits[0] - 0.1 * max_range, y_limits[1] + 0.1 * max_range)
        z_limits = (z_limits[0] - 0.1 * max_range, z_limits[1] + 0.1 * max_range)

        return x_limits, y_limits, z_limits


    def _plot_target(self, sim_time):
        targ_dict = self.episode.target_states.get_all_until(sim_time)
        sorted_t = sorted(targ_dict.keys())
        positions_t = [targ_dict[t].position for t in sorted_t]
        if positions_t:
            xs_t, ys_t, zs_t = zip(*positions_t)
        else:
            xs_t, ys_t, zs_t = [], [], []

        # First-time creation
        if self.target_trace is None:
            self.target_trace = self.fig.add_trace(go.Scatter3d(
                x=xs_t, y=ys_t, z=zs_t,
                mode='lines',
                line=dict(color='red', width=2),
                name='Target'
            )).data[-1]

            self.target_point = self.fig.add_trace(go.Scatter3d(
                x=[xs_t[-1]] if xs_t else [],
                y=[ys_t[-1]] if ys_t else [],
                z=[zs_t[-1]] if zs_t else [],
                mode='markers',
                marker=dict(color='red', size=6),
                name='Target Current Position'
            )).data[-1]
        else:
            self.target_trace.x = xs_t
            self.target_trace.y = ys_t
            self.target_trace.z = zs_t

            if xs_t:
                self.target_point.x = [xs_t[-1]]
                self.target_point.y = [ys_t[-1]]
                self.target_point.z = [zs_t[-1]]
            else:
                self.target_point.x = []
                self.target_point.y = []
                self.target_point.z = []

    def _plot_interceptors(self, sim_time):
        for iid, interceptor in self.episode.interceptors.items():
            int_dict = interceptor.states.get_all_until(sim_time)
            sorted_it = sorted(int_dict.keys())
            positions_i = [int_dict[t].position for t in sorted_it]
            if positions_i:
                xs_i, ys_i, zs_i = zip(*positions_i)
            else:
                xs_i, ys_i, zs_i = [], [], []

            if iid not in self.interceptor_traces:
                trace = self.fig.add_trace(go.Scatter3d(
                    x=xs_i, y=ys_i, z=zs_i,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    name=f'Interceptor {iid}'
                )).data[-1]

                point_trace = self.fig.add_trace(go.Scatter3d(
                    x=[xs_i[-1]] if xs_i else [],
                    y=[ys_i[-1]] if ys_i else [],
                    z=[zs_i[-1]] if zs_i else [],
                    mode='markers',
                    marker=dict(color='blue', size=6),
                    name=f'Interceptor {iid} Current Position'
                )).data[-1]

                self.interceptor_traces[iid] = (trace, point_trace)
            else:
                trace, point_trace = self.interceptor_traces[iid]
                trace.x = xs_i
                trace.y = ys_i
                trace.z = zs_i

                if xs_i:
                    point_trace.x = [xs_i[-1]]
                    point_trace.y = [ys_i[-1]]
                    point_trace.z = [zs_i[-1]]
                else:
                    point_trace.x = []
                    point_trace.y = []
                    point_trace.z = []

    def _plot(self, sim_time):
        """
        Plot the target and interceptors at the given simulation time.
        """
        self._plot_target(sim_time)
        self._plot_interceptors(sim_time)

    def render(self, sim_time: float):
        """
        Render all trajectories up to the given time. Updates traces in the Plotly FigureWidget.
        """
        if self.episode is None:
            raise RuntimeError("No episode data set. Call set_episode_data first.")

        x_limits, y_limits, z_limits = self._get_episode_limits(sim_time)
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=x_limits),
                yaxis=dict(range=y_limits),
                zaxis=dict(range=z_limits)
            ),
            title=f'Episode Visualization at Time: {sim_time:.2f}s'
        )

        self._plot(sim_time)
        
        # display the figure
        if not self.displaying:
            display(self.fig)
            self.displaying = True

    def playback(self, sim_time: float, speed: float = 1.0, fps: int = 1):
        """
        Play back the episode data at a given speed.
        
        Args:
            sim_time (float): The time to play back to.
            speed (float): The playback speed.
        """
        if self.episode is None:
            raise RuntimeError("No episode data set. Call set_episode_data first.")
        
        x_limits, y_limits, z_limits = self._get_episode_limits(sim_time)
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=x_limits),
                yaxis=dict(range=y_limits),
                zaxis=dict(range=z_limits)
            ),
            title=f'Episode Visualization at Time: {sim_time:.2f}s'
        )

        frames = sim_time * fps / speed
        times = np.linspace(0, sim_time, int(frames))

        last_time = time.time()
        interval = 1 / fps
        for t in times:
            now = time.time()
            elapsed = now - last_time
            
            if elapsed < interval:
                time.sleep(interval - elapsed)
            
            last_time = time.time()  # Update last_time after sleeping

            self._plot(t)

            if not self.displaying:
                display(self.fig)
                self.displaying = True

    def save_playback(self, filename: str):
        pass
        
    def reset(self):
        """
        Remove all traces and reset.
        """
        self.fig.data = []
        self.interceptor_traces = {}
        self.target_trace = None
        self.target_point = None

    def close(self):
        """
        Dispose of the figure.
        """
        self.fig = None
