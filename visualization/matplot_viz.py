from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import gym.episode as ep
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from visualization.abstract_viz import AbstractVisualizer

@dataclass
class GifSettings:
    fps: int = 10
    speed: float = 1.0

class MatplotVisualizer(AbstractVisualizer):
    def __init__(self, data: ep.Episode = None, gif_settings: GifSettings = None):
        self.data: ep.Episode = data
        self.gif_settings: GifSettings = gif_settings
        self.limits = None

        if self.data is not None:
            self._compute_limits()

    def reset(self):
        self.data = None
        self.limits = None

    def set_episode_data(self, data: ep.Episode):
        self.data = data
        self._compute_limits()

    def _compute_limits(self):
        if self.data is None:
            raise ValueError("No episode data set. Please set the episode data before computing limits.")

        all_states = [s.position for s in self.data.target_states.all.values()]
        for interceptor in self.data.interceptors.values():
            all_states.extend([s.position for s in interceptor.states.all.values()])

        all_states = np.array(all_states)
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

        self.limits = (x_limits, y_limits, z_limits)


    def _plot(self, time: float, ax):
        target_states = self.data.target_states.get_all_until(time)
        interceptor_states = {
            id: interceptor.states.get_all_until(time)
            for id, interceptor in self.data.interceptors.items()
        }

        # Plot target
        if target_states:
            target_positions = [s.position for _, s in target_states.items()]
            x, y, z = zip(*target_positions)
            ax.plot(x, y, z, c='r', label='Target')
            last = target_states[max(target_states.keys())]
            ax.scatter(*last.position, c='r', marker='o', s=100)

        # Plot interceptors
        for id, states in interceptor_states.items():
            if states:
                positions = [s.position for _, s in states.items()]
                x, y, z = zip(*positions)
                ax.plot(x, y, z, c='b')
                last = states[max(states.keys())]
                ax.scatter(*last.position, c='b', marker='o', s=100)

        if self.limits:
            x_limits, y_limits, z_limits = self.limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altitude')
        ax.set_title(f'Episode Visualization at Time: {time:.2f}s')

        # Fix duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    def render(self, time: float):
        if self.data is None:
            raise ValueError("No episode data set. Please set the episode data before rendering.")
    
        if self.gif_settings is not None:
            interval = (1 / self.gif_settings.fps) * self.gif_settings.speed
            self._render_gif(time, interval=interval)
        
        self._render_single_image(time)

    def render_live(self, current_time: float):
        if self.data is None:
            raise ValueError("No episode data set.")
        self._render_single_image(current_time)

    def close(self):
        plt.close()
        self.reset()

    def _render_gif(self, time: float, interval: float = 0.1):
        if self.data is None:
            raise ValueError("No episode data set. Please set the episode data before creating a GIF.")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        total_duration = max(self.data.target_states.all.keys()) if self.data.target_states.all else 0
        times = np.linspace(0, total_duration, int(total_duration / interval) + 1)

        print(f"Creating GIF with {len(times)} frames.")

        def update_plot(frame_idx):
            ax.clear()
            self._plot(times[frame_idx], ax)
            return ax

        ani = FuncAnimation(fig, update_plot, frames=len(times), blit=False, interval=0.1)
        gif_file_name = 'episode.gif'
        ani.save(gif_file_name, writer='imagemagick', fps=self.gif_settings.fps)
        print(f"GIF saved as {gif_file_name}")

    def _render_single_image(self, time: float):
        if self.data is None:
            raise ValueError("No episode data set. Please set the episode data before creating an image.")

        clear_output(wait=True)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._plot(time, ax)
        display(fig)
        plt.close(fig)
