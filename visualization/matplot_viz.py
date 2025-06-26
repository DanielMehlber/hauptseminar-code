from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import data.episode as ep
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from visualization.abstract_viz import AbstractVisualizer
import os


class MatplotVisualizer(AbstractVisualizer):
    def __init__(self, episode: ep.Episode = None):
        self.episodes: ep.Episode = episode
        self.limits = None

        self.must_recompute_limits = False # used to optimize limit computation of plot

        self.episodes = []

        if episode is not None:
            self.episodes.append(episode)
            self._compute_limits()

    def reset(self):
        self.episodes = []
        self._compute_limits()
        self.must_recompute_limits = True

    def set_episode_data(self, data: ep.Episode):
        self.episodes.clear()
        self.episodes.append(data)
        self.must_recompute_limits = True
        self._compute_limits()

    def set_episodes_data(self, data: list[ep.Episode]):
        self.episodes = data
        self.must_recompute_limits = True
        self._compute_limits()

    def add_episode_data(self, data: ep.Episode):
        self.episodes.append(data)
        self.must_recompute_limits = True
        self._compute_limits()

    def _compute_limits(self):
        x_limits = (0, 0)
        y_limits = (0, 0)
        z_limits = (0, 0)

        if not self.must_recompute_limits:
            return self.limits

        for episode in self.episodes:
            ep_x_limits, ep_y_limits, ep_z_limits = self._compute_episode_limits(episode)
            x_limits = (min(x_limits[0], ep_x_limits[0]), max(x_limits[1], ep_x_limits[1]))
            y_limits = (min(y_limits[0], ep_y_limits[0]), max(y_limits[1], ep_y_limits[1]))
            z_limits = (min(z_limits[0], ep_z_limits[0]), max(z_limits[1], ep_z_limits[1]))

        self.limits = (x_limits, y_limits, z_limits)

    def _compute_episode_limits(self, episode: ep.Episode):
        if episode is None:
            raise ValueError("No episode data set. Please set the episode data before computing limits.")

        all_states = [s.position for s in episode.target_states.all.values()]
        for interceptor in episode.interceptors.values():
            all_states.extend([s.position for s in interceptor.states.all.values()])

        if len(all_states) == 0:
            x_limits = (0, 1)
            y_limits = (0, 1)
            z_limits = (0, 1)

            return x_limits, y_limits, z_limits


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

        return x_limits, y_limits, z_limits

    def _plot(self, time: float, ax: plt.Axes):

        for episode in self.episodes:
            self._plot_episode(episode, time, ax)

        if self.limits:
            x_limits, y_limits, z_limits = self.limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'Episode Visualization at Time: {time:.2f}s')

        # Fix duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())


    def _plot_episode(self, episode: ep.Episode, time: float, ax: plt.Axes):
        target_states = episode.target_states.get_all_until(time)
        interceptor_states = {
            id: interceptor.states.get_all_until(time)
            for id, interceptor in episode.interceptors.items()
        }

        # Plot target
        if target_states:
            target_positions = [s.position for _, s in target_states.items()]
            x, y, z = zip(*target_positions)
            ax.plot(x, y, z, c='r', label='Target')
            last = target_states[max(target_states.keys())]
            ax.scatter(*last.position, c='r', marker='o', s=100)
            # Draw a dotted line from the point to the ground (z = 0) in the same color
            ax.plot([last.position[0], last.position[0]], 
                    [last.position[1], last.position[1]], 
                    [last.position[2], 0], 
                    c='r', linestyle='--')

        # Plot interceptors
        for id, states in interceptor_states.items():
            if states:
                final_state = episode.interceptors[id].states.get(time)
                # trajectory
                positions = [s.position for _, s in states.items()]
                current_distance = final_state.distance
                x, y, z = zip(*positions)
                ax.plot(x, y, z, c='b', label=f'{id}')
                last = states[max(states.keys())]
                ax.scatter(*last.position, c='b', marker='o', s=100)
                # Draw a dotted line from the point to the ground (z = 0) in the same color
                ax.plot([last.position[0], last.position[0]], 
                        [last.position[1], last.position[1]], 
                        [last.position[2], 0], 
                        c='b', linestyle='--')

                # predicted intercept point
                if final_state.predicted_intercept_point is not None:
                    pip = final_state.predicted_intercept_point
                    ax.scatter(*pip, c='gray', marker='X', s=50, label='Predicted Intercept Point')
                    ax.plot([pip[0], pip[0]], [pip[1], pip[1]], [pip[2], 0], c='gray', linestyle=':')


    def render(self, time: float):
        self._compute_limits()

        if len(self.episodes) < 1:
            raise ValueError("No episode data set. Please set the episode data before rendering.")
        
        self._render_single_image(time)

    def save_playback(self, filename, time, speed = 1, fps = 10):
        self._compute_limits()

        interval = (1 / fps) * speed
        self._render_gif(filename, time, interval=interval, fps=fps)

    def playback(self, time: float, speed: float = 1.0, fps: int = 10):
        pass

    def close(self):
        plt.close()
        self.reset()

    def _render_gif(self, filename: str, time: float, interval: float = 0.1, fps: int = 10):
        if len(self.episodes) < 1:
            raise ValueError("No episode data set. Please set the episode data before creating a GIF.")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        total_duration = time
        times = np.linspace(0, total_duration, int(total_duration / interval) + 1)

        print(f"Creating GIF with {len(times)} frames.")

        def update_plot(frame_idx):
            ax.clear()
            self._plot(times[frame_idx], ax)
            return ax

        ani = FuncAnimation(fig, update_plot, frames=len(times), blit=False, interval=0.1)
        gif_file_name = filename if filename.endswith('.gif') else filename + '.gif'
        os.makedirs(os.path.dirname(gif_file_name), exist_ok=True)
        ani.save(gif_file_name, writer='imagemagick', fps=fps)
        print(f"GIF saved as {gif_file_name}")

        fig.clear()
        plt.close(fig)

    def _render_single_image(self, time: float):
        if len(self.episodes) < 1:
            raise ValueError("No episode data set. Please set the episode data before creating an image.")

        clear_output(wait=True)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._plot(time, ax)
        plt.show(fig)
        # plt.close(fig)
