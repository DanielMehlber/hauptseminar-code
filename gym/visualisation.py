import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class MissileVisualizer(threading.Thread):
    def __init__(self, xlim=(-1000, 1000), ylim=(-1000, 1000), zlim=(-1000, 1000), fit_view=True, update_interval=0.05):
        super().__init__()
        self.daemon = True
        self.update_interval = update_interval  # seconds

        self.interceptor_path = []
        self.target_path = []
        self.sim_time = None
        self._data_changed = False
        self._lock = threading.Lock()
        self._running = True

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.fit_view = fit_view

        self.start()

    def __del__(self):
        self.stop()

    def run(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_plot()

        while self._running:
            time.sleep(self.update_interval)

            if not self._data_changed:
                continue

            with self._lock:
                ip = np.array(self.interceptor_path)
                tp = np.array(self.target_path)
                sim_time = self.sim_time
                self._data_changed = False

            if len(ip) > 0:
                self.interceptor_line.set_data(ip[:, 0], ip[:, 1])
                self.interceptor_line.set_3d_properties(ip[:, 2])
                self.interceptor_dot.set_data([ip[-1, 0]], [ip[-1, 1]])
                self.interceptor_dot.set_3d_properties([ip[-1, 2]])

            if len(tp) > 0:
                self.target_line.set_data(tp[:, 0], tp[:, 1])
                self.target_line.set_3d_properties(tp[:, 2])
                self.target_dot.set_data([tp[-1, 0]], [tp[-1, 1]])
                self.target_dot.set_3d_properties([tp[-1, 2]])

            if sim_time is not None:
                self.ax.set_title(f"Simulation Time: {sim_time:.2f}s")

            if self.fit_view:
                # Dynamically adjust axis limits based on combined paths while keeping aspect ratio
                all_points = np.vstack((ip, tp)) if len(ip) > 0 and len(tp) > 0 else (ip if len(ip) > 0 else tp)
                if len(all_points) > 0:
                    margin = 50  # Add some margin so the lines aren't touching the plot edges
                    min_vals = np.min(all_points, axis=0) - margin
                    max_vals = np.max(all_points, axis=0) + margin

                    # Calculate the range for each axis
                    ranges = max_vals - min_vals
                    max_range = max(ranges) / 2.0

                    # Find the midpoints for each axis
                    mid_x = (max_vals[0] + min_vals[0]) / 2.0
                    mid_y = (max_vals[1] + min_vals[1]) / 2.0
                    mid_z = (max_vals[2] + min_vals[2]) / 2.0

                    # Set the limits to keep the aspect ratio
                    self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
                

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        plt.close(self.fig)

    def _init_plot(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)

        self.interceptor_line, = self.ax.plot([], [], [], 'b-', label="Interceptor")
        self.target_line, = self.ax.plot([], [], [], 'r-', label="Target")

        self.interceptor_dot, = self.ax.plot([0], [0], [0], 'bo', markersize=8)
        self.target_dot, = self.ax.plot([0], [0], [0], 'ro', markersize=8)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Missile Interception')
        self.ax.legend()

    def update(self, interceptor_pos, target_pos, sim_time=None):
        with self._lock:
            self.interceptor_path.append(np.array(interceptor_pos))
            self.target_path.append(np.array(target_pos))
            self.sim_time = sim_time
            self._data_changed = True

    def reset(self):
        with self._lock:
            self.interceptor_path.clear()
            self.target_path.clear()
            self.sim_time = None
            self._data_changed = True

    def stop(self):
        self._running = False
