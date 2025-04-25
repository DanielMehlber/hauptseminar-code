import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
import time

class MissileVisualizer(threading.Thread):
    def __init__(self, xlim=(-1000, 1000), ylim=(-1000, 1000), zlim=(-1000, 1000), update_interval=0.05):
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
