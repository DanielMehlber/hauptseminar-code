import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

class MissileVisualizer(threading.Thread):
    def __init__(self, xlim=(-1000, 1000), ylim=(-1000, 1000), zlim=(-1000, 1000), fit_view=True, update_interval=0.05):
        super().__init__()
        self.daemon = True
        self.update_interval = update_interval

        self.interceptor_path = []
        self.target_path = []
        self.sim_time = None
        self.accel_command = None
        self.orientation_matrix = None
        self.los_angles = None  # Added variable to store LOS angles (yaw, pitch)
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
        self.fig = plt.figure(figsize=(12, 8))
        
        # Adjust the spacing between subplots to avoid overlap
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 3], height_ratios=[1, 1, 1], figure=self.fig)

        # Sidebar: Top (acceleration polar plot), Middle (orientation), Bottom (LOS angle)
        self.ax_accel = self.fig.add_subplot(gs[0, 0], polar=True)
        self.ax_orient = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax_los = self.fig.add_subplot(gs[2, 0], polar=True)  # New LOS plot

        # Main 3D plot
        self.ax_main = self.fig.add_subplot(gs[:, 1], projection='3d')
        
        self._init_plots()

        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)  # Adjust margins to fit titles and plots

        self._last_elev = None
        self._last_azim = None
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        while self._running:
            time.sleep(self.update_interval)

            if not self._data_changed:
                continue

            with self._lock:
                ip = np.array(self.interceptor_path)
                tp = np.array(self.target_path)
                sim_time = self.sim_time
                accel_command = self.accel_command
                orientation_matrix = self.orientation_matrix
                los_angles = self.los_angles  # New LOS angle data
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
                self.ax_main.set_title(f"Simulation Time: {sim_time:.2f}s")

            if self.fit_view:
                all_points = np.vstack((ip, tp)) if len(ip) > 0 and len(tp) > 0 else (ip if len(ip) > 0 else tp)
                if len(all_points) > 0:
                    margin = 50
                    min_vals = np.min(all_points, axis=0) - margin
                    max_vals = np.max(all_points, axis=0) + margin
                    ranges = max_vals - min_vals
                    max_range = max(ranges) / 2.0
                    mid = (max_vals + min_vals) / 2.0
                    self.ax_main.set_xlim(mid[0] - max_range, mid[0] + max_range)
                    self.ax_main.set_ylim(mid[1] - max_range, mid[1] + max_range)
                    self.ax_main.set_zlim(mid[2] - max_range, mid[2] + max_range)

            if accel_command is not None:
                self._update_accel_plot(accel_command)

            if orientation_matrix is not None:
                self._update_orientation_plot(orientation_matrix)

            if los_angles is not None:
                self._update_los_plot(los_angles)  # Update the LOS plot

            # Dynamically adjust the layout to prevent overlap
            plt.tight_layout()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        plt.close(self.fig)


    def _init_plots(self):
        self.ax_main.set_xlim(*self.xlim)
        self.ax_main.set_ylim(*self.ylim)
        self.ax_main.set_zlim(*self.zlim)

        self.interceptor_line, = self.ax_main.plot([], [], [], 'b-', label="Interceptor")
        self.target_line, = self.ax_main.plot([], [], [], 'r-', label="Target")
        self.interceptor_dot, = self.ax_main.plot([0], [0], [0], 'bo', markersize=8)
        self.target_dot, = self.ax_main.plot([0], [0], [0], 'ro', markersize=8)

        self.ax_main.set_xlabel('X')
        self.ax_main.set_ylabel('Y')
        self.ax_main.set_zlabel('Z')
        self.ax_main.set_title('Missile Interception')
        self.ax_main.legend()

        # Acceleration command plot setup
        self.ax_accel.set_title("Lateral Accel Command")
        self.ax_accel.set_ylim(0, 1)
        self.accel_dot, = self.ax_accel.plot([], [], 'ro')

        # Orientation plot setup
        self.ax_orient.set_title("Orientation")
        self.ax_orient.set_xlim(-1, 1)
        self.ax_orient.set_ylim(-1, 1)
        self.ax_orient.set_zlim(-1, 1)
        self.ax_orient.set_xlabel("X")
        self.ax_orient.set_ylabel("Y")
        self.ax_orient.set_zlabel("Z")

        # Line of Sight (LOS) angle plot setup
        self.ax_los.set_title("Line of Sight Angle")
        self.ax_los.set_ylim(-np.pi/2, np.pi/2)  # Vertical angle range from -pi/2 to pi/2
        self.ax_los.set_xlim(-np.pi, np.pi)      # Horizontal angle range from -pi to pi
        self.los_dot, = self.ax_los.plot([], [], 'ro')

    def _update_accel_plot(self, accel_cmd):
        x, y = accel_cmd
        r = np.linalg.norm([x, y])
        theta = np.arctan2(y, x)
        self.accel_dot.set_data([theta], [r])
        self.ax_accel.set_ylim(0, 1)

    def _update_orientation_plot(self, R):
        for artist in self.ax_orient.collections + self.ax_orient.lines:
            artist.remove()

        origin = np.array([[0, 0, 0]]).T
        arrow_length = 0.8
        axis_defs = [
            ('Forward (X)', R[:, 0] * 1.0, 'r', 1.5),
            ('Up (Y)',      R[:, 1] * 0.5, 'g', 1),
            ('Right (Z)',   R[:, 2] * 0.5, 'b', 1)
        ]

        for label, vec, color, width in axis_defs:
            self.ax_orient.quiver(
                *origin.ravel(),
                *vec,
                color=color,
                linewidth=width,
                arrow_length_ratio=0.15,
                label=label
            )

        lim = arrow_length * 1.2
        self.ax_orient.set_xlim([-lim, lim])
        self.ax_orient.set_ylim([-lim, lim])
        self.ax_orient.set_zlim([-lim, lim])
        self.ax_orient.legend()

    def _update_los_plot(self, los_angles):
        yaw, pitch = los_angles
        # Convert yaw and pitch to the polar plot range
        theta = yaw  # Horizontal angle (yaw), in radians from -pi to pi
        r = pitch    # Vertical angle (pitch), in radians from -pi/2 to pi/2
        self.los_dot.set_data([theta], [r])
        self.ax_los.set_ylim(-np.pi/2, np.pi/2)  # Vertical angle range from -pi/2 to pi/2
        self.ax_los.set_xlim(-np.pi, np.pi)      # Horizontal angle range from -pi to pi

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax_main:
            return

        elev = self.ax_main.elev
        azim = self.ax_main.azim

        if (elev != self._last_elev) or (azim != self._last_azim):
            self._last_elev = elev
            self._last_azim = azim
            self.ax_orient.view_init(elev, azim)

    def update(self, interceptor_pos, target_pos, sim_time=None, accel_command=None, orientation_matrix=None, los_angles=None):
        with self._lock:
            self.interceptor_path.append(np.array(interceptor_pos))
            self.target_path.append(np.array(target_pos))
            self.sim_time = sim_time
            self.accel_command = accel_command
            self.orientation_matrix = orientation_matrix
            self.los_angles = los_angles  # Store the LOS angles
            self._data_changed = True

    def reset(self):
        with self._lock:
            self.interceptor_path.clear()
            self.target_path.clear()
            self.sim_time = None
            self.accel_command = None
            self.orientation_matrix = None
            self.los_angles = None  # Reset LOS angles
            self._data_changed = True

    def stop(self):
        self._running = False
