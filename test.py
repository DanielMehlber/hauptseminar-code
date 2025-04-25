import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# Set interactive mode on
plt.ion()

# Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize some data
n_points = 100
x = np.random.rand(n_points)
y = np.random.rand(n_points)
z = np.random.rand(n_points)
scatter = ax.scatter(x, y, z)

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

def update_plot():
    # Update the data here (random walk as an example)
    global x, y, z
    x += (np.random.rand(n_points) - 0.5) * 0.01
    y += (np.random.rand(n_points) - 0.5) * 0.01
    z += (np.random.rand(n_points) - 0.5) * 0.01
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    z = np.clip(z, 0, 1)

    # Update the scatter plot
    scatter._offsets3d = (x, y, z)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Loop at ~60 Hz
try:
    while True:
        start_time = time.time()
        update_plot()
        elapsed = time.time() - start_time
        time.sleep(max(0, 1/60 - elapsed))  # Sleep to maintain ~60 FPS
except KeyboardInterrupt:
    print("Stopped.")