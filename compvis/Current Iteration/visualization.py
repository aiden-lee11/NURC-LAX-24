# taken from Vincent's triloc analysis.py 
# https://github.com/v-rim/triloc/blob/main/triloc/analysis.py
import time

import matplotlib.pyplot as plt
import numpy as np


class PointInSpace:
    def __init__(self, x_lim, y_lim=None, z_lim=None, m="o"):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.m = m

        if y_lim is None:
            y_lim = x_lim
        if z_lim is None:
            z_lim = x_lim

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_zlim(z_lim)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        zero = np.array([0])

        # Not sure why assignment needs to be like this
        (self.point,) = self.ax.plot(zero, zero, zero, m, animated=True)
        plt.show(block=False)
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.point)
        self.fig.canvas.blit(self.fig.bbox)

    def draw_point(self, point, delay=0):
        point = np.array(point)[:, None]

        self.fig.canvas.restore_region(self.bg)
        self.point.set_data(point[0], point[1])
        self.point.set_3d_properties(point[2])

        self.ax.draw_artist(self.point)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        if delay > 0:
            plt.pause(delay)


if __name__ == "__main__":
    lim = [-2, 2]
    pp = PointInSpace(lim)
    frame_count = 1000

    tic = time.time()
    for j in range(frame_count):
        x = np.cos((j / 100) * np.pi)
        y = np.sin((j / 100) * np.pi)
        z = 3 * (j % 200.0) / 200.0
        pp.draw_point([x, y, z])

    print(f"Average FPS: {frame_count / (time.time() - tic)}")