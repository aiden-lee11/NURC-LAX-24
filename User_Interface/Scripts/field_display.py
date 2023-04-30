import numpy as np
from matplotlib import pyplot as plt
import random

plt.rcParams["figure.figsize"] = [7.0, 3.5]   # default size from copied code, manipulate for iphone14 later
plt.rcParams["figure.autolayout"] = True
im = plt.imread("/Users/emilygordon/NURC/nurc-lax-robot-2023/User_Interface/media/lax_field.png") # this path likely needs to change for host computer
fig, ax = plt.subplots()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
im = ax.imshow(im, extent=[-14, 14, -15, 4.5])   # units are meters, where to define 0 from?

# function, takes in datasets xpoints, ypoints to plot on image of goal
def field_display(xpoints, ypoints):     # should the function accept arrays or tuples?
    xpoints = np.array(xpoints)
    ypoints = np.array(ypoints)

    for i in range(len(xpoints)):

        # Generating a random number in between 0 and 2^24
        color = random.randrange(0, 2**24)
        # Converting that number from base-10 (decimal) to base-16 (hexadecimal)
        hex_color = hex(color)
        # Converting hex (0x000000) to useable structure (#000000)
        std_color = "#" + hex_color[2:]
        plt.plot(xpoints[i], ypoints[i], marker="o", markersize=10, markeredgecolor="black", markerfacecolor=std_color, linestyle="None") # how big should the "ball" be?
        
    plt.show()

field_display([0, -12, 0, 12, 0], [0, 0, -8, 0, -12])