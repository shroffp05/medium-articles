import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 

def data_gen():
    x_range = [(val/8)*np.pi for val in range(0,49)]
    for cnt in x_range:
        yield cnt, np.sin(cnt)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(right = 6*np.pi)

    ticks = [(val/2)*np.pi for val in range(0,13)]
    labels = [val for val in range(0, 180*6+90, 90)]
    ax.set_xticks(ticks=ticks, labels=labels)

    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
plt.suptitle("Sine Wave")
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    x, sin = data
    xdata.append(x)
    ydata.append(sin)
    xmin, xmax = ax.get_xlim()
    print(x, xmax)

    if x >= xmax:
        ax.set_xlim(xmin, xmax+(np.pi/8))
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

# Only save last 100 frames, but run forever
ani = animation.FuncAnimation(fig, run, data_gen, interval=100, init_func=init, save_count=100, repeat=False)
#writervideo = animation.FFMpegWriter(fps=60)
ani.save("Animation.gif")

#plt.show()