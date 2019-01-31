import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np


def model(x, t):
    y = x[0]
    dy = x[1]
    # k = 30
    xdot = [[], []]
    xdot[0] = dy
    # xdot[1] = -(0.9 + 0.7 * t) * dy - k * y
    xdot[1] = -8 * dy - y
    return xdot
time = np.linspace(0, 2, 100)
z2 = odeint(model, [2, -1], time)

# plot results
plt.plot(time, z2[:, 0], 'g:')
plt.plot(time, z2[:, 1], 'k-.')
plt.legend(['y', 'dy/dt'])
plt.xlabel('Time')
plt.show()

plt.plot(z2[:, 0], z2[:, 1])
plt.xlabel('x')
plt.ylabel('x_dot')
plt.show()
