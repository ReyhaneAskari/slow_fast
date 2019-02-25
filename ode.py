import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np


def model(x_, t):
    x = x_[0]
    x_dot = x_[1]
    system = [x_dot, -3.0 / t * x_dot - x]
    return system

time = np.linspace(0.001, 2, 100)
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
