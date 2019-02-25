import torch
import numpy as np
from torch.optim import SGD
import matplotlib.pyplot as plt
from scipy.integrate import odeint


steps = 50000
init = 2


def func(x_):
    f = 0.5 * x_.pow(2)
    return f


def d_f(x_):
    return x_


def optimization():
    device = torch.device('cpu')
    x_1 = torch.randn(1, 1, device=device, requires_grad=True)
    x_1.data = torch.Tensor([[init]])

    optimizer = SGD([x_1], lr=1e-5, momentum=0.9, dampening=0,
                    weight_decay=0, nesterov=True)

    x_1s = []
    x_dots = []
    for step in range(steps):
        if len(x_1s) != 0:
            x_dots.append(x_1.item() - x_1s[-1])
        x_1s.append(x_1.item())
        f = func(x_1)
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
    return x_1s, x_dots


def ode_solver(x_, t):
    x = x_[0]
    x_dot = x_[1]
    # system = [x_dot, -3.0 / t * x_dot - x]
    system = [x_dot, -5.0 * x_dot - x]
    return system

time = np.linspace(0.001, 20, 1000)

x_1s, x_dots = optimization()
z = odeint(ode_solver, [2, 0], time)
plt.plot(z[:, 0], z[:, 1], label='EL ODE')
# plt.show()
# plt.plot(x_1s)
# plt.show()
# df = analytic_manifold(x_1s)
# plt.plot(x_1s[1:], np.multiply(x_dots, 1e3), label="optim")
# plt.plot(x_1s[1:], x_dots, label="optim")
# import ipdb; ipdb.set_trace()
plt.plot(x_1s[1:], np.multiply(x_1s[1:], -0.24), label="analytic")
# import ipdb; ipdb.set_trace()
plt.xlabel("x")
plt.ylabel("x_dot")
plt.legend()
plt.show()
