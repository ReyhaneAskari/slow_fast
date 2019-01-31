import torch
import numpy as np
from torch.optim import SGD
import matplotlib.pyplot as plt

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

# def analytic_manifold(x):
#     # x = np.linspace(init - 2, init + 2, 100)
#     df = d_f(x)
#     return df

x_1s, x_dots = optimization()
# plt.plot(x_1s)
# plt.show()
# df = analytic_manifold(x_1s)
plt.plot(x_1s[1:], np.multiply(x_dots, 1e3), label="optim")
# import ipdb; ipdb.set_trace()
plt.plot(x_1s[1:], np.multiply(x_1s[1:], -0.24), label="analytic")
plt.xlabel("x")
plt.ylabel("x_dot")
plt.legend()
plt.show()
