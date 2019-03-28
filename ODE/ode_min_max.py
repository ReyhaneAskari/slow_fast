import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import ortho_group
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.figsize"] = (8, 8)
np.random.seed(11)

lambda_ = 100


def plot_surface():
    points = []
    X = np.linspace(-20, 20, 100)
    Y = np.linspace(-20, 20, 100)
    for x in X:
        for y in Y:
            points += [[x, y]]
    points = Variable(torch.FloatTensor(np.array(points)))
    surf = net(points)
    surf = surf.data.numpy().reshape((100, 100)).T
    X, Y = np.meshgrid(X, Y)
    plt.contourf(X, Y, surf, 18, alpha=.5, cmap='RdBu_r')


def SimGD(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    for i in range(1000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        net.x.grad.data = -net.x.grad.data
        opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='#01117C', label='SimGD')


def ODE(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    for i in range(1000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        net.x.grad.data = -net.x.grad.data
        j_row_1 = torch.autograd.grad(net.x.grad, net.parameters(), retain_graph=True, create_graph=True)
        j_row_2 = torch.autograd.grad(net.y.grad, net.parameters(), retain_graph=True, create_graph=True)
        joc = torch.cat((torch.cat(j_row_1).unsqueeze(0), torch.cat(j_row_2).unsqueeze(0)))
        omega = torch.cat((net.x.grad.unsqueeze(0), net.y.grad.unsqueeze(0)))
        j_t_j = torch.mm(joc.transpose(0, 1), joc)
        j_lambda = j_t_j + lambda_ * Variable(torch.eye(2))
        omega = 0.5 * (omega + torch.mm(joc.transpose(0, 1), torch.mm(torch.inverse(j_lambda), torch.mm(joc.transpose(0, 1), omega))))
        # omega = 0.5 * (omega + torch.mm(torch.mm(joc.transpose(0, 1), torch.inverse(joc)), omega))

        net.x.grad.data = omega[0].data
        net.y.grad.data = omega[1].data
        opt.step()
    xys = np.array(xys)
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='g', label="ODE_naive")


def DODE(net, x_init, y_init):
    net.x.data = torch.FloatTensor([x_init])
    net.y.data = torch.FloatTensor([y_init])
    xys = []
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)
    a_n = 0.004  # gamma_1
    b_n = 0.005  # gamma_2
    kesi_1 = 0.9
    for i in range(5000):
        xys += [[net.x.data[0] + 0, net.y.data[0] + 0]]
        loss = net()
        opt.zero_grad()
        loss.backward(create_graph=True)
        # net.x.grad.data = -net.x.grad.data
        omega = torch.cat((-net.x.grad.unsqueeze(0), net.y.grad.unsqueeze(0)))
        f_lambda = kesi_1 * (1 - torch.exp(- torch.norm(omega, p=2)))
        j_row_1 = torch.autograd.grad(omega[0], net.parameters(),
                                      retain_graph=True, create_graph=True)
        j_row_2 = torch.autograd.grad(omega[1], net.parameters(),
                                      retain_graph=True, create_graph=True)
        joc = torch.cat((torch.cat(j_row_1).unsqueeze(0),
                        torch.cat(j_row_2).unsqueeze(0)))

        g_x = torch.autograd.grad(omega[0] * net.v[0] + omega[1] * net.v[1] - loss,
                                  net.x, retain_graph=True)
        g_y = torch.autograd.grad(omega[0] * net.v[0] + omega[1] * net.v[1] + loss,
                                  net.y, retain_graph=True)
        g_v = torch.autograd.grad(torch.norm(torch.mm(
            joc, net.v.unsqueeze(1)) - omega) + f_lambda * torch.norm(net.v), net.v,
            retain_graph=True)
        net.x.data = net.x.data - a_n * g_x[0].data
        net.y.data = net.y.data - a_n * g_y[0].data
        net.v.data = net.v.data - b_n * g_v[0].data
        # opt.step()
    xys = np.array(xys)
    # import ipdb; ipdb.set_trace()
    plt.plot(xys[:, 0], xys[:, 1], lw=2, color='red', label='Discrete ODE')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.x = torch.nn.Parameter(torch.FloatTensor([0]))
        self.y = torch.nn.Parameter(torch.FloatTensor([0]))
        self.v = Variable(torch.FloatTensor([1, 1]))
        self.v.requires_grad = True

    def forward(self, xy=None):
        if xy is not None:
            x = xy[:, 0]
            y = xy[:, 1]
            return (torch.exp(-0.01 * (x ** 2 + y ** 2)) *
                    ((0.3 * x ** 2 + y) ** 2 +
                     (0.5 * y ** 2 + x) ** 2))
        return (torch.exp(-0.01 * (self.x ** 2 + self.y ** 2)) *
                ((0.3 * self.x ** 2 + self.y) ** 2 +
                 (0.5 * self.y ** 2 + self.x) ** 2))
net = Net()
plot_surface()
SimGD(net, -5, -10)
# SimGD(net, 10, -10)
# SimGD(net, -10, 4)
ODE(net, -5, -10)
# ODE(net, 10, -10)
# ODE(net, -10, 4)
DODE(net, -5, -10)
# DODE(net, 10, -10)
# DODE(net, -10, 4)
plt.legend()
plt.show()
