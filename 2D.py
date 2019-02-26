from scipy.integrate import odeint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
plt.rcParams["figure.figsize"] = (18, 5)

init = 3.4
mu = 1.0
eta = 1.0
n_steps = 1000

# def nesterov():


def f_x(x):
    # return x ** 2 / 2.0
    return (x + 4) * (x + 1) * (x - 1) * (x - 3) / 14.0


def nabla_f_x(x):
    # return x
    return (4 * x ** 3 + 3 * x ** 2 - 26 * x - 1) / 14.0


def nabla2_f_x(x):
    # return x
    return (12 * x ** 2 + 6 * x - 26) / 14.0


def manifold(x):
    eps = 1.0 / mu
    return (- eps * eta * nabla_f_x(x) -
            eps ** 3 * eta ** 2 * nabla2_f_x(x) * nabla_f_x(x))


def system(u, t):
    x = u[0]
    y = u[1]
    udot = [[], []]
    udot[0] = y
    udot[1] = -mu * y - eta * nabla_f_x(x)
    # udot[1] = -3.0 * y / t - nabla_f_x(x)
    return udot


def run_discrete_updates(step_size):
    xs = []
    vs = []
    x = init + 0
    v = 0
    for i in range(n_steps):
        xs += [x]
        vs += [v]
        # if i == 100:
        #     v = 0
        #     x = x - 3 * np.sign(nabla_f_x(x))
        #     # x = -1
        # else:
        x = x + np.sqrt(step_size) * v
        v = (v - nabla_f_x(x)) / 2.0

    xs = np.array(xs)
    x_dots = np.array(vs)
    return xs, x_dots

x_rng = np.linspace(-4.5, 3.5, 1000)

xs, x_dots = run_discrete_updates(0.0001)


axes = plt.subplots(1, 3)[1]
axes[0].plot(range(n_steps), xs, 'b')
axes[0].plot(range(n_steps), x_dots, 'r')
axes[0].legend(['x_t', 'x_dot_t'])
axes[0].set_xlabel('Time')

axes[1].plot(x_rng, manifold(x_rng), 'k')
axes[1].plot(xs, x_dots, 'r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('x_dot')

axes[2].plot(x_rng, f_x(x_rng), 'k', lw=2, alpha=0.7)
# axes[2].plot(x_t, f_x(x_t), 'b', lw=3, alpha=0.7)
axes[2].plot(xs, f_x(xs), 'r', lw=3, alpha=0.7)
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')


# argmin_y = np.argmin(y_t)
# axes[0].plot(time[argmin_y: argmin_y + 2], y_t[argmin_y: argmin_y + 2],
#              'g', lw=10, alpha=0.5)
# axes[2].plot(x_t[argmin_y: argmin_y + 2], f_x(x_t[argmin_y: argmin_y + 2]),
#              'g', lw=10, alpha=0.5)
# axes[1].plot(x_t[argmin_y: argmin_y + 2], y_t[argmin_y: argmin_y + 2],
#              'g', lw=10, alpha=0.5)

plt.show()
