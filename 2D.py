import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
plt.rcParams["figure.figsize"] = (18, 5)

init = np.array([[3], [3]])
mu = 8.0
eta = 1.0
n_steps = 1000
step_size = 0.0001
a = 2
b = 100
A = np.array([[a, 0],
              [0, b]])


def f_x_1(x):
    return 0.5 * a * x ** 2


def nabla_f_x_1(x):
    return a * x


def nabla2_f_x_1(x):
    return a


def manifold_1(x):
    eps = 1.0 / mu
    return (- eps * eta * nabla_f_x_1(x) -
            eps ** 3 * eta ** 2 * nabla2_f_x_1(x) * nabla_f_x_1(x))


def f_x_2(x):
    return 0.5 * b * x ** 2


def nabla_f_x_2(x):
    return b * x


def nabla2_f_x_2(x):
    return b


def manifold_2(x):
    eps = 1.0 / mu
    return (- eps * eta * nabla_f_x_2(x) -
            eps ** 3 * eta ** 2 * nabla2_f_x_2(x) * nabla_f_x_2(x))


def run_discrete_updates(step_size):
    x_1_s = []
    x_2_s = []
    v_1_s = []
    v_2_s = []
    x = init + 0
    v = np.array([[0], [0]])
    for i in range(n_steps):
        x_1_s += [x[0]]
        x_2_s += [x[1]]
        v_1_s += [v[0]]
        v_2_s += [v[1]]
        x = x + np.sqrt(step_size) * v
        v = (v - A.dot(x)) / 2.0
    return (np.array(x_1_s)[:, 0], np.array(x_2_s)[:, 0],
            np.array(v_1_s)[:, 0], np.array(v_2_s)[:, 0])

x_1_rng = np.linspace(-4.5, 3.5, 1000)
x_2_rng = np.linspace(-4.5, 3.5, 1000)

x_1_s, x_2_s, v_1_s, v_2_s = run_discrete_updates(step_size)

axes = plt.subplots(1, 3)[1]
axes[0].plot(range(n_steps), x_1_s, 'b')
axes[0].plot(range(n_steps), v_1_s, 'r')
axes[0].legend(['x_t', 'x_dot_t'])
axes[0].set_xlabel('Time')

axes[1].plot(x_1_rng, manifold_1(x_1_rng), 'k')
axes[1].plot(x_1_s, v_1_s, 'r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('x_dot')

axes[2].plot(x_1_rng, f_x_1(x_1_rng), 'k', lw=2, alpha=0.7)
axes[2].plot(x_1_s, f_x_1(x_1_s), 'r', lw=3, alpha=0.7)
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')

plt.show()

axes = plt.subplots(1, 3)[1]
axes[0].plot(range(n_steps), x_2_s, 'b')
axes[0].plot(range(n_steps), v_2_s, 'r')
axes[0].legend(['x_t', 'x_dot_t'])
axes[0].set_xlabel('Time')

axes[1].plot(x_2_rng, manifold_2(x_2_rng), 'k')
axes[1].plot(x_2_s, v_2_s, 'r')
axes[1].set_xlabel('x')
axes[1].set_ylabel('x_dot')

axes[2].plot(x_2_rng, f_x_2(x_2_rng), 'k', lw=2, alpha=0.7)
axes[2].plot(x_2_s, f_x_2(x_2_s), 'r', lw=3, alpha=0.7)
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')

plt.show()
