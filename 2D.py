import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams["figure.figsize"] = (18, 5)

init = np.array([[3], [3]])
mu = 8
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
    k = n_steps
    for i in range(n_steps):
        x_1_s += [x[0]]
        x_2_s += [x[1]]
        v_1_s += [v[0]]
        v_2_s += [v[1]]
        x = x + np.sqrt(step_size) * v
        if i == 700:
            v = np.array([[-20], [-20]])
        else:
            v = (v - A.dot(x)) / (mu + 1.0)
        if (x < np.array([[0.3], [0.3]])).all() and k == n_steps:
            k = i
    print k
    return (np.array(x_1_s)[:, 0], np.array(x_2_s)[:, 0],
            np.array(v_1_s)[:, 0], np.array(v_2_s)[:, 0])

x_1_s, x_2_s, v_1_s, v_2_s = run_discrete_updates(step_size)

x_1_rng = np.linspace(-4.5, 3.5, 100)
x_2_rng = np.linspace(-4.5, 3.5, 100)

fig, axes = plt.subplots(2, 3)
axes[0, 0].plot(range(n_steps), x_1_s, 'b')
axes[0, 0].plot(range(n_steps), v_1_s, 'r')
axes[0, 0].legend(['x_1_t', 'x_1_dot_t'])
axes[0, 0].set_xlabel('Time')

axes[1, 0].plot(range(n_steps), x_2_s, 'b')
axes[1, 0].plot(range(n_steps), v_2_s, 'r')
axes[1, 0].legend(['x_2_t', 'x_2_dot_t'])
axes[1, 0].set_xlabel('Time')

axes[0, 1].plot(x_1_rng, manifold_1(x_1_rng), 'k')
axes[0, 1].plot(x_1_s, v_1_s, 'r')
axes[0, 1].set_xlabel('x_1')
axes[0, 1].set_ylabel('x_1_dot')

axes[1, 1].plot(x_2_rng, manifold_2(x_2_rng), 'k')
axes[1, 1].plot(x_2_s, v_2_s, 'r')
axes[1, 1].set_xlabel('x_2')
axes[1, 1].set_ylabel('x_2_dot')

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
X_1, X_2 = np.meshgrid(x_1_rng, x_2_rng)
Y = f_x_1(X_1) + f_x_2(X_2)
surf = ax.plot_surface(X_1, X_2, Y, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.7)

y = f_x_1(x_1_s) + f_x_2(x_2_s)
for i in range(len(y) - 1):
    ax.plot(x_1_s[i: i + 2], x_2_s[i: i + 2], y[i: i + 2],
            c='red', lw=2, alpha=0.5)

axes[0, 2].plot(v_1_s.tolist())
axes[1, 2].plot(v_2_s.tolist())


plt.show()
