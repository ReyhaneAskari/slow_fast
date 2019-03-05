import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# plt.rcParams["figure.figsize"] = (18, 5)
matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.figsize"] = (10, 5)

init = np.array([[-4], [-4]])
mu = 0.05
eta = 1.0
n_steps = 1000
step_size = 0.00001
condition_number = 1
a = 1
A = np.array([[a, 0],
              [0, condition_number]])
restart_iter = 10
perturb_iter = 10
perturbation = 2

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
    return 0.5 * condition_number * x ** 2


def nabla_f_x_2(x):
    return condition_number * x


def nabla2_f_x_2(x):
    return condition_number


def manifold_2(x):
    eps = 1.0 / mu
    return (- eps * eta * nabla_f_x_2(x) -
            eps ** 3 * eta ** 2 * nabla2_f_x_2(x) * nabla_f_x_2(x))


def run_discrete_updates(step_size, restart=0, perturb=0):
    x_1_s = []
    x_2_s = []
    v_1_s = []
    v_2_s = []
    x = init + 0
    v = np.array([[-80], [+80]])
    # k = n_steps
    distance = []
    for i in range(n_steps):
        x_1_s += [x[0]]
        x_2_s += [x[1]]
        v_1_s += [v[0]]
        v_2_s += [v[1]]
        x = x + np.sqrt(step_size) * v
        if restart and (i + 1) == restart:
            print 'restart'
            v = np.array([[0], [0]])
        elif perturb and (i + 1) == perturb:
            print 'perturb'
            v = np.array([[perturbation], [perturbation]])
        else:
            v = (v - A.dot(x)) / (mu + 1.0)

        distance.append(np.linalg.norm(x))
    return (np.array(x_1_s)[:, 0], np.array(x_2_s)[:, 0],
            np.array(v_1_s)[:, 0], np.array(v_2_s)[:, 0],
            distance)


x_1_s, x_2_s, v_1_s, v_2_s, distance = run_discrete_updates(step_size)
x_1_s_r, x_2_s_r, v_1_s_r, v_2_s_r, distance_r = run_discrete_updates(
    step_size, restart_iter, 0)
x_1_s_p, x_2_s_p, v_1_s_p, v_2_s_p, distance_p = run_discrete_updates(
    step_size, 0, perturb_iter)

x_1_rng = np.linspace(-7, 7, 100)
x_2_rng = np.linspace(-7, 7, 100)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x_1_s, x_2_s, color='b')
ax[0].plot(x_1_s_r, x_2_s_r, color='r')
ax[0].plot(x_1_s_p, x_2_s_p, color='g')
X_1, X_2 = np.meshgrid(x_1_rng, x_2_rng)
Y = f_x_1(X_1) + f_x_2(X_2)
ax[0].contour(X_1, X_2, Y, colors=['#c1c4c9'])
ax[0].legend(['normal', 'restarted', 'perturbed'])

# import ipdb; ipdb.set_trace()
ax[1].plot(range(0, n_steps), distance, color='b')
ax[1].plot(range(0, n_steps), distance_r, color='r')
ax[1].plot(range(0, n_steps), distance_p, color='g')
ax[1].legend(['normal', 'restarted', 'perturbed'])

title = ('Condition number: ' + str(condition_number) +
         ', Restarted at iteration ' + str(restart_iter) +
         ', Perturbed at iteration ' + str(perturb_iter) +
         ' with velocity: ' + str(perturbation))

plt.suptitle(title, fontdict={'horizontalalignment': 'center'})
plt.savefig(title)
plt.show()

# fig, axes = plt.subplots(2, 3)
# axes[0, 0].plot(range(n_steps), x_1_s, 'b')
# axes[0, 0].plot(range(n_steps), v_1_s, 'r')
# axes[0, 0].legend(['x_1_t', 'x_1_dot_t'])
# axes[0, 0].set_xlabel('Time')

# axes[1, 0].plot(range(n_steps), x_2_s, 'b')
# axes[1, 0].plot(range(n_steps), v_2_s, 'r')
# axes[1, 0].legend(['x_2_t', 'x_2_dot_t'])
# axes[1, 0].set_xlabel('Time')

# axes[0, 1].plot(x_1_rng, manifold_1(x_1_rng), 'k')
# axes[0, 1].plot(x_1_s, v_1_s, 'r')
# axes[0, 1].set_xlabel('x_1')
# axes[0, 1].set_ylabel('x_1_dot')

# axes[1, 1].plot(x_2_rng, manifold_2(x_2_rng), 'k')
# axes[1, 1].plot(x_2_s, v_2_s, 'r')
# axes[1, 1].set_xlabel('x_2')
# axes[1, 1].set_ylabel('x_2_dot')

# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# X_1, X_2 = np.meshgrid(x_1_rng, x_2_rng)
# Y = f_x_1(X_1) + f_x_2(X_2)
# surf = ax.plot_surface(X_1, X_2, Y, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False, alpha=0.7)

# y = f_x_1(x_1_s) + f_x_2(x_2_s)
# for i in range(len(y) - 1):
#     ax.plot(x_1_s[i: i + 2], x_2_s[i: i + 2], y[i: i + 2],
#             c='red', lw=2, alpha=0.5)

# axes[0, 2].plot(v_1_s.tolist())
# axes[0, 2].set_xlabel('time')
# axes[0, 2].set_ylabel('v_1')
# axes[1, 2].plot(v_2_s.tolist())
# axes[1, 2].set_xlabel('time')
# axes[1, 2].set_ylabel('v_2')


# plt.show()
