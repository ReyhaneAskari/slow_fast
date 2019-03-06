import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams["figure.figsize"] = (10, 5)

init = np.array([[-4], [-4]])
eta = 1.0
n_steps = 2000
k = 100
a = 1
A = np.array([[a, 0],
              [0, k]])
step_size = (2 / (1 + np.sqrt(k))) ** 2
print 'optimal step_size: ' + str(step_size)
step_size = 0.001
beta = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1)) ** 2
print 'optimal beta: ' + str(beta)
beta = 0.9
mu = 1 / beta - 1
print 'beta: ' + str(beta)
print 'step_size: ' + str(step_size)

restart_iter = 400
perturb_iter = 400
perturbation = 2
init_v_1 = 0
init_v_2 = 0

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
    return 0.5 * k * x ** 2


def nabla_f_x_2(x):
    return k * x


def nabla2_f_x_2(x):
    return k


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
    v = np.array([[init_v_1], [init_v_2]])
    distance = []
    found = False
    for i in range(n_steps):
        x_1_s += [x[0]]
        x_2_s += [x[1]]
        v_1_s += [v[0]]
        v_2_s += [v[1]]
        x = x + np.sqrt(step_size) * v
        if restart and (i + 1) % restart == 0 and np.linalg.norm(x) > 0.2:
        # if restart and (i + 1) == restart:
            v = np.array([[0], [0]])
        elif perturb and (i + 1) % perturb == 0 and np.linalg.norm(x) > 0.2:
        # elif perturb and (i + 1) == perturb:
            v = np.array([[perturbation], [perturbation]])
        else:
            v = (v - A.dot(x)) / (mu + 1.0)

        distance.append(np.linalg.norm(x))
        if not found and np.linalg.norm(x) < 0.0001:
            found = True
            print i
    return (np.array(x_1_s)[:, 0], np.array(x_2_s)[:, 0],
            np.array(v_1_s)[:, 0], np.array(v_2_s)[:, 0],
            distance)

print 'original'
x_1_s, x_2_s, v_1_s, v_2_s, distance = run_discrete_updates(step_size)
print 'restart'
x_1_s_r, x_2_s_r, v_1_s_r, v_2_s_r, distance_r = run_discrete_updates(
    step_size, restart_iter, 0)
print 'perturbed'
x_1_s_p, x_2_s_p, v_1_s_p, v_2_s_p, distance_p = run_discrete_updates(
    step_size, 0, perturb_iter)

x_1_rng = np.linspace(-7, 7, 100)
x_2_rng = np.linspace(-7, 7, 100)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x_1_s, x_2_s, color='b', alpha=0.8)
ax[0].plot(x_1_s_r, x_2_s_r, color='r', alpha=0.8)
ax[0].plot(x_1_s_p, x_2_s_p, color='g', alpha=0.8)
X_1, X_2 = np.meshgrid(x_1_rng, x_2_rng)
Y = f_x_1(X_1) + f_x_2(X_2)
ax[0].contour(X_1, X_2, Y, colors=['#c1c4c9'])
ax[0].legend(['normal', 'restarted', 'perturbed'])

# import ipdb; ipdb.set_trace()
ax[1].plot(range(0, n_steps), distance, color='b')
ax[1].plot(range(0, n_steps), distance_r, color='r')
ax[1].plot(range(0, n_steps), distance_p, color='g')
ax[1].legend(['normal', 'restarted', 'perturbed'])

title = ('Init v_1: ' + str(init_v_1) + ' Init v_2: ' + str(init_v_2) +
         ', Condition number: ' + str(k) +
         ', Restarted at every' + str(restart_iter) +
         ', Perturbed at every' + str(perturb_iter) +
         ' to velocity: ' + str(perturbation))

plt.suptitle(title)
# plt.savefig('results/' + title)
plt.show()

# plt.rcParams["figure.figsize"] = (18, 5)
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
