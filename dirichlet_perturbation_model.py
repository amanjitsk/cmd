import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from latex_utils import latexify, format_axes

# from scipy.special import logsumexp, softmax

np.set_printoptions(precision=3, suppress=True)

# small value to avoid overflow
EPS = np.finfo(np.float64).eps

# parameters
sigma = 0.3
lamda = -sigma
# lamda = 1
# alpha = 1 / lamda


def rho(a, b):
    return 1 + lamda * np.dot(a, b)


def gyroadd(a, b):
    return (a * b) / (a * b).sum()


def primal_to_dual(theta):
    return np.reciprocal(lamda * theta)


def dual_to_primal(eta):
    return np.reciprocal(lamda * eta)


def error(x_true, x_hat):
    return np.linalg.norm(np.log(x_true) - np.log(x_hat), 2)


# dimensions, constants
N = 51
D = N - 1
STEP = 1.0 / D

alpha = np.full(N, (N * sigma) ** (-1))

# parameter of dirichlet perturbation model
# p is in the n-dim unit simplex
# p_star = softmax(np.random.randn(N))
rng = np.random.RandomState(0)
# p_star = 1 + rng.rand(N)
p_star = rng.uniform(low=1.0, high=5.0, size=(N,))
p_star /= p_star.sum()
theta_star = p_star[0] * np.reciprocal(lamda * p_star)[1:]
# theta_star = p_star[D] * np.reciprocal(lamda * p_star)[:D]
eta_star = primal_to_dual(theta_star)
# print(f"p_star = {p_star}")


latexify(4.5, 4)
fig, ax = plt.subplots()
N_REPEAT = 30
NUM_ITERS = 10000
cmap = plt.get_cmap("binary")
cnorm = colors.Normalize(vmin=0, vmax=N_REPEAT - 1)
start = 0.1
end = 0.6

for i in range(N_REPEAT):
    # p_guess = softmax(rng.randn(N))
    # p_guess = np.full(N, 1 / N)
    # theta = p_guess[0] * np.reciprocal(lamda * p_guess)[1:]
    # eta = primal_to_dual(theta)
    eta = np.full(D, 1.0)
    loss = []
    loss.append(error(eta_star, eta))

    for k in range(1, NUM_ITERS + 1):
        # LR = STEP / np.sqrt(k)
        LR = 1.0 / k
        noise = rng.dirichlet(alpha)
        # print(f"noise {k} : {noise}")
        q = gyroadd(p_star, noise)
        # print(f"q {k} : {q}")
        y = q[1:] / q[0]
        # print(f"y {k} : {y}")
        theta = dual_to_primal(eta)
        eta = eta + LR * (rho(theta, eta) / rho(theta, y)) * (y - eta)
        eta = np.abs(eta)
        # eta = eta + LR * (y - eta)
        # theta = dual_to_primal(eta)
        # print(f"eta {k} : {eta}")
        # print(f"theta {k} : {theta}")
        loss.append(error(eta_star, eta))

    color = cmap(start + (end - start) * cnorm(i))
    ax.plot(np.arange(NUM_ITERS + 1), np.log(np.asarray(loss)), color=color)

format_axes(
    ax,
    title=rf"Dirichlet perturbation ($d = {D}$)",
    xlabel=r"Iteration $k$",
    ylabel=r"$\log \mathrm{dist}(\eta_k, \eta^*)$",
    # leg_loc="best",
)
# ax2.set_ylim(0, 1)
ax.set_xscale("log")
ax.grid(alpha=0.1)
plt.tight_layout()
plt.subplots_adjust(left=0.13, bottom=0.13, top=0.91)
# plt.savefig("dir_sim_py.pdf")
plt.show()
