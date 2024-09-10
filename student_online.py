import fire
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
from latex_utils import latexify, format_axes


def dof_to_lamda(k):
    return -2 / (k + 1)


def para_to_theta(para, k):
    mu, sigma = para
    lamda = dof_to_lamda(k)
    return np.array([2 * mu, -1.0]) / ((lamda + 2) * sigma**2 - lamda * mu**2)


def theta_to_para(theta, k):
    lamda = dof_to_lamda(k)
    t1, t2 = theta
    return np.array(
        [
            -t1 / (2 * t2),
            -np.sqrt((lamda * t1**2 - 4 * t2) / (lamda + 2)) / (2 * t2),
        ]
    )


def theta_to_eta(theta, k):
    lamda = dof_to_lamda(k)
    t1, t2 = theta
    e1 = -t1 / (2 * t2)
    e2 = ((lamda + 1) * t1**2 - 2 * t2) / (2 * t2**2 * (lamda + 2))
    return np.array([e1, e2])


def eta_to_theta(eta, k):
    lamda = dof_to_lamda(k)
    e1, e2 = eta
    un_t1 = -2 * e1  # unnormalized
    un_t2 = 1
    denom = 2 * (lamda + 1) * e1**2 - (lamda + 2) * e2
    return np.array([un_t1, un_t2]) / denom


def para_to_eta(para, k):
    return theta_to_eta(para_to_theta(para, k), k)


def eta_to_para(eta, k):
    return theta_to_para(eta_to_theta(eta, k), k)


def update_eta(eta, k, y, lr):
    theta = eta_to_theta(eta, k)
    lamda = dof_to_lamda(k)
    const = (1 + lamda * np.dot(theta, eta)) / (1 + lamda * np.dot(theta, y))
    updated_eta = eta + lr * const * (y - eta)
    # add domain constraints here
    e1, e2 = updated_eta
    e2 = np.maximum(e2, 2 * e1**2 * (lamda + 1) / (lamda + 2))
    ub = np.sqrt((2 * (lamda + 1) * e1**2 - (lamda + 2) * e2) / lamda)
    e1 = np.clip(e1, -ub, ub)
    new_eta = np.array([e1, e2])
    return new_eta


def plot_path(para_true: np.ndarray, k: int, n_samples: int, ax, seed: int, color: str):
    rng = np.random.RandomState(seed)
    loc, scale = para_true
    X = stats.t.rvs(df=k, loc=loc, scale=scale, size=n_samples, random_state=rng)
    Y = np.vstack([X, X**2]).T  # n_samples x 2
    eta_history = []
    para_history = []
    para_init = np.array([0.0, 1.0])
    eta_history.append(para_to_eta(para_init, k))
    para_history.append(eta_to_para(eta_history[-1], k))
    for i in range(n_samples):
        eta = eta_history[-1]
        y = Y[i, :]
        eta_history.append(update_eta(eta, k, y, 1 / (i + 1)))
        para_history.append(eta_to_para(eta_history[-1], k))
    para_history = np.array(para_history)
    ax.plot(para_history[:, 0], para_history[:, 1], color=color)
    return para_history


def main(k=2, n_samples=10000, mu_true=20.0, sigma_true=10.0, num_paths=10, save=False):
    # n_samples = 10000
    para_true = np.array([mu_true, sigma_true])
    # k = 3
    latexify(4.5, 4)
    _, ax = plt.subplots()
    cmap = plt.get_cmap("binary")
    cnorm = colors.Normalize(vmin=1, vmax=num_paths)
    start = 0.2
    end = 0.5
    for i in range(num_paths):
        color = cmap(start + (end - start) * cnorm(i + 1))
        para_history = plot_path(para_true, k, n_samples, ax, i, color)

    ax.scatter(
        para_true[:1],
        para_true[-1:],
        marker="s",
        # markersize=8,
        s=64,
        color="xkcd:blue",
        label=r"$(\mu^*, \sigma^*)$",
        zorder=10,
    )
    ax.scatter(
        para_history[0, :1],
        para_history[0, -1:],
        marker="o",
        # markersize=8,
        s=64,
        color="red",
        label=r"$(\mu_0, \sigma_0)$",
        zorder=10,
    )
    ax = format_axes(
        ax,
        title=rf"Student's $t$-distribution ($\nu={k})$",
        xlabel=r"$\mu_k$",
        ylabel=r"$\sigma_k$",
        leg_loc="best",
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.tight_layout()
    plt.subplots_adjust(left=0.14, bottom=0.13, top=0.91)
    if save:
        name = f"tdist_k{k}_mu{mu_true}_sigma{sigma_true}_paths{num_paths}"
        plt.savefig(f"{name}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    fire.Fire(main)
