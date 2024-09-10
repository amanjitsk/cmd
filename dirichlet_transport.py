import fire
from functools import partial
from jax import config

config.update("jax_enable_x64", True)
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jnn, random as jrd, lax
from jax import grad, jit, vmap, jacfwd
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.ticker import FormatStrFormatter
from latex_utils import latexify, format_axes


def ot_map(p, alpha):
    """The optimal transport map q := T(p)"""
    q_un = p ** (1 - alpha)
    return q_un / q_un.sum()


def ot_inv(q, alpha):
    """The inverse optimal transport map p := T^{-1}(q)"""
    p_un = q ** (1 / (1 - alpha))
    return p_un / p_un.sum()


def portfolio(p, alpha):
    """Compute the portfolio map (p^-1) = (r) using the
    diversity weighted portfolio parameterized by alpha.
    """
    w = p ** (-alpha)
    return w / w.sum()


def dirichlet_cost(p, q):
    """Dirichlet cost on the simplex"""
    return jnp.log(jnp.mean(p / q)) - jnp.mean(jnp.log(p / q))


@partial(jit, static_argnums=(2,))
@partial(vmap, in_axes=(0, None, None, None))
def logarithmic_descent(p, alpha, grad, lr):
    """Logarithmic descent with 𝝺 = 1 (Dirichlet transport special case)"""
    dder = lambda x: grad(x) - jnp.dot(x, grad(x))
    q = ot_map(p, alpha)  # dual
    port = portfolio(p, alpha)
    q_new = q + lr * (-q / (port * p)) * (
        p**2 * dder(p) - q * jnp.dot(p**2, dder(p))
    )  # gradient flow on q
    p_new = ot_inv(q_new, alpha)  # back to primal
    return p_new


@partial(jit, static_argnums=1)
@partial(vmap, in_axes=(0, None, None))
def uwp_descent(p, grad, lr):
    """Conformal descent using the uniformly weighted portfolio"""
    dder = grad(p) - jnp.dot(p, grad(p))
    return jnn.softmax(jnp.log(p) - lr * p * dder)


@partial(jit, static_argnums=1)
@partial(vmap, in_axes=(0, None, None))
def bregman_descent(p, grad, lr):
    """Entropic descent on the primal space"""
    return jnn.softmax(jnp.log(p) - lr * grad(p))


@jit
@partial(vmap, in_axes=(0, None))
def rate(x, xmin):
    return jnp.log10(dirichlet_cost(x, xmin))


def minimize(
    key: jax.Array,
    wstar: jax.Array,
    alphas: jax.Array,
    batch: int,
    iters: int,
):
    """Minimize dirichlet cost to wstar and return convergence rates"""
    xmin = wstar
    # define function and its gradient
    f = partial(dirichlet_cost, q=wstar)
    fgrad = jit(grad(f))
    f = jit(f)

    # initial values
    d = wstar.size
    init_lr = jnp.array(1.0 / d)
    # p_ent = jrd.uniform(key, (batch, d))
    # p_ent /= jnp.sum(p_ent, axis=1, keepdims=True)
    p_ent = jrd.dirichlet(key, jnp.full(d, 10.0), shape=(batch,))
    logdict = dict()
    for alpha in alphas:
        logdict[alpha] = jnp.array(p_ent)

    def scan_fn(carry, step):
        lr = init_lr * jnp.sqrt(1 / step)
        # xe, xb, yb, xl, yl = carry
        p_ent, logdict = carry
        new_p_ent = bregman_descent(p_ent, fgrad, lr)
        for alpha in alphas:
            logdict[alpha] = logarithmic_descent(logdict[alpha], alpha, fgrad, lr)

        new_carry = (new_p_ent, logdict)
        # y = jnp.vstack([rate(f, x, xmin) for x in (new_xe, new_xb, new_xl)])
        bregman_rate = rate(new_p_ent, xmin)
        logarithmic_rates = dict(
            (alpha, rate(logdict[alpha], xmin)) for alpha in alphas
        )
        y = (bregman_rate, logarithmic_rates)
        return new_carry, y

    init_carry = (p_ent, logdict)
    last_state, rates = lax.scan(scan_fn, init_carry, jnp.arange(1, iters + 1))
    # rates = rates
    return last_state, rates


def plot_band(ax, L, **kwargs):
    """Helper function to plot rates of convergence in batches
    Args:
        L: List[np.ndarray(N)]
    """
    L = np.asarray(L)  # NUM_ITERSxB
    x = np.arange(L.shape[0])
    y_vals = np.nanmean(L, axis=-1)  # shape NUM_ITERS
    y_error = np.nanstd(L, axis=-1)  # shape NUM_ITERS
    ax.plot(x, y_vals, **kwargs)
    kwargs.pop("label", None)
    kwargs.pop("marker", None)
    ax.fill_between(x, y_vals - y_error, y_vals + y_error, alpha=0.1, **kwargs)
    return ax


def main(name, save=False, **kwargs):
    batch = kwargs.get("batch", 12)
    iters = kwargs.get("iters", 100)
    seed = kwargs.get("seed", 0)
    key = jrd.PRNGKey(seed)

    alphas = np.linspace(0.0, 0.9, 10)
    # alphas = np.around(np.linspace(0.0, 1.0, 11), 1)
    cmap = plt.get_cmap("cool")
    cNorm = colors.Normalize(vmin=0, vmax=alphas[-1].item())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap.set_array(alphas)

    wstar_list = []
    # random optimal point w = p* (on the simplex)
    d = 50
    if name == "center":
        w = jnp.full(d, 1 / d)  # barycenter
    elif name == "dirichlet-1":
        w = jrd.dirichlet(key, jnp.full(d, 1e0), ())  # dirichlet
    elif name == "dirichlet-2":
        w = jrd.dirichlet(key, jnp.full(d, 5e0), ())  # dirichlet
    elif name == "random":
        w = jrd.dirichlet(key, jnp.arange(d, 0.0, -1), ())  # dirichlet
        # w = jnp.sort(jrd.uniform(key, (d,)))  # random
        # w = w / w.sum()  # random
    else:
        raise NotImplementedError(f"{name} is unknown!")

    w = jnp.sort(w)
    wstar_list.append(w)

    latexify(7.25, 3.25 * len(wstar_list))
    fig, axes = plt.subplots(nrows=len(wstar_list), ncols=2)
    axes = axes.reshape(len(wstar_list), 2)
    for i, wstar in enumerate(wstar_list):
        key, subkey = jrd.split(key)
        d = wstar.size
        (p_ent, p_div), (bregman_rates, logarithmic_rates) = minimize(
            subkey, wstar, alphas, batch, iters
        )
        # print("p_ent: ", p_ent)
        # print("p_div: ", p_div)
        # print("Bregman rate: ", bregman_rates)
        # print("Logarithmic rates: ", logarithmic_rates)
        ax1, ax2 = axes[i]
        ax1 = plot_band(
            ax1, bregman_rates, label="Entropic", color="black", lw=3, zorder=5
        )
        for alpha in alphas:
            # color = scalarMap.to_rgba(alpha.item())
            color = cmap(cNorm(alpha))
            ax1 = plot_band(
                ax1,
                logarithmic_rates[alpha],
                # label=fr"$\alpha = {alpha:.1f}$",
                color=color,
            )
        ax1.set_xscale("log")
        ax1 = format_axes(
            ax1,
            title=rf"$d = {d}$",
            xlabel=r"Iteration $k$",
            ylabel=r"$\log_{10} c(p_k, p^*)$",
            leg_loc="best",
        )
        ax1.grid(alpha=0.2)

        ax2.bar(
            np.arange(d),
            wstar,
            color="grey",
            label=r"$p^*$",
            alpha=0.35,
        )
        ax2.errorbar(
            np.arange(d),
            p_ent.mean(0),
            yerr=p_ent.std(0),
            fmt="o",
            label="Entropic",
            elinewidth=0.5,
            color="black",
        )
        if name in {"dirichlet-1", "dirichlet-2"}:
            random_alpha = jrd.choice(subkey, alphas).item()  # choose smaller alpha
        else:
            random_alpha = jrd.choice(key, alphas[5:]).item()  # choose larger alpha
        final_batch = p_div[random_alpha]
        ax2.errorbar(
            np.arange(d),
            np.nanmean(final_batch, 0),
            yerr=np.nanstd(final_batch, 0),
            fmt="o",
            elinewidth=0.5,
            label=rf"Conformal $(\alpha = {random_alpha:.1f})$",
            color=cmap(cNorm(random_alpha)),
        )
        ax2 = format_axes(
            ax2,
            title="Sorted coordinates",
            xlabel=r"Coordinate $i$",
            ylabel=r"$p_{(\text{final})}^i$",
            leg_loc="best",
        )
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax2.grid(alpha=0.2)

    cax = fig.add_axes([0.125, 0.325, 0.2, 0.025])
    clb = fig.colorbar(scalarMap, cax=cax, orientation="horizontal")
    clb.ax.set_title(r"$\alpha$")
    clb.set_ticks([0.0, 0.9])
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, top=0.9, wspace=0.24)
    if save:
        plt.savefig(f"dc_{name}.pdf")
    # plt.savefig("dc_center.pdf")
    # plt.savefig("dc_dirichlet-1.pdf")
    # plt.savefig("dc_dirichlet-2.pdf")
    # plt.savefig("dc_random.pdf")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
