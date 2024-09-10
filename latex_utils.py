import matplotlib
from math import sqrt


SPINE_COLOR = "black"


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.487 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 100
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + str(fig_height)
            + "so will reduce to"
            + str(MAX_HEIGHT_INCHES)
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES
    # , '\usepackage{amsmath, amsfonts}',
    params = {
        "backend": "QtAgg",
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{gensymb}",
                r"\usepackage{amsfonts,dsfont,mathrsfs}",
                r"\usepackage{amssymb,amsthm,amscd,empheq,amsmath}",
                r"\usepackage{mathtools,mathbbol,bm,bbm,nicefrac,scalerel}",
                r"\def\mbf#1{\mathbf{#1}}",
            ]
        ),
        "axes.labelsize": 14,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 18,
        "font.size": 12,  # was 10
        "legend.fontsize": 10,  # was 10
        "legend.shadow": False,
        "legend.fancybox": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "font.serif": "cm",
        "mathtext.fontset": "cm",
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2,
        "lines.markersize": 5,
    }

    matplotlib.rcParams.update(params)


def format_axes(
    ax, title=None, xlabel=None, ylabel=None, leg_loc=None, leg_title=None, grid=None
):
    for spine in ["top", "right"]:
        # ax.spines[spine].set_visible(False)
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.7)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.7)

    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=0.4)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=3.0)
    if leg_loc is not None:
        ax.legend(loc=leg_loc, title=leg_title)
    if grid is not None:
        ax.grid(grid, lw=0.3)
    return ax
