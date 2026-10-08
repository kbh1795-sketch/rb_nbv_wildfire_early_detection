import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Data
# ============================================================

terrains = np.arange(1, 11)

data = {
    "Random Search": {
        "success": [
            12.0, 6.0, 8.0, 2.0, 6.0,
            0.0, 6.0, 8.0, 4.0, 6.0
        ],
        "step": [
            601.8, 388.3, 557.2, 465.0, 744.0,
            np.nan, 639.0, 588.8, 702.0, 532.3
        ],
        "fail": [
            44, 47, 46, 49, 47,
            50, 47, 46, 48, 47
        ],
    },

    "Lawn-mower Search": {
        "success": [
            12.0, 16.0, 12.0, 14.0, 16.0,
            4.0, 12.0, 14.0, 18.0, 14.0
        ],
        "step": [
            199.7, 306.5, 386.7, 466.4, 352.0,
            155.0, 253.0, 400.3, 468.1, 383.4
        ],
        "fail": [
            44, 42, 44, 43, 42,
            48, 44, 43, 41, 43
        ],
    },

    "Greedy NBV": {
        "success": [
            66.0, 64.0, 58.0, 50.0, 56.0,
            64.0, 80.0, 72.0, 70.0, 70.0
        ],
        "step": [
            287.4, 224.2, 231.3, 283.0, 277.6,
            272.8, 326.9, 212.6, 288.8, 272.9
        ],
        "fail": [
            17, 18, 21, 25, 22,
            18, 10, 14, 15, 15
        ],
    },

    "RH-NBV": {
        "success": [
            78.0, 86.0, 90.0, 84.0, 72.0,
            84.0, 88.0, 74.0, 80.0, 84.0
        ],
        "step": [
            305.2, 329.3, 300.3, 301.2, 327.9,
            316.8, 258.5, 374.6, 309.4, 374.6
        ],
        "fail": [
            11, 7, 5, 8, 14,
            8, 6, 13, 10, 8
        ],
    }
}

methods = list(data.keys())


# ============================================================
# Figure setup
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8.27, 11.69),
    sharex=True
)


# ============================================================
# Plot helper
# ============================================================

def plot_metric(ax, metric, ylabel):

    for method in methods:
        ax.plot(
            terrains,
            data[method][metric],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=method
        )

    ax.set_ylabel(
        ylabel,
        fontsize=11
    )

    ax.set_xticks(terrains)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        alpha=0.7
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )


# ============================================================
# Success rate
# ============================================================

plot_metric(
    axes[0],
    "success",
    "Detection Success Rate (%)"
)

axes[0].set_ylim(0, 100)


# ============================================================
# Mean Detection Step
# ============================================================

plot_metric(
    axes[1],
    "step",
    "Mean Detection Step"
)


# ============================================================
# Failure count
# ============================================================

plot_metric(
    axes[2],
    "fail",
    "Number of Failures"
)

axes[2].set_ylim(0, 52)

axes[2].set_xlabel(
    "Virtual Terrain",
    fontsize=11
)

axes[2].set_xticks(terrains)

axes[2].set_xticklabels(
    [str(i) for i in terrains]
)


# ============================================================
# Shared legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, 0.985),
    fontsize=10
)


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.94,
    bottom=0.07,
    hspace=0.18
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "table4_fullpage_figure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "table4_fullpage_figure.pdf",
    bbox_inches="tight"
)

plt.show()