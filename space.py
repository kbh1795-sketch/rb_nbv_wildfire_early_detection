import numpy as np
import matplotlib.pyplot as plt

GRID = 100


def generate_default_terrain(seed):
    rng = np.random.default_rng(seed)

    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)

    Z = (
        6.0 * np.sin(X / 7.0)
        + 5.0 * np.cos(Y / 9.0)
        + 3.0 * np.sin((X + Y) / 4.0)
    )

    for _ in range(9):
        mux = rng.uniform(0, GRID)
        muy = rng.uniform(0, GRID)
        sigma = rng.uniform(7, 20)
        A = rng.uniform(18, 55)

        Z += A * np.exp(
            -((X - mux) ** 2 + (Y - muy) ** 2) / (2 * sigma ** 2)
        )

    Z -= Z.min()
    Z = 100.0 * Z / Z.max()

    return Z


def elliptical_gaussian(X, Y, cx, cy, amp, sx, sy, theta_deg):
    theta = np.deg2rad(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)

    xr = ct * (X - cx) + st * (Y - cy)
    yr = -st * (X - cx) + ct * (Y - cy)

    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def generate_terrain4_special():
    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)

    Z = (
        5.0 * np.sin(X / 11.0)
        + 4.0 * np.cos(Y / 13.0)
        + 2.5 * np.sin((X + 0.7 * Y) / 8.0)
    )

    Z += elliptical_gaussian(X, Y, 18, 13, 85, 11, 8, 18)
    Z += elliptical_gaussian(X, Y, 29, 25, 55, 13, 7, 42)
    Z += elliptical_gaussian(X, Y, 43, 39, 52, 24, 6, 50)
    Z += elliptical_gaussian(X, Y, 58, 55, 42, 25, 7, 48)
    Z += elliptical_gaussian(X, Y, 78, 67, 48, 19, 12, 20)
    Z += elliptical_gaussian(X, Y, 88, 78, 32, 12, 9, -10)

    Z -= elliptical_gaussian(X, Y, 78, 24, 48, 19, 10, -12)
    Z -= elliptical_gaussian(X, Y, 18, 67, 26, 16, 12, 20)

    Z += elliptical_gaussian(X, Y, 41, 42, 23, 6, 5, 0)
    Z += elliptical_gaussian(X, Y, 63, 63, 19, 7, 5, 35)

    Z -= Z.min()
    Z = 100.0 * Z / Z.max()

    return Z


def generate_terrain7_special():
    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)

    Z = (
        4.5 * np.sin(X / 10.0)
        + 4.0 * np.cos(Y / 12.0)
        + 3.0 * np.sin((0.8 * X + Y) / 7.5)
    )

    Z += elliptical_gaussian(X, Y, 23, 74, 78, 12, 9, -20)
    Z += elliptical_gaussian(X, Y, 38, 62, 55, 17, 7, -35)
    Z += elliptical_gaussian(X, Y, 53, 52, 48, 28, 6, -32)
    Z += elliptical_gaussian(X, Y, 69, 43, 38, 22, 6, -28)
    Z += elliptical_gaussian(X, Y, 82, 22, 58, 12, 10, 8)
    Z += elliptical_gaussian(X, Y, 72, 30, 31, 10, 7, 20)

    Z -= elliptical_gaussian(X, Y, 18, 28, 36, 16, 10, -5)
    Z -= elliptical_gaussian(X, Y, 77, 75, 32, 18, 13, 15)

    Z += elliptical_gaussian(X, Y, 49, 55, 20, 6, 4, -20)
    Z += elliptical_gaussian(X, Y, 83, 19, 16, 5, 4, 5)

    Z -= Z.min()
    Z = 100.0 * Z / Z.max()

    return Z


def generate_terrain(seed):
    # Terrain 4: seed 20260828
    if seed == 20260828:
        return generate_terrain4_special()

    # Terrain 7: seed 20260831
    if seed == 20260831:
        return generate_terrain7_special()

    return generate_default_terrain(seed)


def main():
    seeds = list(range(20260825, 20260835))
    terrains = [generate_terrain(seed) for seed in seeds]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(terrains[i], origin="lower", cmap="terrain")
        ax.set_title(f"Terrain {i+1}\n(seed={seeds[i]})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Normalized Elevation")

    plt.savefig("terrain_10_examples.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()