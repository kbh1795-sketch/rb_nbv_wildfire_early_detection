#!/usr/bin/env python3

import csv
import time
import numpy as np
from pathlib import Path
from multiprocessing import get_context

import rh_nbv_v3 as sim


# ============================================================
# Settings
# ============================================================

R_OBS = 5
H = 5
K_VALUES = [100, 300, 500]

TERRAIN_COUNT = 10
TRIALS_PER_TERRAIN = 50

WORKERS = 6

CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./results_k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Run one K condition and return individual detection steps
# ============================================================

def evaluate_steps(fires, vis, H, K, sim_seed, workers):

    ctx = get_context("fork")

    with ctx.Pool(
        processes=workers,
        initializer=sim._worker_init,
        initargs=(
            vis,
            "rh-nbv",
            H,
            K,
            sim_seed
        )
    ) as pool:

        steps = list(
            pool.imap(
                sim._run_trial,
                enumerate(fires),
                chunksize=max(
                    1,
                    len(fires) // (workers * 8)
                )
            )
        )

    return np.asarray(steps, dtype=int)


# ============================================================
# Wilson 95% confidence interval for success probability
# ============================================================

def wilson_ci(successes, n, z=1.96):

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1 + z**2 / n

    center = (
        p + z**2 / (2 * n)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return (
        100 * (center - half_width),
        100 * (center + half_width)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("RH-NBV K Sensitivity Analysis")
    print(
        f"Terrains={TERRAIN_COUNT}, "
        f"fires/terrain={TRIALS_PER_TERRAIN}, "
        f"R_obs={R_OBS}, H={H}"
    )
    print(f"K values = {K_VALUES}")
    print("=" * 70)

    # Store every individual result for exact aggregate statistics
    all_results = {
        K: []
        for K in K_VALUES
    }

    terrain_rows = []

    total_start = time.time()

    # --------------------------------------------------------
    # 10 terrains
    # --------------------------------------------------------

    for terrain_idx in range(TERRAIN_COUNT):

        terrain_seed = sim.TERRAIN_SEED + terrain_idx

        # Same rule used in multi-terrain experiment
        fire_seed = (
            sim.SIM_SEED
            + 10000 * terrain_idx
        )

        print()
        print(
            f"[Terrain {terrain_idx + 1}/{TERRAIN_COUNT}] "
            f"terrain seed={terrain_seed}, "
            f"fire seed={fire_seed}"
        )

        # Identical 50 fire locations for all K conditions
        fires = sim.generate_fire_cells(
            TRIALS_PER_TERRAIN,
            fire_seed
        )

        # LOS visibility map
        vis = sim.get_visibility(
            terrain_seed,
            R_OBS,
            CACHE_DIR
        )

        # ----------------------------------------------------
        # K = 100 / 300 / 500
        # ----------------------------------------------------

        for K in K_VALUES:

            t0 = time.time()

            steps = evaluate_steps(
                fires=fires,
                vis=vis,
                H=H,
                K=K,
                sim_seed=fire_seed,
                workers=WORKERS
            )

            all_results[K].extend(
                steps.tolist()
            )

            success_mask = steps >= 0

            successes = int(
                success_mask.sum()
            )

            failures = (
                TRIALS_PER_TERRAIN
                - successes
            )

            success_rate = (
                100.0
                * successes
                / TRIALS_PER_TERRAIN
            )

            if successes > 0:
                mean_step = float(
                    steps[success_mask].mean()
                )
            else:
                mean_step = np.nan

            elapsed = time.time() - t0

            terrain_rows.append({
                "terrain": terrain_idx + 1,
                "terrain_seed": terrain_seed,
                "fire_seed": fire_seed,
                "K": K,
                "N": TRIALS_PER_TERRAIN,
                "successes": successes,
                "failures": failures,
                "success_rate_pct": success_rate,
                "mean_detection_step": mean_step,
                "elapsed_sec": elapsed
            })

            print(
                f"  K={K:<3} | "
                f"success={success_rate:5.1f}% | "
                f"mean step={mean_step:7.1f} | "
                f"fail={failures:2d} | "
                f"time={elapsed:6.1f}s"
            )

    # ========================================================
    # Aggregate results over all 10 terrains = 500 trials / K
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    summary_rows = []

    for K in K_VALUES:

        arr = np.asarray(
            all_results[K],
            dtype=int
        )

        success_mask = arr >= 0

        N = len(arr)

        successes = int(
            success_mask.sum()
        )

        failures = int(
            (~success_mask).sum()
        )

        success_rate = (
            100.0
            * successes
            / N
        )

        successful_steps = (
            arr[success_mask]
        )

        if successes > 0:

            mean_step = float(
                successful_steps.mean()
            )

            std_step = float(
                successful_steps.std(
                    ddof=1
                )
            )

            se_step = (
                std_step
                / np.sqrt(successes)
            )

            step_ci_low = (
                mean_step
                - 1.96 * se_step
            )

            step_ci_high = (
                mean_step
                + 1.96 * se_step
            )

        else:

            mean_step = np.nan
            std_step = np.nan
            step_ci_low = np.nan
            step_ci_high = np.nan

        success_ci_low, success_ci_high = (
            wilson_ci(
                successes,
                N
            )
        )

        summary_rows.append({
            "K": K,
            "N": N,
            "successes": successes,
            "failures": failures,
            "success_rate_pct": success_rate,
            "success_CI95_low": success_ci_low,
            "success_CI95_high": success_ci_high,
            "mean_detection_step": mean_step,
            "step_CI95_low": step_ci_low,
            "step_CI95_high": step_ci_high
        })

        print(
            f"K={K}"
        )

        print(
            f"  Success rate      : "
            f"{success_rate:.1f}% "
            f"({successes}/{N})"
        )

        print(
            f"  95% CI success    : "
            f"[{success_ci_low:.1f}%, "
            f"{success_ci_high:.1f}%]"
        )

        print(
            f"  Mean Detection Step: "
            f"{mean_step:.1f}"
        )

        print(
            f"  95% CI mean step  : "
            f"[{step_ci_low:.1f}, "
            f"{step_ci_high:.1f}]"
        )

        print(
            f"  Failures          : "
            f"{failures}"
        )

        print()

    # ========================================================
    # Save per-terrain CSV
    # ========================================================

    terrain_csv = (
        OUTPUT_DIR
        / "k_sensitivity_by_terrain.csv"
    )

    with open(
        terrain_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
            terrain_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            terrain_rows
        )

    # ========================================================
    # Save overall summary CSV
    # ========================================================

    summary_csv = (
        OUTPUT_DIR
        / "k_sensitivity_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
            summary_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print("=" * 70)

    print(
        f"Saved:\n"
        f"  {terrain_csv}\n"
        f"  {summary_csv}"
    )

    print(
        f"Total elapsed time: "
        f"{time.time() - total_start:.1f}s"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()