#!/usr/bin/env python3
"""
Re-run the revision experiments consistently on 10 terrains.

Table 5:
    R_obs in {1, 3, 5, 7}
    H = 5, K = 300
    10 terrains x 50 fire locations
    Random / Lawn-mower / Greedy NBV / RH-NBV

Table 6:
    H in {1, 3, 5, 7}
    R_obs = 5, K = 300
    10 terrains x 50 fire locations
    RH-NBV only

This script imports the existing reconstructed simulator:
    rh_nbv_mp_mountain_v3.py

Put both files in the same directory and run:
    python revision_table5_table6.py
"""

import csv
import time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

import rh_nbv_v3 as sim


# ============================================================
# Settings
# ============================================================

TERRAIN_COUNT = 10
TRIALS_PER_TERRAIN = 50
WORKERS = 6

R_VALUES = [1, 3, 5, 7]
H_VALUES = [1, 3, 5, 7]

BASE_H = 5
BASE_K = 300
BASE_R = 5

METHODS = [
    ("random", "Random Search"),
    ("lawnmower", "Lawn-mower Search"),
    ("greedy", "Greedy NBV"),
    ("rh-nbv", "RH-NBV"),
]

CACHE_DIR = Path("./cache_revision")
OUTPUT_DIR = Path("./results_revision")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Trial runner
# ============================================================

def run_steps(method, fires, vis, H, K, sim_seed, workers):
    """
    Return one Detection Step per fire.
    -1 means detection failure within TMAX.
    """
    ctx = get_context("fork")

    with ctx.Pool(
        processes=workers,
        initializer=sim._worker_init,
        initargs=(vis, method, H, K, sim_seed),
    ) as pool:
        steps = list(
            pool.imap(
                sim._run_trial,
                enumerate(fires),
                chunksize=max(1, len(fires) // (workers * 8)),
            )
        )

    return np.asarray(steps, dtype=int)


def summarize_steps(steps):
    steps = np.asarray(steps, dtype=int)
    ok = steps >= 0

    successes = int(ok.sum())
    failures = int((~ok).sum())
    n = len(steps)

    return {
        "N": n,
        "successes": successes,
        "failures": failures,
        "success_rate_pct": 100.0 * successes / n,
        "mean_detection_step": (
            float(steps[ok].mean()) if successes > 0 else np.nan
        ),
    }


def aggregate_across_terrains(terrain_rows, raw_steps):
    """
    terrain_rows: list of per-terrain result dicts for one condition
    raw_steps: all individual trials pooled across the 10 terrains

    The terrain-to-terrain standard deviation uses sample SD (ddof=1),
    matching the K-sensitivity table already used in the manuscript.
    """
    terrain_rates = np.asarray(
        [r["success_rate_pct"] for r in terrain_rows],
        dtype=float,
    )

    raw_steps = np.asarray(raw_steps, dtype=int)
    ok = raw_steps >= 0

    return {
        "N": len(raw_steps),
        "successes": int(ok.sum()),
        "failures": int((~ok).sum()),
        "mean_success_rate_pct": float(terrain_rates.mean()),
        "terrain_min_success_pct": float(terrain_rates.min()),
        "terrain_max_success_pct": float(terrain_rates.max()),
        "terrain_sd_success_pct_point": float(
            terrain_rates.std(ddof=1)
        ),
        "mean_detection_step": (
            float(raw_steps[ok].mean()) if ok.any() else np.nan
        ),
    }


# ============================================================
# TABLE 5
# R_obs sensitivity across all four methods
# ============================================================

def run_table5():
    print()
    print("=" * 78)
    print("TABLE 5: R_obs sensitivity")
    print(
        f"Terrains={TERRAIN_COUNT}, "
        f"fires/terrain={TRIALS_PER_TERRAIN}, "
        f"H={BASE_H}, K={BASE_K}"
    )
    print(f"R_obs values = {R_VALUES}")
    print("=" * 78)

    per_terrain_rows = []
    summary_rows = []

    # Pre-generate identical fire locations for each terrain.
    fires_by_terrain = {}
    for terrain_idx in range(TERRAIN_COUNT):
        fire_seed = sim.SIM_SEED + 10000 * terrain_idx
        fires_by_terrain[terrain_idx] = (
            fire_seed,
            sim.generate_fire_cells(
                TRIALS_PER_TERRAIN,
                fire_seed,
            ),
        )

    for R in R_VALUES:
        print()
        print(f"[R_obs = {R}]")

        # Collect pooled raw steps separately for every method.
        pooled_steps = {
            method: []
            for method, _ in METHODS
        }

        # Collect terrain-level statistics separately for every method.
        condition_rows = {
            method: []
            for method, _ in METHODS
        }

        for terrain_idx in range(TERRAIN_COUNT):
            terrain_no = terrain_idx + 1
            terrain_seed = sim.TERRAIN_SEED + terrain_idx
            fire_seed, fires = fires_by_terrain[terrain_idx]

            print(
                f"  Terrain {terrain_no:02d}/{TERRAIN_COUNT} "
                f"(terrain seed={terrain_seed})"
            )

            vis = sim.get_visibility(
                terrain_seed,
                R,
                CACHE_DIR,
            )

            for method, display_name in METHODS:
                t0 = time.time()

                steps = run_steps(
                    method=method,
                    fires=fires,
                    vis=vis,
                    H=BASE_H,
                    K=BASE_K,
                    sim_seed=fire_seed,
                    workers=WORKERS,
                )

                stats = summarize_steps(steps)

                row = {
                    "R_obs": R,
                    "terrain": terrain_no,
                    "terrain_seed": terrain_seed,
                    "fire_seed": fire_seed,
                    "method": display_name,
                    **stats,
                    "elapsed_sec": time.time() - t0,
                }

                per_terrain_rows.append(row)
                condition_rows[method].append(row)
                pooled_steps[method].extend(steps.tolist())

                print(
                    f"    {display_name:<20} | "
                    f"success={stats['success_rate_pct']:5.1f}% | "
                    f"mean step={stats['mean_detection_step']:7.1f} | "
                    f"fail={stats['failures']:2d}"
                )

        print()
        print(f"  --- R_obs={R} aggregate ---")

        for method, display_name in METHODS:
            agg = aggregate_across_terrains(
                condition_rows[method],
                pooled_steps[method],
            )

            summary_row = {
                "R_obs": R,
                "method": display_name,
                **agg,
            }
            summary_rows.append(summary_row)

            print(
                f"    {display_name:<20} | "
                f"mean success={agg['mean_success_rate_pct']:5.1f}% | "
                f"range={agg['terrain_min_success_pct']:4.1f}"
                f"~{agg['terrain_max_success_pct']:4.1f}% | "
                f"SD={agg['terrain_sd_success_pct_point']:4.1f}%p | "
                f"mean step={agg['mean_detection_step']:7.1f} | "
                f"fail={agg['failures']:3d}"
            )

    return per_terrain_rows, summary_rows


# ============================================================
# TABLE 6
# H sensitivity for RH-NBV
# ============================================================

def run_table6():
    print()
    print("=" * 78)
    print("TABLE 6: H sensitivity")
    print(
        f"Terrains={TERRAIN_COUNT}, "
        f"fires/terrain={TRIALS_PER_TERRAIN}, "
        f"R_obs={BASE_R}, K={BASE_K}"
    )
    print(f"H values = {H_VALUES}")
    print("=" * 78)

    per_terrain_rows = []
    summary_rows = []

    # Pre-generate same fire locations used above.
    fires_by_terrain = {}
    for terrain_idx in range(TERRAIN_COUNT):
        fire_seed = sim.SIM_SEED + 10000 * terrain_idx
        fires_by_terrain[terrain_idx] = (
            fire_seed,
            sim.generate_fire_cells(
                TRIALS_PER_TERRAIN,
                fire_seed,
            ),
        )

    for H in H_VALUES:
        print()
        print(f"[H = {H}]")

        pooled_steps = []
        condition_rows = []

        for terrain_idx in range(TERRAIN_COUNT):
            terrain_no = terrain_idx + 1
            terrain_seed = sim.TERRAIN_SEED + terrain_idx
            fire_seed, fires = fires_by_terrain[terrain_idx]

            print(
                f"  Terrain {terrain_no:02d}/{TERRAIN_COUNT} "
                f"(terrain seed={terrain_seed})"
            )

            # R=5 visibility will normally already be cached after Table 5.
            vis = sim.get_visibility(
                terrain_seed,
                BASE_R,
                CACHE_DIR,
            )

            t0 = time.time()

            steps = run_steps(
                method="rh-nbv",
                fires=fires,
                vis=vis,
                H=H,
                K=BASE_K,
                sim_seed=fire_seed,
                workers=WORKERS,
            )

            stats = summarize_steps(steps)

            row = {
                "H": H,
                "terrain": terrain_no,
                "terrain_seed": terrain_seed,
                "fire_seed": fire_seed,
                "method": "RH-NBV",
                **stats,
                "elapsed_sec": time.time() - t0,
            }

            per_terrain_rows.append(row)
            condition_rows.append(row)
            pooled_steps.extend(steps.tolist())

            print(
                f"    success={stats['success_rate_pct']:5.1f}% | "
                f"mean step={stats['mean_detection_step']:7.1f} | "
                f"fail={stats['failures']:2d}"
            )

        agg = aggregate_across_terrains(
            condition_rows,
            pooled_steps,
        )

        summary_row = {
            "H": H,
            "method": "RH-NBV",
            **agg,
        }
        summary_rows.append(summary_row)

        print()
        print(
            f"  H={H} aggregate | "
            f"mean success={agg['mean_success_rate_pct']:5.1f}% | "
            f"range={agg['terrain_min_success_pct']:4.1f}"
            f"~{agg['terrain_max_success_pct']:4.1f}% | "
            f"SD={agg['terrain_sd_success_pct_point']:4.1f}%p | "
            f"mean step={agg['mean_detection_step']:7.1f} | "
            f"fail={agg['failures']:3d}"
        )

    return per_terrain_rows, summary_rows


# ============================================================
# CSV helpers
# ============================================================

def save_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    total_start = time.time()

    table5_terrain, table5_summary = run_table5()

    save_csv(
        OUTPUT_DIR / "table5_Robs_by_terrain.csv",
        table5_terrain,
    )
    save_csv(
        OUTPUT_DIR / "table5_Robs_summary.csv",
        table5_summary,
    )

    table6_terrain, table6_summary = run_table6()

    save_csv(
        OUTPUT_DIR / "table6_H_by_terrain.csv",
        table6_terrain,
    )
    save_csv(
        OUTPUT_DIR / "table6_H_summary.csv",
        table6_summary,
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Saved:")
    print("  results_revision/table5_Robs_by_terrain.csv")
    print("  results_revision/table5_Robs_summary.csv")
    print("  results_revision/table6_H_by_terrain.csv")
    print("  results_revision/table6_H_summary.csv")
    print(
        f"Total elapsed time: "
        f"{time.time() - total_start:.1f}s"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
