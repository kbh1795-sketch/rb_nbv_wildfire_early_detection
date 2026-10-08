#!/usr/bin/env python3
"""
Reconstructed simulator for:
"산불 조기감지를 위한 UAV의 Receding-Horizon Next-Best-View 관측계획 연구"

This is a reconstruction from the submitted manuscript, not the lost original source.

Recovered manuscript settings
-----------------------------
- Grid: 100 x 100
- Terrain:
    Z = 6 sin(x/7) + 5 cos(y/9) + 3 sin((x+y)/4)
    + 9 Gaussian components
    Gaussian sigma ~ U[7,20], amplitude ~ U[18,55]
    normalize Z to [0,100]
- terrain seed: 20260825
- simulation/fire seed: 20260824
- UAV start: (50,50), inferred from the manuscript figure
- h_UAV = terrain(current)+5
- target-cell reference height = terrain(target)+0.5
- 8-connected motion, one cell per step
- fire cell: Chebyshev distance 30..45 from start
- Tmax = 1000
- baseline RH-NBV: H=5, K=300
- observation radius R_obs in {1,3,5,7}
- 500 repeated trials
- RH-NBV reward: number of previously-unobserved LOS-visible cells
  over the union of the H-step candidate path
- equal-reward ties: random choice

Implementation details not explicitly fixed by the manuscript
-------------------------------------------------------------
1) LOS rasterisation:
   This reconstruction linearly samples intermediate grid cells and blocks LOS
   when terrain height exceeds the interpolated sight-line height.
2) Random streams:
   Trial-specific streams are derived deterministically from SIM_SEED.
3) Lawn-mower traversal:
   Implemented as a deterministic serpentine coverage path.
4) The manuscript figures contain a "Spiral" legend not defined in Table 3.
   Spiral is intentionally excluded from the four-method benchmark here.

These points should be documented if exact numerical equality with the old
tables is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GRID = 100
START = (50, 50)  # (x, y), inferred from Fig. 2
TERRAIN_SEED = 20260825
SIM_SEED = 20260824
H_UAV = 5.0
TARGET_HEIGHT = 0.5
TMAX = 1000

DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
)


def generate_terrain(seed: int = TERRAIN_SEED) -> np.ndarray:
    """Generate the manuscript's 100x100 synthetic terrain."""
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
        mux = rng.uniform(0.0, GRID)
        muy = rng.uniform(0.0, GRID)
        sigma = rng.uniform(7.0, 20.0)
        amp = rng.uniform(18.0, 55.0)
        Z += amp * np.exp(
            -((X - mux) ** 2 + (Y - muy) ** 2) / (2.0 * sigma ** 2)
        )

    Z -= Z.min()
    Z *= 100.0 / Z.max()
    return Z


def build_neighbors() -> List[List[Tuple[int, int]]]:
    out: List[List[Tuple[int, int]]] = [None] * (GRID * GRID)  # type: ignore
    for y in range(GRID):
        for x in range(GRID):
            out[y * GRID + x] = [
                (x + dx, y + dy)
                for dx, dy in DIRS
                if 0 <= x + dx < GRID and 0 <= y + dy < GRID
            ]
    return out


NEIGHBORS = build_neighbors()


def los_clear(
    Z: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> bool:
    """
    Terrain-aware LOS.

    UAV endpoint height:
        Z[y0,x0] + H_UAV
    Target endpoint height:
        Z[y1,x1] + TARGET_HEIGHT

    Intermediate cells are sampled along the line segment.
    """
    if x0 == x1 and y0 == y1:
        return True

    dx = x1 - x0
    dy = y1 - y0
    n = max(abs(dx), abs(dy))

    z0 = float(Z[y0, x0]) + H_UAV
    z1 = float(Z[y1, x1]) + TARGET_HEIGHT

    visited = set()
    for k in range(1, n):
        a = k / n
        x = int(round(x0 + a * dx))
        y = int(round(y0 + a * dy))
        if (x, y) in visited:
            continue
        visited.add((x, y))

        z_line = z0 + a * (z1 - z0)
        if float(Z[y, x]) > z_line:
            return False
    return True


def visible_mask_at(
    Z: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> int:
    """
    Return LOS-visible cells as one Python integer bit-mask.
    Cell i=(x,y) is bit y*GRID+x.
    """
    mask = 0
    for yy in range(max(0, y - radius), min(GRID, y + radius + 1)):
        for xx in range(max(0, x - radius), min(GRID, x + radius + 1)):
            if max(abs(xx - x), abs(yy - y)) > radius:
                continue
            if los_clear(Z, x, y, xx, yy):
                mask |= 1 << (yy * GRID + xx)
    return mask


def precompute_visibility(
    Z: np.ndarray,
    radius: int,
    cache_dir: Optional[Path] = None,
    terrain_seed: int = TERRAIN_SEED,
) -> List[int]:
    """
    Precompute VisibleCells(p,G,Z,Robs) for all 10,000 cells.
    Optional pickle cache greatly accelerates reruns.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"visibility_seed{terrain_seed}_R{radius}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)

    masks = [0] * (GRID * GRID)
    for y in range(GRID):
        for x in range(GRID):
            masks[y * GRID + x] = visible_mask_at(Z, x, y, radius)

    if cache_path is not None:
        with cache_path.open("wb") as f:
            pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)

    return masks


def generate_fire_cells(
    n: int = 500,
    seed: int = SIM_SEED,
    start: Tuple[int, int] = START,
) -> List[Tuple[int, int]]:
    """Fire points: Chebyshev distance 30..45 from UAV start."""
    rng = np.random.default_rng(seed)
    fires: List[Tuple[int, int]] = []
    while len(fires) < n:
        x = int(rng.integers(0, GRID))
        y = int(rng.integers(0, GRID))
        d = max(abs(x - start[0]), abs(y - start[1]))
        if 30 <= d <= 45:
            fires.append((x, y))
    return fires


def fire_bit(fire: Tuple[int, int]) -> int:
    return 1 << (fire[1] * GRID + fire[0])


def random_search(
    fire: Tuple[int, int],
    vis: Sequence[int],
    rng: np.random.Generator,
    tmax: int = TMAX,
) -> Tuple[bool, Optional[int]]:
    x, y = START
    fb = fire_bit(fire)

    if vis[y * GRID + x] & fb:
        return True, 0

    for t in range(1, tmax + 1):
        nbs = NEIGHBORS[y * GRID + x]
        x, y = nbs[int(rng.integers(0, len(nbs)))]
        if vis[y * GRID + x] & fb:
            return True, t
    return False, None


def greedy_nbv(
    fire: Tuple[int, int],
    vis: Sequence[int],
    rng: np.random.Generator,
    tmax: int = TMAX,
) -> Tuple[bool, Optional[int]]:
    x, y = START
    observed = vis[y * GRID + x]
    fb = fire_bit(fire)

    if observed & fb:
        return True, 0

    for t in range(1, tmax + 1):
        best_reward = -1
        best_moves: List[Tuple[int, int]] = []

        for nx, ny in NEIGHBORS[y * GRID + x]:
            reward = (vis[ny * GRID + nx] & ~observed).bit_count()
            if reward > best_reward:
                best_reward = reward
                best_moves = [(nx, ny)]
            elif reward == best_reward:
                best_moves.append((nx, ny))

        x, y = best_moves[int(rng.integers(0, len(best_moves)))]
        observed |= vis[y * GRID + x]

        if vis[y * GRID + x] & fb:
            return True, t

    return False, None


def rh_nbv(
    fire: Tuple[int, int],
    vis: Sequence[int],
    rng: np.random.Generator,
    H: int = 5,
    K: int = 300,
    tmax: int = TMAX,
) -> Tuple[bool, Optional[int]]:
    """
    Receding-Horizon Next-Best-View:
    - sample K random H-step paths
    - reward = | union visible(path) intersect currently-unobserved |
    - select max reward
    - execute only first step
    - update and replan
    """
    x, y = START
    observed = vis[y * GRID + x]
    fb = fire_bit(fire)

    if observed & fb:
        return True, 0

    for t in range(1, tmax + 1):
        best_reward = -1
        best_first_moves: List[Tuple[int, int]] = []

        for _ in range(K):
            px, py = x, y
            path_visible = 0
            first_move = None

            for h in range(H):
                nbs = NEIGHBORS[py * GRID + px]
                px, py = nbs[int(rng.integers(0, len(nbs)))]
                if h == 0:
                    first_move = (px, py)
                path_visible |= vis[py * GRID + px]

            reward = (path_visible & ~observed).bit_count()

            if reward > best_reward:
                best_reward = reward
                best_first_moves = [first_move]  # type: ignore
            elif reward == best_reward:
                best_first_moves.append(first_move)  # type: ignore

        x, y = best_first_moves[
            int(rng.integers(0, len(best_first_moves)))
        ]

        observed |= vis[y * GRID + x]

        if vis[y * GRID + x] & fb:
            return True, t

    return False, None


def make_lawnmower_path() -> List[Tuple[int, int]]:
    """
    Deterministic serpentine path beginning at START.

    This detail was not fully specified in the manuscript, so this is the
    reconstruction's explicit definition.
    """
    path: List[Tuple[int, int]] = [START]
    x, y = START

    # First reach the left edge of the start row.
    while x > 0:
        x -= 1
        path.append((x, y))

    # Sweep upward to row 99.
    direction = 1
    for yy in range(y, GRID):
        target_x = GRID - 1 if direction == 1 else 0
        while x != target_x:
            x += direction
            path.append((x, yy))
        if yy < GRID - 1:
            path.append((x, yy + 1))
        direction *= -1

    # Then sweep downward below the original start row if needed.
    # This keeps the path defined beyond the upper-half sweep.
    direction = -direction
    for yy in range(START[1] - 1, -1, -1):
        target_x = GRID - 1 if direction == 1 else 0
        while x != target_x:
            x += direction
            path.append((x, yy))
        if yy > 0:
            path.append((x, yy - 1))
        direction *= -1

    return path


LAWNMOWER_PATH = make_lawnmower_path()


def lawnmower_search(
    fire: Tuple[int, int],
    vis: Sequence[int],
    rng: np.random.Generator,  # kept for common signature
    tmax: int = TMAX,
) -> Tuple[bool, Optional[int]]:
    fb = fire_bit(fire)
    x, y = START

    if vis[y * GRID + x] & fb:
        return True, 0

    for t in range(1, min(tmax, len(LAWNMOWER_PATH) - 1) + 1):
        x, y = LAWNMOWER_PATH[t]
        if vis[y * GRID + x] & fb:
            return True, t

    return False, None


def evaluate_method(
    method_name: str,
    fires: Sequence[Tuple[int, int]],
    vis: Sequence[int],
    H: int = 5,
    K: int = 300,
    sim_seed: int = SIM_SEED,
) -> Dict[str, float]:
    methods = {
        "random": random_search,
        "lawnmower": lawnmower_search,
        "greedy": greedy_nbv,
        "rh-nbv": rh_nbv,
    }
    fn = methods[method_name]

    successes = 0
    detection_steps: List[int] = []

    for i, fire in enumerate(fires):
        # Deterministic method/trial stream.
        # Stable offsets avoid using Python's process-randomised hash().
        method_offset = {
            "random": 100_000,
            "lawnmower": 200_000,
            "greedy": 300_000,
            "rh-nbv": 400_000,
        }[method_name]
        rng = np.random.default_rng(sim_seed + method_offset + i)

        if method_name == "rh-nbv":
            ok, step = fn(fire, vis, rng, H=H, K=K)  # type: ignore
        else:
            ok, step = fn(fire, vis, rng)  # type: ignore

        if ok:
            successes += 1
            detection_steps.append(int(step))

    n = len(fires)
    return {
        "method": method_name,
        "N": n,
        "successes": successes,
        "failures": n - successes,
        "success_rate_pct": 100.0 * successes / n,
        "mean_detection_step_success_only": (
            float(np.mean(detection_steps)) if detection_steps else float("nan")
        ),
    }


def write_rows(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_original_benchmark(
    trials: int,
    radii: Sequence[int],
    out_dir: Path,
    cache_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Z = generate_terrain(TERRAIN_SEED)
    fires = generate_fire_cells(trials, SIM_SEED)

    rows = []
    for radius in radii:
        print(f"[visibility] R_obs={radius}")
        vis = precompute_visibility(
            Z, radius, cache_dir=cache_dir, terrain_seed=TERRAIN_SEED
        )

        for method in ("random", "lawnmower", "greedy", "rh-nbv"):
            t0 = time.time()
            result = evaluate_method(method, fires, vis, H=5, K=300)
            result["R_obs"] = radius
            rows.append(result)
            print(
                method,
                f"success={result['success_rate_pct']:.1f}%",
                f"mean_step={result['mean_detection_step_success_only']:.1f}",
                f"time={time.time()-t0:.1f}s",
            )

    write_rows(out_dir / "benchmark_reconstructed.csv", rows)


def run_h_sensitivity(
    trials: int,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    Z = generate_terrain(TERRAIN_SEED)
    fires = generate_fire_cells(trials, SIM_SEED)
    vis = precompute_visibility(
        Z, 5, cache_dir=cache_dir, terrain_seed=TERRAIN_SEED
    )

    rows = []
    for H in (1, 3, 5, 7):
        result = evaluate_method("rh-nbv", fires, vis, H=H, K=300)
        result["H"] = H
        result["K"] = 300
        rows.append(result)
        print("H", H, result)

    write_rows(out_dir / "H_sensitivity_reconstructed.csv", rows)


def run_k_sensitivity(
    trials: int,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    Z = generate_terrain(TERRAIN_SEED)
    fires = generate_fire_cells(trials, SIM_SEED)
    vis = precompute_visibility(
        Z, 5, cache_dir=cache_dir, terrain_seed=TERRAIN_SEED
    )

    rows = []
    for K in (100, 300, 500):
        result = evaluate_method("rh-nbv", fires, vis, H=5, K=K)
        result["H"] = 5
        result["K"] = K
        rows.append(result)
        print("K", K, result)

    write_rows(out_dir / "K_sensitivity_reconstructed.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("benchmark", "H", "K", "all"),
        default="benchmark",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=500,
        help="Use 50 first for a quick check; manuscript used 500.",
    )
    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rh_nbv_results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".rh_nbv_cache"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("benchmark", "all"):
        run_original_benchmark(
            args.trials, args.radii, args.out_dir, args.cache_dir
        )
    if args.mode in ("H", "all"):
        run_h_sensitivity(args.trials, args.out_dir, args.cache_dir)
    if args.mode in ("K", "all"):
        run_k_sensitivity(args.trials, args.out_dir, args.cache_dir)


if __name__ == "__main__":
    main()
