#!/usr/bin/env python3
"""
NumPy + multiprocessing reconstruction of the RH-NBV wildfire simulator.
No Numba required.

Install:
    pip install numpy

Recommended on Ubuntu/WSL:
    python rh_nbv_mp.py --mode benchmark --trials 500 --workers 6
"""

from __future__ import annotations
import argparse, csv, math, os, pickle, time
from pathlib import Path
from multiprocessing import Pool, get_context
from typing import List, Sequence, Tuple, Dict
import numpy as np

GRID = 100
NCELLS = GRID * GRID
START = (50, 50)
START_IDX = START[1] * GRID + START[0]
TERRAIN_SEED = 20260825
SIM_SEED = 20260824
H_UAV = 5.0
TARGET_HEIGHT = 0.5
TMAX = 1000

DIRS = (
    (-1,-1),(0,-1),(1,-1),
    (-1, 0),       (1, 0),
    (-1, 1),(0, 1),(1, 1),
)

PAPER_TABLE4 = {
    1: {"random":(2.6,710.7),"lawnmower":(6.8,435.2),"greedy":(30.2,466.0),"rh-nbv":(40.2,492.9)},
    3: {"random":(4.6,653.2),"lawnmower":(8.6,345.4),"greedy":(52.8,376.3),"rh-nbv":(72.6,430.2)},
    5: {"random":(8.8,689.8),"lawnmower":(11.8,343.9),"greedy":(64.4,272.2),"rh-nbv":(77.6,312.2)},
    7: {"random":(10.8,666.7),"lawnmower":(15.6,317.4),"greedy":(66.0,202.7),"rh-nbv":(78.8,265.6)},
}


def _elliptical_gaussian(X, Y, cx, cy, amp, sx, sy, theta_deg):
    """Rotated elliptical Gaussian used to create peaks/ridges/valleys."""
    th = np.deg2rad(theta_deg)
    ct, st = np.cos(th), np.sin(th)
    xr =  ct * (X - cx) + st * (Y - cy)
    yr = -st * (X - cx) + ct * (Y - cy)
    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def _mountain_terrain_4():
    """
    Replacement for terrain 4 (seed 20260828).
    Designed to look like a clearly mountainous/forested relief:
    one dominant peak, diagonal ridge, secondary highland, and deep valley.
    """
    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)

    # Broad low-frequency relief
    Z = (
        5.0 * np.sin(X / 11.0)
        + 4.0 * np.cos(Y / 13.0)
        + 2.5 * np.sin((X + 0.7 * Y) / 8.0)
    )

    # Dominant mountain mass in lower-left
    Z += _elliptical_gaussian(X, Y, 18, 13, 85, 11, 8, 18)
    Z += _elliptical_gaussian(X, Y, 29, 25, 55, 13, 7, 42)

    # Central ridge running SW -> NE
    Z += _elliptical_gaussian(X, Y, 43, 39, 52, 24, 6, 50)
    Z += _elliptical_gaussian(X, Y, 58, 55, 42, 25, 7, 48)

    # Secondary highland toward upper-right
    Z += _elliptical_gaussian(X, Y, 78, 67, 48, 19, 12, 20)
    Z += _elliptical_gaussian(X, Y, 88, 78, 32, 12, 9, -10)

    # Valleys to create strong LOS occlusion structure
    Z -= _elliptical_gaussian(X, Y, 78, 24, 48, 19, 10, -12)
    Z -= _elliptical_gaussian(X, Y, 18, 67, 26, 16, 12, 20)

    # Sharper local relief
    Z += _elliptical_gaussian(X, Y, 41, 42, 23, 6, 5, 0)
    Z += _elliptical_gaussian(X, Y, 63, 63, 19, 7, 5, 35)

    Z -= Z.min()
    Z *= 100.0 / Z.max()
    return Z


def _mountain_terrain_7():
    """
    Replacement for terrain 7 (seed 20260831).
    Different mountain layout from terrain 4, but with comparable relief.
    """
    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)

    Z = (
        4.5 * np.sin(X / 10.0)
        + 4.0 * np.cos(Y / 12.0)
        + 3.0 * np.sin((0.8 * X + Y) / 7.5)
    )

    # Main massif in upper-left / centre-left
    Z += _elliptical_gaussian(X, Y, 23, 74, 78, 12, 9, -20)
    Z += _elliptical_gaussian(X, Y, 38, 62, 55, 17, 7, -35)

    # Long diagonal ridge crossing the centre
    Z += _elliptical_gaussian(X, Y, 53, 52, 48, 28, 6, -32)
    Z += _elliptical_gaussian(X, Y, 69, 43, 38, 22, 6, -28)

    # Separate mountain group lower-right
    Z += _elliptical_gaussian(X, Y, 82, 22, 58, 12, 10, 8)
    Z += _elliptical_gaussian(X, Y, 72, 30, 31, 10, 7, 20)

    # Valleys
    Z -= _elliptical_gaussian(X, Y, 18, 28, 36, 16, 10, -5)
    Z -= _elliptical_gaussian(X, Y, 77, 75, 32, 18, 13, 15)

    # Local peaks
    Z += _elliptical_gaussian(X, Y, 49, 55, 20, 6, 4, -20)
    Z += _elliptical_gaussian(X, Y, 83, 19, 16, 5, 4, 5)

    Z -= Z.min()
    Z *= 100.0 / Z.max()
    return Z


def generate_terrain(seed=TERRAIN_SEED):
    """
    Original terrain generator is preserved for all terrains except:
      seed 20260828 -> replacement Terrain 4
      seed 20260831 -> replacement Terrain 7
    """
    if seed == 20260828:
        return _mountain_terrain_4()
    if seed == 20260831:
        return _mountain_terrain_7()

    rng = np.random.default_rng(seed)
    x = np.arange(GRID, dtype=float)
    y = np.arange(GRID, dtype=float)
    X, Y = np.meshgrid(x, y)
    Z = 6*np.sin(X/7) + 5*np.cos(Y/9) + 3*np.sin((X+Y)/4)
    for _ in range(9):
        mux = rng.uniform(0, GRID)
        muy = rng.uniform(0, GRID)
        sigma = rng.uniform(7, 20)
        amp = rng.uniform(18, 55)
        Z += amp*np.exp(-((X-mux)**2 + (Y-muy)**2)/(2*sigma**2))
    Z -= Z.min()
    Z *= 100.0 / Z.max()
    return Z


def build_neighbors():
    out = [[] for _ in range(NCELLS)]
    for y in range(GRID):
        for x in range(GRID):
            out[y*GRID+x] = [(x+dx,y+dy) for dx,dy in DIRS
                             if 0 <= x+dx < GRID and 0 <= y+dy < GRID]
    return out

NEIGHBORS = build_neighbors()

def los_clear(Z, x0, y0, x1, y1):
    if x0 == x1 and y0 == y1:
        return True
    dx, dy = x1-x0, y1-y0
    n = max(abs(dx), abs(dy))
    z0 = float(Z[y0,x0]) + H_UAV
    z1 = float(Z[y1,x1]) + TARGET_HEIGHT
    seen = set()
    for k in range(1, n):
        a = k/n
        x = int(round(x0 + a*dx))
        y = int(round(y0 + a*dy))
        if (x,y) in seen:
            continue
        seen.add((x,y))
        if float(Z[y,x]) > z0 + a*(z1-z0):
            return False
    return True

def get_visibility(terrain_seed, radius, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"vis_bigint_mountain_v2_seed{terrain_seed}_R{radius}.pkl"
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    Z = generate_terrain(terrain_seed)
    masks = [0] * NCELLS
    t0 = time.time()
    for y in range(GRID):
        for x in range(GRID):
            mask = 0
            for yy in range(max(0,y-radius), min(GRID,y+radius+1)):
                for xx in range(max(0,x-radius), min(GRID,x+radius+1)):
                    if max(abs(xx-x), abs(yy-y)) <= radius and los_clear(Z,x,y,xx,yy):
                        mask |= 1 << (yy*GRID+xx)
            masks[y*GRID+x] = mask
    with p.open("wb") as f:
        pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[visibility] seed={terrain_seed} R={radius} built in {time.time()-t0:.1f}s")
    return masks

def generate_fire_cells(n, seed):
    rng = np.random.default_rng(seed)
    fires = []
    while len(fires) < n:
        x = int(rng.integers(0,GRID))
        y = int(rng.integers(0,GRID))
        d = max(abs(x-START[0]), abs(y-START[1]))
        if 30 <= d <= 45:
            fires.append((x,y))
    return fires

def fire_bit(fire):
    return 1 << (fire[1]*GRID + fire[0])

def make_lawnmower_path():
    path=[START]
    x,y=START
    while x>0:
        x-=1; path.append((x,y))
    direction=1
    for yy in range(y,GRID):
        tx=GRID-1 if direction==1 else 0
        while x!=tx:
            x += direction; path.append((x,yy))
        if yy<GRID-1: path.append((x,yy+1))
        direction *= -1
    direction = -direction
    for yy in range(START[1]-1,-1,-1):
        tx=GRID-1 if direction==1 else 0
        while x!=tx:
            x += direction; path.append((x,yy))
        if yy>0: path.append((x,yy-1))
        direction *= -1
    return path

LAWNMOWER_PATH = make_lawnmower_path()

# Worker globals, inherited efficiently via fork on Ubuntu/WSL.
_G_VIS = None
_G_METHOD = None
_G_H = 5
_G_K = 300
_G_SIM_SEED = SIM_SEED

def _worker_init(vis, method, H, K, sim_seed):
    global _G_VIS, _G_METHOD, _G_H, _G_K, _G_SIM_SEED
    _G_VIS = vis
    _G_METHOD = method
    _G_H = H
    _G_K = K
    _G_SIM_SEED = sim_seed

def random_one(fire, rng):
    x,y=START
    fb=fire_bit(fire)
    if _G_VIS[y*GRID+x] & fb: return 0
    for t in range(1,TMAX+1):
        nbs=NEIGHBORS[y*GRID+x]
        x,y=nbs[int(rng.integers(0,len(nbs)))]
        if _G_VIS[y*GRID+x] & fb: return t
    return -1

def lawnmower_one(fire, rng):
    fb=fire_bit(fire)
    x,y=START
    if _G_VIS[y*GRID+x] & fb: return 0
    for t,(x,y) in enumerate(LAWNMOWER_PATH[1:TMAX+1],1):
        if _G_VIS[y*GRID+x] & fb: return t
    return -1

def greedy_one(fire, rng):
    x,y=START
    observed=_G_VIS[y*GRID+x]
    fb=fire_bit(fire)
    if observed & fb: return 0
    for t in range(1,TMAX+1):
        best=-1; moves=[]
        for nx,ny in NEIGHBORS[y*GRID+x]:
            reward=(_G_VIS[ny*GRID+nx] & ~observed).bit_count()
            if reward>best:
                best=reward; moves=[(nx,ny)]
            elif reward==best:
                moves.append((nx,ny))
        x,y=moves[int(rng.integers(0,len(moves)))]
        observed |= _G_VIS[y*GRID+x]
        if _G_VIS[y*GRID+x] & fb: return t
    return -1

def rh_nbv_one(fire, rng):
    x,y=START
    observed=_G_VIS[y*GRID+x]
    fb=fire_bit(fire)
    if observed & fb: return 0
    H,K=_G_H,_G_K
    for t in range(1,TMAX+1):
        best=-1; firsts=[]
        for _ in range(K):
            px,py=x,y
            union=0
            first=None
            for h in range(H):
                nbs=NEIGHBORS[py*GRID+px]
                px,py=nbs[int(rng.integers(0,len(nbs)))]
                if h==0: first=(px,py)
                union |= _G_VIS[py*GRID+px]
            reward=(union & ~observed).bit_count()
            if reward>best:
                best=reward; firsts=[first]
            elif reward==best:
                firsts.append(first)
        x,y=firsts[int(rng.integers(0,len(firsts)))]
        observed |= _G_VIS[y*GRID+x]
        if _G_VIS[y*GRID+x] & fb: return t
    return -1

def _run_trial(arg):
    i, fire = arg
    offsets={"random":100000,"lawnmower":200000,"greedy":300000,"rh-nbv":400000}
    rng=np.random.default_rng(_G_SIM_SEED + offsets[_G_METHOD] + i)
    if _G_METHOD=="random": return random_one(fire,rng)
    if _G_METHOD=="lawnmower": return lawnmower_one(fire,rng)
    if _G_METHOD=="greedy": return greedy_one(fire,rng)
    return rh_nbv_one(fire,rng)

def evaluate(method, fires, vis, H, K, sim_seed, workers):
    t0=time.time()
    ctx=get_context("fork")
    with ctx.Pool(processes=workers, initializer=_worker_init,
                  initargs=(vis,method,H,K,sim_seed)) as pool:
        steps=list(pool.imap(_run_trial, enumerate(fires), chunksize=max(1,len(fires)//(workers*8))))
    arr=np.asarray(steps,dtype=int)
    ok=arr>=0
    return {
        "N":len(arr),
        "successes":int(ok.sum()),
        "failures":int((~ok).sum()),
        "success_rate_pct":100.0*ok.mean(),
        "mean_detection_step_success_only":float(arr[ok].mean()) if ok.any() else float("nan"),
        "elapsed_sec":time.time()-t0,
    }

def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def run_benchmark(trials,radii,workers,cache_dir,out_dir):
    fires=generate_fire_cells(trials,SIM_SEED)
    rows=[]; comp=[]
    for R in radii:
        vis=get_visibility(TERRAIN_SEED,R,cache_dir)
        for method in ("random","lawnmower","greedy","rh-nbv"):
            s=evaluate(method,fires,vis,5,300,SIM_SEED,workers)
            row={"R_obs":R,"method":method,**s}; rows.append(row)
            print(f"R={R:<2} {method:<10} success={s['success_rate_pct']:.1f}% "
                  f"mean_step={s['mean_detection_step_success_only']:.1f} time={s['elapsed_sec']:.1f}s")
            if trials==500:
                ps,pm=PAPER_TABLE4[R][method]
                comp.append({"R_obs":R,"method":method,
                             "paper_success_pct":ps,
                             "reconstructed_success_pct":s["success_rate_pct"],
                             "success_diff_pp":s["success_rate_pct"]-ps,
                             "paper_mean_step":pm,
                             "reconstructed_mean_step":s["mean_detection_step_success_only"],
                             "mean_step_diff":s["mean_detection_step_success_only"]-pm})
    write_csv(out_dir/"benchmark_mp.csv",rows)
    if comp: write_csv(out_dir/"benchmark_vs_paper_mp.csv",comp)

def run_multi_terrain(terrain_count,trials_per_terrain,R,workers,cache_dir,out_dir):
    rows=[]
    for j in range(terrain_count):
        ts=TERRAIN_SEED+j
        fs=SIM_SEED+10000*j
        print(f"\n[terrain {j+1}/{terrain_count}] seed={ts}")
        fires=generate_fire_cells(trials_per_terrain,fs)
        vis=get_visibility(ts,R,cache_dir)
        for method in ("random","lawnmower","greedy","rh-nbv"):
            s=evaluate(method,fires,vis,5,300,fs,workers)
            rows.append({"terrain_index":j,"terrain_seed":ts,"fire_seed":fs,
                         "R_obs":R,"method":method,**s})
            print(f"{method:<10} success={s['success_rate_pct']:.1f}% "
                  f"mean_step={s['mean_detection_step_success_only']:.1f} time={s['elapsed_sec']:.1f}s")
    write_csv(out_dir/"multi_terrain_raw_mp.csv",rows)


def preview_replacement_terrains(out_dir: Path):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for label, seed in (("terrain4", 20260828), ("terrain7", 20260831)):
        Z = generate_terrain(seed)

        fig, ax = plt.subplots(figsize=(7, 6))
        c = ax.contourf(
            np.arange(GRID),
            np.arange(GRID),
            Z,
            levels=np.linspace(0, 100, 18),
        )
        ax.contour(
            np.arange(GRID),
            np.arange(GRID),
            Z,
            levels=np.linspace(0, 100, 18),
            linewidths=0.45,
        )
        ax.scatter(
            [START[0]], [START[1]],
            s=35, marker="o", label="UAV start"
        )
        ax.set_title(
            f"{label.capitalize()} replacement (seed {seed})"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend(loc="upper right")
        fig.colorbar(
            c, ax=ax,
            label="Synthetic elevation (relative scale)"
        )
        fig.tight_layout()
        path = out_dir / f"{label}_replacement.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[preview] {path}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("benchmark","multi-terrain","preview"),default="multi-terrain")
    p.add_argument("--trials",type=int,default=500)
    p.add_argument("--radii",type=int,nargs="+",default=[1,3,5,7])
    p.add_argument("--workers",type=int,default=max(1,(os.cpu_count() or 4)-1))
    p.add_argument("--terrain-count",type=int,default=10)
    p.add_argument("--trials-per-terrain",type=int,default=50)
    p.add_argument("--multi-terrain-radius",type=int,default=5)
    p.add_argument("--cache-dir",type=Path,default=Path(".rh_nbv_cache_mp"))
    p.add_argument("--out-dir",type=Path,default=Path("rh_nbv_results_mp"))
    a=p.parse_args()
    print(f"[setup] workers={a.workers}; no numba required")
    if a.mode=="benchmark":
        run_benchmark(a.trials,a.radii,a.workers,a.cache_dir,a.out_dir)
    elif a.mode=="multi-terrain":
        run_multi_terrain(a.terrain_count,a.trials_per_terrain,a.multi_terrain_radius,
                          a.workers,a.cache_dir,a.out_dir)
    else:
        preview_replacement_terrains(a.out_dir)

if __name__=="__main__":
    main()
