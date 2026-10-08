# RH-NBV for UAV-Based Early Wildfire Detection

This repository contains the simulation code and analysis scripts used for the study:

**Receding-Horizon Next-Best-View Observation Planning for UAV Early Wildfire Detection**

The study formulates early wildfire detection as an active observation-planning problem and evaluates a Receding-Horizon Next-Best-View (RH-NBV) strategy under terrain-dependent Line-of-Sight (LOS) constraints.

---

## Overview

Early wildfire detection is challenging because initial smoke signals can be weak and the surveillance area can be large.

This study considers a UAV searching for an unknown wildfire ignition point in a synthetic forest environment. The environment is represented as a two-dimensional grid with terrain elevation, and terrain-dependent LOS constraints determine which cells are observable from each UAV position.

The RH-NBV planner repeatedly:

1. Generates multiple candidate paths from the current UAV position.
2. Predicts the observable area along each candidate path.
3. Evaluates the cumulative number of newly observable cells.
4. Selects the candidate path with the highest observation reward.
5. Executes only the first movement of the selected path.
6. Updates the observation history.
7. Replans from the new UAV position.

This receding-horizon structure allows the UAV to continuously adapt its search path according to newly obtained observations.

---

## Compared Search Methods

Four search strategies are evaluated.

### Random Search

At each step, the UAV randomly selects one feasible movement direction.

### Lawn-mower Search

The UAV systematically scans the search area using a deterministic coverage pattern.

### Greedy NBV

The UAV selects the next position according to immediate observation gain.

### RH-NBV

The UAV evaluates cumulative observation gain over multiple candidate paths within a finite prediction horizon and executes the first movement of the highest-reward path.

---

## Simulation Environment

The simulation is implemented in Python using a 100 × 100 grid environment.

| Parameter | Value |
|---|---:|
| Grid size | 100 × 100 |
| Number of virtual terrains | 10 |
| Fire locations per terrain | 50 |
| Maximum simulation steps | 1000 |
| UAV relative altitude | 5 |
| Movement model | 8-connected grid |

Each grid cell contains a relative terrain elevation value.

Synthetic terrains are generated using combinations of periodic elevation patterns and Gaussian-based terrain components. The resulting elevation maps are normalized to a relative elevation range of 0–100 and are designed to represent different terrain structures, including ridges and valleys.

The elevation values represent relative terrain height rather than physical altitude in meters.

---

## Line-of-Sight Observation Model

The UAV has a limited observation radius, denoted by `R_obs`.

A cell is considered observable only when:

1. the cell lies within the observation radius, and
2. the Line-of-Sight between the UAV and the cell is not blocked by terrain.

Therefore, cells located within the nominal observation radius may remain unobservable because of terrain occlusion.

The baseline observation radius is `R_obs = 5`.

---

## Fire Generation

One wildfire ignition point is generated for each simulation run.

To avoid trivial detections close to the initial UAV position, fire locations are generated within a constrained distance from the UAV starting position.

Within the same terrain, the same fire locations are used across the compared search methods to provide consistent comparison conditions.

---

## RH-NBV Parameters

The RH-NBV planner is controlled by three main parameters.

### Observation Radius

`R_obs`

Defines the maximum observation range around the UAV.

### Prediction Horizon

`H`

Defines the number of future movement steps included in each candidate path.

### Number of Candidate Paths

`K`

Defines the number of candidate paths sampled and evaluated at each replanning step.

The baseline configuration is:

| Parameter | Value |
|---|---:|
| `R_obs` | 5 |
| `H` | 5 |
| `K` | 300 |

---

## Parameter Sensitivity Analysis

The effect of the main RH-NBV parameters is evaluated using additional simulations.

### Observation Radius

`R_obs ∈ {1, 3, 5, 7}`

with:

- `H = 5`
- `K = 300`

### Prediction Horizon

`H ∈ {1, 3, 5, 7}`

with:

- `R_obs = 5`
- `K = 300`

### Number of Candidate Paths

`K ∈ {100, 300, 500}`

with:

- `R_obs = 5`
- `H = 5`

Each condition is evaluated across **10 terrains × 50 fire locations**.

---

## Main Results

Under the baseline configuration:

- `R_obs = 5`
- `H = 5`
- `K = 300`

the mean detection success rates across the ten virtual terrains were:

| Method | Mean Success Rate |
|---|---:|
| Random Search | 5.8% |
| Lawn-mower Search | 13.2% |
| Greedy NBV | 65.0% |
| RH-NBV | 82.0% |

RH-NBV achieved the highest detection success rate across all ten evaluated terrains.

---

## Observation Radius Analysis

The mean RH-NBV success rates for different observation radii were:

| `R_obs` | Mean Success Rate |
|---:|---:|
| 1 | 39.0% |
| 3 | 72.0% |
| 5 | 82.0% |
| 7 | 80.8% |

The highest mean success rate was obtained at **`R_obs = 5`**.

---

## Prediction Horizon Analysis

The RH-NBV results for different prediction horizons were:

| `H` | Mean Success Rate | Terrain-wise Range | Between-terrain SD |
|---:|---:|---:|---:|
| 1 | 59.6% | 44.0–72.0% | 9.6%p |
| 3 | 65.2% | 62.0–68.0% | 1.9%p |
| 5 | 82.0% | 72.0–90.0% | 5.9%p |
| 7 | 79.8% | 66.0–88.0% | 7.5%p |

The highest mean success rate was obtained at **`H = 5`**.

---

## Candidate Path Analysis

The RH-NBV results for different numbers of candidate paths were:

| `K` | Mean Success Rate | Terrain-wise Range | Between-terrain SD |
|---:|---:|---:|---:|
| 100 | 75.2% | 62.0–86.0% | 7.4%p |
| 300 | 82.0% | 72.0–90.0% | 5.9%p |
| 500 | 76.6% | 62.0–84.0% | 7.2%p |

The highest mean success rate was obtained at **`K = 300`**.

Increasing the number of candidate paths does not necessarily increase detection performance, while computational cost increases as more candidate paths are evaluated.

---

## Performance Metrics

The primary performance metric is **Detection Success Rate**.

A simulation is considered successful when the wildfire ignition point enters the UAV's LOS-constrained observation region within the maximum number of simulation steps.

Additional metrics include:

- Mean Detection Step
- Number of Failures
- Terrain-wise Success-rate Range
- Between-terrain Standard Deviation

Mean Detection Step is calculated only from successful simulation runs.

---

## Repository Structure

```text
rb_nbv_wildfire_early_detection/
├── rh_nbv_mp_mountain_v3.py
├── revision_table5_table6.py
├── figures.py
├── results_revision/
├── README.md
└── .gitignore
```

### `rh_nbv_mp_mountain_v3.py`

Main simulation implementation for the multi-terrain comparison of Random Search, Lawn-mower Search, Greedy NBV, and RH-NBV.

### `revision_table5_table6.py`

Parameter sensitivity analysis for observation radius and prediction horizon.

### `figures.py`

Generates figures for simulation results and parameter sensitivity analysis.

### `results_revision/`

Contains simulation outputs and aggregated result files.

---

## Requirements

Python 3 is required.

Main dependencies:

- NumPy
- Matplotlib

A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

## Running the Simulation

Run the main simulation with:

```bash
python3 rh_nbv_mp_mountain_v3.py
```

---

## Running the Parameter Analysis

Run:

```bash
python3 revision_table5_table6.py
```

This script performs the multi-terrain sensitivity analyses for `R_obs` and `H`.

---

## Generating Figures

Run:

```bash
python3 figures.py
```

---

## Reproducibility

Fixed random seeds are used for terrain generation and wildfire ignition-point generation.

The same fire locations are used across the compared search methods within each terrain so that all methods are evaluated under consistent conditions.

---

## Scope

This repository focuses on evaluating observation-planning performance for UAV-based early wildfire detection under terrain-dependent LOS constraints.

The current implementation uses simplified UAV motion and synthetic terrain models to provide a controlled and reproducible environment for comparing search strategies.

Future work may extend the framework to real terrain data, more realistic UAV dynamics, and dynamic environmental conditions.

---

## Citation

If you use this repository in academic work, please cite the corresponding study:

**Sebin Lee and Byunghee Kim.**  
**“Receding-Horizon Next-Best-View Observation Planning for UAV Early Wildfire Detection.”**  
**2026.**

Formal publication information will be added after publication.

---

## Authors

**Sebin Lee**  
Republic of Korea Army

**Byunghee Kim**  
Faculty of Engineering and Information Technology  
University of Technology Sydney