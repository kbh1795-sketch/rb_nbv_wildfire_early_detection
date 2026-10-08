import csv
import statistics
import matplotlib.pyplot as plt


# ============================================================
# CSV paths
# ============================================================

TABLE5_PATH = "results_revision/table5_Robs_by_terrain.csv"
TABLE6_PATH = "results_revision/table6_H_by_terrain.csv"

# ============================================================
# Read RH-NBV terrain-level success rates
# ============================================================

def read_success_rates(path, parameter_name, parameter_values):
    data = {v: [] for v in parameter_values}

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("method", "") != "RH-NBV":
                continue

            parameter = int(float(row[parameter_name]))

            if parameter in data:
                data[parameter].append(float(row["success_rate_pct"]))

    return data


# ============================================================
# Table 5: R_obs
# ============================================================

R_VALUES = [1, 3, 5, 7]

r_data = read_success_rates(
    TABLE5_PATH,
    "R_obs",
    R_VALUES
)


# ============================================================
# Table 6: H
# ============================================================

H_VALUES = [1, 3, 5, 7]

h_data = read_success_rates(
    TABLE6_PATH,
    "H",
    H_VALUES
)


# ============================================================
# Table 7: K
# ============================================================

K_VALUES = [100, 300, 500]

k_data = {
    100: [72.0, 72.0, 86.0, 72.0, 80.0, 74.0, 78.0, 62.0, 70.0, 86.0],
    300: [78.0, 86.0, 90.0, 84.0, 72.0, 84.0, 88.0, 74.0, 80.0, 84.0],
    500: [78.0, 72.0, 68.0, 80.0, 84.0, 62.0, 78.0, 82.0, 84.0, 78.0]
}


# ============================================================
# Check means
# ============================================================

print("\nR_obs data")
for value in R_VALUES:
    print(value, r_data[value], "mean =", round(statistics.mean(r_data[value]), 1))

print("\nH data")
for value in H_VALUES:
    print(value, h_data[value], "mean =", round(statistics.mean(h_data[value]), 1))

print("\nK data")
for value in K_VALUES:
    print(value, k_data[value], "mean =", round(statistics.mean(k_data[value]), 1))


# ============================================================
# Plot helper
# ============================================================

def draw_single_boxplot(data_dict, parameter_values, title, xlabel, output_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    plot_data = [data_dict[value] for value in parameter_values]

    ax.boxplot(
        plot_data,
        widths=0.55,
        showmeans=True,
        showfliers=True
    )

    ax.set_xticks(range(1, len(parameter_values) + 1))
    ax.set_xticklabels([str(v) for v in parameter_values])

    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.grid(axis="y", linestyle=":", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# Generate 3 separate figures
# ============================================================

draw_single_boxplot(
    r_data,
    R_VALUES,
    "Success Rate Distribution by Observation Radius",
    r"$R_{obs}$",
    "figure_Robs.png"
)

draw_single_boxplot(
    h_data,
    H_VALUES,
    "Success Rate Distribution by Prediction Horizon",
    r"$H$",
    "figure_H.png"
)

draw_single_boxplot(
    k_data,
    K_VALUES,
    "Success Rate Distribution by Number of Candidate Paths",
    r"$K$",
    "figure_K.png"
)