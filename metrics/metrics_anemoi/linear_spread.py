"""
spread_comparison.py
=====================
Compares the spatial spread (ensemble standard deviation) of several SDEdit
configurations against the AROME reference ensemble, as a time series.

For each SDEdit configuration listed in SDEDIT_CONFIGS, this script:

  - Loads all matching NetCDF files (one per ensemble member)
  - Selects timesteps at midnight (00h), from DATE_MIN_SDEDIT onwards
  - Computes the spatial spread (mean of the per-pixel standard deviation
    across members) for each variable in VARS, at each selected timestep

For the AROME reference ensemble, this script:

  - Reads the CSV catalogue (AROME_CSV) to find available members at
    lead time 0, for each date in [DATE_MIN_AROME, DATE_MAX_AROME]
  - Randomly samples n_members members per date
  - Computes the spatial spread for each variable, at each date

Finally, the spread time series for all SDEdit configurations and AROME are
plotted together, one subplot per variable.

Configuration
-------------
SDEDIT_CONFIGS : list of configurations to compare, each with a label,
                  directory, file pattern, and plot color
AROME_DIR      : directory containing the AROME reference ensemble (.npy files)
AROME_CSV      : CSV catalogue referencing AROME files (Name, Date, Member, LeadTime)
OUTPUT_DIR     : directory where the output figure is saved
OUTPUT_FILE    : name of the output figure
VARS           : variable names and their index in the output arrays
INVALID_VAL    : value used to mark invalid/missing data (treated as NaN)
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator


# ── Config ───────────────────────────────────────────────────────────────

# Define the configurations to compare.
SDEDIT_CONFIGS = [
    {
        "label":   "config 1",
        "dir":     "dir/netcdf/files",
        "pattern": "pattern_of_the_file.nc",  # e.g. SDEdit_Xsteps_*.nc
        "color":   "steelblue",
    },
    {
        "label":   "config 2",
        "dir":     "dir/netcdf/files",
        "pattern": "pattern_of_the_file.nc",
        "color":   "steelblue",
    },
    {
        "label":   "config 3",
        "dir":     "dir/netcdf/files",
        "pattern": "pattern_of_the_file.nc",
        "color":   "steelblue",
    },
]

AROME_DIR = "path/to/AROME/dataset"  # reference ensemble dataset
AROME_CSV = "path/to/csv.csv"        # CSV referencing files in AROME_DIR (Name, Date, Member, LeadTime)

OUTPUT_DIR = "path/to/output/dir/"
OUTPUT_FILE = "name_of_file.png"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable indices in the output NetCDF / NumPy arrays
VARS = {
    "10u":   0,
    "10v":   1,
    "2t":    2,
    "z_500": 7,
}

INVALID_VAL = 9999  # treated as NaN

DATE_MIN_SDEDIT = pd.Timestamp("2020-06-15", tz="UTC")
DATE_MIN_AROME = pd.Timestamp("2020-06-01", tz="UTC")
DATE_MAX_AROME = pd.Timestamp("2020-07-02", tz="UTC")


def spatial_spread(members: np.ndarray, mask_valid: np.ndarray) -> float:
    """Compute the mean per-pixel ensemble standard deviation.

    Parameters
    ----------
    members : np.ndarray
        Array of shape (n_members, n_points) with one row per ensemble member.
    mask_valid : np.ndarray
        Boolean mask of valid grid points (currently unused, kept for API
        consistency with compute_arome_spread).

    Returns
    -------
    float
        Spatial mean of the per-pixel standard deviation across members.
    """
    std_map = members.std(axis=0, ddof=1)
    return float(std_map.mean())


def compute_sdedit_spread(netcdf_dir, netcdf_pat, label="SDEdit"):
    """Compute the spread time series for one SDEdit configuration.

    Parameters
    ----------
    netcdf_dir : str
        Directory containing the NetCDF files (one per member).
    netcdf_pat : str
        Glob pattern to select the NetCDF files.
    label : str
        Label used for progress bar display.

    Returns
    -------
    tuple[dict, pd.DatetimeIndex]
        Spread time series per variable, and the corresponding dates.
    """
    nc_files = sorted(glob.glob(os.path.join(netcdf_dir, netcdf_pat)))
    print(f"{label}: {len(nc_files)} NetCDF files found")

    datasets = [xr.open_dataset(f) for f in nc_files]

    times = datasets[0]["time"].values
    times_dates = pd.to_datetime(times).tz_localize("UTC")

    # Keep only midnight timesteps from DATE_MIN_SDEDIT onwards
    mask = (times_dates >= DATE_MIN_SDEDIT) & (times_dates.hour == 0)

    indices_midnight = np.where(mask)[0]
    times_filtered = times_dates[mask]

    spread = {v: np.zeros(len(indices_midnight)) for v in VARS}

    for k, ti in enumerate(tqdm(indices_midnight, desc=f"{label} leadtimes")):
        for varname in VARS:
            members = np.array([
                ds[varname].isel(time=int(ti)).values
                for ds in datasets
            ])
            spread[varname][k] = spatial_spread(
                members, np.ones(members.shape[1], dtype=bool)
            )

    return spread, times_filtered


def compute_arome_spread(n_members=5):
    """Compute the AROME reference spread time series.

    Parameters
    ----------
    n_members : int
        Number of ensemble members to randomly sample per date.

    Returns
    -------
    tuple[dict, pd.DatetimeIndex]
        Spread time series per variable, and the corresponding dates.
    """
    csv = pd.read_csv(AROME_CSV)
    csv_lt0 = csv[csv["LeadTime"] == 0].copy()

    csv_lt0["Date_dt"] = pd.to_datetime(csv_lt0["Date"])
    csv_lt0 = csv_lt0[(csv_lt0["Date_dt"] >= DATE_MIN_AROME) & (csv_lt0["Date_dt"] <= DATE_MAX_AROME)]

    dates_uniq = csv_lt0["Date"].unique()

    first_file = os.path.join(AROME_DIR, csv_lt0.iloc[0]["Name"])
    sample = np.load(first_file)
    mask_valid = (sample[:, :, 0] != INVALID_VAL) & np.isfinite(sample[:, :, 0])

    spread_arome = {v: [] for v in VARS}
    dates_used = []

    for date in tqdm(dates_uniq, desc="AROME dates"):
        rows = csv_lt0[csv_lt0["Date"] == date]
        if len(rows) < n_members:
            continue
        rows_sampled = rows.sample(n=n_members, random_state=None)

        members_by_var = {v: [] for v in VARS}
        for _, row in rows_sampled.iterrows():
            fpath = os.path.join(AROME_DIR, row["Name"])
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath)
            if data.shape != (717, 1121, 8):
                continue
            for varname, idx in VARS.items():
                field = data[:, :, idx]
                field = np.where(mask_valid, field, np.nan)
                members_by_var[varname].append(field[mask_valid])

        if all(len(members_by_var[v]) == n_members for v in VARS):
            dates_used.append(pd.Timestamp(date))
            for varname in VARS:
                members = np.array(members_by_var[varname])
                spread_arome[varname].append(spatial_spread(members, mask_valid))

    spread_arome_ts = {v: np.array(vals) for v, vals in spread_arome.items()}
    dates_used = pd.DatetimeIndex(dates_used)
    return spread_arome_ts, dates_used


def plot_spread(all_sdedit_results, spread_arome_ts, dates_arome):
    """Plot spread time series for all SDEdit configurations vs AROME.

    Parameters
    ----------
    all_sdedit_results : list[dict]
        Each dict has keys "label", "color", "spread" (dict per variable),
        and "times" (DatetimeIndex).
    spread_arome_ts : dict
        AROME spread time series per variable.
    dates_arome : pd.DatetimeIndex
        Dates corresponding to spread_arome_ts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, varname in zip(axes.flat, VARS):

        for res in all_sdedit_results:
            ax.plot(res["times"], res["spread"][varname],
                    label=res["label"], color=res["color"],
                    linewidth=2)

        ax.plot(dates_arome, spread_arome_ts[varname],
                label="AROME", color="darkorange",
                linewidth=2, linestyle="--")

        ax.set_title(varname, fontsize=13)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Spread", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Spread SDEdit vs AROME", fontsize=14)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


if __name__ == "__main__":

    all_sdedit_results = []
    for cfg in SDEDIT_CONFIGS:
        print(f"\n=== Computing SDEdit spread — {cfg['label']} ===")
        spread, times = compute_sdedit_spread(cfg["dir"], cfg["pattern"], cfg["label"])
        all_sdedit_results.append({
            "label":  cfg["label"],
            "color":  cfg["color"],
            "spread": spread,
            "times":  times,
        })

    print("\n=== Computing AROME spread ===")
    spread_arome_ts, dates_arome = compute_arome_spread(n_members=5)

    print("\n=== Plotting ===")
    plot_spread(all_sdedit_results, spread_arome_ts, dates_arome)

