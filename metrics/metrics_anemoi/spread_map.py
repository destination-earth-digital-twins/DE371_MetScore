"""
spread_map.py
==================
Compares the spatial spread (ensemble standard deviation map) of three SDEdit
configurations against the AROME reference ensemble, for a single variable,
date, and lead time.

For the AROME reference ensemble, this script:

  - Reads the CSV catalogue to find available members for TARGET_DATE and
    TARGET_LEADTIME
  - Randomly samples N_MEMBERS members (fixed seed)
  - Computes the per-pixel standard deviation across members, normalized by
    its maximum value

For each SDEdit configuration listed in SDEDIT_CONFIGS, this script:

  - Loads the first N_MEMBERS matching NetCDF files
  - Pivots each member's flat (lat, lon, value) data into a 2D field
  - Computes the per-pixel standard deviation across members, normalized by
    its maximum value

Finally, the four spread maps (AROME + 3 SDEdit configurations) are plotted
side by side in a 2x2 grid.

Configuration
-------------
NETCDF_DIR      : directory containing the SDEdit NetCDF files
SDEDIT_CONFIGS  : list of configurations to compare (label + file pattern)
AROME_DIR       : directory containing the AROME reference ensemble (.npy files)
AROME_CSV       : CSV catalogue referencing AROME files (Name, Date, Member, LeadTime)
OUTPUT_DIR      : directory where the output figure is saved
VARNAME         : variable to plot
TARGET_DATE     : date used to select the AROME ensemble
TARGET_LEADTIME : lead time used to select the AROME ensemble
N_MEMBERS       : number of ensemble members used to compute the spread
RANDOM_SEED     : seed used when sampling AROME members
TIME_IDX        : time index used to select the SDEdit timestep
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── Config ───────────────────────────────────────────────────────────────

NETCDF_DIR = "path/to/netcdf/dir/" #dir containing the generated netcdf

SDEDIT_CONFIGS = [
    {"label": "config 1 ",  "pattern": "pattern_config_1.nc"},
    {"label": "config 2", "pattern": "pattern_config_2.nc"},
    {"label": "config 3", "pattern": "pattern_config_3.nc"},
]

AROME_DIR = "path/to/AROME/dataset"  # reference ensemble dataset
AROME_CSV = "path/to/csv.csv"        # CSV referencing files in AROME_DIR (Name, Date, Member, LeadTime)

OUTPUT_DIR = "path/to/output/dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variable to plot
VARNAME = "10u"

# Target date and lead time for AROME selection
TARGET_DATE = "YYYY-MM-DDTHH:mm:ssZ"
TARGET_LEADTIME = 3

VARS = {
    "10u":   0,
    "10v":   1,
    "2t":    2,
    "z_500": 7,
}

GRID_SHAPE = (717, 1121)
INVALID_VAL = 9999
N_MEMBERS = 10
RANDOM_SEED = 42
TIME_IDX = 1  # time index in the SDEdit NetCDF files


# ── Helpers ──────────────────────────────────────────────────────────────

def add_cbar(fig, ax, mappable, size="4%", pad=0.05):
    """Add a colorbar to the right of an axis without resizing it."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    return fig.colorbar(mappable, cax=cax)


def get_mask(sample_2d):
    """Return a boolean mask of valid (non-missing, finite) grid points."""
    return (sample_2d != INVALID_VAL) & np.isfinite(sample_2d)


# ── 1. AROME spread map ─────────────────────────────────────────────────

def compute_arome_spread_map(varname):
    """Compute the normalized AROME ensemble spread map.

    Parameters
    ----------
    varname : str
        Variable name to compute the spread for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized spread map (H, W), and the corresponding valid-points mask.
    """
    csv = pd.read_csv(AROME_CSV)
    rows = csv[(csv["Date"] == TARGET_DATE) & (csv["LeadTime"] == TARGET_LEADTIME)]

    members_available = rows["Member"].unique()
    print(f"AROME: {len(members_available)} members available")

    if len(members_available) < N_MEMBERS:
        raise ValueError(f"Not enough AROME members: {len(members_available)} < {N_MEMBERS}")

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(members_available, size=N_MEMBERS, replace=False)
    rows_sel = rows[rows["Member"].isin(selected)]

    idx = VARS[varname]
    fields = []
    mask_valid = None

    for _, row in rows_sel.iterrows():
        fpath = os.path.join(AROME_DIR, row["Name"])
        if not os.path.exists(fpath):
            print(f"  [WARN] missing file: {fpath}")
            continue
        data = np.load(fpath)
        if data.shape != (*GRID_SHAPE, 8):
            print(f"  [WARN] unexpected shape: {data.shape} — skipped")
            continue
        field = data[:, :, idx].astype(float)
        if mask_valid is None:
            mask_valid = get_mask(data[:, :, 0])
        field[~mask_valid] = np.nan
        field = field[1:, :]
        fields.append(field)

    stack = np.array(fields)
    spread_map = np.nanstd(stack, axis=0, ddof=1)
    spread_map = spread_map / np.nanmax(spread_map)

    mask_valid = mask_valid[1:, :]
    return spread_map, mask_valid


# ── 2. SDEdit spread map (one configuration) ────────────────────────────

def compute_sdedit_spread_map(pattern, varname):
    """Compute the normalized SDEdit ensemble spread map for one configuration.

    Parameters
    ----------
    pattern : str
        Glob pattern to select the NetCDF files for this configuration.
    varname : str
        Variable name to compute the spread for.

    Returns
    -------
    np.ndarray
        Normalized spread map (H, W).
    """
    nc_files = sorted(glob.glob(os.path.join(NETCDF_DIR, pattern)))
    print(f"  {pattern}: {len(nc_files)} files found")

    if len(nc_files) < N_MEMBERS:
        raise ValueError(f"Not enough NetCDF files: {len(nc_files)} < {N_MEMBERS}")

    fields = []
    for f in nc_files[:N_MEMBERS]:
        ds = xr.open_dataset(f)
        flat = ds[varname].isel(time=TIME_IDX).values
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        df = pd.DataFrame({
            "lat":   np.round(lat, 3),
            "lon":   np.round(lon, 3),
            "value": flat,
        })
        field_2D = df.pivot(index="lat", columns="lon", values="value")
        fields.append(field_2D.values)

    stack = np.array(fields)
    spread_map = np.nanstd(stack, axis=0, ddof=1)
    spread_map = spread_map / np.nanmax(spread_map)
    return spread_map


# ── 3. 2x2 plot ──────────────────────────────────────────────────────────

def plot_2x2(spread_arome, sdedit_results, mask_valid, varname):
    """Plot AROME and SDEdit spread maps side by side in a 2x2 grid.

    Parameters
    ----------
    spread_arome : np.ndarray
        AROME spread map, shape (H, W).
    sdedit_results : list[dict]
        Each dict has keys "label" and "spread" (H, W array).
    mask_valid : np.ndarray
        Boolean mask of valid grid points, shape (H, W).
    varname : str
        Variable name, used in the title and output filename.
    """
    panels = [{"label": "AROME", "spread": spread_arome}] + sdedit_results

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Ensemble spread - {varname} - 10 members",
        fontsize=13, fontweight="bold"
    )

    for ax, panel in zip(axes.flat, panels):
        data_plot = np.where(mask_valid, panel["spread"], np.nan)
        vmin = np.nanmin(data_plot)
        vmax = np.nanmax(data_plot)
        im = ax.imshow(data_plot, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(panel["label"], fontsize=11)
        ax.axis("off")
        add_cbar(fig, ax, im)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"spread_map_2x2_{varname}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figure saved: {out}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== AROME spread — {VARNAME} ===")
    spread_arome, mask_valid = compute_arome_spread_map(VARNAME)

    print(f"\n=== SDEdit spread — {VARNAME} ===")
    sdedit_results = []
    for cfg in SDEDIT_CONFIGS:
        print(f"  -> {cfg['label']}")
        spread = compute_sdedit_spread_map(cfg["pattern"], VARNAME)
        sdedit_results.append({"label": cfg["label"], "spread": spread})

    print("\n=== 2x2 plot ===")
    plot_2x2(spread_arome, sdedit_results, mask_valid, VARNAME)

