"""
quantile_comparison.py
=======================
Compares the distribution of inference outputs against the AROME reference
dataset, for a set of surface variables (2t, 10u, 10v).

For a given period (DATE_START to DATE_END), this script:

  1. Computes spatial quantile maps (Q10, Q50, Q90) from a random sample of
     AROME reference dates (drawn from the Zarr dataset), and saves them as
     .npy files and PNG maps.

  2. Computes the same quantile maps from the inference NetCDF files
     (concatenating all available samples), and saves them as .npy files
     and PNG maps.

  3. Computes and plots the difference (inference - AROME) for each quantile,
     as spatial maps.

  4. Produces QQ-plots comparing the full distribution of inference outputs
     against the AROME reference distribution, for each variable.

Configuration
-------------
NETCDF     : directory containing the inference NetCDF files
ZARR_PATH  : path to the reference Zarr dataset (e.g. AROME)
OUTPUT_DIR : directory where .npy arrays and PNG plots are saved

Output files
-------------
- q{10,50,90}_<var>_test_arome.npy / .png : AROME reference quantile maps
- q{10,50,90}_<var>.npy / .png            : inference quantile maps
- diff_<var>_inf_*_train_arome.png        : difference maps (inference - AROME)
- qqplot_inf_arome_*.png                  : QQ-plots (inference vs AROME)
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import netCDF4 as nc
import glob
from tqdm import tqdm


NETCDF = 'path/to/netcdf/dir'   # dir containing the generated netcdf files
ZARR_PATH = 'path/to/zarr/dataset'  # reference zarr dataset (e.g. AROME)
OUTPUT_DIR = 'path/to/output/dir'

files = sorted(glob.glob(f'{NETCDF}/pattern.nc'))  # find all files matching the chosen pattern
print(f"{len(files)} files found")


# ─────────────────────────────────────────────────────────────────────────
# 1. AROME reference quantile maps
# ─────────────────────────────────────────────────────────────────────────

ds = zarr.open(ZARR_PATH, mode='r')
dates = ds['dates'][:]  # array of datetime64

date_start = np.datetime64('2020-06-01')
date_end = np.datetime64('2020-07-01')

mask = (dates >= date_start) & (dates <= date_end)
available_idx = np.where(mask)[0]

# Randomly sample reference dates to compute AROME quantiles
n_samples = min(500, len(available_idx))
np.random.seed(42)
indices_dates = np.random.choice(available_idx, size=n_samples, replace=False)
indices_dates.sort()

variables = list(ds.attrs['variables'])
vars_to_plot = ['2t', '10u', '10v']
indices_vars = {v: variables.index(v) for v in vars_to_plot}

lats = ds['latitudes'][:]
lons = ds['longitudes'][:]
triang = tri.Triangulation(lons, lats)

for varname, var_idx in indices_vars.items():
    print(f"Processing {varname}...")
    data = ds['data'].oindex[indices_dates, var_idx, 0, :]  # shape (n_samples, n_points)

    q10 = np.nanpercentile(data, 10, axis=0)
    q50 = np.nanpercentile(data, 50, axis=0)
    q90 = np.nanpercentile(data, 90, axis=0)

    vmin = np.nanmin(q10)
    vmax = np.nanmax(q90)

    np.save(f'{OUTPUT_DIR}/q10_{varname}_test_arome.npy', q10)
    np.save(f'{OUTPUT_DIR}/q50_{varname}_test_arome.npy', q50)
    np.save(f'{OUTPUT_DIR}/q90_{varname}_test_arome.npy', q90)

    # Plot AROME quantile maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, values, title in zip(axes, [q10, q50, q90], ['Q10', 'Q50', 'Q90']):
        im = ax.tripcolor(triang, values, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=varname, shrink=0.8)
        ax.set_title(f'{title} — {varname}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.suptitle(f'AROME quantiles: {varname}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/quantiles_{varname}_test_arome.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
# 2. Inference quantile maps
# ─────────────────────────────────────────────────────────────────────────

ds_ref = nc.Dataset(files[0])
lats = ds_ref['latitude'][:]
lons = ds_ref['longitude'][:]
ds_ref.close()

triang = tri.Triangulation(lons, lats)

vars_config = {
    '2t':  {'cmap': 'coolwarm', 'label': '2t (K)'},
    '10u': {'cmap': 'coolwarm', 'label': '10u (m/s)'},
    '10v': {'cmap': 'coolwarm', 'label': '10v (m/s)'},
}

for varname, cfg in vars_config.items():
    print(f"Loading {varname}...")
    data = []
    files_valid = 0
    for f in tqdm(files, desc=f"  read {varname}"):
        try:
            d = nc.Dataset(f)
            data.append(np.array(d[varname][1:, ]))  # skip the initial state
            d.close()
            files_valid += 1
        except Exception as e:
            print(f"\n  ignored file: {f} ({e})")

    data = np.concatenate(data, axis=0)  # shape (n_total_samples, n_points)

    print("  computing quantiles...")
    q10 = np.nanpercentile(data, 10, axis=0)
    q50 = np.nanpercentile(data, 50, axis=0)
    q90 = np.nanpercentile(data, 90, axis=0)

    np.save(f'{OUTPUT_DIR}/q10_{varname}.npy', q10)
    np.save(f'{OUTPUT_DIR}/q50_{varname}.npy', q50)
    np.save(f'{OUTPUT_DIR}/q90_{varname}.npy', q90)
    print(f"  saved .npy files for {varname}")

    vmin = np.nanmin(q10)
    vmax = np.nanmax(q90)

    # Plot inference quantile maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, values, title in zip(axes, [q10, q50, q90], ['Q10', 'Q50', 'Q90']):
        im = ax.tripcolor(triang, values, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=cfg['label'], shrink=0.8)
        ax.set_title(f'{title} — {varname}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.suptitle(f'Quantiles of {varname} — {len(files)} samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/quantiles_{varname}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
# 3. Difference maps (inference - AROME) and AROME-only plots
# ─────────────────────────────────────────────────────────────────────────

ds_ref = nc.Dataset(files[0])
lats = ds_ref['latitude'][:]
lons = ds_ref['longitude'][:]
ds_ref.close()

triang = tri.Triangulation(lons, lats)

vars_config = {
    '2t':  {'cmap': 'coolwarm', 'label': '2t (K)'},
    '10u': {'cmap': 'coolwarm', 'label': '10u (m/s)'},
    '10v': {'cmap': 'coolwarm', 'label': '10v (m/s)'},
}

for varname, cfg in vars_config.items():
    q10_inf = np.load(f'{OUTPUT_DIR}/q10_{varname}_inference_193K.npy')
    q50_inf = np.load(f'{OUTPUT_DIR}/q50_{varname}_inference_193K.npy')
    q90_inf = np.load(f'{OUTPUT_DIR}/q90_{varname}_inference_193K.npy')

    q10_arome = np.load(f'{OUTPUT_DIR}/q10_{varname}_test_arome.npy')
    q50_arome = np.load(f'{OUTPUT_DIR}/q50_{varname}_test_arome.npy')
    q90_arome = np.load(f'{OUTPUT_DIR}/q90_{varname}_test_arome.npy')

    diff_q10 = q10_inf - q10_arome
    diff_q50 = q50_inf - q50_arome
    diff_q90 = q90_inf - q90_arome

    # Symmetric colorbar centered on 0
    vmax = np.nanmax(np.abs([diff_q10, diff_q50, diff_q90]))
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, diff, title in zip(axes, [diff_q10, diff_q50, diff_q90], ['Q10', 'Q50', 'Q90']):
        im = ax.tripcolor(triang, diff, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=cfg['label'], shrink=0.8)
        ax.set_title(f'Diff {title} — {varname} (inf - AROME)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.suptitle(f'Difference inference - AROME: {varname}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/diff_{varname}_inf_193K_train_arome.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/diff_{varname}_inf_193K_train_arome.png")

    # AROME-only quantile maps for reference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, values, title in zip(axes, [q10_arome, q50_arome, q90_arome], ['Q10', 'Q50', 'Q90']):
        vmin = np.nanmin(q10_arome)
        vmax = np.nanmax(q90_arome)
        im = ax.tripcolor(triang, values, cmap=cfg['cmap'], vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=varname, shrink=0.8)
        ax.set_title(f'{title} — {varname}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.suptitle(f'Quantiles of {varname} — AROME (500 dates)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/quantiles_{varname}_arome.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: quantiles_{varname}_arome.png")


# ─────────────────────────────────────────────────────────────────────────
# 4. QQ-plots: inference vs AROME (full distribution)
# ─────────────────────────────────────────────────────────────────────────

print("QQ-plots")

vars_config = {
    '2t':  {'label': '2t (K)',    'color': '#D85A30'},
    '10u': {'label': '10u (m/s)', 'color': '#378ADD'},
    '10v': {'label': '10v (m/s)', 'color': '#1D9E75'},
}

percentiles = np.arange(1, 100, 1)

# Reload AROME reference dates/variables
ds_zarr = zarr.open(ZARR_PATH, mode='r')
variables = list(ds_zarr.attrs['variables'])
dates = ds_zarr['dates'][:]

mask = (dates >= date_start) & (dates <= date_end)
available_idx = np.where(mask)[0]
n_samples = min(500, len(available_idx))
np.random.seed(42)
indices_dates = np.random.choice(available_idx, size=n_samples, replace=False)
indices_dates.sort()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (varname, cfg) in zip(axes, vars_config.items()):
    # Load and flatten all inference samples and grid points
    data_inf = []
    for f in tqdm(files, desc=f'Loading inference {varname}'):
        try:
            d = nc.Dataset(f)
            arr = np.array(d[varname][1:, :])  # skip the initial state
            data_inf.append(arr.reshape(-1))
            d.close()
        except Exception as e:
            print(f"Ignored file: {f} ({e})")

    data_inf = np.concatenate(data_inf)

    # Load and flatten AROME reference samples and grid points
    var_idx = variables.index(varname)
    data_arome = ds_zarr['data'].oindex[indices_dates, var_idx, 0, :]  # shape (n_samples, n_points)
    data_arome = data_arome.flatten()

    # Compute percentiles over the full distribution
    q_inf = np.nanpercentile(data_inf, percentiles)
    q_arome = np.nanpercentile(data_arome, percentiles)

    # y=x reference line
    vmin = min(q_inf.min(), q_arome.min())
    vmax = max(q_inf.max(), q_arome.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, label='y=x')

    # QQ-plot
    ax.scatter(q_arome, q_inf, color=cfg['color'], s=20, alpha=0.7)

    ax.set_xlabel(f'AROME — {cfg["label"]}')
    ax.set_ylabel(f'Inference — {cfg["label"]}')
    ax.set_title(f'QQ-plot {varname}')
    ax.legend()
    ax.set_aspect('equal')

plt.suptitle('QQ-plot inference vs AROME', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/qqplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: qqplot_inf_arome.png")

