#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble evaluation on a given situation

"""
import argparse
import os

import matplotlib

# import artistic as art
# import numpy as np
# import random
# import matplotlib as mpl
# import matplotlib.font_manager as fm# Collect all the font names available to matplotlib
# import argparse
# font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# mpl.rcParams['font.family'] = 'Helvetica'


matplotlib.use("Agg")
plt.rcParams["font.size"] = 20
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["figure.autolayout"] = True
torch.manual_seed(42)  # reproducibility of runs


def str2intlist(li):
    if type(li) is list:
        li2 = [int(p) for p in li]
        return li2

    elif type(li) is str:
        li2 = li[1:-1].split(",")
        li3 = [int(p) for p in li2]
        return li3

    else:
        raise ValueError(
            "li argument must be a string or a list, not '{}'".format(type(li))
        )


var_names = ["u", "v", "t2m"]
dict_var = {"u": 1, "v": 2, "t2m": 3}
colormap_var = ["viridis", "viridis", "coolwarm"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument(
        "--real_data_dir",
        type=str,
        default="/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/",
    )
    # Output Directory - PATH where the output of the inversion will be saved
    ########################## CONTROL of Data to invert ######################
    parser.add_argument(
        "--dates_file", type=str, default="Large_lt_test_labels_ens.csv"
    )
    parser.add_argument("--date_start", type=str, default="2021-10-01")
    parser.add_argument("--date_stop", type=str, default="2021-10-02")
    parser.add_argument("--leadtimes", type=str2intlist, default=[36])
    parser.add_argument("--var_indices", type=str2intlist, default=[0, 1, 2, 3])
    parser.add_argument("--var_data", type=str2intlist, default=["rr", "u", "v", "t2m"])
    parser.add_argument(
        "--orography_path",
        type=str,
        default="/project/home/p200177/DE_371/resources/AROME_orography/PEARO_EURW1S40_Orography.npy",
    )

    params = parser.parse_args()
    directory = (
        params.output_dir + "scores_subjective_comparison_conditionning_method/test/"
    )
    # create output directories
    if not os.path.exists(directory):
        os.makedirs(directory)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date["Date"] = pd.to_datetime(df_date["Date"])
    df_extract = df_date[
        (df_date["Date"] >= params.date_start) & (df_date["Date"] <= params.date_stop)
    ]

    list_dates = df_extract["Date"].unique()

    orog = np.load(params.orography_path)[180:436, 500:756]
    #################### main loop ##################
    for date_ in list_dates:
        datename = date_.strftime("%Y-%m-%d")
        for lt in params.leadtimes:
            # Loading AROME ensemble
            df0 = df_extract[
                (df_extract["Date"] == date_) & (df_extract["LeadTime"] == lt - 1)
            ]
            Nb = len(df0)
            Ens_AROME = np.zeros((Nb,) + tuple((4, 256, 256)))
            for i, s in enumerate(df0["Name"]):
                sn = np.load(f"{params.real_data_dir}{s}.npy")[
                    params.var_indices, :, :
                ].astype(np.float32)
                Ens_AROME[i] = sn

            # Loading Generated ensemble
            crop = [0, 256, 0, 256]
            orog = orog[crop[0] : crop[1], crop[2] : crop[3]]
            print(f"Importing 4var_fake_ensemble_{datename}_{lt}.npy")
            title_info = f"{datename}_{lt}"
            figname_info = directory + f"{datename}_{lt}"
            coord = [229, 44]
            fig, ax = plt.subplots(
                figsize=(5 * (len(var_names) + 1), 5), nrows=1, ncols=len(var_names) + 1
            )
            x = y = np.arange(0, 256)
            X, Y = np.meshgrid(x, y)
            CS = ax[3].contourf(X, Y, np.where(orog < 10, orog, 0), 5, cmap=plt.cm.bone)
            scatter = np.zeros_like(Ens_AROME)
            for id, var in enumerate(var_names):
                var_id = dict_var[var]
                vmin = np.min(Ens_AROME[0, var_id])
                vmax = np.max(Ens_AROME[0, var_id])
                clim = (vmin, vmax)
                ax[id].set_title(f"{var} real")
                im = ax[id].imshow(
                    Ens_AROME[0, var_id][crop[0] : crop[1], crop[2] : crop[3]],
                    origin="lower",
                    cmap=colormap_var[id],
                    clim=clim,
                )
                fig.colorbar(im, ax=ax[id], shrink=0.5)
                # rect1 = matplotlib.patches.Rectangle((50,200), 50, 50, color='blue', fc = 'none',lw = 2)
                # ax[id].add_patch(rect1)
                scatter[0, var_id][coord[0], coord[1]] = 1
                ax[id].imshow(
                    scatter[0, var_id][crop[0] : crop[1], crop[2] : crop[3]],
                    origin="lower",
                    cmap="Greys",
                    alpha=0.5,
                )
                # ax[id].scatter(coord[1], coord[0], color='red', linewidth=1)
                ax[id].contour(CS, levels=CS.levels[::2], colors="black")

            fig.suptitle(title_info)
            fig.tight_layout()
            fig.savefig(figname_info + ".png", dpi=100)
            plt.close()
