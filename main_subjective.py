#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import torch
import argparse
import os
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from ast import literal_eval as make_tuple
import glob

torch.manual_seed(42) #reproducibility of runs
def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def plot_quantiles(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          axis_title_global='',
          quantiles_list=[0.01,0.1,0.9,0.99]
          ):

        cmap = plt.get_cmap("PiYG", 8)
        quantiles_arome = np.quantile(packsample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_gen = np.quantile(pert_sample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        vmins = np.zeros((len(quantiles_list), len(var_names)))
        vmaxs = np.zeros((len(quantiles_list), len(var_names)))

        # Quantiles of AROME
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=3, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                vmins[quantile_idx][var_idx] = np.min(
                    [np.min(quantiles_arome[quantile_idx][var_idx])]
                )
                vmaxs[quantile_idx][var_idx] = np.min(
                    [np.max(quantiles_arome[quantile_idx][var_idx])]
                )

                clim = (vmins[quantile_idx][var_idx],vmaxs[quantile_idx][var_idx])

                ax[var_idx][quantile_idx].set_title(f"{axis_title_global}{var} real - Q{quantile}")
                im = ax[var_idx][quantile_idx].imshow(quantiles_arome[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                fig.colorbar(im, ax=ax[var_idx][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of AROME ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_AROME.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'_AROME.png'}")
        plt.close()
        
        # Quantiles of Generated samples
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=3, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                clim = (vmins[quantile_idx][var_idx],vmaxs[quantile_idx][var_idx])

                ax[var_idx][quantile_idx].set_title(f"{axis_title_global}{var} GEN - Q{quantile}")
                im = ax[var_idx][quantile_idx].imshow(quantiles_gen[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                fig.colorbar(im, ax=ax[var_idx][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of Generated ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_GEN.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'_GEN.png'}")
        plt.close()
        return

def plot_max_temperature(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u','v','t2m'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          quantiles_list=[0.01,0.1,0.5,0.9,0.99],
          threshold_t2m=303.15 # 30°C
          ):

        cmap = plt.get_cmap("PiYG", 8)
        max_arome = np.zeros_like(packsample[:,dict_var['t2m']]) 
        max_gen = np.zeros_like(pert_sample[:,dict_var['t2m']])

        fig, ax = plt.subplots(figsize=(10,15), nrows=1, ncols=2)

        for id_member, member_arome in enumerate(packsample[:,dict_var['t2m']]):
            max_arome[id_member] = np.clip(member_arome,threshold_t2m)
            ax[0].imshow(max_arome[id_member], origin="lower")

        for id_member, member_gen in enumerate(pert_sample[:,dict_var['t2m']]):
            max_gen[id_member] = np.clip(member_gen,threshold_t2m)
            ax[1].imshow(max_arome[id_member], origin="lower")
        
        fig.suptitle('Maximum comparison of AROME end Generated ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'.png'}")
        plt.close()
        
        return

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--gen_data_dir', type = str,default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit_ED/sampling_1steps/samples/')
    # Output Directory - PATH where the output of the inversion will be saved
    parser.add_argument('--output_dir',type = str, default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit_ED/subjective_scores/')

    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default='Large_lt_val_labels_ens.csv')
    parser.add_argument("--date_start", type=str, default = "2020-07-01")
    parser.add_argument("--date_stop", type=str, default = "2020-07-02")
    parser.add_argument("--leadtimes", type=str2intlist, default=[3])
    parser.add_argument("--var_indices", type=str2intlist, default=[0,1,2,3])
    parser.add_argument("--var_data", type=str2intlist, default=['rr','u','v','t2m'])
    
    params = parser.parse_args()

    # create output directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()


    #################### main loop ##################
    for date_ in list_dates:
        datename = date_.strftime('%Y-%m-%d')
        for lt in params.leadtimes:
            # Loading AROME ensemble   
            df0 = df_extract[(df_extract['Date']==date_) & (df_extract['LeadTime']==lt-1)]
            Nb = len(df0)
            Ens_AROME = np.zeros((Nb,) + tuple((4,256,256)))
            for i,s in enumerate(df0['Name']):
                sn = np.load(f'{params.real_data_dir}{s}.npy')[params.var_indices,:,:].astype(np.float32)
                Ens_AROME[i] = sn

            # Loading Generated ensemble
            print(f'Importing 4var_fake_ensemble_{datename}_{lt}.npy')
            Ens_Gen = np.load(f'{params.gen_data_dir}4var_fake_ensemble_{datename}_{lt}.npy')

            # # Maximum/Minimum temperature map
            # max_t2m_map = np.zeros((Ens_AROME.shape[-2], Ens_AROME.shape[-1]))
            # id_t2m = 3
            # a = np.array([[[0,4],[1,5]],[[2,1],[1,2]],[[0,1],[3,3]]])
            # # Warning : If the maximum is not unique, it is choosing the first encountered
            # print([np.unravel_index(np.argmax(a[i]), (2,2)) for i in range(3)])
            # # print(np.max(Ens_AROME[:,id_t2m], axis=0).shape)
            # # brightest_pixel_index = [np.unravel_index(np.max(Ens_AROME[i,id_t2m], axis=0).argmax(), (Ens_AROME.shape[-2], Ens_AROME.shape[-1])) for i in range(16)]
            # # print(brightest_pixel_index)
            # max_t2m_map = np.clip(Ens_AROME[:,id_t2m],a_min=303.15)
            plot_max_temperature(
                packsample, 
                pert_sample, 
                title_info=f" ", 
                figname_info=".png",  
                var_names=['rr', 'u','v','t2m'], 
                dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
                axis_title_global='',
                quantiles_list=[0.01,0.1,0.5,0.9,0.99],
                threshold_t2m=303.15 # 30°C
            )
            
            











