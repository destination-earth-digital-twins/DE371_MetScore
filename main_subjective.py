#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
# import artistic as art
# import numpy as np
# import random
# import matplotlib as mpl
# import matplotlib.font_manager as fm# Collect all the font names available to matplotlib
# import argparse
# font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.pyplot as plt
#mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["figure.autolayout"] = True
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
import matplotlib as mpl

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
          quantiles_list=[0.01,0.1,0.5,0.9,0.99]
          ):

        cmap = plt.get_cmap("PiYG", 8)
        quantiles_arome = np.quantile(packsample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_gen = np.quantile(pert_sample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        vmins = np.zeros((len(quantiles_list), len(var_names)))
        vmaxs = np.zeros((len(quantiles_list), len(var_names)))

        # Quantiles of AROME
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=len(var_names), ncols=len(quantiles_list))
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
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=len(var_names), ncols=len(quantiles_list))
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

def plot_probability_exceeding_threshold_wind(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u','v','t2m'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          threshold_wind=[5,10,15,20,25,30] # 15m/s = 54km/h
          ):
        r'''
        Plot a map representing the spatial probability to exceed a given threshold.

        '''
        cmap = plt.get_cmap("Blues", 5)
        wind_arome = np.sqrt(packsample[:,dict_var['u']]**2+packsample[:,dict_var['v']]**2)
        wind_gen = np.sqrt(packsample[:,dict_var['u']]**2+packsample[:,dict_var['v']]**2)

        fig, ax = plt.subplots(figsize=(10*len(threshold_wind),15), ncols=len(threshold_wind), nrows=2)
        for threshold_id, threshold in enumerate(threshold_wind):
            max_arome = np.zeros_like(packsample[:,dict_var['u']]) 
            max_gen = np.zeros_like(pert_sample[:,dict_var['u']])

            for id_member, member_arome in enumerate(wind_arome):
                max_arome[id_member] = np.where(member_arome>=threshold, 1, 0)
            im = ax[0][threshold_id].imshow(np.mean(max_arome, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[0][threshold_id].set_title(f'AROME {len(packsample)} members - P(Wind>{threshold*3.6}km/h)')
            fig.colorbar(im, ax=ax[0][threshold_id], shrink=0.5)

            for id_member, member_gen in enumerate(wind_gen):
                max_gen[id_member] = np.where(member_gen>=threshold, 1, 0)
            ax[1][threshold_id].imshow(np.mean(max_gen, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[1][threshold_id].set_title(f'Generated {len(pert_sample)} members - P(Wind>{threshold*3.6}km/h)')
            fig.colorbar(im, ax=ax[1][threshold_id], shrink=0.5)
        
        fig.suptitle(f'Probability Exceeding Wind Threshold map comparison for '+title_info)
        fig.tight_layout()
        fig.savefig(figname_info+'.png', dpi=100)
        plt.close()
        
        return


def plot_probability_exceeding_threshold_temperature(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u','v','t2m'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          threshold_t2m=[10,15,20,30,35,40]
          ):
        r'''
        Plot a map representing the spatial probability to exceed a given threshold.

        '''
        cmap = plt.get_cmap("Reds", 5)
        threshold_t2m = 273.15+np.array(threshold_t2m)

        fig, ax = plt.subplots(figsize=(10*len(threshold_t2m),15), ncols=len(threshold_t2m), nrows=2)
        for threshold_id, threshold in enumerate(threshold_t2m):
            max_arome = np.zeros_like(packsample[:,dict_var['t2m']]) 
            max_gen = np.zeros_like(pert_sample[:,dict_var['t2m']])

            for id_member, member_arome in enumerate(packsample[:,dict_var['t2m']]):
                max_arome[id_member] = np.where(member_arome>=threshold, 1, 0)
            im = ax[0][threshold_id].imshow(np.mean(max_arome, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[0][threshold_id].set_title(f'AROME {len(packsample)} members - P(T>{threshold-273.15}°C)')
            fig.colorbar(im, ax=ax[0][threshold_id], shrink=0.5)

            for id_member, member_gen in enumerate(pert_sample[:,dict_var['t2m']]):
                max_gen[id_member] = np.where(member_gen>=threshold, 1, 0)
            ax[1][threshold_id].imshow(np.mean(max_gen, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[1][threshold_id].set_title(f'Generated {len(pert_sample)} members - P(T>{threshold-273.15}°C)')
            fig.colorbar(im, ax=ax[1][threshold_id], shrink=0.5)
        
        fig.suptitle(f'Probability Exceeding Temperature Threshold map comparison for '+title_info)
        fig.tight_layout()
        fig.savefig(figname_info+'.png', dpi=100)
        plt.close()
        
        return



def plot_panache_min_max(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u','v','t2m'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          quantiles_list=[0.01,0.1,0.5,0.9,0.99],
          coord = [63,38] # Toulouse
          ):
        
        arome = packsample[ :, :, :, coord[0], coord[1]]
        lt,_,v = arome.shape
        arome_min_max = np.zeros((lt,v,2))
        for id_lt in range(lt):
            for id_var, var_name in enumerate(var_names):
                arome_min_max[id_lt][id_var][0] = np.min(arome[id_lt,:,id_var])
                arome_min_max[id_lt][id_var][1] = np.max(arome[id_lt,:,id_var])

        gen = pert_sample[:,:,:,coord[0],coord[1]]
        lt,_,v = gen.shape
        gen_min_max = np.zeros((lt,v,2))
        for id_lt in range(lt):
            for id_var, var_name in enumerate(var_names):
                gen_min_max[id_lt][id_var][0] = np.min(gen[id_lt,:,id_var])
                gen_min_max[id_lt][id_var][1] = np.max(gen[id_lt,:,id_var])

        fig, ax = plt.subplots(figsize=(10,15), ncols=1, nrows=3)
        for id_var, var_name in enumerate(var_names):
            if id_var>0:
                # ax[id_var-1].plot(arome[:,:,id_var], c='blue')
                # ax[id_var-1].plot(gen[:,:,id_var], c='red', alpha=0.5)
                ax[id_var-1].fill_between(list(range(lt)), arome_min_max[:,id_var,0], arome_min_max[:,id_var,1], linewidth=0, color='blue')
                ax[id_var-1].fill_between(list(range(lt)), gen_min_max[:,id_var,0], gen_min_max[:,id_var,1], alpha=.5, linewidth=0, color='red')
        fig.savefig(figname_info+'_.png', dpi=100)
        plt.close()

        
        return

def plot_panache_density(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u (m/s)','v (m/s)','t2m (K)'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          quantiles_list=[0.01,0.1,0.5,0.9,0.99],
          coord = [63,38] # Toulouse
          ):
        
        arome = packsample[ :, :, :, coord[0], coord[1]]
        lt,_,v = arome.shape
        levels=10
        arome_density = np.zeros((lt,v,2,levels))
        for id_lt in range(lt):
            for id_var, var_name in enumerate(var_names):
                minimum = np.min(arome[id_lt,:,id_var])
                maximum = np.max(arome[id_lt,:,id_var])
                mean = np.mean(arome[id_lt,:,id_var])
                # mean = minimum + (maximum-minimum)/2
                density_mem = 0
                for level_id, level in enumerate(np.linspace(0,1,levels)):
                    borne_inf_level = mean - level*(mean-minimum)
                    borne_sup_level = mean + level*(maximum-mean)
                    density = np.mean(np.multiply(arome[id_lt,:,id_var]>=borne_inf_level, arome[id_lt,:,id_var]<=borne_sup_level))
                    print(f'[{borne_inf_level},{borne_sup_level}]', int((density-density_mem)*16))
                    arome_density[id_lt][id_var][0][level_id] = mean - density*(mean-minimum)
                    arome_density[id_lt][id_var][1][level_id] = mean + density*(maximum-mean)
                    density_mem=density
            # raise NotImplementedError

        gen = pert_sample[:,:,:,coord[0],coord[1]]
        gen_density = np.zeros((lt,v,2,levels))
        for id_lt in range(lt):
            for id_var, var_name in enumerate(var_names):
                minimum = np.min(gen[id_lt,:,id_var])
                maximum = np.max(gen[id_lt,:,id_var])
                mean = np.mean(gen[id_lt,:,id_var])
                # mean = minimum + (maximum-minimum)/2
                density_mem = 0
                for level_id, level in enumerate(np.linspace(0,1,levels)):
                    borne_inf_level = mean - level*(mean-minimum)
                    borne_sup_level = mean + level*(maximum-mean)
                    density = np.mean(np.multiply(gen[id_lt,:,id_var]>=borne_inf_level, gen[id_lt,:,id_var]<=borne_sup_level))
                    print(f'[{borne_inf_level},{borne_sup_level}]', int((density-density_mem)*16))
                    gen_density[id_lt][id_var][0][level_id] = mean - density*(mean-minimum)
                    gen_density[id_lt][id_var][1][level_id] = mean + density*(maximum-mean)
                    density_mem=density

        colors=['purple', 'purple', 'magenta','magenta', 'red','red', 'orange','orange', 'gold','gold']
        # colors=['purple', 'magenta', 'red', 'orange','gold']
        colors_lines=['black', 'black', 'dimgrey','dimgrey', 'grey', 'grey', 'silver', 'silver', 'lightgray', 'lightgray']
        fig, ax = plt.subplots(figsize=(25,15), ncols=3, nrows=3)
        ax[0][0].set_title('Panache AROME')
        ax[0][1].set_title('Panache AROME&Generated')
        ax[0][2].set_title('Panache Generated')
        
        for id_var, var_name in enumerate(var_names):
            if id_var>0:
                ax[id_var-1][0].set_xlabel('leadtimes (h)')
                ax[id_var-1][1].set_xlabel('leadtimes (h)')
                ax[id_var-1][2].set_xlabel('leadtimes (h)')
                ax[id_var-1][0].grid(True, linestyle='--')
                ax[id_var-1][0].set_ylabel(var_name)
                ax[id_var-1][0].plot(np.mean(arome[:,:,id_var], axis=1), c='black', linewidth=2)
                ax[id_var-1][1].grid(True, linestyle='--')
                ax[id_var-1][1].set_ylabel(var_name)
                ax[id_var-1][2].grid(True, linestyle='--')
                ax[id_var-1][2].set_ylabel(var_name)
                ax[id_var-1][2].plot(np.mean(gen[:,:,id_var], axis=1), c='black', linewidth=2)
                # ax[id_var-1].plot(arome[:,:,id_var], c='blue', linewidth=1, alpha=0.2)
                # ax[id_var-1].plot(gen[:,:,id_var], c='red', alpha=0.5)
                for level_id in reversed(range(0,levels)):
                    # AROME panache
                    ax[id_var-1][0].fill_between(list(range(lt)), arome_density[:,id_var,0, level_id], arome_density[:,id_var,1, level_id], linewidth=0, color=colors[level_id])
                    ax[id_var-1][1].fill_between(list(range(lt)), arome_density[:,id_var,0, level_id], arome_density[:,id_var,1, level_id], linewidth=0, color=colors[level_id])
                    ax[id_var-1][2].fill_between(list(range(lt)), gen_density[:,id_var,0, level_id], gen_density[:,id_var,1, level_id], linewidth=0, color=colors[level_id])
                    # Generated panache (only lines)
                    if (level_id+1)%2==0 :
                        ax[id_var-1][1].plot(list(range(lt)), gen_density[:,id_var,0, level_id], linewidth=1, color=colors_lines[level_id])
                        ax[id_var-1][1].plot(list(range(lt)), gen_density[:,id_var,1, level_id], linewidth=1, color=colors_lines[level_id])
                # ax[id_var-1].fill_between(list(range(lt)), gen_min_max[:,id_var,0], gen_min_max[:,id_var,1], alpha=.5, linewidth=0, color='red')
                
        fig.savefig(figname_info+'__test.png', dpi=100)
        plt.close()

        
        return

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--gen_data_dir', type = str,default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit/sampling_sdedit_ddpm/sampling_10steps/')
    # Output Directory - PATH where the output of the inversion will be saved
    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default='Large_lt_val_labels_ens.csv')
    parser.add_argument("--date_start", type=str, default = "2020-08-01")
    parser.add_argument("--date_stop", type=str, default = "2020-08-10")
    parser.add_argument("--leadtimes", type=str2intlist, default=[3,6,9,12,15,18,21,24,27,30,33,36,39,42])
    parser.add_argument("--var_indices", type=str2intlist, default=[0,1,2,3])
    parser.add_argument("--var_data", type=str2intlist, default=['rr','u','v','t2m'])
    
    params = parser.parse_args()
    output_dir = params.gen_data_dir + f'subjective_scores/{params.date_start}_to_{params.date_stop}/'
    params.gen_data_dir += 'unbiased_samples/'
    # create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ################## loading dates and file names ##
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date['Date'] = pd.to_datetime(df_date['Date'])
    df_extract = df_date[(df_date['Date']>=params.date_start) & (df_date['Date']<=params.date_stop)]

    list_dates = df_extract['Date'].unique()


    #################### main loop ##################
    for date_ in list_dates:
        datename = date_.strftime('%Y-%m-%d')
        Ens_AROME_date = []
        Ens_Gen_date = []
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

            # plot_probability_exceeding_threshold_temperature(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'{datename}_{lt}',
            #     figname_info=output_dir+f'probability_temperature_threshold_{datename}_{lt}',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
            #     axis_title_global=''
            # )

            # plot_probability_exceeding_threshold_wind(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'{datename}_{lt}',
            #     figname_info=output_dir+f'wind_zone_over_5_{datename}_{lt}',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
            #     axis_title_global='',
            #     threshold_wind=[5,10,15,20,25,30]
            # )

            # plot_quantiles(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'Quantile {datename}_{lt}',
            #     figname_info=output_dir+f'quantiles_{datename}_{lt}',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
            #     axis_title_global='',
            #     quantiles_list=[0.01,0.1,0.5,0.9,0.99]      
            # )
            
            # data2plot = Ens_AROME[0]

            # canvas = art.canvasHolder("MassifCentral",256,256)
            # Datamax = data2plot.max(axis=(0,-1, -2))
            # Datamin = data2plot.min(axis=(0,-1, -2))
            # var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
            # #data2plot0 = data2plot[(0,1,2,4),:,:,:]
            # canvas.plot_data_normal(data2plot, var_names, output_dir, f"{'test'}.pdf", contrast=True,
            #                     cvalues=(Datamin, Datamax))

            # var_names = [('ff', 'm/s'), ('t2m', 'K')]
            # canvas.plot_data_ff_t2m(data2plot, var_names, output_dir, f"{'test'}_fft2m.pdf", contrast=False,
            #                 )
            Ens_AROME_date.append(Ens_AROME)
            Ens_Gen_date.append(Ens_Gen)
        plot_panache_density(np.array(Ens_AROME_date), np.array(Ens_Gen_date),figname_info=output_dir+f'panache_{datename}',)
            
            











