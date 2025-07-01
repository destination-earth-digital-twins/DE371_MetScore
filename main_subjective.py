#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs ensemble evaluation on a given situation

"""
# import artistic as art
# import numpy as np
# import random
# import matplotlib as mpl
# import matplotlib.font_manager as fm# Collect all the font names available to matplotlib
# import argparse
# font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
from core.useful_funcs import obs_clean
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


def plot_mean(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          figtitle=" ", 
          figname=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','viridis','coolwarm'],
          clim_global=[(0,0),(-5,5),(-5,5),(270,300)],
          axis_title_global=''
          ):
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)-5), nrows=3, ncols=len(var_names)-1)
        for id, var in enumerate(var_names):
            if id>0:
                var_id = dict_var[var]
                if not clim_global :
                    vmin = np.min([np.min(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                    vmax = np.min([np.max(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                    clim_mean = (vmin, vmax)
                else :
                    clim_mean = clim_global[id-1]
                ax[0][id-1].set_title(f"mean {axis_title_global}{var} real")
                im = ax[0][id-1].imshow(np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0), origin="lower", cmap=colormap_var[id-1], clim=clim_mean)
                fig.colorbar(im, ax=ax[0][id-1], shrink=0.5)

                ax[1][id-1].set_title(f"mean {axis_title_global}{var} generated")
                im = ax[1][id-1].imshow(np.mean(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0),  origin="lower", cmap=colormap_var[id-1], clim=clim_mean)
                fig.colorbar(im, ax=ax[1][id-1], shrink=0.5)

                diff = np.mean(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0) - np.mean(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0)
                ax[2][id-1].set_title(f"diff of mean {axis_title_global}{var}")
                im = ax[2][id-1].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
                fig.colorbar(im, ax=ax[2][id-1], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def plot_var(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          figtitle=" ", 
          figname=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          clim_global=[(0,5),(0,5),(0,5)],
          axis_title_global=''
          ):
        fig, ax = plt.subplots(figsize=(15,5*len(var_names)-5), nrows=3, ncols=len(var_names)-1)
        for id, var in enumerate(var_names):
            if id>0:
                var_id = dict_var[var]
                if not clim_global :
                    vmin = np.min([np.min(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                    vmax = np.min([np.max(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]]))], axis=0)
                    clim_var = (vmin, vmax)
                else :
                    clim_var = clim_global[id-1]
                ax[0][id-1].set_title(f"var {axis_title_global}{var} real")
                im = ax[0][id-1].imshow(np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0), origin="lower", cmap=colormap_var[id-1], clim=clim_var)
                fig.colorbar(im, ax=ax[0][id-1], shrink=0.5)

                ax[1][id-1].set_title(f"var {axis_title_global}{var} generated")
                im = ax[1][id-1].imshow(np.var(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0),  origin="lower", cmap=colormap_var[id-1], clim=clim_var)
                fig.colorbar(im, ax=ax[1][id-1], shrink=0.5)

                diff = np.var(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0) - np.var(pert_sample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]], axis=0)
                ax[2][id-1].set_title(f"diff var {axis_title_global}{var}")
                im = ax[2][id-1].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
                fig.colorbar(im, ax=ax[2][id-1], shrink=0.5)

        fig.suptitle(figtitle)
        fig.tight_layout()
        try:
            fig.savefig(figname, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname}")
        plt.close()
        return


def plot_samples(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          mem_idx=0, 
          mem_pert_idx=0,
          title_info=" ", 
          figname_info=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          colormap_var=['viridis','viridis','coolwarm'],
          clim_global=[(-1,1), (-5,5),(-5,5),(270,300)],
          axis_title_global=''
          ):

        fig, ax = plt.subplots(figsize=(15,5*(len(var_names)-1)), nrows=3, ncols=(len(var_names)-1))
        for id, var in enumerate(var_names):
            if id>0:
                var_id = dict_var[var]
                if not clim_global :
                    vmin = np.min([np.min(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                    vmax = np.min([np.max(packsample[:,var_id,crop[0]:crop[1],crop[2]:crop[3]])])
                    clim = (vmin, vmax)
                else :
                    clim = clim_global[id]
                ax[0][id-1].set_title(f"{axis_title_global}{var} real")
                im = ax[0][id-1].imshow(packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]], origin="lower", cmap=colormap_var[id], clim=clim)
                fig.colorbar(im, ax=ax[0][id-1], shrink=0.5)

                ax[1][id-1].set_title(f"{axis_title_global}{var} generated")
                im = ax[1][id-1].imshow(pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]],  origin="lower", cmap=colormap_var[id], clim=clim)
                fig.colorbar(im, ax=ax[1][id-1], shrink=0.5)

                diff = packsample[mem_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]] - pert_sample[mem_pert_idx,var_id,crop[0]:crop[1],crop[2]:crop[3]]
                ax[2][id-1].set_title(f"diff {axis_title_global}{var}")
                im = ax[2][id-1].imshow(diff, origin="lower", cmap="RdYlGn", clim=(-2,2))
                fig.colorbar(im, ax=ax[2][id-1], shrink=0.5)

        fig.suptitle(title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info, dpi=100)
        except Exception as e : 
            print(f"unable to save figure: {figname_info}")
        plt.close()
        return


def plot_quantiles(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          quantiles_list=[0.01,0.1,0.5,0.9,0.99,1]
          ):

        cmap = plt.get_cmap("PiYG", 8)
        quantiles_arome = np.quantile(packsample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_gen = np.quantile(pert_sample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        vmins = np.zeros((len(quantiles_list), len(var_names)-1))
        vmaxs = np.zeros((len(quantiles_list), len(var_names)-1))

        # Quantiles of AROME
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=len(var_names)-1, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                if var_idx>0:
                    vmins[quantile_idx][var_idx-1] = np.min(
                        [np.min(quantiles_arome[quantile_idx][var_idx])]
                    )
                    vmaxs[quantile_idx][var_idx-1] = np.min(
                        [np.max(quantiles_arome[quantile_idx][var_idx])]
                    )

                    clim = (vmins[quantile_idx][var_idx-1],vmaxs[quantile_idx][var_idx-1])

                    ax[var_idx-1][quantile_idx].set_title(f"{var} real - Q{quantile}")
                    im = ax[var_idx-1][quantile_idx].imshow(quantiles_arome[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                    fig.colorbar(im, ax=ax[var_idx-1][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of AROME ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_AROME.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'_AROME.png'}")
        plt.close()
        
        # Quantiles of Generated samples
        fig, ax = plt.subplots(figsize=(5*len(quantiles_list),15), nrows=len(var_names)-1, ncols=len(quantiles_list))
        for quantile_idx, quantile in enumerate(quantiles_list):
            for var_idx, var in enumerate(var_names):
                if var_idx>0:
                    clim = (vmins[quantile_idx][var_idx-1],vmaxs[quantile_idx][var_idx-1])

                    ax[var_idx-1][quantile_idx].set_title(f"{var} GEN - Q{quantile}")
                    im = ax[var_idx-1][quantile_idx].imshow(quantiles_gen[quantile_idx][var_idx], origin="lower", cmap=cmap, clim=clim)
                    fig.colorbar(im, ax=ax[var_idx-1][quantile_idx], shrink=0.5)

        fig.suptitle('Quantiles of Generated ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_GEN.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'_GEN.png'}")
        plt.close()
        return

def plot_quantiles_comparison(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['u','v','t2m'], 
          dict_var={'u': 0, 'v': 1, 't2m': 2},
          dict_var_lim={'u': [-1,1], 'v': [-1,1], 't2m': [-5,5]},
          quantiles_list=[0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]
          ):

        cmap = plt.get_cmap("PiYG", 8)
        quantiles_arome = np.quantile(packsample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_gen = np.quantile(pert_sample[:,:,crop[0]:crop[1],crop[2]:crop[3]], quantiles_list, axis=0)
        quantiles_diff = np.zeros((len(var_names), len(quantiles_list)))
        for var_idx, var in enumerate(var_names):
            for q in range(len(quantiles_list)):
                quantiles_diff[var_idx][q] = quantiles_gen[q][var_idx].mean() - quantiles_arome[q][var_idx].mean()

        # Diff Quantiles to AROME
        fig, ax = plt.subplots(figsize=(20,15), nrows=len(var_names)-1, ncols=1)
        for var_idx, var in enumerate(var_names):
            if var_idx > 0:
                ax[var_idx-1].scatter(range(len(quantiles_list)), quantiles_diff[var_idx], linewidth=20, c='r')
                ax[var_idx-1].plot(range(len(quantiles_list)), np.zeros(np.size(quantiles_list)), linewidth=5, c='b')
                list_ticks=[str(i*100)+'%' for i in quantiles_list]
                ax[var_idx-1].set_xticks(range(len(quantiles_list)) ,labels=list_ticks)
                ax[var_idx-1].set_ylabel(var)
                ax[var_idx-1].set_xlabel('Quantiles')
                ax[var_idx-1].set_ylim(dict_var_lim[var])

        fig.suptitle('Diff Quantiles with AROME ensemble for '+title_info)
        fig.tight_layout()
        try:
            fig.savefig(figname_info+'_Diff.png', dpi=100)
        except Exception:
            print(f"unable to save figure: {figname_info+'_Diff.png'}")
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
          threshold_wind=[5,10,15,20], # 15m/s = 54km/h
          wind_obs_data=None
          ):
        r'''
        Plot a map representing the spatial probability to exceed a given threshold.

        '''
        cmap = plt.get_cmap("Blues", 5)
        wind_arome = np.sqrt(packsample[:,dict_var['u']]**2+packsample[:,dict_var['v']]**2)
        wind_gen = np.sqrt(pert_sample[:,dict_var['u']]**2+pert_sample[:,dict_var['v']]**2)
        fig, ax = plt.subplots(figsize=(10*len(threshold_wind),15), ncols=len(threshold_wind), nrows=2)
        
        for threshold_id, threshold in enumerate(threshold_wind):
            if wind_obs_data is not None:
                coords_obs = np.where(wind_obs_data>=threshold)
            max_arome = np.zeros_like(packsample[:,dict_var['u']]) 
            max_gen = np.zeros_like(pert_sample[:,dict_var['u']])

            for id_member, member_arome in enumerate(wind_arome):
                max_arome[id_member] = np.where(member_arome>=threshold, 1, 0)
            im = ax[0][threshold_id].imshow(np.mean(max_arome, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[0][threshold_id].set_title(f'AROME {len(packsample)} members - P(Wind>{round(threshold*3.6,1)}km/h)')
            fig.colorbar(im, ax=ax[0][threshold_id], shrink=0.5)

            for id_member, member_gen in enumerate(wind_gen):
                max_gen[id_member] = np.where(member_gen>=threshold, 1, 0)
            ax[1][threshold_id].imshow(np.mean(max_gen, axis=0), origin="lower", cmap=cmap, clim=(0,1))
            ax[1][threshold_id].set_title(f'Generated {len(pert_sample)} members - P(Wind>{round(threshold*3.6,1)}km/h)')
            fig.colorbar(im, ax=ax[1][threshold_id], shrink=0.5)
            if wind_obs_data is not None:
                for x,y in zip(coords_obs[0], coords_obs[1]):
                    # ax[0][threshold_id].imshow(np.where(wind_obs_data>=threshold,1,0), origin="lower", cmap='Reds', alpha=0.5)
                    # ax[1][threshold_id].imshow(np.where(wind_obs_data>=threshold,1,0), origin="lower", cmap='Reds', alpha=0.5)
                    ax[0][threshold_id].text(y,x,round(wind_obs_data[x,y]*3.6,2), alpha=0.5)
                    ax[1][threshold_id].text(y,x,round(wind_obs_data[x,y]*3.6,2), alpha=0.5)
            
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
          threshold_t2m=[15,20,30,35],
          obs_data=None
          ):
        r'''
        Plot a map representing the spatial probability to exceed a given threshold.

        '''
        cmap = plt.get_cmap("Reds", 5)
        threshold_t2m = 273.15+np.array(threshold_t2m)

        fig, ax = plt.subplots(figsize=(10*len(threshold_t2m),15), ncols=len(threshold_t2m), nrows=2)
        for threshold_id, threshold in enumerate(threshold_t2m):
            if obs_data is not None:
                coords_obs = np.where(obs_data>=273.15+threshold)
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
            if obs_data is not None:
                for x,y in zip(coords_obs[0], coords_obs[1]):
                    ax[0][threshold_id].text(y,x,round(obs_data[x,y]-273.15,2), alpha=0.5)
                    ax[1][threshold_id].text(y,x,round(obs_data[x,y]-273.15,2), alpha=0.5)
        
        fig.suptitle(f'Probability Exceeding Temperature Threshold map comparison for '+title_info)
        fig.tight_layout()
        fig.savefig(figname_info+'.png', dpi=100)
        plt.close()
        
        return

def compute_density(density):
    return np.where(
                density<0.01,
                'white',
                np.where(
                    density<=0.05,
                    'gold',
                    np.where(
                        density<0.25,
                        'orange',
                        np.where(
                            density<0.5,
                            'red',
                            np.where(
                                density<0.75,
                                'magenta',
                                np.where(
                                    density<1.,
                                    'purple',
                                    'black'
                                )
                            )
                        )
                    )
                )
            )

def plot_panache_density_dynamic(
          packsample, 
          pert_sample, 
          crop=[0,-1,0,-1], 
          title_info=" ", 
          figname_info=".png",  
          var_names=['rr', 'u (m/s)','v (m/s)','t2m (K)'], 
          dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
          axis_title_global='',
          coord = [63,38] # Toulouse
          ):
        
        arome = packsample[ :, :, :, coord[0], coord[1]]
        gen = pert_sample[ :, :, :, coord[0], coord[1]]
        lt_arome, num_member_arome, v = arome.shape
        arome_enveloppe = np.zeros((lt_arome, v, 3))
        minimum_arome = np.min(arome.reshape((lt_arome*num_member_arome,v)), axis=0)
        maximum_arome = np.max(arome.reshape((lt_arome*num_member_arome,v)), axis=0)
        
        lt_gen, num_member_gen, v = gen.shape
        gen_enveloppe = np.zeros((lt_gen, v, 3))
        minimum_gen = np.min(gen.reshape((lt_gen*num_member_gen,v)), axis=0)
        maximum_gen = np.max(gen.reshape((lt_gen*num_member_gen,v)), axis=0)

        dL_arome = (maximum_arome - minimum_arome)/20
        dL_gen = (maximum_gen - minimum_gen)/20
        
        
        fig, axes = plt.subplots(figsize=(25,15), ncols=3, nrows=3, layout='constrained')
        axes[0][0].set_title('Panache AROME')
        axes[0][1].set_title('Panache AROME&Generated')
        axes[0][2].set_title('Panache Generated')
        threshold_density = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 1]
        colors=['white', 'gold', 'orange', 'red', 'magenta','purple']
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(threshold_density, cmap.N)

        for id_var, var_name in enumerate(var_names):
            if id_var>0:
                levels_arome = np.arange(minimum_arome[id_var],maximum_arome[id_var], dL_arome[id_var]) 
                levels_gen = np.arange(minimum_gen[id_var],maximum_gen[id_var], dL_gen[id_var])  
                minimum_global = np.min((minimum_arome[id_var], minimum_gen[id_var]))
                maximum_global = np.max((maximum_arome[id_var], maximum_gen[id_var]))

                axes[id_var-1][0].set_xlabel('leadtimes (h)')
                axes[id_var-1][0].set_ylim([minimum_global, maximum_global])

                axes[id_var-1][1].set_xlabel('leadtimes (h)')
                axes[id_var-1][1].set_ylim([minimum_global, maximum_global])
                
                axes[id_var-1][2].set_xlabel('leadtimes (h)')
                axes[id_var-1][2].set_ylim([minimum_global, maximum_global])

                axes[id_var-1][0].grid(True, linestyle='--')
                axes[id_var-1][0].set_ylabel(var_name)
                axes[id_var-1][1].grid(True, linestyle='--')
                axes[id_var-1][1].set_ylabel(var_name)
                axes[id_var-1][2].grid(True, linestyle='--')
                axes[id_var-1][2].set_ylabel(var_name)

                min_enveloppe_arome_lt = []
                max_enveloppe_arome_lt = []
                min_enveloppe_gen_lt = []
                max_enveloppe_gen_lt = []
                for id_lt in range(lt_arome):
                    min_enveloppe_arome_lt.append(np.min(arome[id_lt,:,id_var]))
                    max_enveloppe_arome_lt.append(np.max(arome[id_lt,:,id_var]))
                    min_enveloppe_gen_lt.append(np.min(gen[id_lt,:,id_var]))
                    max_enveloppe_gen_lt.append(np.max(gen[id_lt,:,id_var]))

                    for level_id in range(len(levels_arome)-1):
                        borne_inf_level=levels_arome[level_id]
                        borne_sup_level=levels_arome[level_id+1]
                        density = np.mean(np.multiply(arome[id_lt,:,id_var]>=borne_inf_level, arome[id_lt,:,id_var]<=borne_sup_level))
                        color = compute_density(density)
                        axes[id_var-1][0].scatter(id_lt*3, borne_inf_level + (borne_sup_level-borne_inf_level)/2, color=str(color), linewidth=10)
                        
                        borne_inf_level=levels_gen[level_id]
                        borne_sup_level=levels_gen[level_id+1]
                        density = np.mean(np.multiply(gen[id_lt,:,id_var]>=borne_inf_level, gen[id_lt,:,id_var]<=borne_sup_level))
                        color = compute_density(density)
                        axes[id_var-1][2].scatter(id_lt*3, borne_inf_level + (borne_sup_level-borne_inf_level)/2, color=str(color), linewidth=10)
                
                axes[id_var-1][0].fill_between(
                    list(range(0, lt_arome*3, 3)),
                    min_enveloppe_arome_lt,
                    max_enveloppe_arome_lt,
                    linewidth=0,
                    color='lightgray',
                    alpha=0.2,
                    )
                axes[id_var-1][0].plot(list(range(0, lt_arome*3, 3)),min_enveloppe_arome_lt,color='black')
                axes[id_var-1][0].plot(list(range(0, lt_arome*3, 3)),max_enveloppe_arome_lt,color='black')
                
                axes[id_var-1][1].fill_between(
                    list(range(0, lt_arome*3, 3)),
                    min_enveloppe_arome_lt,
                    max_enveloppe_arome_lt,
                    linewidth=0,
                    color='gray'
                    )
                
                axes[id_var-1][1].plot(list(range(0, lt_arome*3, 3)),min_enveloppe_arome_lt,color='black')
                axes[id_var-1][1].plot(list(range(0, lt_arome*3, 3)),max_enveloppe_arome_lt,color='black')
                axes[id_var-1][1].plot(list(range(0, lt_arome*3, 3)),min_enveloppe_gen_lt,color='black',linestyle='--')
                axes[id_var-1][1].plot(list(range(0, lt_arome*3, 3)),max_enveloppe_gen_lt,color='black',linestyle='--')

                axes[id_var-1][2].fill_between(
                    list(range(0, lt_arome*3, 3)),
                    min_enveloppe_gen_lt,
                    max_enveloppe_gen_lt,
                    linewidth=0,
                    color='lightgray',
                    alpha=0.2,
                    )
                axes[id_var-1][2].plot(list(range(0, lt_arome*3, 3)),min_enveloppe_gen_lt,color='black',linestyle='--')
                axes[id_var-1][2].plot(list(range(0, lt_arome*3, 3)),max_enveloppe_gen_lt,color='black',linestyle='--')


                axes[id_var-1][0].plot(list(range(0, lt_arome*3, 3)), arome[:,:,id_var], color='blue', alpha=0.05)
                axes[id_var-1][2].plot(list(range(0, lt_arome*3, 3)), gen[:,:,id_var], color='blue', alpha=0.05)

        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cax,
            orientation='vertical',
            label='Density of members',
        )

        fig.suptitle(f'Plumes comparison for '+title_info)
        fig.savefig(figname_info+'.png', dpi=100)
        plt.close()
        

        
        return

if __name__=="__main__" :
    
    parser = argparse.ArgumentParser()

    # Real Data Directory - PATH to samples of the dataset
    parser.add_argument('--real_data_dir', type = str,default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--gen_data_dir', type = str,default='/project/home/p200177/DE_371/experiments_WP1/DIFFUSION_experiments_AROME/sdedit_ED/sampling_3steps/')
    parser.add_argument('--obs_dir', type=str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/obs_databases/')
    # Output Directory - PATH where the output of the inversion will be saved
    ########################## CONTROL of Data to invert ######################
    parser.add_argument("--dates_file", type=str, default='Large_lt_val_labels_ens.csv')
    parser.add_argument("--date_start", type=str, default = "2020-10-01")
    parser.add_argument("--date_stop", type=str, default = "2020-10-02")
    parser.add_argument("--leadtimes", type=str2intlist, default=[3,6,9,12,18,21,24,27,30,33,36,39,42])
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
            
            directory = output_dir+'/samples_aromes/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            # plot_samples(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'{datename}_{lt}',
            #     figname_info=directory+f'samples_{datename}_{lt}.png',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
            #     colormap_var=['viridis','viridis','viridis','coolwarm'],
            #     clim_global=[(0,0),(-5,5),(-5,5),(270,300)]
            # )


            directory = output_dir+'/statistics/'
            if not os.path.exists(directory):
                os.makedirs(directory)
        #     plot_mean(
        #         Ens_AROME, 
        #         Ens_Gen, 
        #         figtitle=f'{datename}_{lt}',
        #         figname=directory+f'mean_2D_{datename}_{lt}.png',
        #         var_names=['rr', 'u','v','t2m'], 
        #         dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
        #         colormap_var=['viridis','viridis','viridis','coolwarm'],
        #         clim_global=[(0,0),(-5,5),(-5,5),(270,300)]
        #     )

            plot_var(
                Ens_AROME, 
                Ens_Gen, 
                figtitle=f'{datename}_{lt}',
                figname=directory+f'var_2D_{datename}_{lt}.png',
                var_names=['rr', 'u','v','t2m'], 
                dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
                colormap_var=['viridis','viridis','viridis','coolwarm']
            )


            directory = output_dir+'/probability_threshold_map/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            obs_data=None

            obs_date_name = date_.strftime('%Y%m%d') 
            if lt>=24:
                obs_date_name = obs_date_name[:-1] + str(int(obs_date_name[-1:])+1)
                print(obs_date_name)
            obs_date_filename = f'obs{obs_date_name}_{lt%24}.npy'
            print(obs_date_filename)
            obs_data = obs_clean(np.load(params.obs_dir+obs_date_filename).astype(np.float32), [180, 436, 500, 756])

            # plot_probability_exceeding_threshold_temperature(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'{datename}_{lt}',
            #     figname_info=directory+f'probability_temperature_threshold_{datename}_{lt}',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3} ,   
            #     threshold_t2m=[10,15,20,30],
            #     obs_data=obs_data[2] if obs_data is not None else None
            # )
            plot_probability_exceeding_threshold_wind(
                Ens_AROME, 
                Ens_Gen, 
                title_info=f'{datename}_{lt}',
                figname_info=directory+f'probability_wind_threshold_{datename}_{lt}',
                var_names=['rr', 'u','v','t2m'], 
                dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
                threshold_wind=[10/3.6,20/3.6,30/3.6,40/3.6,50/3.6],
                wind_obs_data=obs_data[0] if obs_data is not None else None
            )

            directory = output_dir+'/quantiles/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            plot_quantiles_comparison(
                Ens_AROME, 
                Ens_Gen, 
                title_info=f'{datename}_{lt}',
                figname_info=directory+f'diff_quantiles_{datename}_{lt}',
                var_names=['rr', 'u','v','t2m'], 
                dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
                quantiles_list=[0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]     
            )

            # plot_quantiles(
            #     Ens_AROME, 
            #     Ens_Gen, 
            #     title_info=f'{datename}_{lt}',
            #     figname_info=directory+f'quantiles_{datename}_{lt}',
            #     var_names=['rr', 'u','v','t2m'], 
            #     dict_var={'rr': 0, 'u': 1, 'v': 2, 't2m': 3},
            #     quantiles_list=[0.01,0.1,0.5,0.9,0.99,1]      
            # )
            

            # Projection on orographie
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

        directory = output_dir+'/panaches/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot_panache_density_dynamic(
            np.array(Ens_AROME_date),
            np.array(Ens_Gen_date),
            title_info=f'{datename}',
            figname_info=directory+f'panache_{datename}',
            coord=[60,240]
        )
            
            











