# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:54:22 2025

@author: Jing
"""


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Step 1: Read the shapefile
basin_shapefile_path = r'basin.shp'
gdf_basin = gpd.read_file(basin_shapefile_path)

watersheld_shapefile_path = r'Watersheld.shp'
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)

hru_shapefile_path = r'hru.shp'
gdf_hru = gpd.read_file(hru_shapefile_path)

reach_shapefile_path = r'reach.shp'
gdf_reach = gpd.read_file(reach_shapefile_path)

zhjs_shapefile_path = r'zhjs_station.shp'
gdf_zhjs= gpd.read_file(zhjs_shapefile_path)

# Display the first few rows to understand the existing data
print("Shapefile attribute table:")
print(gdf_hru.head())

# Step 2: Read the Morris sensitivy analysis results 
# Make sure  has 'GRIDCODE' column and your new values column, e.g., 'New_Values'
n_subbasins = 39
n_sb_params = 1

n_hrus = 630
n_hru_params = 4

param_sb_names = ['CH_K1']
param_hru_names = ['RCHRG_DP', 'CANMX', 'ESCO', 'CN2']
param_names = param_sb_names + param_hru_names

ST_sb_tmp = np.load('Sobol_ST_NSE_spatialHRUs_streamed_design.npy')[:int(n_sb_params*n_subbasins)]
ST_sb = ST_sb_tmp.reshape(n_sb_params, n_subbasins)       
ST_sb_transposed = ST_sb.T



df_ST_sb = pd.DataFrame(ST_sb_transposed, columns=param_sb_names)
df_ST_sb.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))

df_ST_sb = df_ST_sb.rename(columns={col: col + '_ST' for col in param_sb_names})

# Merge the SA resulst with the GeoDataFrame based on 'GRIDCODE'
gdf_watersheld = gdf_watersheld.merge(df_ST_sb, on='Subbasin', how='left')

ST_hru_tmp = np.load('Sobol_ST_NSE_spatialHRUs_streamed_design.npy')[n_sb_params*n_subbasins:]
ST_hru = ST_hru_tmp.reshape(n_hru_params, n_hrus)       
ST_hru_transposed = ST_hru.T

df_ST_hru = pd.DataFrame(ST_hru_transposed, columns=param_hru_names)
df_ST_hru.insert(0, 'HRU_ID', np.arange(1, n_hrus + 1))

df_ST_hru = df_ST_hru.rename(columns={col: col + '_ST' for col in param_hru_names})


# Merge the SA resulst with the GeoDataFrame based on 'GRIDCODE'
gdf_hru = gdf_hru.merge(df_ST_hru, on='HRU_ID', how='left')




# Create figure
fig = plt.figure(figsize=(12, 7))

# Define a 2-row, 5-column grid with different height ratios
# e.g., row 1 (maps) shorter, row 2 (bar plots) taller
gs = gridspec.GridSpec(2, 5, height_ratios=[1, 0.6], 
                       figure=fig, hspace=-0.2, wspace=0.30)

axes = []
for r in range(2):
    for c in range(5):
        axes.append(fig.add_subplot(gs[r, c]))
cmap = 'hot_r'  # colormap for the maps


# ---------- Row 1: spatial maps of S1 ----------
for i, col in enumerate([p + '_ST' for p in param_names]):
    ax = axes[i]
    if i==0:
        gdf_watersheld.plot(column=col, cmap=cmap, vmin=0, vmax=0.15, legend=False, ax=ax)
        gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, legend=False, ax=ax)
        gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, legend=False, ax=ax)
    

    else:
        gdf_hru.plot(column=col, cmap=cmap, vmin=0, vmax=0.15, legend=False, ax=ax)
        gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, legend=False, ax=ax)
        gdf_hru.plot(edgecolor='none', facecolor='none', linewidth=0.2, legend=False, ax=ax)
        gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, legend=False, ax=ax)
        
    #if 'AreaC' in gdf_reach.columns:
    #    gdf_reach.plot(ax=ax, color='b', lw=(gdf_reach['AreaC'])/gdf_reach['AreaC'].max() * 1.25)
    if 'Linewidth' in gdf_reach.columns:
        gdf_reach.plot(ax=ax, color='b', lw=gdf_reach['Linewidth'], alpha=0.75)
        
    else:
        gdf_reach.plot(ax=ax, color='b', lw=0.8)
    gdf_zhjs.plot(ax=ax, color='k', markersize=15, marker='s')

    ax.set_title(f"{param_names[i]}", fontsize=12)
    ax.set_axis_off()

# ---------- Row 2: bar plots (TOP 10 only, colormap coloring) ----------
norm = plt.Normalize(vmin=0, vmax=0.15)  # normalize for color mapping
cmap_obj = plt.cm.get_cmap(cmap)

for i, p in enumerate(param_names):
    ax = axes[i + 5]
    col = f"{p}_ST"

    if i==0:
        bars_df = df_ST_sb[['Subbasin', col]].copy()
        bars_df['Subbasin'] = bars_df['Subbasin'].astype(int)
        bars_df = bars_df.sort_values(by=col, ascending=False)
    
        top10 = bars_df.head(10)
    
        # Assign colors according to colormap
        colors = [cmap_obj(norm(val)) for val in top10[col]]
    
        ax.barh(y=top10['Subbasin'].astype(str), 
                width=top10[col], 
                color=colors, 
                edgecolor='k')
        ax.invert_yaxis()
    
        ax.set_xlabel('Total-effect $S_{Ti}$', fontsize=12)
        ax.set_ylabel('Subbasin/HRU ID (Top 10 $S_{Ti}$)', fontsize=12)
        
    else:
        bars_df = df_ST_hru[['HRU_ID', col]].copy()
        bars_df['HRU_ID'] = bars_df['HRU_ID'].astype(int)
        bars_df = bars_df.sort_values(by=col, ascending=False)
    
        top10 = bars_df.head(10)
    
        # Assign colors according to colormap
        colors = [cmap_obj(norm(val)) for val in top10[col]]
    
        ax.barh(y=top10['HRU_ID'].astype(str), 
                width=top10[col], 
                color=colors, 
                edgecolor='k')
        
        if i == 1:
            sb_texts = ['Sub. 33', 'Sub. 34', 'Sub. 30', 'Sub. 32', 'Sub. 33', 'Sub. 30', 'Sub. 34', 'Sub. 8', 'Sub. 35', 'Sub. 1']
            for ii in range(10):
                ax.text(top10[col].values[ii]+0.01, ii, sb_texts[ii], va='center')
                
        if i == 2:
            sb_texts = ['Sub. 33', 'Sub. 7', 'Sub. 34', 'Sub. 34', 'Sub. 12', 'Sub. 33', 'Sub. 30', 'Sub. 30', 'Sub. 36', 'Sub. 35']
            for ii in range(10):
                ax.text(top10[col].values[ii]+0.01, ii, sb_texts[ii], va='center')
                
        if i == 3:
            sb_texts = ['Sub. 33', 'Sub. 34', 'Sub. 30', 'Sub. 33', 'Sub. 34', 'Sub. 30', 'Sub. 7', 'Sub. 7', 'Sub. 34', 'Sub. 33']
            for ii in range(10):
                ax.text(top10[col].values[ii]+0.01, ii, sb_texts[ii], va='center')
                
        if i == 4:
            sb_texts = ['Sub. 33', 'Sub. 14', 'Sub. 30', 'Sub. 34', 'Sub. 7', 'Sub. 11', 'Sub. 33', 'Sub. 7', 'Sub. 8', 'Sub. 33']
            for ii in range(10):
                ax.text(top10[col].values[ii]+0.01, ii, sb_texts[ii], va='center')
            
            
        ax.invert_yaxis()
        
        ax.set_xlabel('Total-effect $S_{Ti}$', fontsize=12)
        

    #ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_xlim(0, 0.15)  # fixed x-axis limit
    ax.set_xticks([0, 0.15]) 
    ax.set_title("")  # no title for bar plots


# ---- Add horizontal colorbar at bottom ----
cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])  # [left, bottom, width, height]
norm = Normalize(vmin=0, vmax=0.15)
sm = ScalarMappable(norm=norm, cmap='hot_r')

sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Total-effect $S_{Ti}$", fontsize=12)
cbar.ax.tick_params(labelsize=10)



fig.suptitle('(b) Spatial distribution of Sobol\' $\mathbf{S_{Ti}}$ by HRU-scale SSA',
             x=0.075, y=0.85, ha='left', va='top',
             fontsize=14, fontweight='bold')
plt.savefig('spatial_dist_par_hru_scale_ST.png', dpi=300, bbox_inches='tight')
plt.show()