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

watersheld_shapefile_path = r'watersheld.shp'
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)

reach_shapefile_path = r'reach.shp'
gdf_reach = gpd.read_file(reach_shapefile_path)

zhjs_shapefile_path = r'zhjs_station.shp'
gdf_zhjs= gpd.read_file(zhjs_shapefile_path)


# Display the first few rows to understand the existing data
print("Shapefile attribute table:")
print(gdf_watersheld.head())

# Step 2: Read the Morris sensitivy analysis results 
# Make sure  has 'GRIDCODE' column and your new values column, e.g., 'New_Values'
n_subbasins = 39
n_params = 5
param_names = ['CH_K1', 'RCHRG_DP', 'CANMX', 'ESCO', 'CN2']



S1_tmp = np.load('Sobol_S1_NSE_spatialBASINs.npy')
S1 = S1_tmp.reshape(n_params, n_subbasins)       
S1_transposed = S1.T

df_S1 = pd.DataFrame(S1_transposed, columns=param_names)
df_S1.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))

df_S1 = df_S1.rename(columns={col: col + '_S1' for col in param_names})


ST_tmp = np.load('Sobol_ST_NSE_spatialBASINs.npy')
ST = ST_tmp.reshape(n_params, n_subbasins)      
ST_transposed = ST.T

df_ST = pd.DataFrame(ST_transposed, columns=param_names)
df_ST.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))

df_ST = df_ST.rename(columns={col: col + '_ST' for col in param_names})

# Merge the SA resulst with the GeoDataFrame based on 'GRIDCODE'
gdf_watersheld = gdf_watersheld.merge(df_S1, on='Subbasin', how='left')
gdf_watersheld = gdf_watersheld.merge(df_ST, on='Subbasin', how='left')


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
    gdf_watersheld.plot(column=col, cmap=cmap, vmin=0, vmax=0.15, legend=False, ax=ax)
    gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, legend=False, ax=ax)
    gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, legend=False, ax=ax)
    
    #if 'AreaC' in gdf_reach.columns:
    #    gdf_reach.plot(ax=ax, color='b', lw=(gdf_reach['AreaC'])/gdf_reach['AreaC'].max() * 1.25)
    if 'Linewidth' in gdf_reach.columns:
        gdf_reach.plot(ax=ax, color='b', lw=(gdf_reach['Linewidth']))
        
    else:
        gdf_reach.plot(ax=ax, color='b', lw=0.8)
    gdf_zhjs.plot(ax=ax, color='k', markersize=20, marker='s')

    ax.set_title(f"{param_names[i]}", fontsize=12)
    ax.set_axis_off()

# ---------- Row 2: bar plots (TOP 10 only, colormap coloring) ----------
norm = plt.Normalize(vmin=0, vmax=0.15)  # normalize for color mapping
cmap_obj = plt.cm.get_cmap(cmap)

for i, p in enumerate(param_names):
    ax = axes[i + 5]
    col = f"{p}_ST"

    bars_df = df_ST[['Subbasin', col]].copy()
    bars_df['Subbasin'] = bars_df['Subbasin'].astype(int)
    bars_df = bars_df.sort_values(by=col, ascending=False)

    top10 = bars_df.head(10)
    tail10 = bars_df.tail(10)
    
    print(tail10)

    # Assign colors according to colormap
    colors = [cmap_obj(norm(val)) for val in top10[col]]

    ax.barh(y=top10['Subbasin'].astype(str), 
            width=top10[col], 
            color=colors, 
            edgecolor='k')
    ax.invert_yaxis()
    
    ax.set_xlabel('Total-effect $S_{Ti}$', fontsize=12)
    if i==0:
        ax.set_ylabel('Subbasin (Top 10 $S_{Ti}$)', fontsize=12)
    else:
        ax.set_ylabel('')
    #ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_xlim(0, 0.15)  # fixed x-axis limit
    ax.set_xticks([0, 0.15]) 
    ax.set_title("")  # no title for bar plots

'''
# ---- Add horizontal colorbar at bottom ----
cbar_ax = fig.add_axes([0.25, 0.00, 0.5, 0.02])  # [left, bottom, width, height]
norm = Normalize(vmin=0, vmax=0.15)
sm = ScalarMappable(norm=norm, cmap='cool')

sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Sobol's $S_{i}$", fontsize=11)
cbar.ax.tick_params(labelsize=10)
'''

fig.suptitle('(a) Spatial distribution of Sobol\'s $\mathbf{S_{Ti}}$ by subbasin-scale SSA',
             x=0.075, y=0.85, ha='left', va='top',
             fontsize=14, fontweight='bold')
plt.savefig('spatial_dist_par_subbasin_scale_ST.png', dpi=300, bbox_inches='tight')
plt.show()