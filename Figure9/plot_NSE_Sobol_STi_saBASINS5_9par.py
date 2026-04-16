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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Step 1: Read the shapefile
basin_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\basin.shp'
gdf_basin = gpd.read_file(basin_shapefile_path)

watersheld_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\watersheld.shp'
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)

reach_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\reach.shp'
gdf_reach = gpd.read_file(reach_shapefile_path)

zhjs_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\zhjs_station.shp'
gdf_zhjs = gpd.read_file(zhjs_shapefile_path)

print("Shapefile attribute table:")
print(gdf_watersheld.head())

# Step 2: Read Sobol results
n_subbasins = 39
n_params = 9
param_names = ['CH_K1', 'RCHRG_DP', 'CANMX', 'ESCO', 'CN2', 'SOL_AWC', 'SOL_BD', 'GWQMN', 'GW_DELAY']

S1_tmp = np.load('Sobol_S1_NSE_spatialBASINs.npy')
S1 = S1_tmp.reshape(n_params, n_subbasins).T

df_S1 = pd.DataFrame(S1, columns=param_names)
df_S1.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))
df_S1 = df_S1.rename(columns={col: col + '_S1' for col in param_names})

ST_tmp = np.load('Sobol_ST_NSE_spatialBASINs.npy')
ST = ST_tmp.reshape(n_params, n_subbasins).T

df_ST = pd.DataFrame(ST, columns=param_names)
df_ST.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))
df_ST = df_ST.rename(columns={col: col + '_ST' for col in param_names})

# Merge
gdf_watersheld = gdf_watersheld.merge(df_S1, on='Subbasin', how='left')
gdf_watersheld = gdf_watersheld.merge(df_ST, on='Subbasin', how='left')

# =========================
# Figure layout
# =========================
fig = plt.figure(figsize=(12, 5))

outer_gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[1, 0.2],
    hspace=-0.6,
    figure=fig
)

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 9,
    subplot_spec=outer_gs[0],
    wspace=0.12
)

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 9,
    subplot_spec=outer_gs[1],
    wspace=0.65
)

axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(9)]
axes_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(9)]

cmap = 'hot_r'

# ---------- Row 1: maps ----------
for i, col in enumerate([p + '_ST' for p in param_names]):
    ax = axes_top[i]

    gdf_watersheld.plot(column=col, cmap=cmap, vmin=0, vmax=0.15, ax=ax)
    gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, ax=ax)
    gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, ax=ax)

    if 'Linewidth' in gdf_reach.columns:
        gdf_reach.plot(ax=ax, color='b', lw=gdf_reach['Linewidth'], alpha=0.75)
    else:
        gdf_reach.plot(ax=ax, color='b', lw=0.8)

    gdf_zhjs.plot(ax=ax, color='#C31B23', markersize=20, marker='s')

    ax.set_title(param_names[i], fontsize=12, y=1.02)
    ax.set_axis_off()

# ---------- Row 2: bar plots (Top 5) ----------
norm = Normalize(vmin=0, vmax=0.15)
cmap_obj = plt.cm.get_cmap(cmap)

top_n = 5

for i, p in enumerate(param_names):
    ax = axes_bottom[i]
    col = f"{p}_ST"

    bars_df = df_ST[['Subbasin', col]].copy()
    bars_df['Subbasin'] = bars_df['Subbasin'].astype(int)
    bars_df = bars_df.sort_values(by=col, ascending=False)

    topN = bars_df.head(top_n)

    colors = [cmap_obj(norm(val)) for val in topN[col]]

    ax.barh(
        y=topN['Subbasin'].astype(str),
        width=topN[col],
        color=colors,
        edgecolor='k'
    )

    ax.invert_yaxis()

    #ax.set_xlabel('$S_{Ti}$', fontsize=12)

    if i == 0:
        ax.set_ylabel(f'Subbasin ID \n(Top {top_n} $S_{{Ti}}$)', fontsize=12)
    else:
        ax.set_ylabel('')

    ax.set_xlim(0, 0.15)
    ax.set_xticks([0, 0.15])
    ax.set_xticklabels(['0', '0.15'])

fig.suptitle(
    '(a) Spatial distribution of NSE-based $\mathbf{S_{Ti}}$ at the subbasin scale for the 9 screened parameters',
    x=0.075, y=0.675, ha='left', va='top',
    fontsize=14, fontweight='bold'
)

# =========================
plt.savefig('spatial_dist_par_subbasin_scale.jpg', dpi=300)
plt.show()