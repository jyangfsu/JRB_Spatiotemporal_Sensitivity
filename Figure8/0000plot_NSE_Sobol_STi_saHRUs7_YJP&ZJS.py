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



# =========================
# HRU -> Subbasin mapping
# =========================
hru_ranges = {
    1: range(1, 18),
    2: range(18, 42),
    3: range(42, 65),
    4: range(65, 84),
    5: range(84, 102),
    6: range(102, 114),
    7: range(114, 124),
    8: range(124, 136),
    9: range(136, 145),
    10: range(145, 166),
    11: range(166, 195),
    12: range(195, 204),
    13: range(204, 214),
    14: range(214, 232),
    15: range(232, 252),
    16: range(252, 271),
    17: range(271, 285),
    18: range(285, 297),
    19: range(297, 314),
    20: range(314, 324),
    21: range(324, 347),
    22: range(347, 362),
    23: range(362, 379),
    24: range(379, 388),
    25: range(388, 394),
    26: range(394, 414),
    27: range(414, 434),
    28: range(434, 443),
    29: range(443, 461),
    30: range(461, 476),
    31: range(476, 488),
    32: range(488, 505),
    33: range(505, 522),
    34: range(522, 533),
    35: range(533, 556),
    36: range(556, 578),
    37: range(578, 599),
    38: range(599, 619),
    39: range(619, 631)
}

hru_to_sub = {}
for sub, hrus in hru_ranges.items():
    for hru in hrus:
        hru_to_sub[hru] = sub



# Step 1: Read the shapefile
basin_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\basin.shp'
gdf_basin = gpd.read_file(basin_shapefile_path)

watersheld_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\Watersheld.shp'
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)

hru_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\hru.shp'
gdf_hru = gpd.read_file(hru_shapefile_path)

reach_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\reach.shp'
gdf_reach = gpd.read_file(reach_shapefile_path)


zhjs_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\zhjs_station.shp'
gdf_zhjs= gpd.read_file(zhjs_shapefile_path)

yjps_shapefile_path = r'D:\Jinghe_SWAT\code_sptp_SA\yjps.shp'
gdf_yjps= gpd.read_file(yjps_shapefile_path)


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




# =========================
# Figure layout
# =========================
fig = plt.figure(figsize=(6.667, 5))

outer_gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[1, 0.2],
    hspace=-0.6,
    figure=fig
)

gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=outer_gs[0],
    wspace=0.12
)

gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=outer_gs[1],
    wspace=0.65
)


axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(5)]
axes_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(5)]

cmap = 'hot_r'

# =========================
# Row 1: spatial maps
# =========================
for i, col in enumerate([p + '_ST' for p in param_names]):
    ax = axes_top[i]

    if i == 0:
        gdf_watersheld.plot(column=col, cmap=cmap, vmin=0, vmax=0.30, legend=False, ax=ax)
        gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, legend=False, ax=ax)
        gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, legend=False, ax=ax)
    else:
        gdf_hru.plot(column=col, cmap=cmap, vmin=0, vmax=0.15, legend=False, ax=ax)
        gdf_watersheld.plot(edgecolor='grey', facecolor='none', linewidth=0.2, legend=False, ax=ax)
        gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.5, legend=False, ax=ax)

    if 'Linewidth' in gdf_reach.columns:
        gdf_reach.plot(ax=ax, color='b', lw=gdf_reach['Linewidth'], alpha=0.75)
    else:
        gdf_reach.plot(ax=ax, color='b', lw=0.8)

    gdf_zhjs.plot(ax=ax, color='#C31B23', markersize=15, marker='s')
    gdf_yjps.plot(ax=ax, color='#C31B23', markersize=15, marker='s')

    ax.set_title(param_names[i], fontsize=12, y=1.02)
    ax.set_axis_off()

# =========================
# Row 2: Top 5 bar plots
# =========================
norm = Normalize(vmin=0, vmax=0.30)
cmap_obj = plt.cm.get_cmap(cmap)
top_n = 5

for i, p in enumerate(param_names):
    ax = axes_bottom[i]
    col = f"{p}_ST"

    if i == 0:
        bars_df = df_ST_sb[['Subbasin', col]].copy()
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
        #ax.set_ylabel(f'Subbasin/HRU ID \n(Top {top_n} $S_{{Ti}}$)', fontsize=12)

    else:
        bars_df = df_ST_hru[['HRU_ID', col]].copy()
        bars_df['HRU_ID'] = bars_df['HRU_ID'].astype(int)
        bars_df['Subbasin'] = bars_df['HRU_ID'].map(hru_to_sub)
        bars_df = bars_df.sort_values(by=col, ascending=False)

        topN = bars_df.head(top_n)
        colors = [cmap_obj(norm(val)) for val in topN[col]]

        ax.barh(
            y=topN['HRU_ID'].astype(str),
            width=topN[col],
            color=colors,
            edgecolor='k'
        )

        # automatic subbasin annotation
        offset = 0.02 * 0.15
        for ii, (val, sub) in enumerate(zip(topN[col].values, topN['Subbasin'].values)):
            ax.text(val + offset, ii, f"sb.{sub}", va='center', fontsize=10)

        ax.invert_yaxis()
        #ax.set_xlabel('$S_{Ti}$', fontsize=12)
        ax.set_ylabel('')

    ax.set_xlim(0, 0.30)
    ax.set_xticks([0, 0.30])
    ax.set_xticklabels(['0', '0.30'])
    ax.set_title("")

'''
# =========================
# Horizontal colorbar
# =========================
cbar_ax = fig.add_axes([0.25, -0.00, 0.5, 0.025])
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Sobol\' $S_{Ti}$", fontsize=12)
cbar.ax.tick_params(labelsize=10)
'''

# =========================
# Title and save
# =========================
fig.suptitle(
    '(d) HRU-scale $S_{Ti}$ based on aggregated NSE at YJP&ZJS',
    x=0.075, y=0.675, ha='left', va='top',
    fontsize=12, fontweight='bold'
)

plt.savefig('spatial_dist_par_hru_scale.jpg', dpi=300, bbox_inches='tight')
plt.show()