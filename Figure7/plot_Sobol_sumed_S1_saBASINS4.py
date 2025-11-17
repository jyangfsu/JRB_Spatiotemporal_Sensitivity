# -*- coding: utf-8 -*-
"""
One-figure view of summed first-order Sobol indices (S1) across five parameters.
Left: spatial distribution of SUM(S1) over subbasins (with reach + ZHJS overlays)
Right: bar plot of top-15 subbasins by SUM(S1)

Inputs expected:
- Shapefiles: basin.shp, watersheld.shp, reach.shp, zhjs_station.shp
- S1 array file: Sobol_S1_NSE_spatialBASINs.npy
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# -----------------------------
# Paths (update if needed)
# -----------------------------
basin_shapefile_path      = r'basin.shp'
watersheld_shapefile_path = r'watersheld.shp'
reach_shapefile_path      = r'reach.shp'
zhjs_shapefile_path       = r'zhjs_station.shp'
s1_npy_path               = 'Sobol_S1_NSE_spatialBASINs.npy'

# -----------------------------
# Read base layers
# -----------------------------
gdf_basin      = gpd.read_file(basin_shapefile_path)
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)
gdf_reach      = gpd.read_file(reach_shapefile_path)
gdf_zhjs       = gpd.read_file(zhjs_shapefile_path)

# -----------------------------
# S1 data → sum across parameters per subbasin
# -----------------------------
n_subbasins = 39
n_params    = 5
param_names = ['CH_K1', 'RCHRG_DP', 'CANMX', 'ESCO', 'CN2']  # only used for reference/logging

S1_raw = np.load(s1_npy_path)                          # shape expected: n_params * n_subbasins
S1     = S1_raw.reshape(n_params, n_subbasins)         # (n_params, n_subbasins)
S1_T   = S1.T                                          # (n_subbasins, n_params)

df_S1  = pd.DataFrame(S1_T, columns=[f'{p}_S1' for p in param_names])
df_S1.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))
df_S1['S1_sum'] = df_S1[[f'{p}_S1' for p in param_names]].sum(axis=1)

df_S1.to_csv('df_sumed_S1_saBASIN.csv')

# Merge to watershed polygons by 'Subbasin'
gdf_w = gdf_watersheld.merge(df_S1[['Subbasin', 'S1_sum']], on='Subbasin', how='left')

# -----------------------------
# Robust color limits for map
# -----------------------------
vals = gdf_w['S1_sum'].dropna().to_numpy()
if vals.size == 0:
    vmin, vmax = 0.0, 1.0
else:
    q2, q98 = np.nanquantile(vals, [0.02, 0.98])
    vmin    = float(q2) if np.isfinite(q2) else float(np.nanmin(vals))
    vmax    = float(q98) if np.isfinite(q98) else float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1e-3  # avoid zero range

cmap  = 'hot_r'
norm  = Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# Prepare bar data: top-15 by S1_sum
# -----------------------------
bars_df = df_S1[['Subbasin', 'S1_sum']].copy()
bars_df['Subbasin'] = bars_df['Subbasin'].astype(int)
bars_df = bars_df.sort_values('S1_sum', ascending=False)
top15   = bars_df.head(15)

# -----------------------------
# Figure layout: 1x2 (map | bars)
# -----------------------------
fig = plt.figure(figsize=(10, 5), dpi=100)
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.3, 0.9], wspace=0.25, figure=fig)

ax_map = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])

# -----------------------------
# Left: spatial map of SUM(S1)
# -----------------------------
gdf_w.plot(column='S1_sum', cmap=cmap, norm=norm, legend=False, ax=ax_map)
gdf_w.plot(edgecolor='grey', facecolor='none', linewidth=0.25, ax=ax_map)
gdf_basin.plot(edgecolor='k', facecolor='none', linewidth=0.6, ax=ax_map)

# Overlay reaches and stations
if 'Linewidth' in gdf_reach.columns:
    gdf_reach.plot(ax=ax_map, color='b', lw=gdf_reach['Linewidth'], label='Reach')
else:
    gdf_reach.plot(ax=ax_map, color='b', lw=0.8, label='Reach')

gdf_zhjs.plot(ax=ax_map, color='k', markersize=20, marker='s', label='ZHJS')

ax_map.set_axis_off()
ax_map.set_title("Sum of Sobol's $S_{i}$ (5 parameters)")

# In-map colorbar (top-right inside)
pos     = ax_map.get_position()
cb_ax   = fig.add_axes([pos.x1 - 0.06, pos.y1 - 0.32, 0.012, 0.24])
sm      = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb      = fig.colorbar(sm, cax=cb_ax, orientation='vertical')
cb.set_label("SUM $S_{i}$", fontsize=9)

# Optional legend inside map (lower-right)
handles, labels = ax_map.get_legend_handles_labels()
if handles:
    leg = ax_map.legend(loc="lower right", frameon=True, framealpha=0.7, fontsize=8)
    leg.get_frame().set_linewidth(0.3)

# -----------------------------
# Right: bar plot (top-15 subbasins)
# -----------------------------
ax_bar.barh(
    y=top15['Subbasin'].astype(str).values,
    width=top15['S1_sum'].values,
    color=plt.cm.get_cmap(cmap)(norm(top15['S1_sum'].values)),
    edgecolor='k',
    linewidth=0.4
)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("SUM $S_{i}$")
ax_bar.set_ylabel("Subbasin (Top 15)")
# Align bar x-limits to map color scale for easier comparison
ax_bar.set_xlim(0, 0.3)
ax_bar.grid(axis='x', linestyle='--', alpha=0.4)

# -----------------------------
# Save / show
# -----------------------------
fig.suptitle("Spatial distribution and ranking of SUM of Sobol's $S_{i}$ (5 parameters)", y=0.98, fontsize=12, fontweight='bold')
fig.savefig('spatial_S1sum_map_bar.png', dpi=300, bbox_inches='tight')
plt.show()
