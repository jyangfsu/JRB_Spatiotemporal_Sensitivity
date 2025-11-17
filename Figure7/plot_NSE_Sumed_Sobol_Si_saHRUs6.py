# -*- coding: utf-8 -*-
"""
HRU-scale SUM(S1) map + ranking bar plot.

Goal:
- CH_K1 is a subbasin-scale parameter. For HRU-scale SUM(S1),
  assign each HRU the CH_K1 S1 of its parent subbasin (per provided HRU ID ranges),
  then add the four HRU-scale S1s (RCHRG_DP, CANMX, ESCO, CN2).
- Plot a single figure:
    Left  : spatial distribution of SUM(S1) over HRUs (with reaches + ZHJS overlays)
    Right : bar plot of top-15 HRUs by SUM(S1)

Inputs expected:
- Shapefiles: basin.shp, Watersheld.shp, hru.shp, reach.shp, zhjs_station.shp
- NPY file:  Sobol_S1_NSE_spatialHRUs_streamed_design.npy
    * first (n_subbasins * 1) values are CH_K1 (subbasin-scale)
    * next  (n_hrus * 4) values are the four HRU parameters (in order: RCHRG_DP, CANMX, ESCO, CN2)

Author: Jing (adapted)
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# -----------------------------
# Paths (update if needed)
# -----------------------------
basin_shapefile_path      = r'basin.shp'
watersheld_shapefile_path = r'Watersheld.shp'
hru_shapefile_path        = r'hru.shp'
reach_shapefile_path      = r'reach.shp'
zhjs_shapefile_path       = r'zhjs_station.shp'

s1_npy_path               = 'Sobol_S1_NSE_spatialHRUs_streamed_design.npy'

# -----------------------------
# Constants / names
# -----------------------------
n_subbasins    = 39
n_sb_params    = 1          # only CH_K1 at subbasin scale
n_hrus         = 630
n_hru_params   = 4          # RCHRG_DP, CANMX, ESCO, CN2 (HRU-scale)
param_sb_names = ['CH_K1']
param_hru_names = ['RCHRG_DP', 'CANMX', 'ESCO', 'CN2']
CMAP = 'hot_r'

# -----------------------------
# Read base layers
# -----------------------------
gdf_basin      = gpd.read_file(basin_shapefile_path)
gdf_watersheld = gpd.read_file(watersheld_shapefile_path)
gdf_hru        = gpd.read_file(hru_shapefile_path)
gdf_reach      = gpd.read_file(reach_shapefile_path)
gdf_zhjs       = gpd.read_file(zhjs_shapefile_path)

# -----------------------------
# Load S1 arrays
# -----------------------------
S1_all = np.load(s1_npy_path)

# First chunk: subbasin-scale CH_K1
S1_sb_flat = S1_all[: (n_sb_params * n_subbasins)]
S1_sb      = S1_sb_flat.reshape(n_sb_params, n_subbasins)          # shape: (1, n_subbasins)
S1_sb_T    = S1_sb.T                                               # shape: (n_subbasins, 1)
df_S1_sb   = pd.DataFrame(S1_sb_T, columns=[f'{p}_S1' for p in param_sb_names])
df_S1_sb.insert(0, 'Subbasin', np.arange(1, n_subbasins + 1))

# Second chunk: HRU-scale four parameters
S1_hru_flat = S1_all[(n_sb_params * n_subbasins):]
S1_hru      = S1_hru_flat.reshape(n_hru_params, n_hrus)            # shape: (4, n_hrus)
S1_hru_T    = S1_hru.T                                             # shape: (n_hrus, 4)
df_S1_hru   = pd.DataFrame(S1_hru_T, columns=[f'{p}_S1' for p in param_hru_names])
df_S1_hru.insert(0, 'HRU_ID', np.arange(1, n_hrus + 1))

# -----------------------------
# Map: subbasin → HRU range (inclusive)
# (from the table you provided)
# -----------------------------
sub_to_hru_ranges = [
    ( 1,   1,  17), ( 2,  18,  41), ( 3,  42,  64), ( 4,  65,  83), ( 5,  84, 101),
    ( 6, 102, 113), ( 7, 114, 123), ( 8, 124, 135), ( 9, 136, 144), (10, 145, 165),
    (11, 166, 194), (12, 195, 203), (13, 204, 213), (14, 214, 231), (15, 232, 251),
    (16, 252, 270), (17, 271, 284), (18, 285, 296), (19, 297, 313), (20, 314, 323),
    (21, 324, 346), (22, 347, 361), (23, 362, 378), (24, 379, 387), (25, 388, 393),
    (26, 394, 413), (27, 414, 433), (28, 434, 442), (29, 443, 460), (30, 461, 475),
    (31, 476, 487), (32, 488, 504), (33, 505, 521), (34, 522, 532), (35, 533, 555),
    (36, 556, 577), (37, 578, 598), (38, 599, 618), (39, 619, 630),
]

# Build a Series that assigns each HRU its parent Subbasin
hru_to_sub = pd.Series(index=np.arange(1, n_hrus + 1), dtype="Int64")
for sub_id, hru_start, hru_end in sub_to_hru_ranges:
    hru_to_sub.loc[hru_start:hru_end] = sub_id

# Build Series of CH_K1_S1 per Subbasin
ch_k1_by_sub = df_S1_sb.set_index('Subbasin')['CH_K1_S1']

# Map CH_K1 S1 from Subbasin to each HRU via the ranges above
ch_k1_for_hru = hru_to_sub.map(ch_k1_by_sub)

# -----------------------------
# Compute SUM(S1) at HRU-scale:
#   SUM = CH_K1_S1(mapped) + RCHRG_DP_S1 + CANMX_S1 + ESCO_S1 + CN2_S1
# -----------------------------
sum_cols = [f'{p}_S1' for p in param_hru_names]  # the 4 HRU-scale params
df_sum = df_S1_hru[['HRU_ID'] + sum_cols].copy()
df_sum['CH_K1_S1_from_sub'] = ch_k1_for_hru.values
df_sum['S1_sum'] = df_sum['CH_K1_S1_from_sub'] + df_sum[sum_cols].sum(axis=1)


df_sum.to_csv("df_sumed_S1_saHRU.csv")


# Merge SUM(S1) to HRU polygons
gdf_hru_sum = gdf_hru.merge(df_sum[['HRU_ID', 'S1_sum']], on='HRU_ID', how='left')

# -----------------------------
# Robust color normalization for map
# -----------------------------
vals = gdf_hru_sum['S1_sum'].dropna().to_numpy()
if vals.size == 0:
    vmin, vmax = 0.0, 1.0
else:
    q2, q98 = np.nanquantile(vals, [0.02, 0.98])
    vmin    = float(q2) if np.isfinite(q2) else float(np.nanmin(vals))
    vmax    = float(q98) if np.isfinite(q98) else float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1e-3

norm = Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# Top-15 HRUs by SUM(S1)
# -----------------------------
top15 = (df_sum[['HRU_ID', 'S1_sum']]
         .dropna()
         .sort_values('S1_sum', ascending=False)
         .head(15))

# -----------------------------
# Plot: 1x2 (map | bars) with in-map colorbar (top-right)
# -----------------------------
fig = plt.figure(figsize=(10.5, 5.2), dpi=100)
gs  = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)

ax_map = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])

# Left: HRU SUM(S1) map
gdf_hru_sum.plot(column='S1_sum', cmap=CMAP, norm=norm, linewidth=0, ax=ax_map)
# Overlays
try:
    gdf_watersheld.boundary.plot(ax=ax_map, color='0.6', linewidth=0.25)
except Exception:
    pass
gdf_basin.boundary.plot(ax=ax_map, color='k', linewidth=0.6)

# Reaches + ZHJS
if 'Linewidth' in gdf_reach.columns:
    gdf_reach.plot(ax=ax_map, color='b', lw=gdf_reach['Linewidth'], alpha=0.8, label='Reach')
else:
    gdf_reach.plot(ax=ax_map, color='b', lw=0.8, alpha=0.8, label='Reach')
gdf_zhjs.plot(ax=ax_map, color='k', markersize=18, marker='s', edgecolor='white', linewidth=0.4, label='ZHJS')

ax_map.set_axis_off()
ax_map.set_title("HRU-scale SUM of Sobol's $S_{i}$ (CH_K1 from Subbasin)")

# In-map colorbar (top-right)
pos   = ax_map.get_position()
cb_ax = fig.add_axes([pos.x1 - 0.06, pos.y1 - 0.32, 0.012, 0.24])
sm    = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cb    = fig.colorbar(sm, cax=cb_ax, orientation='vertical')
cb.set_label("SUM $S_{i}$", fontsize=9)

# Optional legend inside map
handles, labels = ax_map.get_legend_handles_labels()
if handles:
    leg = ax_map.legend(loc="lower right", frameon=True, framealpha=0.7, fontsize=8)
    leg.get_frame().set_linewidth(0.3)

# Right: bar plot
ax_bar.barh(
    y=top15['HRU_ID'].astype(str).values,
    width=top15['S1_sum'].values,
    color=plt.cm.get_cmap(CMAP)(norm(top15['S1_sum'].values)),
    edgecolor='k',
    linewidth=0.4
)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("SUM $S_{i}$")
ax_bar.set_ylabel("HRU (Top 15)")
ax_bar.set_xlim(vmin, 0.3)
ax_bar.grid(axis='x', linestyle='--', alpha=0.4)

# Save / show
fig.suptitle("HRU-scale SUM(S1): CH_K1 mapped from Subbasins + HRU parameters (RCHRG_DP, CANMX, ESCO, CN2)",
             y=0.98, fontsize=12, fontweight='bold')
fig.savefig('hru_SUM_S1_map_bar.png', dpi=300, bbox_inches='tight')
plt.show()
