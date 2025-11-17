# -*- coding: utf-8 -*-
"""
HRU-scale time-varying sensitivity figure
- Left panels: ST_i heatmaps (subbasin or HRU × time), with runoff overlay
- Right panels: Lagged Spearman-ρ heatmaps (rows = subbasins/HRUs, x = lag in INTERVALS)
  * intervals are in window steps (e.g., 3m/step=1 => 1 interval = 1 month;
    24m/step=6 => 1 interval = 6 months)
  * stars (*) mark p < 0.05
  * rows with very small ST_i across time (max < sti_thresh) are masked (NaN)

Windows shown: (a) 3-month (step=1), (b) 24-month (step=6)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the HRUs for each subbasin based the HRU Analysis Report in ArcSWAT
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

# -------------------
# Style
# -------------------
cmap_st  = 'hot_r'   # for ST_i
cmap_lag = 'coolwarm'  # for lag-ρ
plt.rcParams.update({
    "axes.spines.top": True,
    "axes.spines.right": True,
    "font.size": 10
})

# -------------------
# Time axis
# -------------------
n_months = 180
months = np.arange('1972-01', '1987-01', dtype='datetime64[M]')
desired_years = ['1973','1975','1977','1979','1981','1983','1985']
xtick_locs   = [i for i, m in enumerate(months) if str(m)[:4] in desired_years and str(m)[5:7]=='01']
xtick_labels = [str(months[i])[:4] for i in xtick_locs]

# -------------------
# Model metadata
# -------------------
param_labels   = ['CH_K1', 'RCHRG_DP', 'CANMX', 'ESCO', 'CN2']
n_params       = len(param_labels)   # 5
sub_per_param  = 39                  # CH_K1 at subbasin scale
hru_per_param  = 630                 # HRU-scale parameters
total_features = 2559                # 39 + 4*630

# ST heatmap scale
vmin_st, vmax_st = 0.0, 0.01

# Windows to plot
WIN_SPECS = [
    dict(name="                (a) 3-month window with 1-month step",  win_len=3,  step=1,  label_short="3m"),
    dict(name="                  (b) 24-month window with 6-month step", win_len=24, step=6, label_short="24m"),
]

# Lag domain (in INTERVALS)
MAX_LAG_INT = 2  # will display -2..+2
lag_ticks   = np.arange(-MAX_LAG_INT, MAX_LAG_INT+1, 3)

# Skip rows whose ST_i never rises above this threshold
sti_thresh = 1e-3

# -------------------
# Helpers
# -------------------
def compute_centers(win_len, step):
    starts  = np.arange(0, n_months - win_len + 1, step)
    centers = starts + win_len // 2
    return starts, centers

def expand_window_series_to_months(values, centers, half_block):
    """Expand windowed series to a monthly array via symmetric block filling."""
    arr = np.full(n_months, np.nan)
    for v, c in zip(values, centers):
        s, e = max(c - half_block, 0), min(c + half_block, n_months)
        arr[s:e] = v
    return arr

def normalize_to_range(x, ymin=0.0, ymax=39.0):
    """Normalize a 1D array to [ymin, ymax], preserving NaNs."""
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)): 
        return np.full_like(x, np.nan)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return np.full_like(x, np.nan)
    if np.isclose(xmax - xmin, 0):
        out = np.full_like(x, (ymin + ymax) / 2.0)
        out[np.isnan(x)] = np.nan
        return out
    return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin

def spearman_nan_safe(x, y, min_n=6):
    """NaN-robust Spearman; returns (rho, p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan, np.nan
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p)

def compute_lag_corr_intervals(series_2D, q_monthly, step_size_months, max_lag_int=3, min_n=6, sti_thresh=1e-5):
    """
    Lagged Spearman ρ and p in INTERVALS.

    series_2D : (N, 180) ST_i time series (rows = subbasins or HRUs; monthly grid; may contain NaNs)
    q_monthly : (180,)  matching runoff series (with NaNs where window undefined)
    step_size_months : months per interval (e.g., 1 for 3m; 6 for 24m)
    """
    N, T = series_2D.shape
    lags_int = np.arange(-max_lag_int, max_lag_int + 1)
    rho_lag = np.full((N, lags_int.size), np.nan)
    p_lag   = np.full((N, lags_int.size), np.nan)
    for j in range(N):
        Sj = np.asarray(series_2D[j, :], dtype=float)
        # skip rows with tiny sensitivity everywhere
        if not (np.nanmax(Sj) > sti_thresh):
            continue
        for k, tau_int in enumerate(lags_int):
            shift = int(tau_int * step_size_months)  # convert intervals to months
            if shift > 0:
                x, y = Sj[:-shift], q_monthly[shift:]
            elif shift < 0:
                x, y = Sj[-shift:], q_monthly[:shift]
            else:
                x, y = Sj, q_monthly
            rho, p = spearman_nan_safe(x, y, min_n=min_n)
            rho_lag[j, k] = rho
            p_lag[j, k]   = p
    return rho_lag, p_lag, lags_int

# -------------------
# Load runoff
# -------------------
obs_flow = pd.read_csv('obs_flow_36.csv')['Value'].values  # (180,)

# Build windowed runoff arrays (block-constant monthly series) per window
q_monthlies = []   # length-180 arrays (with NaNs at edges)
q_overlays  = []   # normalized for plotting on twin axis

for spec in WIN_SPECS:
    W, step = spec['win_len'], spec['step']
    starts, centers = compute_centers(W, step)
    q_vals = np.array([obs_flow[s:s+W].mean() for s in starts])
    half_block = max(1, step // 2)  # symmetric block width around center
    q_monthly = expand_window_series_to_months(q_vals, centers, half_block)
    q_monthlies.append(q_monthly)
    # overlay range: 39 (CH_K1) by default; we rescale per panel below
    q_overlays.append(q_monthly.copy())

# -------------------
# Load ST arrays and expand to monthly grids
# (HRU-scale files: 2559 columns total: 39 (CH_K1 at subbasin) + 4*630 (HRU))
# -------------------
st_3m  = np.load("ST_NSE_temp_3mo_windows.npy")               # (178, 2559)
st_24m = np.load("ST_NSE_temp_24mo_windows_6mo_step.npy")     # (27,  2559)

# Expand to monthly grids
# 3m: put rows at months 1..178
exp_3m = np.full((n_months, st_3m.shape[1]), np.nan); exp_3m[1:179, :] = st_3m

# 24m: block-fill around centers with step=6
starts24, centers24 = compute_centers(24, 6)
exp_24m = np.full((n_months, st_24m.shape[1]), np.nan)
half_block_24 = max(1, 6 // 2)  # = 3 months on each side -> 6-month block
for k, c in enumerate(centers24):
    s, e = max(c - half_block_24, 0), min(c + half_block_24, n_months)
    exp_24m[s:e, :] = st_24m[k, :]

st_matrices = [exp_3m, exp_24m]  # per WIN_SPECS order

# -------------------
# Figure layout: 5 rows (parameters) × 4 cols per row: [ST | LAG] × 2 windows
# -------------------
fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(n_params, 4, width_ratios=[1, 0.25, 1, 0.25],
                        wspace=0.1, hspace=0.15)

plt.subplots_adjust(left=0.10)

fig.text(
    0.02, 0.5,                          
    "Time-varying SSA at the HRU scale",
    rotation=90,                        
    va='center', ha='left',             
    fontsize=12, fontweight='bold'      
)

for i, p in enumerate(param_labels):
    # Slice indices per parameter:
    # CH_K1 (subbasin-level): first 39 cols
    # Others (HRU-level): consecutive 630-col blocks after the first 39
    if i == 0:
        # CH_K1 -> subbasin slice
        param_start = 0
        param_end   = sub_per_param
        yticks_pos  = np.arange(4, sub_per_param, 10) + 0.5
        yticks_lab  = np.arange(5, sub_per_param+1, 10)
        n_rows_this = sub_per_param
    else:
        # HRU block i-1
        param_start = sub_per_param + (i-1)*hru_per_param
        param_end   = sub_per_param + i*hru_per_param
        yticks_pos  = 630 - np.arange(49, hru_per_param, 150) + 0.5 + 1
        yticks_lab  = np.arange(50, hru_per_param+1, 150)
        n_rows_this = hru_per_param

    for w_idx, spec in enumerate(WIN_SPECS):
        title      = spec['name']
        step_month = spec['step']
        exp_mat    = st_matrices[w_idx]    # (time x 2559)
        q_mon      = q_monthlies[w_idx]    # length 180
        data       = exp_mat[:, param_start:param_end].T  # rows=subs/HRUs, cols=time (180)

        # ---------- (Left) ST_i heatmap ----------
        ax_hm = fig.add_subplot(gs[i, 2*w_idx])
        sns.heatmap(
            data[::-1],            # flip vertically to show ID ascending bottom->top
            ax=ax_hm,
            cmap=cmap_st, vmin=vmin_st, vmax=vmax_st,
            xticklabels=False, yticklabels=False, cbar=False,
            linewidths=0.0
        )
        if i == 0:
            ax_hm.set_title(title, fontsize=12, weight='bold')
        # y-axis labels only on first column
        if w_idx == 0:
            ax_hm.set_ylabel(f"{p}\n" + ("Subbasin ID" if i==0 else "HRU ID"), fontsize=11)
            # add sparse y ticks (optional)
            ax_hm.set_yticks(yticks_pos)
            # For flipped heatmap, reverse labels for subbasins; HRU can keep forward labels
            if i == 0:
                ax_hm.set_yticklabels(yticks_lab[::-1])
            else:
                ax_hm.set_yticklabels(yticks_lab)  # match visual orientation
                
        if i == n_params-1:
            ax_hm.set_xticks(xtick_locs)
            ax_hm.set_xticklabels(xtick_labels, rotation=60)
            ax_hm.set_xlabel("Year")
            
        #ax_hm.axhline(y = 631-np.array(hru_ranges[33]).min(), color='b', linestyle='-', lw=0.5)
        #ax_hm.axhline(y = 631-np.array(hru_ranges[34]).max(), color='b', linestyle='-', lw=0.5)
        
        # Overlay runoff (normalized to row count for visual alignment)
        ax_flow = ax_hm.twiny()
        q_overlay = normalize_to_range(q_mon, 0, n_rows_this-1)  # 0..(rows-1)
        ax_flow.plot(np.arange(n_months), q_overlay, color='k', lw=1.0, alpha=0.85)
        ax_flow.set_xlim(ax_hm.get_xlim())
        ax_flow.set_xticks(xtick_locs); ax_flow.set_xticklabels([])
        ax_flow.tick_params(axis='x', length=0)
        ax_flow.invert_yaxis()

        # ---------- (Right) Lag–correlation heatmap (intervals) ----------
        rho_lag, p_lag, lags_int = compute_lag_corr_intervals(
            series_2D=data, q_monthly=q_mon,
            step_size_months=step_month, max_lag_int=MAX_LAG_INT,
            min_n=6, sti_thresh=sti_thresh
        )

        ax_lag = fig.add_subplot(gs[i, 2*w_idx + 1])
        hm = sns.heatmap(
            rho_lag,               # NOT flipped; rows are 0..N-1 (top..bottom)
            ax=ax_lag,
            cmap=cmap_lag, vmin=-1, vmax=1, center=0,
            cbar=False,
            xticklabels=[str(v) for v in lags_int],
            yticklabels=False,
            linewidths=0.0, linecolor='0.85'  # full cell borders
        )
        # strengthen spines
        for _, spine in ax_lag.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("k")

        # zero-lag vertical guide (if desired, draw between cells)
        # zero_idx = np.where(lags_int == 0)[0][0]
        # ax_lag.vlines(zero_idx + 0.5, *ax_lag.get_ylim(), colors='k', lw=0.8, alpha=0.6)

        # significance stars
        J, K = rho_lag.shape
        for r in range(J):          # r = row index as displayed
            for c in range(K):
                pval = p_lag[r, c]
                if np.isfinite(pval) and (pval < 0.05):
                    ax_lag.text(
                        c + 0.5, r + 0.5, 'x',
                        ha='center', va='center',
                        fontsize=3, color='k'
                    )

        # x-axis labels on bottom row only
        if i == n_params - 1:
            ax_lag.set_xlabel("Lag\n(interval)")  # intervals: multiply by step-months to get months
        else:
            ax_lag.set_xticklabels([])
            

# --- Shared colorbars ---
# ST_i
cax_st = fig.add_axes([0.12, 0.02, 0.30, 0.018])
norm_st = Normalize(vmin=vmin_st, vmax=vmax_st)
sm_st = ScalarMappable(norm=norm_st, cmap=cmap_st); sm_st.set_array([])
cbar_st = fig.colorbar(sm_st, cax=cax_st, orientation='horizontal', extend='both')
cbar_st.set_label("Total-effect $S_{Ti}$", fontsize=11)
cbar_st.ax.tick_params(labelsize=10)

# Lag rho
cax_lag = fig.add_axes([0.515, 0.02, 0.30, 0.018])
norm_lag = Normalize(vmin=-1, vmax=1)
sm_lag = ScalarMappable(norm=norm_lag, cmap=cmap_lag); sm_lag.set_array([])
cbar_lag = fig.colorbar(sm_lag, cax=cax_lag, orientation='horizontal', extend='both')
cbar_lag.set_label("Spearman's r", fontsize=11)
cbar_lag.ax.tick_params(labelsize=10)

plt.savefig("Fig_all_windows_STi_with_LagCorr_intervals.png", dpi=300, bbox_inches='tight')
plt.show()
