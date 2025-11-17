# -*- coding: utf-8 -*-
"""
All barplots replaced by lag–correlation heatmaps (x-axis = intervals).
Adds: (i) significance stars for p < 0.05 and (ii) grid lines on the heatmaps.

For each parameter (rows) and each window (columns), we show:
  [left]  ST_i heatmap (subbasin × time)
  [right] Lagged Spearman rho heatmap (subbasin × lag in INTERVALS)

Interpretation:
  tau_int > 0  => sensitivity leads flow by tau_int * step_size_months
  tau_int < 0  => flow leads sensitivity by |tau_int| * step_size_months
"""

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




# -------------------
# Styles
# -------------------
cmap_st  = 'hot_r'      # ST_i heatmaps
cmap_lag = 'coolwarm'   # lag heatmaps
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
xtick_locs = [i for i, m in enumerate(months) if str(m)[:4] in desired_years and str(m)[5:7]=='01']
xtick_labels = [str(months[i])[:4] for i in xtick_locs]

# -------------------
# Model metadata
# -------------------
param_labels = ['CH_K1', 'RCHRG_DP', 'CANMX', 'ESCO', 'CN2']
n_params = len(param_labels)      # 5
sub_per_param = 39
vmin_st, vmax_st = 0.0, 0.1

# Windows to show
WIN_SPECS = [
    dict(name="                (a) 3-month window with 1-month step",  win_len=3,  step=1),
    dict(name="                  (b) 24-month window with 6-month step", win_len=24, step=6),
]

# Lag range in INTERVALS for all windows (constant)
MAX_LAG_INT = 2   # show -3..+3 intervals on x-axis

# -------------------
# Helpers
# -------------------
def spearman_nan_safe(x, y, min_n=6):
    """Return scalar (rho, p) with robust NaN/constant-series handling."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan, np.nan
    x = x[m]; y = y[m]
    # Spearman is undefined if either series is (nearly) constant
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return (float(rho) if np.isscalar(rho) else np.nan,
            float(p)   if np.isscalar(p)   else np.nan)

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

def expand_window_series_to_months(values, centers, half_block):
    """Expand windowed series (values at centers) to monthly array by block fill."""
    arr = np.full(n_months, np.nan)
    for v, c in zip(values, centers):
        s, e = max(c - half_block, 0), min(c + half_block, n_months)
        arr[s:e] = v
    return arr

def compute_centers(win_len, step):
    """Return start indices and center indices (in months)."""
    starts = np.arange(0, n_months - win_len + 1, step)
    centers = starts + win_len // 2
    return starts, centers

def compute_lag_corr_intervals(series_39x180, q_monthly, step_size_months,
                               max_lag_int=3, min_n=6, sti_thresh=0.01):
    """
    Lagged Spearman correlation in UNITS OF INTERVALS.
    Skips subbasins whose STi are all < sti_thresh (or undefined).
    Returns: rho_lag (J×L), p_lag (J×L), lags_int (L,)
    """
    series_39x180 = np.asarray(series_39x180, dtype=float)
    q_monthly = np.asarray(q_monthly, dtype=float).ravel()

    J, T = series_39x180.shape
    lags_int = np.arange(-max_lag_int, max_lag_int + 1, dtype=int)
    L = lags_int.size
    rho_lag = np.full((J, L), np.nan, dtype=float)
    p_lag   = np.full((J, L), np.nan, dtype=float)

    for j in range(J):
        Sj = np.asarray(series_39x180[j, :], dtype=float).ravel()

        # --- robust skip: all NaN/inf or max <= threshold
        if not np.isfinite(Sj).any():
            continue
        max_sti = np.nanmax(Sj)
       
        if max_sti > sti_thresh:
            print(max_sti)
            for k, tau_int in enumerate(lags_int):
                shift = int(tau_int * step_size_months)  # intervals → months
    
                # Build aligned segments; guard against empty after shift
                if shift > 0:
                    if shift >= T: 
                        continue
                    x, y = Sj[:-shift], q_monthly[shift:]
                elif shift < 0:
                    if -shift >= T:
                        continue
                    x, y = Sj[-shift:], q_monthly[:shift]
                else:
                    x, y = Sj, q_monthly
    
                # Compute rho,p (function also checks NaNs, min_n, and zero-variance)
                rho, p = spearman_nan_safe(x, y, min_n=min_n)
                rho_lag[j, k] = rho
                p_lag[j, k]   = p
        else:
            print('Not met')

    return rho_lag, p_lag, lags_int
# -------------------
# Load runoff
# -------------------
obs_flow = pd.read_csv('obs_flow_36.csv')['Value'].values  # length 180

# Prepare per-window runoff series (monthly block-fill) and overlays
q_monthlies, q_overlays = [], []
st_matrices = []

# Load ST arrays
st_3m  = np.load("ST_NSE_temp_3mo_windows.npy")         # (178, 195), centers 1..178
st_24m = np.load("ST_NSE_24mo_window_6mo_step.npy")     # (27, 195)

# Build window-wise runoff and expanded ST matrices
for spec in WIN_SPECS:
    W, step = spec['win_len'], spec['step']
    starts, centers = compute_centers(W, step)
    q_vals = np.array([obs_flow[s:s+W].mean() for s in starts])
    half_block = max(1, step // 2)
    q_monthly = expand_window_series_to_months(q_vals, centers, half_block)
    q_monthlies.append(q_monthly)
    print(np.nanmin(q_monthly), np.nanmax(q_monthly))
    
    q_overlays.append(normalize_to_range(q_monthly, 0, 39))

# Expanded ST to monthly grid matching windows
# 3-month window
exp_3m = np.full((n_months, st_3m.shape[1]), np.nan)
exp_3m[1:179, :] = st_3m
st_matrices.append(exp_3m)

# 24-month window
_, centers24 = compute_centers(24, 6)
exp_24m = np.full((n_months, st_24m.shape[1]), np.nan)
for k, c in enumerate(centers24):
    s, e = max(c - max(1, 6 // 2), 0), min(c + max(1, 6 // 2), n_months)
    exp_24m[s:e, :] = st_24m[k, :]
st_matrices.append(exp_24m)

# -------------------
# Figure: 5 rows (parameters) × 4 columns ([ST | LAG] × 2 windows)
# -------------------
fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(n_params, 4, width_ratios=[1, 0.25, 1, 0.25],
                        wspace=0.1, hspace=0.15)

plt.subplots_adjust(left=0.10)

fig.text(
    0.02, 0.5,                          
    "Time-varying SSA at the subbasin scale",
    rotation=90,                        
    va='center', ha='left',             
    fontsize=12, fontweight='bold'      
)


lag_xticks = np.arange(-MAX_LAG_INT, MAX_LAG_INT + 1, 1)

for i, p in enumerate(param_labels):
    param_start = i * sub_per_param
    param_end   = (i + 1) * sub_per_param
    ytick_pos   = np.arange(4, sub_per_param, 10) + 0.5
    ytick_lab   = np.arange(5, sub_per_param+1, 10)

    for w_idx, spec in enumerate(WIN_SPECS):
        title = spec['name']
        step  = spec['step']        # months per interval
        exp_mat    = st_matrices[w_idx]   # (time × 195)
        q_monthly  = q_monthlies[w_idx]
        q_overlay  = q_overlays[w_idx]

        # --- (Left) ST_i heatmap (subbasin × time) ---
        ax_hm = fig.add_subplot(gs[i, 2*w_idx])
        data  = exp_mat[:, param_start:param_end].T  # (39, 180)
        sns.heatmap(data[::-1], ax=ax_hm, cmap=cmap_st, vmin=vmin_st, vmax=vmax_st,
                    xticklabels=False, yticklabels=5 if w_idx==0 else False, cbar=False)
        if i == 0:
            ax_hm.set_title(f"{title}", fontsize=12, weight='bold')
        if w_idx == 0:
            ax_hm.set_ylabel(f"{p}\nSubbasin ID", fontsize=11)
            ax_hm.set_yticks(ytick_pos); ax_hm.set_yticklabels(ytick_lab[::-1])
        else:
            ax_hm.set_ylabel("")
        if i == n_params-1:
            ax_hm.set_xticks(xtick_locs); ax_hm.set_xticklabels(xtick_labels, rotation=60)
            ax_hm.set_xlabel('Year')
            
        

        # Optional runoff overlay on twiny (normalized 0..39)
        ax_flow = ax_hm.twiny()
        ax_flow.plot(np.arange(n_months), q_overlay, color='k', lw=1.0, alpha=0.85)
        ax_flow.set_xlim(ax_hm.get_xlim())
        ax_flow.set_xticks(xtick_locs); ax_flow.set_xticklabels([])
        ax_flow.tick_params(axis='x', length=0)
        ax_flow.invert_yaxis()

        # --- (Right) Lag–correlation heatmap (rows=subbasins, cols=lag intervals) ---
        rho_lag, p_lag, lags_int = compute_lag_corr_intervals(
            series_39x180=data,     # shape (39, 180)
            q_monthly=q_monthly,    # shape (180,)
            step_size_months=step,
            max_lag_int=MAX_LAG_INT,
            min_n=6
        )
        ax_lag = fig.add_subplot(gs[i, 2*w_idx + 1])
        
        # Do not flip rows; top row is subbasin 1, consistent with your ST panel
        rho_plot = rho_lag
        p_plot   = p_lag
        J, K     = rho_plot.shape
        
        # Force ticks to cell centers and show integer lag labels
        xticks_centers = np.arange(K) + 0.5
        
        hm = sns.heatmap(
            rho_plot,
            ax=ax_lag,
            cmap=cmap_lag,
            vmin=-1, vmax=1, center=0,
            cbar=False,
            xticklabels=lags_int,          # labels only; we set tick positions next
            yticklabels=False,             # hide y tick labels
            linewidths=0.6, linecolor='0.85'  # full cell borders
        )
        
        # Put x ticks at cell centers (align labels with cells)
        ax_lag.set_xticks(xticks_centers)
        ax_lag.set_xticklabels([str(v) for v in lags_int], rotation=0)
        
        # Remove y ticks entirely
        ax_lag.set_yticks([])
        
        # Draw thin frame
        for _, spine in ax_lag.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("k")
        
        # Vertical line at zero lag boundary
        #zero_idx = int(np.where(lags_int == 0)[0][0])
        #ax_lag.axvline(zero_idx + 0.5, color='k', lw=0.8, alpha=0.6)
        
        # Significance stars at cell centers
        alpha = 0.05
        for r in range(J):
            for c in range(K):
                pval = p_plot[r, c]
                if np.isfinite(pval) and pval < alpha:
                    ax_lag.text(
                        c + 0.5,           # cell center x
                        r + 0.5,           # cell center y
                        'x',
                        ha='center', va='center',
                        fontsize=6, color='k',
                        zorder=5, clip_on=False
                    )
                
                # tidy x ticks
                ax_lag.set_xticks(np.arange(len(lags_int)) + 0.5)
                ax_lag.set_xticklabels([str(v) for v in lags_int], rotation=60)
                if i == n_params - 1:
                    ax_lag.set_xlabel("Lag\n(interval)")   # multiply by step for months if you want
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
