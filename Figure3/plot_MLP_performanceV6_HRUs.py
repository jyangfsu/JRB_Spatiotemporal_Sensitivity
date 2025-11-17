# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 16:36:51 2025

@author: Jing
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Set random seed
seed = 42

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Load Data ===
X = np.load('LHS_param_values_spatialHRUs_200000.npy')                       # Shape: (N_samples, 2559)
Y = np.load('LHS_Y_flow_SpatialHRUs_200K.npy')  # Shape: (N_samples, 180)


# === Normalize ===
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

# === Train-test split ===
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=seed)

# === Convert to PyTorch tensors ===
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# MLP Model for Subbasin Scale
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

class HRUMLP(nn.Module):
    def __init__(self, input_dim=2559, output_dim=180):
        super(HRUMLP, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: Reduce dimensionality aggressively
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Layer
            nn.Linear(256, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

# === Load trained model ===
model = HRUMLP().to(device)
model.load_state_dict(torch.load("mlp_hru_model_best.pth"))
model.eval()
loss_fn = RMSELoss()

# === Predict ===
with torch.no_grad():
    y_pred = model(X_test)

# === Inverse transform ===
y_pred_np = y_scaler.inverse_transform(y_pred.cpu().numpy())     # [n_samples, 180]
y_true_np = y_scaler.inverse_transform(Y_test.cpu().numpy())     # [n_samples, 180]

rmse_global = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
print("Global RMSE:", rmse_global)

# === Load observed flow ===
df_obs = pd.read_csv('obs_flow_36.csv')
flow_values = df_obs['Value'].values
assert flow_values.shape[0] == 180

# === Time axis: Jan 1972 to Dec 1986 ===
time_axis = pd.date_range(start='1972-01-01', periods=180, freq='M')
time_vals = mdates.date2num(time_axis)

# === Select best sample (lowest RMSE to observation) ===
rmse_vs_obs = np.sqrt(np.mean((y_true_np - flow_values) ** 2, axis=1))
best_idx = np.argmin(rmse_vs_obs)
best_pred = y_pred_np[best_idx]
best_true = y_true_np[best_idx]

# === Errors (physical units) ===
errors = (y_pred_np - y_true_np)   # [n_samples, 180]

# >>> RMSE per time step (replace mean-error usage)
rmse_t = np.sqrt(np.mean((y_pred_np - y_true_np)**2, axis=0))    # shape (180,)

# === Prepare grid for heatmap using pcolormesh ===
n_samples, n_months = errors.shape
delta = (time_vals[-1] - time_vals[0]) / (n_months - 1)
time_vals_edges = np.concatenate([time_vals, [time_vals[-1] + delta]])  # (181,)
sample_edges = np.arange(n_samples + 1)
Xh, Yh = np.meshgrid(time_vals_edges, sample_edges)

# === Figure layout: 2 rows x 2 cols
# Left column: top = series; bottom = heatmap
# Right column: top = scatter (outside main plots); bottom = PDF of error
fig = plt.figure(figsize=(12, 5.2))
gs = gridspec.GridSpec(
    nrows=2, ncols=2,
    height_ratios=[2.2, 1.0],
    width_ratios=[4.0, 1.35],
    hspace=0.15, wspace=0.15
)

# === Top-left: model predictions vs observation ===
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_axis, best_true, label='SWAT simulated with\nlowest RMSE', linestyle='-', lw=2.0, color='k')
ax1.plot(time_axis, best_pred, label='MLP predicted', linestyle='--', lw=2.0)
ax1.scatter(time_axis, flow_values, label='Observed', marker='o', edgecolors='black',
            facecolors='none', s=25, linewidth=0.8)

for i in range(40000):
    ax1.plot(time_axis, y_pred_np[i, :], linestyle='-', lw=1.0, color='grey', alpha=0.01, zorder=-1)



ax1.set_ylim([0, 400])
ax1.set_ylabel("Monthly Runoff (m³/s)", fontsize=11)
ax1.legend(loc='upper right', fontsize=10, frameon=False)
ax1.grid(False)
ax1.tick_params(axis='both', labelsize=10)
ax1.xaxis.set_tick_params(labelbottom=False)


ax1.text(0.02, 0.96, "(b1)",
        transform=ax1.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))


# === Bottom-left: error heatmap with RMSE(t) on the right axis ===
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

vmax = np.max(np.abs(errors))
# Set a symmetric range or fixed if you prefer
pcm = ax2.pcolormesh(Xh, Yh, errors, cmap='bwr', vmin=-40, vmax=40, shading='auto')

ax2.set_yticks([])
ax2.annotate('', xy=(-0.020, 0.9), xytext=(-0.020, 0.05),
             xycoords='axes fraction',
             arrowprops=dict(arrowstyle='<|-', lw=1.0, color='black'))
ax2.text(-0.05, 0.5, f'{n_samples}\ntest cases',
         transform=ax2.transAxes, rotation=90, va='center', ha='center', fontsize=11)

ax2.tick_params(axis='x', labelsize=10, rotation=60)
ax2.grid(False)

# Format x-axis as datetime (shared with ax1)
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_xticks(time_vals[::12])

# >>> Right y-axis: RMSE per time step
ax2_right = ax2.twinx()
ax2_right.plot(time_axis, rmse_t, label='RMSE (per time)', lw=2)
ax2_right.set_ylim([0, 20])  # RMSE is non-negative
ax2_right.set_ylabel("RMSE (m³/s)", fontsize=11, color='#1F77B4')
ax2_right.tick_params(axis='y', labelsize=10, labelcolor='#1F77B4')
ax2_right.spines['right'].set_color('black')

ax2.text(0.02, 0.93, "(b2)",
        transform=ax2.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))


# === Top-right: scatter plot (outside main) ===
ax3 = fig.add_subplot(gs[0, 1])

y_true_flat = y_true_np.flatten()
y_pred_flat = y_pred_np.flatten()
r2 = r2_score(y_true_flat, y_pred_flat)

ax3.scatter(y_true_flat, y_pred_flat, alpha=0.4, s=8)
mn = min(y_true_flat.min(), y_pred_flat.min())
mx = max(y_true_flat.max(), y_pred_flat.max())
ax3.plot([mn, mx], [mn, mx], linestyle='--', lw=2, color='#F97878')

ax3.set_xlabel("SWAT sim. (m³/s)", fontsize=11)
ax3.set_ylabel("MLP pred. (m³/s)", fontsize=11)

# >>> move y-axis ticks and labels to the right
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
#ax3.spines['left'].set_visible(False)
ax3.tick_params(axis='x', labelsize=10)
ax3.xaxis.set_label_position("top")
ax3.xaxis.tick_top()

ax3.set_xlim([0, 500])
ax3.set_ylim([0, 500])
ax3.tick_params(axis='both', labelsize=10)
ax3.text(0.35, 0.255, f"$R^2$ = {r2:.4f}\nRMSE = {rmse_global: .2f}m³/s\nN={n_samples*errors.shape[1]} points",
         transform=ax3.transAxes, ha='left', va='top', fontsize=10)

ax3.text(0.05, 0.96, "(b3)",
        transform=ax3.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

# === Bottom-right: PDF of the error ===
ax4 = fig.add_subplot(gs[1, 1])

err_flat = (y_pred_flat - y_true_flat)

# ---- bins via Freedman–Diaconis ----
q75, q25 = np.percentile(err_flat, [75, 25])
iqr = q75 - q25
bin_width = 2 * iqr * (len(err_flat) ** (-1/3)) if iqr > 0 else None
if bin_width and bin_width > 0:
    bins = max(10, int(np.ceil((err_flat.max() - err_flat.min()) / bin_width)))
else:
    bins = 50

# histogram (PDF)
ax4.hist(err_flat, bins=bins, density=True, alpha=0.8)

# axes labels & ticks (right y-axis)
ax4.set_xlabel("Error = MLP pred. − SWAT sim. (m³/s)", fontsize=11)
ax4.set_ylabel("Density", fontsize=11)
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
ax4.set_xlim([-10, 10])
ax4.set_ylim([0, 0.45])
ax4.tick_params(axis='both', labelsize=10)

# ==== NEW: percentage in [-5, 5] ====
within = (err_flat >= -5) & (err_flat <= 5)
pct = 100.0 * np.mean(within)

# vertical lines at -5 and 5
ax4.axvline(-5, linestyle=':', color='k')
ax4.axvline(5, linestyle=':', color='k')

# percentage text at (0, 0.2) in black
txt = f"{pct:.2f}%"
ax4.text(0, 0.2, txt, ha='center', va='bottom', color='black', fontsize=11)

# arrows from (-5, 0.2) and (5, 0.2) pointing to the text at (0, 0.2)
ax4.annotate("",
             xy=(0, 0.2), xytext=(-5, 0.2),
             arrowprops=dict(arrowstyle="<-", lw=1.2), clip_on=False)
ax4.annotate("",
             xy=(0, 0.2), xytext=(5, 0.2),
             arrowprops=dict(arrowstyle="<-", lw=1.2), clip_on=False)


ax4.text(0.05, 0.93, "(b4)",
        transform=ax4.transAxes, ha='left', va='top',
        fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))


# ---- Add horizontal colorbar at bottom ----
cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.03])  # [left, bottom, width, height]
norm = Normalize(vmin=-50, vmax=50)

sm = ScalarMappable(norm=norm, cmap='bwr')


sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label("Error (MLP pred. - SWAT sim.) (m³/s)", fontsize=12)
cbar.ax.tick_params(labelsize=10)


fig.suptitle('(b) HRU-scale MLP surrogate',
             x=0.10, y=0.97, ha='left', va='top',
             fontsize=13, fontweight='bold')


# Tight layout AFTER suptitle placement: adjust to give label space
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig('hru_scale_MLP_performance.png', dpi=300)
plt.show()
