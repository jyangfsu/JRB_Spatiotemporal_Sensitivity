# -*- coding: utf-8 -*-
"""
Two-panel MLP architecture figure (WRR style) with **fixed overall width**
and **fixed widths & gaps for Input and Hidden layers** across both subplots.

Outputs:
  - mlp_architecture_wrr_fixed.pdf  (vector, 600 dpi)
  - mlp_architecture_wrr_fixed.jpg  (raster, 300 dpi)
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.lines import Line2D

import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../lib')

from   str2tex         import str2tex  # in lib/
usetex = True
 
# ---------- WRR-like minimalist defaults ----------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']   # fallback if missing
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ---------- Colors (muted) ----------
COL_IN_BG   = "#b7cfe4"   # input block interior
COL_IN_BAR  = "#4c78a8"   # input title stripe
COL_HID_BG  = "#a7c693"   # hidden block interior
COL_OUT_BG  = "#f3c39e"   # output block interior
COL_OUT_BAR = "#f58518"   # output title stripe / output pill
COL_FRAME   = "#d1d5db"   # light gray frame outline
COL_EDGE    = "k"   # connectors
COL_TEXT    = "#111827"   # text

# ---------- Geometry: FIXED widths & gaps ----------
# Global logical axis width kept fixed as [0, 12]
AX_W, AX_H = 12.0, 3.2

# Fixed sizes used for BOTH panels
START_X         = 1.10         # left start for the first block
BLOCK_H         = 2.15         # block height
TITLE_H_RATIO   = 0.18         # title stripe ratio inside a block
PILL_W          = 0.45         # width of orange/blue vertical pill
LEFT_PILL_W     = 0.55         # width of left blue input-pill
DOT_R           = 0.10

# Fixed widths for layer blocks (same for both panels)
W_INPUT  = 1.50
W_HIDDEN = 1.40
W_OUTPUT = 1.20

# Fixed gaps after blocks (same for both panels)
GAP_AFTER_INPUT  = 0.35
GAP_AFTER_HIDDEN = 0.35
GAP_AFTER_OUTPUT = 0.25
GAP_AFTER_PILL   = 0.10   # trailing gap after the final pill (not important visually)

# ---------- Drawing primitives ----------
def draw_pill(ax, x, y, w, h, color, n_dots=4, dot_r=DOT_R, vpad=0.10):
    pill = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={min(w, h)*0.25}",
                          ec=COL_EDGE, fc=color, lw=0.8)
    ax.add_patch(pill)
    usable = h - 2*vpad
    gap = (usable - 2*dot_r*n_dots) / (n_dots - 1) if n_dots > 1 else 0.0
    cy = y + (h - (2*dot_r*n_dots + gap*(n_dots-1))) / 2 + dot_r
    cx = x + w/2
    for _ in range(n_dots):
        ax.add_patch(Circle((cx, cy), dot_r, ec=COL_EDGE, fc="white", lw=0.6))
        cy += 2*dot_r + gap

def draw_group_box(ax, x, y, w, h, title, facecolor, barcolor):
    # outer card
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=COL_FRAME, lw=1.0))
    # title stripe
    t_h = h * TITLE_H_RATIO
    ax.add_patch(Rectangle((x, y + h - t_h), w, t_h, fc=barcolor, ec=COL_FRAME, lw=1.0))
    # inner pane
    pad = 0.04 * h
    ax.add_patch(Rectangle((x + pad, y + pad), w - 2*pad, h - t_h - 2*pad,
                           fc=facecolor, ec=COL_FRAME, lw=0.8))
    ax.text(x + w/2, y + h - t_h/2, str2tex(title, usetex=usetex), ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

def write_lines(ax, x, y, w, h, lines):
    n = len(lines)
    if n == 0: return
    step = h / (n + 1)
    for i, txt in enumerate(lines, 1):
        ax.text(x + w/2, y + h - i*step, txt, ha="center", va="center",
                fontsize=8, color=COL_TEXT)

def arrow(ax, x0, y0, x1, y1):
    ax.add_line(Line2D([x0, x1], [y0, y1], lw=0.8, color=COL_EDGE))
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.0, color=COL_EDGE, mutation_scale=8))


def connect_edge_to_edge(ax, x_left, w_left, y_left, x_right, y_right):
    """Arrow from right edge of left block to left edge of right block."""
    x0 = x_left + w_left
    x1 = x_right
    ax.annotate("",
                xy=(x1, y_right), xytext=(x0, y_left),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=COL_EDGE,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=6))
    
# ---------- Panel drawer using fixed widths/gaps ----------
def draw_panel(ax, title, left_label, right_label, n_hidden, input_n, hidden_ns, output_n, dropouts=None, use_bn=True):
    ax.axis("off")
    ax.set_xlim(0, AX_W)
    ax.set_ylim(0, AX_H)

    # Title (top-left)
    ax.text(0.00, AX_H + 0.05, title, ha="left", va="top", fontsize=10,
            fontweight="bold", color=COL_TEXT)

    base_y = 0.65
    x = START_X
    # Track spans for connectors
    
    
    # Left blue pill + vertical label
    draw_pill(ax, x - 0.85, base_y - 0.03, LEFT_PILL_W, BLOCK_H + 0.06, COL_IN_BAR)
    ax.text(x - 1.15, base_y + BLOCK_H/2, left_label,
            ha="center", va="center", rotation=90, fontsize=9, color=COL_TEXT)
    
    spans = []  # list of (x_left, width, y_center)
    centers = []
    
    spans.append((x-0.85, LEFT_PILL_W, base_y + BLOCK_H/2))


    # Input block
    draw_group_box(ax, x, base_y, W_INPUT, BLOCK_H, "Input", COL_IN_BG, COL_IN_BAR)
    t_h = BLOCK_H * TITLE_H_RATIO
    pad = 0.04 * BLOCK_H
    write_lines(ax, x + pad, base_y + pad, W_INPUT - 2*pad, BLOCK_H - t_h - 2*pad,
                [f"Linear ({input_n})", "+", "ReLU"])
    centers.append((x + W_INPUT/2, base_y + BLOCK_H/2))
    
    spans.append((x, W_INPUT, base_y + BLOCK_H/2))
    
    x += W_INPUT + GAP_AFTER_INPUT
    
    
    

    # Hidden blocks (fixed width/gap)
    for i in range(n_hidden):
        nval = hidden_ns[i]
        draw_group_box(ax, x, base_y, W_HIDDEN, BLOCK_H, f"Hidden {i+1}", COL_HID_BG, COL_HID_BG)
        lines = [f"Linear ({nval})"]
        if use_bn:
            lines += ["+", "BN"]
        lines += ["+", "ReLU"]
        if dropouts is not None and i < len(dropouts) and dropouts[i] is not None:
            lines += ["+", f"Dropout ({dropouts[i]})"]
        write_lines(ax, x + pad, base_y + pad, W_HIDDEN - 2*pad, BLOCK_H - t_h - 2*pad, lines)
        centers.append((x + W_HIDDEN/2, base_y + BLOCK_H/2))
        
        if i == 0:
            spans.append((x, W_HIDDEN , base_y + BLOCK_H/2))
        else:
            spans.append((x, W_HIDDEN , base_y + BLOCK_H/2))

        x += W_HIDDEN + GAP_AFTER_HIDDEN
        
        

    # Output block
    draw_group_box(ax, x, base_y, W_OUTPUT, BLOCK_H, "Output", COL_OUT_BG, COL_OUT_BAR)
    write_lines(ax, x + pad, base_y + pad, W_OUTPUT - 2*pad, BLOCK_H - t_h - 2*pad,
                [f"Linear ({output_n})"])
    centers.append((x + W_OUTPUT/2, base_y + BLOCK_H/2))
    spans.append((x, W_OUTPUT, base_y + BLOCK_H/2))
    
    x += W_OUTPUT + GAP_AFTER_OUTPUT
    
    
    # Orange pill (right)
    draw_pill(ax, x, base_y - 0.03, PILL_W, BLOCK_H + 0.06, COL_OUT_BAR)
    centers.append((x + PILL_W/2, base_y + BLOCK_H/2))
    
    spans.append((x, W_OUTPUT, base_y + BLOCK_H/2))
    
    x += PILL_W + GAP_AFTER_PILL  # (trailing space only)
        
    # Edge-to-edge connectors
    for i in range(len(spans) - 1):
        (xl, wl, yc_l) = spans[i]
        (xr, wr, yc_r) = spans[i+1]
        connect_edge_to_edge(ax, xl, wl, yc_l, xr, yc_r)

    # Right vertical label (rotate)
    ax.text(x + 0.00, base_y + BLOCK_H/2, right_label,
            ha="center", va="center", rotation=90, fontsize=9, color=COL_TEXT)

# ---------- Build both panels (fixed width & fixed gaps for input/hidden) ----------
fig = plt.figure(figsize=(8.6, 4.6), dpi=600)
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.0)

# (a) Subbasin-scale: 3 hidden layers
ax_a = fig.add_subplot(gs[0, 0])
draw_panel(
    ax=ax_a,
    title="(a) Subbasin-scale MLP architecture",
    left_label="Standardized\nparameters (195-dim)",
    right_label="\n\nMonthly average runoff\n(180-dim)",
    n_hidden=4,
    input_n=195,
    hidden_ns=[1024, 512, 512, 256],
    output_n=180,
    dropouts=[0.2, 0.2, 0.1, None],   # match your annotation style
    use_bn=True                   # set False if you don't want BN shown in (a)
)

# (b) HRU-scale: 4 hidden layers (same fixed widths/gaps)
ax_b = fig.add_subplot(gs[1, 0])
draw_panel(
    ax=ax_b,
    title="(b) HRU-scale MLP architecture",
    left_label="Standardized\nparameters (2559-dim)",
    right_label="\n\nMonthly average runoff\n(180-dim)",
    n_hidden=4,
    input_n=2559,                  # you can change to 2559 if you prefer the true input size
    hidden_ns=[2048, 1024, 512, 256],
    output_n=180,
    dropouts=[0.3, 0.3, 0.3, 0.2],
    use_bn=True
)

# ---------- Save (fixed overall width preserved) ----------
plt.savefig("mlp_architecture.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)
