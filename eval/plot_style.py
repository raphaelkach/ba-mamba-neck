"""Einheitlicher Plot-Style für alle Thesis-Figures.

Prinzipien:
- Tufte: Maximale Data-Ink-Ratio, kein Chartjunk
- Nature/Science: Despined, minimales Grid, Direct Labeling
- Wong 2011 (Nature Methods): Colorblind-safe Palette
- LaTeX-kompatibel: Schriftgröße passt zu \\caption{}
"""

import matplotlib.pyplot as plt


PALETTE = {
    'blue':      '#0072B2',
    'orange':    '#E69F00',
    'green':     '#009E73',
    'red':       '#D55E00',
    'skyblue':   '#56B4E9',
    'yellow':    '#F0E442',
    'purple':    '#CC79A7',
    'black':     '#000000',
}

TEXTWIDTH = 5.9
HALFWIDTH = 2.85


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'mathtext.default': 'regular',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'axes.titlepad': 8,
        'axes.labelpad': 4,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.pad': 3,
        'ytick.major.pad': 3,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.4,
        'legend.labelspacing': 0.3,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.transparent': False,
    })


def fig_single():
    return plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.6))

def fig_half():
    return plt.subplots(figsize=(HALFWIDTH, HALFWIDTH * 0.75))

def fig_wide(nrows=1, ncols=2):
    return plt.subplots(nrows, ncols,
                        figsize=(TEXTWIDTH, TEXTWIDTH * 0.45 * nrows))
