"""Figure 5 replacement: PI-DeepONet (Variant B) schematic with the corrected
latent dimension (R^64) and the channel-wise LPA head (K=3, N_p=16 per
channel). Editable source for the schematic previously drawn by hand."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig
from paths import RESULTS
import matplotlib.pyplot as plt

OUT = f"{RESULTS}/manuscript_revisions/figures"
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply_style()
fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')

def box(x, y, w, h, text, fc='#eef3fa', ec='#2E5B8F', fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                 mutation_scale=14, lw=1.2, color='#333333'))

box(2, 30, 10, 8, 'Branch\ninput\n$Re$')
box(16, 30, 16, 8, '3 hidden layers\n(tanh, width $w$)')
box(36, 30, 14, 8, 'Linear\n$b \\in \\mathbb{R}^{64}$')
box(2, 6, 10, 8, 'Trunk\ninput\n$(x,y)$')
box(16, 6, 16, 8, '3 hidden layers\n(tanh, width $w$)')
box(36, 6, 14, 8, 'Linear\n$t \\in \\mathbb{R}^{64}$')
box(55, 18, 12, 8, 'Hadamard\n$b \\odot t \\in \\mathbb{R}^{64}$')
box(71, 18, 10, 8, 'Dense(16)\n(linear)')
box(84.5, 18, 14, 9, 'channel-wise LPA\n$K{=}3,\\ N_p{=}16$\nper channel\n+ Dense(3)',
    fc='#fdeaea', ec='#b23434', fs=10.5)
ax.text(91.5, 14.0, '$(u, v, p)$', ha='center', fontsize=12)

arrow(12, 34, 16, 34); arrow(32, 34, 36, 34)
arrow(12, 10, 16, 10); arrow(32, 10, 36, 10)
arrow(50, 34, 56.5, 26.5); arrow(50, 10, 56.5, 17.5)
arrow(67, 22, 71, 22); arrow(81, 22, 84.5, 22)
ax.text(50, 41, 'Physics-informed DeepONet (Variant B) with LPA head',
        ha='center', fontsize=13)
savefig(fig, f'{OUT}/fig5_replacement.png')
print('saved fig5')
