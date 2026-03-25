from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sequential_state_model import load_kintek_txt, fit_dataset, simulate

DATA = Path('/home/wei/.openclaw/workspace/Telomerase_kinetics/manuscript/data_dropbox/extracted/2019-6-24 PC 1 dATP/Rep 1/6-24-2019 dATP + control 1uM.txt')
OUTDIR = Path('/home/wei/.openclaw/workspace/Telomerase_kinetics/results/sequential_validation')
OUTDIR.mkdir(parents=True, exist_ok=True)

df = load_kintek_txt(DATA)
use_columns = [c for c in df.keys() if c != 'Time']
result = fit_dataset(df, use_columns=use_columns, shared_koff=False)

times = np.array(df['Time'], dtype=float)
data = np.column_stack([np.array(df[c], dtype=float) for c in use_columns])
kf = np.array(result.k_forward)
koff = np.array(result.k_off)
pred, rel = simulate(times, kf, koff)

(Path(OUTDIR/'fit_summary.json')).write_text(json.dumps(result.__dict__, indent=2))

# plot a subset of species for readability
species_to_plot = list(range(min(8, data.shape[1]))) + [9, 13, data.shape[1]-1]
seen = []
for i in species_to_plot:
    if 0 <= i < data.shape[1] and i not in seen:
        seen.append(i)
species_to_plot = seen

fig, axes = plt.subplots(len(species_to_plot), 1, figsize=(7, 2.2*len(species_to_plot)), sharex=True)
if len(species_to_plot) == 1:
    axes = [axes]
for ax, idx in zip(axes, species_to_plot):
    ax.plot(times, data[:, idx], 'o-', label=f'Observed S{idx+1}', color='black', markersize=4)
    ax.plot(times, pred[:, idx], '--', label=f'Predicted S{idx+1}', color='tab:red', linewidth=1.5)
    ax.set_ylabel(f'S{idx+1}')
    ax.legend(loc='best', fontsize=8)
axes[-1].set_xlabel('Time')
fig.suptitle('Sequential-state fit: observed vs predicted')
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(OUTDIR/'observed_vs_predicted_subset.png', dpi=200)
fig.savefig(OUTDIR/'observed_vs_predicted_subset.pdf')
plt.close(fig)

# heatmap-like overview
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(data, aspect='auto', origin='lower')
axs[0].set_title('Observed data')
axs[0].set_xlabel('Species index')
axs[0].set_ylabel('Time index')
fig.colorbar(im0, ax=axs[0], fraction=0.046)
im1 = axs[1].imshow(pred, aspect='auto', origin='lower')
axs[1].set_title('Predicted data')
axs[1].set_xlabel('Species index')
axs[1].set_ylabel('Time index')
fig.colorbar(im1, ax=axs[1], fraction=0.046)
fig.tight_layout()
fig.savefig(OUTDIR/'observed_vs_predicted_heatmap.png', dpi=200)
fig.savefig(OUTDIR/'observed_vs_predicted_heatmap.pdf')
plt.close(fig)

# simple residual summary
rss_per_species = ((pred - data)**2).sum(axis=0)
with open(OUTDIR/'validation_summary.txt','w') as f:
    f.write(f'Data file: {DATA}\n')
    f.write(f'SSE: {result.sse}\n')
    f.write('Top residual species (1-indexed):\n')
    order = np.argsort(rss_per_species)[::-1]
    for idx in order[:10]:
        f.write(f'S{idx+1}: RSS={rss_per_species[idx]:.6g}\n')
    f.write('\nInterpretation:\n')
    f.write('Compare subset plots and heatmaps to judge whether the sequential chain captures the main redistribution pattern.\n')

print('Saved validation outputs to', OUTDIR)
print('SSE', result.sse)
print('Top residual species:', [f'S{i+1}' for i in np.argsort(rss_per_species)[::-1][:5]])
