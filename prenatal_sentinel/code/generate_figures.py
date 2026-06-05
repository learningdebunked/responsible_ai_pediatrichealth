"""
PrenatalSentinel — Figure Generation (Figures 2–5)
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

with open('/home/sandbox/results.json') as f:
    R = json.load(f)

COLORS = {
    'LR':  '#2196F3',
    'RF':  '#4CAF50',
    'GB':  '#FF9800',
    'FL1': '#9C27B0',
    'FL5': '#E91E63',
    'FLi': '#607D8B',
    'main':'#1565C0',
    'eq':  ['#D32F2F','#F57C00','#388E3C','#1976D2','#7B1FA2'],
    'ab':  ['#B0BEC5','#78909C','#546E7A','#37474F','#263238'],
}

# ─────────────────────────────────────────────────────────────────────────
# Figure 2 — Privacy-Utility Tradeoff (AUC vs ε)
# ─────────────────────────────────────────────────────────────────────────
pu = R['privacy_utility_tradeoff']
eps_vals, auc_vals = [], []
for k, v in pu.items():
    eps_vals.append(float(k) if k != 'inf' else 60)
    auc_vals.append(v['AUC'])

# Sort by epsilon
order = np.argsort(eps_vals)
eps_sorted = [eps_vals[i] for i in order]
auc_sorted = [auc_vals[i] for i in order]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(eps_sorted, auc_sorted, 'o-', color=COLORS['main'],
        linewidth=2.5, markersize=8, markerfacecolor='white',
        markeredgewidth=2.5, zorder=5)

# Shade privacy region
ax.axvspan(0, 5, alpha=0.08, color='#D32F2F', label='Strong privacy (ε≤5)')
ax.axvspan(5, 20, alpha=0.06, color='#FF9800', label='Moderate privacy (5<ε≤20)')
ax.axhline(y=auc_sorted[-1], color='#607D8B', linestyle='--',
           linewidth=1.5, alpha=0.7, label=f'No-DP ceiling (AUC={auc_sorted[-1]:.3f})')

ax.set_xlabel('Privacy Budget (ε)', fontsize=13)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_title('Fig. 2: Privacy-Utility Tradeoff in PrenatalSentinel\n'
             '(Federated Learning with Gaussian DP, δ=10⁻⁵, K=5 clients)',
             fontsize=11, pad=10)
ax.set_xscale('log')
ax.set_xlim(0.3, 80)
ax.set_ylim(min(auc_sorted)*0.97, max(auc_sorted)*1.02)

# Label points
for ex, ay in zip(eps_sorted, auc_sorted):
    lbl = f'ε={ex:.0f}' if ex < 60 else 'ε=∞'
    ax.annotate(f'{ay:.3f}', (ex, ay), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9, color=COLORS['main'])

ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('/home/sandbox/fig2_privacy_utility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")

# ─────────────────────────────────────────────────────────────────────────
# Figure 3 — Equity Audit (AUC by SES Quintile)
# ─────────────────────────────────────────────────────────────────────────
eq = R['equity_audit']
quintiles = [f'Q{i}' for i in range(1,6) if f'Q{i}' in eq]
q_aucs = [eq[q]['AUC'] for q in quintiles]
q_fds  = [eq[q]['food_desert_mean'] for q in quintiles]
q_labels = ['Q1\n(Lowest SES)', 'Q2', 'Q3\n(Median)', 'Q4', 'Q5\n(Highest SES)'][:len(quintiles)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: AUC by SES quintile
bars = ax1.bar(q_labels, q_aucs, color=COLORS['eq'][:len(quintiles)],
               width=0.6, edgecolor='white', linewidth=1.5, zorder=3)
ax1.set_ylim(0.45, 0.70)
ax1.axhline(y=np.mean(q_aucs), color='black', linestyle='--',
            linewidth=1.5, alpha=0.6, label=f'Mean AUC={np.mean(q_aucs):.3f}')
for bar, val in zip(bars, q_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('SES Quintile', fontsize=12)
ax1.set_ylabel('AUC-ROC', fontsize=12)
ax1.set_title('(a) ASD Risk AUC by SES Quintile', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=10)

# Right: Food Desert Score by SES quintile
ax2.bar(q_labels, q_fds, color=COLORS['eq'][:len(quintiles)],
        width=0.6, edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85)
for i, (fd_val, auc_val) in enumerate(zip(q_fds, q_aucs)):
    ax2.text(i, fd_val + 0.005, f'FD={fd_val:.3f}\nAUC={auc_val:.3f}',
             ha='center', va='bottom', fontsize=9)
ax2.set_xlabel('SES Quintile', fontsize=12)
ax2.set_ylabel('Mean Food Desert Score', fontsize=12)
ax2.set_title('(b) Food Desert Score by SES Quintile', fontsize=11)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=10)

fig.suptitle('Fig. 3: Equity Audit — PrenatalSentinel Performance Across Socioeconomic Strata',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('/home/sandbox/fig3_equity_audit.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")

# ─────────────────────────────────────────────────────────────────────────
# Figure 4 — Signal Ablation Bar Chart
# ─────────────────────────────────────────────────────────────────────────
ab = R['ablation_study']
ab_keys = list(ab.keys())
ab_aucs = [ab[k]['AUC'] for k in ab_keys]
ab_labels = [k.replace('\n', '\n') for k in ab_keys]
ab_nfeat  = [ab[k]['n_features'] for k in ab_keys]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ab_keys))
bars = ax.bar(x, ab_aucs, color=COLORS['ab'], width=0.6,
              edgecolor='white', linewidth=1.5, zorder=3)

# Baseline LR AUC
lr_auc = R['baseline_models']['Logistic Regression']['AUC']
ax.axhline(y=lr_auc, color='#2196F3', linestyle='--', linewidth=1.8,
           alpha=0.8, label=f'LR Baseline (AUC={lr_auc:.3f})')

for bar, val, nf in zip(bars, ab_aucs, ab_nfeat):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.3f}\n({nf}f)', ha='center', va='bottom', fontsize=9.5)

ax.set_xticks(x)
ax.set_xticklabels(ab_labels, fontsize=10)
ax.set_ylim(0.55, 0.72)
ax.set_ylabel('AUC-ROC (5-fold CV)', fontsize=12)
ax.set_xlabel('Signal Configuration', fontsize=12)
ax.set_title('Fig. 4: Signal Ablation Study\n'
             'F=Food-as-Medicine, M=OTC/Supplements, E=SES/Food Desert, B=Basket Features',
             fontsize=11, pad=8)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('/home/sandbox/fig4_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved.")

# ─────────────────────────────────────────────────────────────────────────
# Figure 5 — ROC Curves
# ─────────────────────────────────────────────────────────────────────────
roc = R['roc_curves']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Baseline models
baseline_keys = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
bl_colors = [COLORS['LR'], COLORS['RF'], COLORS['GB']]
bl_labels = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']

for key, color, lbl in zip(baseline_keys, bl_colors, bl_labels):
    if key in roc:
        fpr = roc[key]['fpr']
        tpr = roc[key]['tpr']
        auc = roc[key]['auc']
        ax1.plot(fpr, tpr, color=color, linewidth=2.5,
                 label=f'{lbl} (AUC={auc:.3f})')

ax1.plot([0,1],[0,1],'k--',linewidth=1.2,alpha=0.5,label='Random (AUC=0.500)')
ax1.set_xlim([-0.01,1.01]); ax1.set_ylim([-0.01,1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('(a) Baseline Models — ROC Curves', fontsize=11)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=10)

# Right: FL with DP
fl_keys = ['FL (ε=1)', 'FL (ε=5)', 'FL (ε=10)', 'FL (No DP)']
fl_colors = [COLORS['FL1'], COLORS['FL5'], '#00BCD4', COLORS['FLi']]
fl_labels_disp = ['FL ε=1 (Strict DP)', 'FL ε=5', 'FL ε=10', 'FL No-DP (Ceiling)']

for key, color, lbl in zip(fl_keys, fl_colors, fl_labels_disp):
    if key in roc:
        fpr = roc[key]['fpr']
        tpr = roc[key]['tpr']
        auc = roc[key]['auc']
        ls = '-' if 'No DP' not in key else '--'
        ax2.plot(fpr, tpr, color=color, linewidth=2.5, linestyle=ls,
                 label=f'{lbl} (AUC={auc:.3f})')

ax2.plot([0,1],[0,1],'k--',linewidth=1.2,alpha=0.5,label='Random (AUC=0.500)')
ax2.set_xlim([-0.01,1.01]); ax2.set_ylim([-0.01,1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('(b) Federated Learning with Differential Privacy', fontsize=11)
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=10)

fig.suptitle('Fig. 5: ROC Curves — PrenatalSentinel Baseline and Federated Models',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('/home/sandbox/fig5_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved.")

print("\n✅ All figures generated.")
