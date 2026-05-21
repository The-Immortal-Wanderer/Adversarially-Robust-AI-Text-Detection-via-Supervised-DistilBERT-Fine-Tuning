"""
Research Paper Figure Generation Script
Figures 4-9: Python-based charts using Plotly and Seaborn
All data values extracted verbatim from result documents.

Run from: c:/Users/madha/source/repos/ANN_Project/
Output:   figures/figure_N_*.png  (300 DPI)
          figures/figure_N_*.html (interactive Plotly)

Requirements:
    pip install plotly kaleido seaborn matplotlib pandas numpy
"""

import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# SHARED FIGURE DESIGN SYSTEM — apply to ALL figures
# ============================================================
PALETTE   = ['#2E4057', '#048A81', '#54C6EB', '#EF7B45', '#D64045', '#8B5CF6']
# navy       teal        sky         orange      crimson     purple
FONT      = 'Times New Roman'
BG        = '#FAFAFA'
GRID      = '#E5E5E5'
TITLE_SZ  = 16
LABEL_SZ  = 13
TICK_SZ   = 11
LEGEND_SZ = 11
DPI       = 300
FIG_W     = 900
FIG_H     = 560

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotly global template
PLOTLY_LAYOUT = dict(
    font=dict(family=FONT, size=LABEL_SZ, color='#1A1A2E'),
    plot_bgcolor=BG,
    paper_bgcolor='white',
    title_font=dict(family=FONT, size=TITLE_SZ, color='#1A1A2E'),
    title_x=0.5,
    title_xanchor='center',
    bargap=0.35,
    xaxis=dict(
        gridcolor=GRID,
        linecolor='#CCCCCC',
        tickfont=dict(family=FONT, size=TICK_SZ),
        title_font=dict(family=FONT, size=LABEL_SZ)
    ),
    yaxis=dict(
        gridcolor=GRID,
        linecolor='#CCCCCC',
        tickfont=dict(family=FONT, size=TICK_SZ),
        title_font=dict(family=FONT, size=LABEL_SZ)
    ),
    margin=dict(l=70, r=40, t=80, b=70)
)

# Separate base legend style — merge per-figure rather than unpacking with PLOTLY_LAYOUT
BASE_LEGEND = dict(
    font=dict(family=FONT, size=LEGEND_SZ),
    bgcolor='rgba(255,255,255,0.9)',
    bordercolor=GRID,
    borderwidth=1
)


def save_fig(fig, name, plotly=True):
    html_path = os.path.join(OUTPUT_DIR, f"{name}.html")
    png_path  = os.path.join(OUTPUT_DIR, f"{name}.png")
    if plotly:
        fig.write_html(html_path)
        try:
            fig.write_image(png_path, width=FIG_W, height=FIG_H, scale=3)
            print(f"  Saved: {png_path}")
        except Exception as e:
            print(f"  PNG export failed (install kaleido): {e}")
            print(f"  HTML saved: {html_path}")
    else:
        fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {png_path}")
        plt.close(fig)


# ============================================================
# FIGURE 3 — Best-Epoch Validation F1 by Ablation Configuration
# Source: artifacts/distilbert_detector/summary.csv (verbatim)
# Note: Per-epoch series was not persisted to disk during training.
#       This figure uses the verified best-epoch value for each config.
# ============================================================
print("Generating Figure 3: Best-epoch Validation F1...")

# Verbatim from artifacts/distilbert_detector/summary.csv
best_val_f1 = {
    'baseline1':  0.9442224442224442,
    'ablation_a': 0.9421151184564032,
    'ablation_b': 0.9487728545544392,   # highest — validation-selected config
    'ablation_c': 0.9484021572888996,
}

cfg_labels = ['baseline1', 'ablation_a', 'ablation_b\n(val-selected)', 'ablation_c']
vals       = list(best_val_f1.values())
bar_colors3 = [PALETTE[0], PALETTE[0], PALETTE[1], PALETTE[0]]

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=cfg_labels,
    y=vals,
    marker_color=bar_colors3,
    marker_line_color='white',
    marker_line_width=1.2,
    text=[f'{v:.4f}' for v in vals],
    textposition='outside',
    textfont=dict(family=FONT, size=LABEL_SZ, color='#1A1A2E'),
    showlegend=False
))

# Best-config annotation arrow
fig3.add_annotation(
    x='ablation_b\n(val-selected)', y=0.9487728545544392 + 0.006,
    text='<b>Highest validation F1</b><br>selected by checkpoint criterion',
    showarrow=True, arrowhead=2, arrowcolor=PALETTE[0],
    ax=60, ay=-40,
    font=dict(family=FONT, size=TICK_SZ, color=PALETTE[0]),
    bgcolor='rgba(255,255,255,0.85)', bordercolor=PALETTE[0], borderwidth=1
)

# Horizontal mean line
mean_val = sum(vals) / len(vals)
fig3.add_hline(
    y=mean_val, line_dash='dot', line_color='#AAAAAA', line_width=1.5,
    annotation_text=f'Mean = {mean_val:.4f}',
    annotation_font=dict(family=FONT, size=TICK_SZ, color='#AAAAAA'),
    annotation_position='bottom right'
)

fig3.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 3: Best-Epoch Validation F1 by Ablation Configuration</b><br>'
             '<span style="font-size:10px; color:#666666;">Per-epoch series not logged; values are the best checkpoint F1 over 3 epochs '
             '(source: artifacts/distilbert_detector/summary.csv)</span>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Ablation Configuration',
    yaxis_title='Validation F1 (best epoch)',
    yaxis_range=[0.930, 0.960],
    width=FIG_W, height=FIG_H,
)
save_fig(fig3, 'figure3_best_val_f1')


# ============================================================
# FIGURE 4 — Ablation Comparison: Seen vs Unseen F1
# ============================================================
print("Generating Figure 4: Ablation Seen vs Unseen F1...")

configs     = ['baseline1', 'ablation_a', 'ablation_b\n(val-selected)', 'ablation_c']
configs_leg = ['baseline1', 'ablation_a', 'ablation_b (val-selected)', 'ablation_c']
seen_f1     = [0.9238, 0.9409, 0.9481, 0.9472]
unseen_f1   = [0.9135, 0.9267, 0.9184, 0.9263]

x = list(range(len(configs)))
width = 0.35

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    name='Seen Test F1',
    x=configs_leg, y=seen_f1,
    marker_color=PALETTE[1],
    text=[f'{v:.4f}' for v in seen_f1],
    textposition='outside',
    textfont=dict(family=FONT, size=TICK_SZ),
    offsetgroup=0
))
fig4.add_trace(go.Bar(
    name='Unseen Test F1',
    x=configs_leg, y=unseen_f1,
    marker_color=PALETTE[0],
    text=[f'{v:.4f}' for v in unseen_f1],
    textposition='outside',
    textfont=dict(family=FONT, size=TICK_SZ),
    offsetgroup=1
))

fig4.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 4: Seen-Generator vs. Held-Out-Generator F1 by Ablation Configuration</b>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Ablation Configuration',
    yaxis_title='F1 Score',
    yaxis_range=[0.88, 0.97],
    barmode='group',
    bargroupgap=0.1,
    width=FIG_W, height=FIG_H,
    legend=dict(**BASE_LEGEND, orientation='h', x=0.5, xanchor='center', y=-0.18)
)
save_fig(fig4, 'figure4_ablation_f1_comparison')


# ============================================================
# FIGURE 5 — Relative Robustness Degradation by Configuration
# ============================================================
print("Generating Figure 5: RRD by Configuration...")

rrd_vals  = [1.11, 1.51, 3.13, 2.21]
bar_colors = [PALETTE[1] if v < 2.0 else PALETTE[3] if v < 3.0 else PALETTE[4] for v in rrd_vals]

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=configs_leg,
    y=rrd_vals,
    marker_color=bar_colors,
    text=[f'{v:.2f}%' for v in rrd_vals],
    textposition='outside',
    textfont=dict(family=FONT, size=LABEL_SZ, color='#1A1A2E'),
    showlegend=False
))
# 5% threshold line
fig5.add_hline(
    y=5.0, line_dash='dash', line_color=PALETTE[4], line_width=2,
    annotation_text='5% generalisation threshold',
    annotation_font=dict(family=FONT, size=TICK_SZ, color=PALETTE[4]),
    annotation_position='top right'
)

legend_items = [
    dict(name='RRD < 2% (strong generalisation)', marker=dict(color=PALETTE[1])),
    dict(name='RRD 2-3% (moderate)',              marker=dict(color=PALETTE[3])),
    dict(name='RRD > 3% (weaker)',                marker=dict(color=PALETTE[4])),
]
for item in legend_items:
    fig5.add_trace(go.Bar(x=[None], y=[None], name=item['name'],
                          marker_color=item['marker']['color']))

fig5.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 5: Relative Robustness Degradation (RRD) by Ablation Configuration</b>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Ablation Configuration',
    yaxis_title='RRD (%)',
    yaxis_range=[0, 6.0],
    width=FIG_W, height=FIG_H,
    legend=dict(**BASE_LEGEND, orientation='h', x=0.5, xanchor='center', y=-0.18)
)
save_fig(fig5, 'figure5_rrd_by_config')


# ============================================================
# FIGURE 6 — Efficiency Speedup Summary
# ============================================================
print("Generating Figure 6: Efficiency Speedup...")

# Removed TC4 latency reduction (%) because mixing x-factor and % on the same axis breaks scale.
# 54% latency reduction = 2.18x throughput, so they represent the same gain.
tc_labels  = ['Parallel\nPreprocessing', 'AMP + Large Batch\nTraining', 'AMP + Inference Mode\nThroughput']
speedups   = [3.58, 3.32, 2.19]
units      = ['x speedup', 'x speedup', 'x throughput']
colors_tc  = [PALETTE[3], PALETTE[0], PALETTE[2]]  # Orange (Preprocessing), Navy (Training), Sky Blue (Inference)

fig6 = go.Figure()
for i, (label, val, unit, color) in enumerate(zip(tc_labels, speedups, units, colors_tc)):
    fig6.add_trace(go.Bar(
        name=label.replace('\n', ' '),
        x=[label], y=[val],
        marker_color=color,
        text=[f'{val}x'],
        textposition='outside',
        textfont=dict(family=FONT, size=LABEL_SZ),
        showlegend=False
    ))

fig6.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 6: Systems Efficiency Optimisation Gains</b><br>'
             '<span style="font-size:10px; color:#666666;">Note: DataLoader pinning and workers increased GPU utilisation from ~30% to ~90%+ (not plotted).</span>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Optimisation Stage',
    yaxis_title='Speedup Factor (x)',
    yaxis_range=[0, 4.5],
    width=FIG_W, height=FIG_H
)
save_fig(fig6, 'figure6_efficiency_speedup')


# ============================================================
# FIGURE 7 — RAID AUROC: Attack Type x Scoring Model
# ============================================================
print("Generating Figure 7: RAID AUROC Comparison...")

scorers        = ['gpt-j-6B', 'gpt2-xl']
homo_auroc     = [0.6689, 0.5257]
para_auroc     = [0.7869, 0.7893]
homo_ci_lo     = [0.6370, 0.4912]
homo_ci_hi     = [0.7036, 0.5601]
para_ci_lo     = [0.7565, 0.7575]
para_ci_hi     = [0.8147, 0.8156]

fig7 = go.Figure()
# Homoglyph bars
fig7.add_trace(go.Bar(
    name='Homoglyph Attack',
    x=scorers, y=homo_auroc,
    marker_color=PALETTE[4],
    error_y=dict(
        type='data',
        symmetric=False,
        array=[hi - val for hi, val in zip(homo_ci_hi, homo_auroc)],
        arrayminus=[0, 0],
        color='#333333', thickness=1.5, width=6
    ),
    text=[f'{v:.4f}' for v in homo_auroc],
    textposition='inside',
    insidetextanchor='end',
    textfont=dict(family=FONT, size=LABEL_SZ, color='white'),
    offsetgroup=0
))
# Paraphrase bars
fig7.add_trace(go.Bar(
    name='Paraphrase Attack',
    x=scorers, y=para_auroc,
    marker_color=PALETTE[0],
    error_y=dict(
        type='data',
        symmetric=False,
        array=[hi - val for hi, val in zip(para_ci_hi, para_auroc)],
        arrayminus=[0, 0],
        color='#333333', thickness=1.5, width=6
    ),
    text=[f'{v:.4f}' for v in para_auroc],
    textposition='inside',
    insidetextanchor='end',
    textfont=dict(family=FONT, size=LABEL_SZ, color='white'),
    offsetgroup=1
))
# Reference line: clean benchmark
fig7.add_hline(
    y=0.9370, line_dash='dot', line_color=PALETTE[1], line_width=2,
    annotation_text='Clean benchmark (reproduced, 0.9370)',
    annotation_font=dict(family=FONT, size=TICK_SZ, color=PALETTE[1]),
    annotation_position='top left'
)
# Random chance
fig7.add_hline(
    y=0.5, line_dash='dash', line_color='#AAAAAA', line_width=1,
    annotation_text='Random (0.50)',
    annotation_font=dict(family=FONT, size=TICK_SZ, color='#AAAAAA'),
    annotation_position='bottom right'
)

fig7.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 7: Fast-DetectGPT AUROC on RAID Adversarial Subsets</b><br>'
             '<span style="font-size:10px; color:#666666;">Error bars = 95% bootstrap CI (1,000 iterations)</span>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Scoring Model',
    yaxis_title='ROC-AUC',
    yaxis_range=[0.40, 1.0],
    barmode='group',
    bargroupgap=0.1,
    width=FIG_W, height=FIG_H,
    legend=dict(**BASE_LEGEND, orientation='h', x=0.5, xanchor='center', y=-0.18)
)
save_fig(fig7, 'figure7_raid_auroc_comparison')


# ============================================================
# FIGURE 8 — Three-Way AUROC Comparison
# ============================================================
print("Generating Figure 8: Three-way AUROC...")

conditions = [
    'Original paper<br>(clean)',
    'Reproduced<br>(clean)',
    'Paraphrase<br>gpt-j-6B',
    'Paraphrase<br>gpt2-xl',
    'Homoglyph<br>gpt-j-6B',
    'Homoglyph<br>gpt2-xl'
]
aurocs = [0.9338, 0.9370, 0.7869, 0.7893, 0.6689, 0.5257]
# Colors unified with Figure 7: Clean=Teal, Paraphrase=Navy, Homoglyph=Red
bar_colors8 = [PALETTE[1], PALETTE[1], PALETTE[0], PALETTE[0], PALETTE[4], PALETTE[4]]
groups = ['Clean Baseline', 'Clean Baseline', 'Paraphrase Attack', 'Paraphrase Attack', 'Homoglyph Attack', 'Homoglyph Attack']

fig8 = go.Figure()
fig8.add_trace(go.Bar(
    x=conditions, y=aurocs,
    marker_color=bar_colors8,
    text=[f'{v:.4f}' for v in aurocs],
    textposition='outside',
    textfont=dict(family=FONT, size=LABEL_SZ),
    showlegend=False
))
# Region annotations
for x0, x1, label, color in [
    (-0.5, 1.5, 'Clean Benchmarks', PALETTE[1]), # Teal to match Clean bars
    (1.5, 3.5, 'Paraphrase Attack', PALETTE[0]), # Navy to match Paraphrase bars
    (3.5, 5.5, 'Homoglyph Attack', PALETTE[4])   # Crimson to match Homoglyph bars
]:
    fig8.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, layer='below', line_width=0)
    fig8.add_annotation(
        x=(x0 + x1) / 2, y=0.96, xref='x', yref='paper',
        text=f'<b>{label}</b>',
        showarrow=False,
        font=dict(family=FONT, size=TICK_SZ, color=color),
        align='center'
    )

fig8.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='<b>Figure 8: Three-Way AUROC Comparison — Original / Reproduced / Adversarial</b>',
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Evaluation Condition',
    yaxis_title='ROC-AUC',
    yaxis_range=[0.40, 1.02],
    width=FIG_W + 100, height=FIG_H,
)
save_fig(fig8, 'figure8_three_way_auroc')


# ============================================================
# FIGURE 9 — Quantization Efficiency: Time/Sample and VRAM
# ============================================================
print("Generating Figure 9: Quantization Efficiency...")

run_labels_html = [
    'Repro. Baseline<br>(fp16, Kaggle T4)',
    'Paraphrase<br>gpt-j-6B<br>(4-bit NF4)',
    'Homoglyph<br>gpt-j-6B<br>(4-bit NF4)',
    'Paraphrase<br>gpt2-xl<br>(4-bit NF4)',
    'Homoglyph<br>gpt2-xl<br>(4-bit NF4)'
]
time_per_sample = [2.690, 0.818, 2.327, 1.917, 2.158]
vram_gib        = [14.0,  3.5,   3.5,   1.8,   1.8]
colors9         = [PALETTE[1], PALETTE[0], PALETTE[4], PALETTE[0], PALETTE[4]]

fig9 = make_subplots(rows=1, cols=2, 
                     subplot_titles=('<b>Inference Time per Sample</b>', '<b>Estimated Peak VRAM</b>'),
                     horizontal_spacing=0.1)

# Left: Time
fig9.add_trace(go.Bar(
    x=run_labels_html, y=time_per_sample,
    marker_color=colors9,
    text=[f'{v:.3f}s' for v in time_per_sample],
    textposition='outside',
    textfont=dict(family=FONT, size=LABEL_SZ),
    showlegend=False
), row=1, col=1)

# Right: VRAM
fig9.add_trace(go.Bar(
    x=run_labels_html, y=vram_gib,
    marker_color=colors9,
    text=[f'{v:.1f} GiB<br>(est.)' for v in vram_gib],
    textposition='outside',
    textfont=dict(family=FONT, size=LABEL_SZ),
    showlegend=False
), row=1, col=2)

# Dummy traces for legend (must be added AFTER main traces to preserve axis inference)
fig9.add_trace(go.Bar(x=[None], y=[None], name='Repro. Baseline (fp16)', marker_color=PALETTE[1]), row=1, col=1)
fig9.add_trace(go.Bar(x=[None], y=[None], name='Paraphrase Attack (NF4)', marker_color=PALETTE[0]), row=1, col=1)
fig9.add_trace(go.Bar(x=[None], y=[None], name='Homoglyph Attack (NF4)', marker_color=PALETTE[4]), row=1, col=1)

# 8 GB RTX 3050 line
fig9.add_hline(
    y=8.0, line_dash='dash', line_color=PALETTE[3], line_width=1.5,
    annotation_text='RTX 3050 (8 GiB)',
    annotation_font=dict(family=FONT, size=TICK_SZ, color=PALETTE[3]),
    annotation_position='top right',
    row=1, col=2
)

# 3.28x speedup annotation
fig9.add_annotation(
    x=run_labels_html[1], y=1.05,
    ax=-65, ay=-180,
    xref='x1', yref='y1',
    text='<b>3.28x<br>speedup</b>',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1.5,
    arrowcolor=PALETTE[2],
    font=dict(family=FONT, size=TICK_SZ, color=PALETTE[2]),
    align='center',
)

fig9.update_layout(**PLOTLY_LAYOUT)
fig9.update_layout(
    title=dict(
        text='<b>Figure 9: Quantization Efficiency — Inference Time and VRAM by Run</b><br>'
             '<span style="font-size:10px; color:#666666;">(VRAM figures are estimates derived from model-size parameters)</span>',
        x=0.5,
        xanchor='center'
    ),
    yaxis_title='Inference Time per Sample (s)',
    yaxis_range=[0, 3.8],
    yaxis2=dict(title='Estimated Peak VRAM (GiB)', range=[0, 18], gridcolor=GRID, gridwidth=0.8, showline=True, linewidth=1, linecolor=GRID),
    width=FIG_W * 1.5, height=FIG_H,
    legend=dict(**BASE_LEGEND, orientation='h', x=0.5, xanchor='center', y=-0.25),
    showlegend=True,
    bargap=0.35
)

# Update subplot title fonts
for annotation in fig9['layout']['annotations']:
    if annotation['text'] in ['<b>Inference Time per Sample</b>', '<b>Estimated Peak VRAM</b>']:
        annotation['font'] = dict(family=FONT, size=TITLE_SZ-1, color='#1A1A2E')

save_fig(fig9, 'figure9_quantization_efficiency')

print("\nAll available figures generated successfully.")
print("Figure 3 (Training Dynamics) skipped — epoch-level log data required.")
print("See FIGURE 3 INCOMPLETE comment at top of script for details.")
