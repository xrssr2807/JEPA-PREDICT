"""
JEPA Prediction Architecture — ultra-clean, clear arrows.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

C = {
    'ecg':    '#2563EB',
    'ppg':    '#A1467E',
    'pred':   '#DC2626',
    'z':      '#EAB308',
    'loss':   '#374151',
    'ema':    '#8B5CF6',
    'bg':     '#FFFFFF',
    'card':   '#F1F5F9',
    'text':   '#1E293B',
    'gray':   '#94A3B8',
}

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

def B(x, y, w, h, c, t, s=14):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                        facecolor=c, edgecolor='none')
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=s, color='white', fontweight='bold')

def M(x, y, w, h, c, t, s=12):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                        facecolor=c, edgecolor='#CBD5E1', linewidth=0.8)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=s, color=C['text'], fontweight='bold')

def T(x, y, t, s=11, c=None, b=False):
    ax.text(x, y, t, ha='center', va='center', fontsize=s, color=c or C['text'], fontweight='bold' if b else 'normal')

def A(x1, y1, x2, y2, c='#64748B', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.2', color=c, lw=lw))

# ═══ INPUT ═══
B(0.8, 7.5, 2.2, 0.7, C['ecg'], 'ECG 信号', 15)
B(14.8, 7.5, 2.2, 0.7, C['ppg'], 'PPG 信号', 15)

# ═══ ENCODERS ═══
ew, eh = 3.6, 2.8

# Context Encoder
lx, ly = 0.2, 4.2
r = FancyBboxPatch((lx, ly), ew, eh, boxstyle="round,pad=0.06,rounding_size=0.12",
                    facecolor=C['card'], edgecolor=C['ecg'], linewidth=2.5)
ax.add_patch(r)
M(lx+0.2, ly+2.1, ew-0.4, 0.5, '#DBEAFE', 'CNN Stem', 12)
M(lx+0.2, ly+1.45, ew-0.4, 0.5, '#E0E7FF', 'Position Encoding', 12)
M(lx+0.2, ly+0.55, ew-0.4, 0.75, '#E0E7FF', 'Transformer ×8', 12)
M(lx+0.2, ly+0.1, ew-0.4, 0.35, C['ecg'], 'Pooling', 12)
T(lx+ew/2, ly+eh+0.2, 'Context Encoder', 13, C['ecg'], True)
T(lx+ew/2, ly+eh, '梯度更新', 10, C['gray'])

A(1.9, 7.5, lx+ew/2, ly+eh+0.25)

# Target Encoder
rx, ry = 14.2, 4.2
r = FancyBboxPatch((rx, ry), ew, eh, boxstyle="round,pad=0.06,rounding_size=0.12",
                    facecolor='#FDF5FA', edgecolor=C['ppg'], linewidth=2.5)
ax.add_patch(r)
M(rx+0.2, ry+2.1, ew-0.4, 0.5, '#F3E8F0', 'CNN Stem', 12)
M(rx+0.2, ry+1.45, ew-0.4, 0.5, '#F3E8F0', 'Position Encoding', 12)
M(rx+0.2, ry+0.55, ew-0.4, 0.75, '#F3E8F0', 'Transformer ×8', 12)
M(rx+0.2, ry+0.1, ew-0.4, 0.35, C['ppg'], 'Pooling', 12)
T(rx+ew/2, ry+eh+0.2, 'Target Encoder', 13, C['ppg'], True)
T(rx+ew/2, ry+eh, 'EMA 更新 · 无梯度', 10, C['gray'])

A(15.9, 7.5, rx+ew/2, ry+eh+0.25)

# ═══ EMBEDDINGS ═══
ce_x = lx+ew+0.5
ce_y = 5.5
B(ce_x, ce_y, 2.2, 0.55, '#3B82F6', '上下文嵌入', 13)

te_x = rx-2.7
te_y = 5.3
B(te_x, te_y, 2.2, 0.55, C['ppg'], '目标嵌入', 13)

A(lx+ew, ly+eh/2+0.3, ce_x, ce_y+0.27, C['ecg'], 2.2)
A(rx, ry+eh/2+0.3, te_x+2.2, te_y+0.27, C['ppg'], 2.2)
T(te_x+1.1, te_y-0.25, 'stop_gradient', 10, C['pred'], True)

# ═══ PREDICTOR ═══
px = 6.5
py = 3.5
pw = 5.0
r = FancyBboxPatch((px, py), pw, 1.3, boxstyle="round,pad=0.06,rounding_size=0.1",
                    facecolor='#FEF2F2', edgecolor=C['pred'], linewidth=2.5)
ax.add_patch(r)
T(px+pw/2, py+1.0, 'Predictor', 16, C['pred'], True)
M(px+0.3, py+0.1, pw-0.6, 0.7, '#FEE2E2', '上下文嵌入 + 隐变量 z → MLP → 预测嵌入', 13)
A(ce_x+2.2, ce_y+0.27, px, py+1.0, C['pred'], 2.2)

# ═══ LATENT z ═══
zx = px+1.5
zy = 5.5
B(zx, zy, 2.0, 0.55, C['z'], '隐变量 z', 14)
T(zx+1.0, zy+0.45, 'z ~ N(0, I)  采样 4 次', 10, C['gray'])
A(zx+1.0, zy, px+2.5, py+1.3, C['z'], 2)

# ═══ LOSS ═══
ly2 = 1.8
B(5.5, ly2, 7.0, 0.7, C['loss'], '损失  L = min MSE (预测嵌入, 目标嵌入)', 14)
A(px+pw/2, py, 7.0, ly2+0.7, C['loss'], 2.2)
A(te_x+1.1, te_y, 11.0, ly2+0.7, C['loss'], 2.2)

# ═══ EMA ═══
ax.annotate('', xy=(rx-0.3, ry+eh/2), xytext=(lx+ew+0.3, ly+eh/2),
            arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.2',
                           color=C['ema'], lw=2, linestyle=(0, (5, 4)),
                           connectionstyle="arc3,rad=-0.3"))
T(9, 3.3, 'EMA 参数同步', 12, C['ema'], True)
T(9, 3.0, 'θ_target = m·θ_target + (1-m)·θ_context', 11, C['gray'])

# ═══ STRUCTURE NOTE ═══
T(9, 6.5, '两个编码器结构完全相同', 12, C['gray'], False)

plt.tight_layout(pad=0.3)
plt.savefig(r"F:\skills\jepa-ecg-ppg\jepa_architecture.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Done: jepa_architecture.png")
