"""
Overall Model Architecture — ultra-clean, clear arrows, labels in empty space.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

ECG   = '#2563EB'
PPG   = '#A1467E'
PRED  = '#DC2626'
Z_C   = '#EAB308'
LOSS  = '#374151'
DOWN  = '#16A34A'
EMA_C = '#8B5CF6'
TEXT  = '#1E293B'
GRAY  = '#94A3B8'
CARD  = '#F1F5F9'

fig, ax = plt.subplots(figsize=(22, 11))
ax.set_xlim(0, 22)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

def B(x, y, w, h, c, t, s=14):
    r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.05,rounding_size=0.1",
                        facecolor=c, edgecolor='none')
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=s, color='white', fontweight='bold')

def M(x, y, w, h, c, t, s=12):
    r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.03,rounding_size=0.06",
                        facecolor=c, edgecolor='#CBD5E1', linewidth=0.8)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=s, color=TEXT, fontweight='bold')

def T(x, y, t, s=12, c=None, b=False, ha='center'):
    ax.text(x, y, t, ha=ha, va='center', fontsize=s, color=c or TEXT, fontweight='bold' if b else 'normal')

def A(x1, y1, x2, y2, c='#64748B', lw=2.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.2', color=c, lw=lw))

# ═══ TITLE (top center, empty area) ═══
T(11, 10.7, 'JEPA ECG-PPG 模型整体架构', 22, TEXT, True)

# ═══ INPUTS ═══
B(1.5, 9.3, 2.5, 0.65, ECG, 'ECG 信号', 15)
B(18.0, 9.3, 2.5, 0.65, PPG, 'PPG 信号', 15)

# ═══ ENCODERS ═══
ew, eh = 5.0, 2.6

# Left
lx, ly = 0.5, 6.2
r = FancyBboxPatch((lx, ly), ew, eh, boxstyle="round,pad=0.06,rounding_size=0.12",
                    facecolor=CARD, edgecolor=ECG, linewidth=2.5)
ax.add_patch(r)
M(lx+0.3, ly+1.9, ew-0.6, 0.5, '#DBEAFE', 'CNN Stem', 12)
M(lx+0.3, ly+1.25, ew-0.6, 0.5, '#E0E7FF', 'Position Encoding', 12)
M(lx+0.3, ly+0.3, ew-0.6, 0.8, '#E0E7FF', 'Transformer ×8', 12)
M(lx+0.3, ly-0.1, ew-0.6, 0.3, ECG, 'Pooling', 12)
T(lx+ew/2, ly-0.5, 'Context Encoder', 13, ECG, True)
T(lx+ew/2, ly-0.75, '梯度更新', 10, GRAY)

A(2.75, 9.3, lx+ew/2, ly+eh+0.25)

# Right
rx, ry = 16.5, 6.2
r = FancyBboxPatch((rx, ry), ew, eh, boxstyle="round,pad=0.06,rounding_size=0.12",
                    facecolor='#FDF5FA', edgecolor=PPG, linewidth=2.5)
ax.add_patch(r)
M(rx+0.3, ry+1.9, ew-0.6, 0.5, '#F3E8F0', 'CNN Stem', 12)
M(rx+0.3, ry+1.25, ew-0.6, 0.5, '#F3E8F0', 'Position Encoding', 12)
M(rx+0.3, ry+0.3, ew-0.6, 0.8, '#F3E8F0', 'Transformer ×8', 12)
M(rx+0.3, ry-0.1, ew-0.6, 0.3, PPG, 'Pooling', 12)
T(rx+ew/2, ry-0.5, 'Target Encoder', 13, PPG, True)
T(rx+ew/2, ry-0.75, 'EMA 更新 · 无梯度', 10, GRAY)

A(19.25, 9.3, rx+ew/2, ry+eh+0.25)

# ═══ SAME STRUCTURE (empty space between encoders) ═══
T(11, 7.3, '结构完全相同', 12, GRAY)

# ═══ EMA (empty space above encoders) ═══
ax.annotate('', xy=(rx-0.3, ry+eh/2), xytext=(lx+ew+0.3, ly+eh/2),
            arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.2',
                           color=EMA_C, lw=2, linestyle=(0, (5, 4)),
                           connectionstyle="arc3,rad=-0.2"))
T(11, 5.5, 'EMA 参数同步', 13, EMA_C, True)
T(11, 5.2, 'θ_target = m·θ_target + (1-m)·θ_context', 11, GRAY)

# ═══ EMBEDDINGS ═══
ce_x = lx+ew+0.8
ce_y = 4.5
B(ce_x, ce_y, 2.2, 0.55, '#3B82F6', '上下文嵌入', 13)

te_x = rx-3.0
te_y = 4.3
B(te_x, te_y, 2.2, 0.55, PPG, '目标嵌入', 13)
T(te_x+1.1, te_y-0.28, 'stop_gradient', 10, PRED, True)

A(lx+ew, ly+eh/2, ce_x, ce_y+0.27, ECG, 2.2)
A(rx, ry+eh/2, te_x+2.2, te_y+0.27, PPG, 2.2)

# ═══ PREDICTOR ═══
px = 7.0
py = 2.9
pw = 8.0
r = FancyBboxPatch((px, py), pw, 1.1, boxstyle="round,pad=0.06,rounding_size=0.1",
                    facecolor='#FEF2F2', edgecolor=PRED, linewidth=2.5)
ax.add_patch(r)
T(px+pw/2, py+0.7, 'Predictor', 18, PRED, True)
T(px+pw/2, py+0.25, '上下文嵌入 + 隐变量 z → MLP → 预测嵌入', 13, GRAY)
A(ce_x+2.2, ce_y+0.27, px, py+0.9, PRED, 2.2)

# ═══ LATENT z (empty area above predictor) ═══
zx = px+3.0
zy = 4.5
B(zx, zy, 2.0, 0.5, Z_C, '隐变量 z', 14)
T(zx+1.0, zy+0.35, 'z ~ N(0, I)  采样 4 次', 10, GRAY)
A(zx+1.0, zy, px+4.0, py+1.1, Z_C, 2)

# ═══ LOSS ═══
ly2 = 1.3
B(5.0, ly2, 12.0, 0.7, LOSS, '损失  L = min MSE (预测嵌入, 目标嵌入)', 15)
A(px+pw/2, py, 7.5, ly2+0.7, LOSS, 2.2)
A(te_x+1.1, te_y, 14.0, ly2+0.7, LOSS, 2.2)

# ═══ GRADIENT (empty space, left side) ═══
ax.annotate('', xy=(ce_x+0.3, 4.0), xytext=(7.0, ly2+0.7),
            arrowprops=dict(arrowstyle='->,head_width=0.25,head_length=0.2',
                           color=ECG, lw=1.5, linestyle=(0, (5, 4)),
                           connectionstyle="arc3,rad=0.3"))
T(5.5, 3.0, '反向传播 梯度更新', 11, ECG, True)

# ═══ DOWNSTREAM ═══
r = FancyBboxPatch((2.0, -0.2), 18.0, 1.2, boxstyle="round,pad=0.06,rounding_size=0.1",
                    facecolor='#F0FDF4', edgecolor=DOWN, linewidth=2.5)
ax.add_patch(r)
T(11, 0.85, '下游任务: CHD 分类', 16, DOWN, True)
T(11, 0.45, '预训练编码器 → 特征融合 → 分类头 → 二分类输出', 13, GRAY)

A(11, ly2, 11, 1.2, DOWN, 2.2)

plt.tight_layout(pad=0.3)
plt.savefig(r"F:\skills\jepa-ecg-ppg\jepa_overall_architecture.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Done: jepa_overall_architecture.png")
