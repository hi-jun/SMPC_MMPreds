"""One-slide reading of constraint (1c) in overleaf/formulation_twosided_conditional.tex."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import norm

plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path", "axes.unicode_minus": False})
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
KR = fm.FontProperties(family="Noto Sans CJK JP", size=11)      # the .ttc exposes the CJK faces under JP; Hangul glyphs are included
KRB = fm.FontProperties(family="Noto Sans CJK JP", size=12, weight="bold")
KRS = fm.FontProperties(family="Noto Sans CJK JP", size=9)

INK, SAFE, RISK, OUT, MUTED = "#1C2430", "#3F7CC4", "#D9480F", "#C9CED6", "#5A6576"
beta = 0.6
b = -norm.ppf(0.5 * (1 + beta))          # safety boundary s + alpha*d_safe, in sigma units from s_bar
x = np.linspace(-3.4, 3.2, 800)
y = norm.pdf(x)

fig = plt.figure(figsize=(8.0, 4.7))
fig.text(0.04, 0.925, r"(1c)", fontsize=15, color=MUTED, va="center")
fig.text(0.125, 0.925, r"$\Pr\left[\,\hat{s}-s\ \geq\ \alpha\,d_{\mathrm{safe}}(v)\ \ \mid\ \ \hat{s}\leq\bar{s}\,\right]\ \geq\ \beta$",
         fontsize=19, color=INK, va="center")
fig.text(0.125, 0.845, "앞차가 예측보다 가까이 오는 경우만 보면, 그중 β 이상에서 줄인 안전거리를 지킨다",
         fontproperties=KRB, color=INK, va="center")

ax = fig.add_axes([0.04, 0.24, 0.92, 0.54])
ax.set_xlim(-5.0, 3.3); ax.set_ylim(-0.13, 0.60); ax.axis("off")

ax.fill_between(x, 0, y, where=x >= 0, color=OUT, lw=0)
ax.fill_between(x, 0, y, where=(x >= b) & (x <= 0), color=SAFE, alpha=0.85, lw=0)
ax.fill_between(x, 0, y, where=x <= b, color=RISK, alpha=0.85, lw=0)
ax.plot(x, y, color=INK, lw=1.8)
ax.plot([-4.75, 3.25], [0, 0], color=INK, lw=1.0)
# axis break between the ego and the distribution (distances not to scale)
for dx in (-3.75, -3.62):
    ax.plot([dx - 0.05, dx + 0.05], [-0.025, 0.025], color=INK, lw=1.0)

ax.plot([0, 0], [0, 0.43], color=MUTED, lw=1.3, ls=(0, (4, 3)))
ax.plot([b, b], [0, 0.43], color=INK, lw=2.0)
ax.plot([-4.6, -4.6], [0, 0.06], color=INK, lw=2.0)

ax.text(0, 0.445, r"$\bar{s}$", ha="center", fontsize=15, color=INK)
ax.text(0.12, 0.40, "예측 위치", fontproperties=KRS, color=MUTED, ha="left")
ax.text(b - 0.08, 0.445, r"$s+\alpha\,d_{\mathrm{safe}}(v)$", ha="right", fontsize=14, color=INK)
ax.text(-4.6, 0.085, r"$s$", ha="center", fontsize=15, color=INK)
ax.text(-4.6, -0.05, "ego", fontproperties=KRS, color=MUTED, ha="center")
ax.text(3.25, -0.05, r"앞차 위치 $\hat{s}$", fontproperties=KRS, color=MUTED, ha="right")

# region labels
ax.text(-2.45, 0.20, "위반", fontproperties=KRB, color=RISK, ha="center")
ax.text(-2.45, 0.14, r"조건부 $\leq 1-\beta$", fontproperties=KR, color=INK, ha="center")
ax.annotate("", xy=(-1.3, 0.04), xytext=(-2.0, 0.125), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax.text(b / 2, 0.20, "안전", fontproperties=KRB, color="white", ha="center")
ax.text(b / 2, 0.14, r"$\geq\beta$", fontsize=14, color="white", ha="center")
ax.text(0.85, 0.115, "조건 밖", fontproperties=KRB, color=MUTED, ha="center")
ax.text(0.95, 0.055, "앞차가 멀어짐", fontproperties=KR, color=MUTED, ha="center")

# condition bracket over the near half
yb = 0.545
ax.plot([-3.3, 0], [yb, yb], color=INK, lw=1.0)
ax.plot([-3.3, -3.3], [yb, yb - 0.02], color=INK, lw=1.0)
ax.plot([0, 0], [yb, yb - 0.02], color=INK, lw=1.0)
ax.text(-1.65, yb + 0.012, r"조건  $\hat{s}\leq\bar{s}$ : 앞차가 예측보다 가까이 옴", fontproperties=KR, color=INK, ha="center", va="bottom")

# distance brackets under the axis
yd = -0.095
for x0, x1, lab in ((-4.6, b, r"$\alpha\,d_{\mathrm{safe}}(v)$"), (b, 0, r"$q(\beta)\,\sigma$")):
    ax.plot([x0, x1], [yd, yd], color=INK, lw=1.0)
    ax.plot([x0, x0], [yd - 0.012, yd + 0.012], color=INK, lw=1.0)
    ax.plot([x1, x1], [yd - 0.012, yd + 0.012], color=INK, lw=1.0)
    ax.text((x0 + x1) / 2, yd - 0.022, lab, ha="center", va="top", fontsize=14, color=INK)

fig.text(0.125, 0.105, r"$\Leftrightarrow\quad \bar{s}-s\ \geq\ \alpha\,d_{\mathrm{safe}}(v)\ +\ q(\beta)\,\sigma,\qquad q(\beta)=\Phi^{-1}\left(\frac{1+\beta}{2}\right)$",
         fontsize=17, color=INK, va="center")
fig.text(0.125, 0.03, r"전체로 보면 위반 확률 $\leq(1-\beta)/2$     ·     그림은 $\beta=0.6$, 거리는 축척 아님, 첨자 $j,k$ 생략",
         fontproperties=KRS, color=MUTED, va="center")

out = "/home/core-dev/HJ/ACC/SMPC_MMPreds/overleaf/ppt/constraint_1c_meaning.svg"
fig.savefig(out, facecolor="white")
fig.savefig("/tmp/claude-1000/-home-core-dev-HJ-ACC-SMPC-MMPreds/3821f262-ebd1-4f7f-aed3-df9f9a6bc6d0/scratchpad/constraint_1c_meaning.png", dpi=170, facecolor="white")
print("wrote", out)
