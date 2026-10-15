"""Why the risk budget sum_j p_j Phi(eta_j) >= 1 - eps is guaranteed convex only for eta >= 0.

(a) one mode's risk 1 - Phi(eta): convex for eta >= 0 (the midpoint of two allocations is no
riskier than their average), concave for eta < 0 (the midpoint can be riskier).
(b) a three-mode budget, p = (0.92, 0.04, 0.04), 1 - eps = 0.95, sliced at eta_1 = 3.2: giving up
mode 2 or mode 3 fully is within budget, giving up half of each is not.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import norm

plt.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path", "axes.unicode_minus": False})
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
KR = fm.FontProperties(family="Noto Sans CJK JP", size=10)
KRB = fm.FontProperties(family="Noto Sans CJK JP", size=12, weight="bold")
KRS = fm.FontProperties(family="Noto Sans CJK JP", size=8.5)
INK, SAFE, RISK, MUTED, OK_BG, NG_BG = "#1C2430", "#3F7CC4", "#D9480F", "#5A6576", "#E3EDF8", "#FBE9E1"

P = np.array([0.92, 0.04, 0.04]); BUDGET = 0.95; ETA1 = 3.2
need = (BUDGET - P[0] * norm.cdf(ETA1)) / P[1]           # Phi(eta2) + Phi(eta3) >= need
A = np.array([-3.0, 0.75]); B = A[::-1]; M = 0.5 * (A + B)
assert norm.cdf(A).sum() >= need and norm.cdf(M).sum() < need
risk = lambda e: 1.0 - norm.cdf(e)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.9), gridspec_kw=dict(wspace=0.28))
fig.text(0.5, 0.985, "η ≥ 0 이면 볼록이 보장되는 이유: 그 구간에서 위험 곡선이 아래로 볼록이라, "
         "두 배분의 중간은 위험이 평균 이하다", fontproperties=KRB, color=INK, ha="center", va="top")

# (a) one mode's risk curve
e = np.linspace(-3.3, 3.3, 600)
ax.axvspan(-3.3, 0, color=NG_BG, lw=0); ax.axvspan(0, 3.3, color=OK_BG, lw=0)
ax.plot(e, risk(e), color=INK, lw=2.0)
for (x0, x1), c in (((0.5, 2.5), SAFE), ((A[0], A[1]), RISK)):
    xm = 0.5 * (x0 + x1); chord = 0.5 * (risk(x0) + risk(x1))
    ax.plot([x0, x1], [risk(x0), risk(x1)], color=c, lw=1.4, ls="--")
    ax.plot([x0, x1], [risk(x0), risk(x1)], "o", color=c, ms=5)
    ax.plot([xm], [chord], "s", color=c, ms=6, mfc="white", mew=1.4)
    ax.plot([xm], [risk(xm)], "o", color=c, ms=7)
    ax.annotate("", xy=(xm, risk(xm)), xytext=(xm, chord),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
ax.text(1.55, 0.36, "중간 배분의 위험 %.3f\n≤ 평균 %.3f" % (risk(1.5), 0.5 * (risk(0.5) + risk(2.5))),
        fontproperties=KRS, color=SAFE, va="center")
xm = M[0]
ax.text(xm - 0.15, 0.40, "중간 배분의 위험 %.3f\n> 평균 %.3f" % (risk(xm), 0.5 * (risk(A[0]) + risk(A[1]))),
        fontproperties=KRS, color=RISK, ha="center", va="center")
ax.text(-1.65, 1.06, "η < 0: 위험이 1에 포화 → 위로 볼록", fontproperties=KRS, color=RISK, ha="center")
ax.text(1.65, 1.06, "η ≥ 0: 아래로 볼록", fontproperties=KRS, color=SAFE, ha="center")
ax.set_xlim(-3.3, 3.3); ax.set_ylim(-0.03, 1.13)
ax.set_xlabel(r"$\eta$  (여유 = $d_{\mathrm{safe}} + \eta\,\sigma$)", fontproperties=KR)
ax.set_ylabel(r"모드 하나의 위험  $1-\Phi(\eta)$", fontproperties=KR)
ax.set_title("(a) 위험 곡선", fontproperties=KR, loc="left")
ax.grid(alpha=0.25, lw=0.5)

# (b) feasible set of a three-mode budget, sliced at eta_1 = 3.2
g = np.linspace(-3.5, 3.5, 701); G2, G3 = np.meshgrid(g, g)
ok = norm.cdf(G2) + norm.cdf(G3) >= need
bx.contourf(G2, G3, ok.astype(float), levels=[-0.5, 0.5, 1.5], colors=[NG_BG, OK_BG])
bx.contour(G2, G3, norm.cdf(G2) + norm.cdf(G3), levels=[need], colors=[INK], linewidths=1.6)
bx.axhline(0, color=MUTED, lw=0.8); bx.axvline(0, color=MUTED, lw=0.8)
bx.plot([A[0], B[0]], [A[1], B[1]], color=RISK, lw=1.4, ls="--")
for pt, name, dx, dy in ((A, "A", 0.15, 0.15), (B, "B", 0.15, 0.15)):
    bx.plot(*pt, "o", color=SAFE, ms=7); bx.text(pt[0] + dx, pt[1] + dy, name, fontsize=12, color=SAFE, weight="bold")
bx.plot(*M, "o", color=RISK, ms=8); bx.text(M[0] - 0.95, M[1] - 0.45, "M (중간)", fontproperties=KR, color=RISK)
bx.text(-3.3, 2.1, "A: 모드 2를 다 포기\n(η2 = −3), 예산 안", fontproperties=KRS, color=SAFE)
bx.text(1.05, -2.55, "B: 모드 3을 다 포기\n(η3 = −3), 예산 안", fontproperties=KRS, color=SAFE)
bx.text(-3.3, -3.35, "M: 둘을 반씩 포기 → 예산 초과", fontproperties=KRS, color=RISK)
bx.text(1.6, 0.25, "η ≥ 0 사분면: 전부 만족", fontproperties=KRS, color=MUTED, ha="center")
bx.set_xlim(-3.5, 3.5); bx.set_ylim(-3.5, 3.5); bx.set_aspect("equal")
bx.set_xlabel(r"$\eta_2$", fontsize=12); bx.set_ylabel(r"$\eta_3$", fontsize=12)
bx.set_title(r"(b) $p=(0.92,\,0.04,\,0.04)$, $\Sigma_j\, p_j\Phi(\eta_j)\geq 0.95$, $\eta_1=3.2$ 고정",
             fontproperties=KR, loc="left")
bx.text(3.35, 3.35, "파랑: 예산 만족\n주황: 예산 초과", fontproperties=KRS, color=MUTED, ha="right", va="top")

for ext in ("svg", "png"):
    fig.savefig("overleaf/ppt/eta_convexity.%s" % ext, dpi=200, bbox_inches="tight")
print("need = %.4f, A risk sum ok, M = %s" % (need, np.round(M, 3)))
