"""Validity, sensitivity and calibration-fluctuation experiments for novelty CP.

Everything here works on the .npz feature files written by extract_features.py
and the fitted detectors written by fit_novelty.py. Sections (each --only-able):

  validity   Are calibration (Val) and test (Test) scores exchangeable, and does
             the alarm rate at alpha land inside the Beta(l, n+1-l) interval a
             single calibration draw allows? Reported for the official split
             AND for a track-level random re-split of Val+Test (P4b), because
             Test vehicles enter the recordings later than Val vehicles.
  shift      AUROC / FPR@TPR95 and their conformal versions on the input-scaled
             Test sets (x2.5, x0.2), feature-based vs output-based scores.
  bates      Novello Fig. 1 / Bates Prop. 1: K disjoint TRACK folds of the
             Val+Test pool, FPR(0.1; D_cal) per fold vs Beta(l, n+1-l); and the
             cumulative histogram of corrected p-values (Fig. 2) for
             Simes / asymptotic / Monte-Carlo.
  controls   The reviewer's failure mode, measured: reference and calibration
             drawn from the SAME tracks (frame-random split of a dense Val
             extraction) vs from different tracks; with- vs without-replacement
             calibration; and the score autocorrelation along a track (how far
             apart two frames must be to count as two samples).
  neighbour  Do Val samples that share a (recording, frame) with Train vehicles
             score differently from those that do not? (leakage vs dependence)

Run (hmg env, from this directory):
    python eval_novelty.py            # all sections
    python eval_novelty.py --only validity shift
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/core-dev/HJ/graduate_project/oodeel")
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from predictor.uq import conformal_ood as C  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
FEAT_DIR = os.path.join(CKPT, "features")
FIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "uq_novelty"))
DATA = "/home/core-dev/HJ/dataset/raw_data_ngsim/dataset_t_v_t"

FEATURES = ("agent", "social", "concat", "values_last", "pre_logit")
OUTPUT_SCORES = ("mls", "msp", "energy")


def feature_matrix(npz, name):
    if name == "concat":
        return np.concatenate([npz["agent"], npz["social"]], axis=1).astype(np.float32)
    return np.asarray(npz[name], dtype=np.float32)


def track_key(npz):
    return npz["ds_id"].astype(np.int64) * 100000 + npz["veh_id"].astype(np.int64)


def load_scorer(kind, feat=None, tag=""):
    t = f"_{tag}" if tag else ""
    if kind in OUTPUT_SCORES:
        d = np.load(os.path.join(CKPT, f"novelty_{kind}{t}.npz"))
        return ("output", kind, np.asarray(d["cal_scores"]), np.asarray(d["b_seq"]),
                C.MondrianCalibration.from_dict(d))
    path = os.path.join(CKPT, f"novelty_{kind}_{feat}{t}.npz")
    return ("feature", C.load_novelty(path), None, None)


_SCORE_CACHE = {}


def score_with(scorer, npz):
    """Scores for one (scorer, feature file); memoised - every section reuses them."""
    key = (id(scorer), id(npz))
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    kind = scorer[0]
    if kind == "output":
        from fit_novelty import output_scores
        out = output_scores(npz["logits"], scorer[1])
    else:
        obj = scorer[1]
        out = obj.score(feature_matrix(npz, obj.feature))
    _SCORE_CACHE[key] = out
    return out


def cal_scores_of(scorer):
    return scorer[2] if scorer[0] == "output" else scorer[1].cal


def mondrian_of(scorer):
    if scorer[0] == "output":
        return scorer[4]
    return getattr(scorer[1], "mondrian", None)


def group_of(npz):
    """Mondrian group = the predictor's argmax class (observable at run time)."""
    return np.asarray(npz["logits"]).argmax(1)


def pvalues_of(scorer, npz, mondrian=True):
    """(u, u_cc). Group-conditional when a Mondrian calibration is available."""
    s = score_with(scorer, npz)
    M = mondrian_of(scorer)
    if mondrian and M is not None:
        return M.p_values(s, group_of(npz))
    u = C.marginal_pvalues(cal_scores_of(scorer), s)
    return u, correction_of(scorer)(u)


def correction_of(scorer):
    if scorer[0] == "output":
        return C.ConditionalCorrection(scorer[3])
    return scorer[1].correction


def all_scorers(tag=""):
    out = []
    for f in FEATURES:
        for k in ("maha", "knn"):
            try:
                out.append((f"{k}_{f}", load_scorer(k, f, tag)))
            except FileNotFoundError:
                pass
    for k in OUTPUT_SCORES:
        try:
            out.append((k, load_scorer(k, tag=tag)))
        except FileNotFoundError:
            pass
    return out


def _korean_font(plt):
    """Register the system Malgun Gothic explicitly: matplotlib's cache in the
    hmg env does not list it, so name lookup alone silently falls back."""
    from matplotlib import font_manager as fm
    for path in ("/usr/share/fonts/malgun.ttf", "/usr/share/fonts/NanumGothic.ttf"):
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


# ------------------------------------------------------------------ sections


def ks_track_permutation(scores, keys, is_cal, n_perm=300, seed=0):
    """Two-sample KS with a TRACK-level permutation null.

    Samples within a track are correlated (~26 thinned frames per vehicle), so
    the iid KS p-value over-rejects. Re-assigning whole tracks between the two
    groups keeps the dependence structure and gives an honest p-value; also
    returns the effective-sample-size ratio implied by the null spread.
    """
    from scipy.stats import ks_2samp
    rng = np.random.RandomState(seed)
    obs = ks_2samp(scores[is_cal], scores[~is_cal]).statistic
    uniq, inv = np.unique(keys, return_inverse=True)
    n_cal_tracks = int(np.unique(keys[is_cal]).size)
    null = np.empty(n_perm)
    for i in range(n_perm):
        pick = np.zeros(len(uniq), bool)
        pick[rng.choice(len(uniq), n_cal_tracks, replace=False)] = True
        m = pick[inv]
        null[i] = ks_2samp(scores[m], scores[~m]).statistic
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    # iid KS null: D * sqrt(n1 n2/(n1+n2)) has Kolmogorov distribution; median ~0.83
    n1, n2 = int(is_cal.sum()), int((~is_cal).sum())
    scale_iid = 0.8276 / np.sqrt(n1 * n2 / (n1 + n2))
    return {"D": float(obs), "p_perm": p, "null_median_D": float(np.median(null)),
            "iid_median_D": float(scale_iid),
            "neff_ratio": float((scale_iid / np.median(null)) ** 2)}


def sec_validity(scorers, test, cal, pool_seed=0):
    """Official Val->Test and a track-level re-split of the pool."""
    res = {"official": {}, "resplit": {}}
    print("\n[validity] official split: calibration = Val, test = Test")
    print(f"  {'score':<20}{'ks2 p':>8}{'mean p':>8}{'@0.01':>14}{'@0.05':>14}{'@0.10':>14}")
    for name, sc in scorers:
        s_test = score_with(sc, test)
        r = C.validity_report(cal_scores_of(sc), s_test)
        res["official"][name] = r
        cells = "".join(f"{r['alphas'][a]['alarm_rate']:>8.4f}{'✓' if r['alphas'][a]['inside'] else '✗':>6}"
                        for a in ("0.01", "0.05", "0.1"))
        print(f"  {name:<20}{r['ks2_p']:>8.3f}{r['mean_p']:>8.4f}{cells}")

    # track-level re-split of Val+Test (same time distribution on both sides)
    keys = np.concatenate([track_key(cal), track_key(test)])
    uniq = np.unique(keys)
    rng = np.random.RandomState(pool_seed)
    half = set(rng.choice(uniq, len(uniq) // 2, replace=False).tolist())
    in_c = np.array([k in half for k in keys])
    print("\n[validity] track re-split of Val+Test pool "
          f"(cal {int(in_c.sum())} rows / test {int((~in_c).sum())} rows)")
    print(f"  {'score':<20}{'ks2 p':>8}{'perm p':>8}{'n_eff/n':>8}{'mean p':>8}{'@0.01':>14}{'@0.05':>14}{'@0.10':>14}")
    for name, sc in scorers:
        s_all = np.concatenate([score_with(sc, cal), score_with(sc, test)])
        r = C.validity_report(s_all[in_c], s_all[~in_c])
        r["track_perm"] = ks_track_permutation(s_all, keys, in_c, n_perm=200)
        res["resplit"][name] = r
        cells = "".join(f"{r['alphas'][a]['alarm_rate']:>8.4f}{'✓' if r['alphas'][a]['inside'] else '✗':>6}"
                        for a in ("0.01", "0.05", "0.1"))
        print(f"  {name:<20}{r['ks2_p']:>8.3f}{r['track_perm']['p_perm']:>8.3f}"
              f"{r['track_perm']['neff_ratio']:>8.2f}{r['mean_p']:>8.4f}{cells}")
    # official split too, permutation over Val-vs-Test track labels is not
    # meaningful (the split is not random), so only iid KS is reported there
    return res


def sec_mondrian_resplit(scorers, cal, test, alphas=(0.05,), seed=0):
    """Same per-class comparison, but on a track-level re-split of Val+Test.

    The official Val->Test split is not exchangeable (Test vehicles enter the
    recordings later), so a per-class alarm rate measured across it mixes the
    Mondrian effect with that drift. Re-splitting by track puts the same time
    distribution on both sides; the calibration scores are recomputed from the
    new calibration half, per predicted class.
    """
    CLS = ("직진", "좌변경", "우변경")
    keys = np.concatenate([track_key(cal), track_key(test)])
    g_all = np.concatenate([group_of(cal), group_of(test)])
    uniq = np.unique(keys)
    rng = np.random.RandomState(seed)
    half = set(rng.choice(uniq, len(uniq) // 2, replace=False).tolist())
    in_c = np.array([k in half for k in keys])
    res = {}
    print("\n[mondrian-resplit] 트랙 재분할에서 예측 클래스별 경보율 @0.05 (주변 → Mondrian)")
    print(f"  {'score':<18}" + "".join(f"{c:>22}" for c in CLS) + f"{'n_cal(직/좌/우)':>20}")
    ncal = "/".join(str(int(np.sum(in_c & (g_all == c)))) for c in range(3))
    for name, sc in scorers:
        s_all = np.concatenate([score_with(sc, cal), score_with(sc, test)])
        M = C.MondrianCalibration({int(c): s_all[in_c & (g_all == c)]
                                   for c in range(3)
                                   if np.sum(in_c & (g_all == c)) > 0},
                                  delta=0.05, method="simes")
        u_g, _ = M.p_values(s_all[~in_c], g_all[~in_c])
        u_m = C.marginal_pvalues(s_all[in_c], s_all[~in_c])
        g_te = g_all[~in_c]
        cells, row = "", {}
        for c in range(3):
            m = g_te == c
            if m.sum() < 20:
                cells += f"{'-':>22}"; continue
            a, b = float(np.mean(u_m[m] <= 0.05)), float(np.mean(u_g[m] <= 0.05))
            row[CLS[c]] = {"marginal": a, "mondrian": b, "n_test": int(m.sum())}
            cells += f"{a:>10.3f} → {b:<9.3f}"
        res[name] = row
        print(f"  {name:<18}{cells}{ncal:>20}")
    return res


def sec_mondrian(scorers, test, alphas=(0.05,)):
    """Per-predicted-class alarm rate, marginal vs group-conditional calibration.

    The marginal detector inherits the road's class imbalance: on the reference
    data itself, samples the predictor calls a lane change alarm far above alpha
    while lane-keeping samples alarm below it. Mondrian by predicted class is
    the fix; the cost is a much smaller n for the rare groups.
    """
    CLS = ("직진", "좌변경", "우변경")
    g = group_of(test)
    res = {}
    print("\n[mondrian] 예측 클래스별 경보율 @0.05  (주변 보정 → Mondrian 보정)")
    print(f"  {'score':<18}" + "".join(f"{c:>22}" for c in CLS) + f"{'n_cal(직/좌/우)':>20}")
    for name, sc in scorers:
        M = mondrian_of(sc)
        u_m, _ = pvalues_of(sc, test, mondrian=False)
        u_g, ucc_g = pvalues_of(sc, test, mondrian=True)
        cells, row = "", {}
        for c in range(3):
            m = g == c
            if m.sum() < 20:
                cells += f"{'-':>22}"; continue
            a, b = float(np.mean(u_m[m] <= 0.05)), float(np.mean(u_g[m] <= 0.05))
            row[CLS[c]] = {"marginal": a, "mondrian": b, "n_test": int(m.sum()),
                           "cc_mondrian": float(np.mean(ucc_g[m] <= 0.05))}
            cells += f"{a:>10.3f} → {b:<9.3f}"
        ncal = "/".join(str(M.n_of(c)) if M else "-" for c in range(3))
        res[name] = row
        print(f"  {name:<18}{cells}{ncal:>20}")
    return res


def sec_shift(scorers, shifts, test=None):
    """Detection metrics under input scaling.

    With a Mondrian calibration the natural read is per predicted class: an
    alarm rate on shifted data at the level alpha that the same detector holds
    on ID data. AUROC is reported against the calibration scores as before, and
    additionally as the shifted-data alarm rate at alpha = 0.05 (which is
    directly comparable across scorers because they are all calibrated to 0.05
    on ID data - see the mondrian section).
    """
    res = {}
    print("\n[shift] AUROC / FPR@TPR95 (classical | conformal, MC delta=0.05)")
    hdr = f"  {'score':<20}" + "".join(f"{'x'+str(s):>34}" for s in shifts)
    print(hdr)
    print(f"  {'':<20}" + "".join(f"{'auroc':>8}{'c-auroc':>9}{'fpr95':>8}{'c-fpr95':>9}" for _ in shifts))
    for name, sc in scorers:
        cal_s = cal_scores_of(sc); corr = correction_of(sc)
        row = f"  {name:<20}"; res[name] = {}
        for s, npz in shifts.items():
            m = C.ood_metrics(cal_s, score_with(sc, npz), correction=corr)
            res[name][str(s)] = m
            row += f"{m['auroc']:>8.3f}{m['conf_auroc']:>9.3f}{m['fpr95tpr']:>8.3f}{m['conf_fpr95tpr']:>9.3f}"
        print(row)
    # detection rate at the operating point the detector is actually calibrated
    # to, using the Mondrian (group-conditional) p-values
    print("\n[shift] Mondrian p-값 기준 검출률 @0.05  (ID에서는 0.05여야 함)")
    hdr = f"  {'score':<18}{'ID(NGSIM)':>11}" + "".join(f"{'x'+str(s):>11}" for s in shifts)
    print(hdr)
    for name, sc in scorers:
        cells = ""
        if test is not None:
            u_id, _ = pvalues_of(sc, test, mondrian=True)
            cells += f"{np.mean(u_id <= 0.05):>11.3f}"
        for s, npz in shifts.items():
            u, _ = pvalues_of(sc, npz, mondrian=True)
            cells += f"{np.mean(u <= 0.05):>11.3f}"
            res[name][f"detect_{s}"] = float(np.mean(u <= 0.05))
        print(f"  {name:<18}{cells}")
    return res


def sec_bates(scorers, cal, test, K=20, tau=0.1, delta=0.05, out_png=None):
    """Bates Prop. 1: FPR(tau; D_cal) across disjoint track folds vs Beta."""
    from scipy.stats import beta as beta_dist
    keys = np.concatenate([track_key(cal), track_key(test)])
    uniq = np.unique(keys)
    rng = np.random.RandomState(0)
    perm = rng.permutation(uniq)
    fold_of_track = {int(k): i % K for i, k in enumerate(perm)}
    fold = np.array([fold_of_track[int(k)] for k in keys])
    res = {}
    fig = None
    if out_png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _korean_font(plt)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    print(f"\n[bates] K={K} track folds, tau={tau}: FPR(tau; D_cal) vs Beta(l, n+1-l)")
    print(f"  {'score':<20}{'n_cal':>7}{'obs mean':>10}{'obs sd':>8}{'Beta sd':>9}{'sd ratio':>9}"
          f"{'MC cons.':>10}{'Simes cons.':>12}{'asymp cons.':>12}{'Simes(n_eff)':>12}{'n_eff':>8}")
    for name, sc in scorers:
        s_all = np.concatenate([score_with(sc, cal), score_with(sc, test)])
        fprs, n_cals = [], []
        cons = {"monte_carlo": [], "simes": [], "asymptotic": []}
        seqs = {}
        for f in range(K):
            c_mask = fold == f
            cal_s, te_s = s_all[c_mask], s_all[~c_mask]
            u = C.marginal_pvalues(cal_s, te_s)
            fprs.append(float(np.mean(u <= tau))); n_cals.append(int(c_mask.sum()))
            n = int(c_mask.sum())
            if n not in seqs:
                seqs[n] = {"simes": C.simes_sequence(n, delta),
                           "asymptotic": C.asymptotic_sequence(n, delta),
                           "monte_carlo": C.monte_carlo_sequence(n, delta, n_sim=5000)}
            for m, b in seqs[n].items():
                ucc = C.ConditionalCorrection(b)(u)
                # conservative for all t on a grid? The guarantee is about the
                # TRUE conditional CDF; the empirical one over n_test points has
                # sampling noise ~sqrt(t(1-t)/n_test), so allow 2 SE of it.
                ts = np.linspace(0.005, 0.5, 100)
                tol = 2.0 * np.sqrt(ts * (1 - ts) / len(u))
                cons[m].append(bool(np.all([np.mean(ucc <= t) <= t + e for t, e in zip(ts, tol)])))
        n_med = int(np.median(n_cals)); l = int(np.floor((n_med + 1) * tau))
        th_sd = float(beta_dist.std(l, n_med + 1 - l))
        # dependence-adjusted variant: build the Simes sequence for n_eff =
        # n / ratio^2 (ratio = observed/theoretical fluctuation) and map the
        # marginal p-values through it. Heuristic - Bates assume iid - but it
        # tells whether "count only the independent information" restores delta.
        ratio_sd = float(np.std(fprs, ddof=1) / th_sd)
        cons_eff = []
        for f in range(K):
            c_mask = fold == f
            cal_s, te_s = s_all[c_mask], s_all[~c_mask]
            u = C.marginal_pvalues(cal_s, te_s)
            n = int(c_mask.sum()); n_eff = max(50, int(n / ratio_sd ** 2))
            b_eff = C.simes_sequence(n_eff, delta)
            # h_eff(t) = b_eff[ceil((n_eff+1) t)]: same map, coarser grid
            idx = np.clip(np.ceil((n_eff + 1) * u).astype(int), 0, n_eff + 1)
            ucc = np.concatenate([[0.0], b_eff, [1.0]])[idx]
            ts = np.linspace(0.005, 0.5, 100)
            tol = 2.0 * np.sqrt(ts * (1 - ts) / len(u))
            cons_eff.append(bool(np.all([np.mean(ucc <= t) <= t + e for t, e in zip(ts, tol)])))
        r = {"n_cal_median": n_med, "fpr_obs": fprs, "obs_mean": float(np.mean(fprs)),
             "obs_sd": float(np.std(fprs, ddof=1)), "beta_sd": th_sd,
             "sd_ratio": float(np.std(fprs, ddof=1) / th_sd),
             "conservative_frac": {m: float(np.mean(v)) for m, v in cons.items()},
             "conservative_frac_simes_neff": float(np.mean(cons_eff)),
             "n_eff_median": int(n_med / ratio_sd ** 2)}
        res[name] = r
        print(f"  {name:<20}{n_med:>7}{r['obs_mean']:>10.4f}{r['obs_sd']:>8.4f}{th_sd:>9.4f}"
              f"{r['sd_ratio']:>9.2f}{r['conservative_frac']['monte_carlo']:>10.2f}"
              f"{r['conservative_frac']['simes']:>12.2f}{r['conservative_frac']['asymptotic']:>12.2f}"
              f"{r['conservative_frac_simes_neff']:>12.2f}{r['n_eff_median']:>8}")
        if fig is not None and name in ("maha_concat", "msp"):
            ax = axes[0] if name.startswith("maha") else axes[1]
            xs = np.linspace(max(0, tau - 6 * th_sd), tau + 6 * th_sd, 300)
            ax.hist(fprs, bins=10, density=True, alpha=.55, color="#4a90d9", label=f"관측 FPR(0.1), K={K} 폴드")
            ax.plot(xs, beta_dist.pdf(xs, l, n_med + 1 - l), "r-", lw=2,
                    label=f"이론 Beta(ℓ, n+1−ℓ), n={n_med}")
            ax.axvline(tau, color="k", ls=":"); ax.set_title(name); ax.legend(fontsize=8)
            ax.set_xlabel("FPR(0.1; D_cal)")
    if fig is not None:
        fig.suptitle("보정셋 하나로 잰 FPR은 요동한다 (Bates Prop.1 / Novello Fig.1 재현)")
        fig.tight_layout(); fig.savefig(out_png, dpi=140); print("  saved", out_png)
    return res


def sec_controls(dense, test, feat="concat", k=50, gap_frames=30, alphas=(0.01, 0.05, 0.1)):
    """Same-track vs different-track reference/calibration; replacement; autocorrelation."""
    from fit_novelty import fit_mahalanobis, fit_knn, oodeel_scores
    X = feature_matrix(dense, feat); y = np.asarray(dense["labels"], dtype=np.int64)
    keys = track_key(dense); frames = dense["frame"].astype(np.int64)
    Xt = feature_matrix(test, feat)
    rng = np.random.RandomState(0)
    res = {}

    def run(ref_mask, cal_mask, label, replace=False):
        det, mu, prec, _ = fit_mahalanobis(X[ref_mask], y[ref_mask])
        m = C.MahalanobisNovelty(mu, prec, cal_scores=[0.0], method="dkwm")
        cal_s = m.score(X[cal_mask])
        if replace:
            cal_s = cal_s[rng.randint(len(cal_s), size=len(cal_s))]
        u = C.marginal_pvalues(cal_s, m.score(Xt))
        det_k = fit_knn(X[ref_mask], y[ref_mask], k)
        kk = C.KnnNovelty(X[ref_mask], k, cal_scores=[0.0], method="dkwm")
        cal_k = kk.score(X[cal_mask])
        if replace:
            cal_k = cal_k[rng.randint(len(cal_k), size=len(cal_k))]
        uk = C.marginal_pvalues(cal_k, kk.score(Xt))
        r = {"maha": {str(a): float(np.mean(u <= a)) for a in alphas},
             "knn": {str(a): float(np.mean(uk <= a)) for a in alphas},
             "n_ref": int(ref_mask.sum()), "n_cal": int(cal_mask.sum())}
        res[label] = r
        print(f"  {label:<38} n_ref={r['n_ref']:>6} n_cal={r['n_cal']:>6} | maha alarm "
              + " ".join(f"@{a}={r['maha'][str(a)]:.3f}" for a in alphas)
              + " | knn " + " ".join(f"@{a}={r['knn'][str(a)]:.3f}" for a in alphas))

    print(f"\n[controls] ID alarm rate on Test (feature={feat}); target ≈ alpha")
    # (i) frame-random split of the dense Val extraction: same tracks on both sides
    perm = rng.permutation(len(X)); half = len(X) // 2
    m_ref = np.zeros(len(X), bool); m_ref[perm[:half]] = True
    run(m_ref, ~m_ref, "frame-random split (SAME tracks)")
    # thinned version of the same leaky split (every 30th frame) - does thinning alone help?
    thin = np.zeros(len(X), bool)
    order = np.lexsort((frames, keys)); ks = keys[order]
    start = np.concatenate([[True], ks[1:] != ks[:-1]])
    pos = np.arange(len(X)) - np.maximum.accumulate(np.where(start, np.arange(len(X)), 0))
    thin[order[pos % gap_frames == 0]] = True
    run(m_ref & thin, (~m_ref) & thin, "frame-random split, thinned 3 s")
    # (ii) track split of the same dense data
    uniq = np.unique(keys); tr_half = set(rng.choice(uniq, len(uniq) // 2, replace=False).tolist())
    m_trk = np.array([kk_ in tr_half for kk_ in keys])
    run(m_trk & thin, (~m_trk) & thin, "track split, thinned 3 s (DIFFERENT tracks)")
    run(m_trk, ~m_trk, "track split, no thinning")
    # (iii) with replacement calibration
    run(m_trk & thin, (~m_trk) & thin, "track split, thinned, cal WITH replacement", replace=True)

    # (iv) autocorrelation of the score along a track (thinning-gap diagnostic)
    det, mu, prec, _ = fit_mahalanobis(X[m_trk], y[m_trk])
    m = C.MahalanobisNovelty(mu, prec, cal_scores=[0.0], method="dkwm")
    s = m.score(X)
    lags = [1, 5, 10, 20, 30, 50, 100]
    acf = {lag: [] for lag in lags}
    for a_, b_ in _track_ranges(order, ks):
        rows = order[a_:b_]
        if len(rows) < 120:
            continue
        v = s[rows] - s[rows].mean()
        den = (v * v).sum()
        if den <= 0:
            continue
        for lag in lags:
            if len(v) > lag + 10:
                acf[lag].append(float((v[:-lag] * v[lag:]).sum() / den))
    rho = {str(lag / 10.0): float(np.mean(v)) for lag, v in acf.items() if v}
    res["autocorr_s"] = rho
    print("  score autocorrelation along a track (lag s: rho): "
          + ", ".join(f"{k}s:{v:.2f}" for k, v in rho.items()))
    return res


def _track_ranges(order, ks):
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(ks)]])
    return zip(starts, ends)


def sec_neighbour(scorers, cal):
    """Val samples co-occurring with Train vehicles vs not: same score distribution?"""
    import scipy.io as sio
    from scipy.stats import ks_2samp
    try:
        T = np.asarray(sio.loadmat(os.path.join(DATA, "TrainSet.mat"))["traj"])
    except Exception:
        import h5py
        T = np.asarray(h5py.File(os.path.join(DATA, "TrainSet.mat"), "r")["traj"]).T
    train_frames = set(map(tuple, T[:, [0, 2]].astype(np.int64)))
    co = np.array([(int(d), int(f)) in train_frames for d, f in zip(cal["ds_id"], cal["frame"])])
    res = {"co_occur_frac": float(co.mean())}
    print(f"\n[neighbour] Val samples sharing (recording, frame) with Train: {co.mean():.3f}")
    print(f"  {'score':<20}{'ks p':>8}{'median co':>11}{'median not':>11}{'IQR':>9}{'shift/IQR':>10}")
    for name, sc in scorers:
        s = score_with(sc, cal)
        p = ks_2samp(s[co], s[~co]).pvalue
        iqr = float(np.subtract(*np.percentile(s, [75, 25])))
        d = float(np.median(s[co]) - np.median(s[~co]))
        res[name] = {"ks_p": float(p), "median_shift_over_iqr": d / iqr if iqr > 0 else None}
        print(f"  {name:<20}{p:>8.3f}{np.median(s[co]):>11.3f}{np.median(s[~co]):>11.3f}{iqr:>9.3f}{d/iqr:>10.3f}")
    return res


def sec_runtime(scorers, test, sets, alphas=(0.01, 0.05, 0.1), out_png=None):
    """Q3: are the hand-crafted CommonRoad benchmarks OOD for STDAN?

    Runtime features dumped by the planner (STDAN_FEATURE_DUMP) are scored with
    the SAME calibrated detectors; the alarm rate at alpha is compared with the
    NGSIM Test baseline (which is ~alpha by validity). Rows are split by whether
    the 33-step history was fully observed (n_real_hist >= 33) or partly
    back-extrapolated at constant velocity (the planner's warm-up), because the
    synthetic prefix is a preprocessing artefact, not a driving situation.
    """
    import glob
    res = {}
    loaded = {}
    for name, d in sets.items():
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            continue
        parts = [np.load(f) for f in files]
        loaded[name] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
        loaded[name]["scenario"] = np.concatenate(
            [np.full(len(p["oid"]), i) for i, p in enumerate(parts)])
    if not loaded:
        print("[runtime] no dumps found"); return res
    print(f"\n[runtime] alarm rate (p <= alpha) on planner-side features; NGSIM Test ≈ alpha")
    for name, sc in scorers:
        cal_s = cal_scores_of(sc); corr = correction_of(sc)
        base_u, _ = pvalues_of(sc, test, mondrian=True)
        row = {"ngsim_test": {str(a): float(np.mean(base_u <= a)) for a in alphas}}
        line = f"  {name:<18} NGSIM " + " ".join(f"@{a}={row['ngsim_test'][str(a)]:.3f}" for a in alphas)
        for sname, d in loaded.items():
            u_marg = C.marginal_pvalues(cal_s, score_with(sc, d))
            u, ucc = pvalues_of(sc, d, mondrian=True)
            real = d["n_real_hist"] >= 33
            r = {"n": int(len(u)), "n_real": int(real.sum()),
                 "all": {str(a): float(np.mean(u <= a)) for a in alphas},
                 "real_hist": {str(a): float(np.mean(u[real] <= a)) for a in alphas},
                 "padded_hist": {str(a): float(np.mean(u[~real] <= a)) for a in alphas},
                 "cc_real_hist": {str(a): float(np.mean(ucc[real] <= a)) for a in alphas},
                 "marginal_real_hist": {str(a): float(np.mean(u_marg[real] <= a)) for a in alphas},
                 "median_p_real": float(np.median(u[real])) if real.any() else None,
                 "median_p_padded": float(np.median(u[~real])) if (~real).any() else None}
            row[sname] = r
            line += f" | {sname}: real " + "/".join(f"{r['real_hist'][str(a)]:.3f}" for a in alphas)                     + f" pad " + "/".join(f"{r['padded_hist'][str(a)]:.3f}" for a in alphas)
        res[name] = row
        print(line)
    if out_png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _korean_font(plt)
        pick = [n for n, _ in scorers if n in ("maha_concat", "knn_social", "knn_values_last", "msp")]
        fig, axes = plt.subplots(1, len(pick), figsize=(4.4 * len(pick), 3.8), sharey=True)
        for ax, pname in zip(np.atleast_1d(axes), pick):
            sc = dict(scorers)[pname]; cal_s = cal_scores_of(sc)
            labels, data = ["NGSIM Test"], [C.marginal_pvalues(cal_s, score_with(sc, test))]
            for sname, d in loaded.items():
                real = d["n_real_hist"] >= 33
                u = C.marginal_pvalues(cal_s, score_with(sc, d))
                labels.append(sname.replace("cutin_", "").replace("nocutin", "no-cutin") + "\n(real hist)")
                data.append(u[real])
            ax.boxplot(data, labels=labels, showfliers=False, widths=0.6)
            ax.axhline(0.05, color="crimson", ls="--", lw=1)
            ax.set_title(pname, fontsize=10); ax.tick_params(axis="x", labelsize=7.5)
        axes_flat = np.atleast_1d(axes); axes_flat[0].set_ylabel("novelty p-값 (작을수록 낯섦)")
        fig.suptitle("우리 합성 시나리오는 예측기에게 낯선가? — 스텝별 p-값 분포 (빨간선 α=0.05)")
        fig.tight_layout(); fig.savefig(out_png, dpi=140); print("  saved", out_png)
    return res


def sec_figures(scorers, test, cal, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _korean_font(plt)
    pick = [n for n, _ in scorers if n in ("maha_concat", "knn_concat", "maha_pre_logit", "msp")]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.6))
    for ax, name in zip(np.atleast_1d(axes), pick):
        sc = dict(scorers)[name]
        u = C.marginal_pvalues(cal_scores_of(sc), score_with(sc, test))
        ax.hist(u, bins=20, range=(0, 1), density=True, color="#4a90d9", alpha=.8)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_title(f"{name}\nID Test p-값 (균등이면 평평)", fontsize=10)
        ax.set_xlabel("p-값"); ax.set_ylim(0, 1.6)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); print("  saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["validity", "shift", "bates", "controls", "neighbour", "figures", "runtime", "mondrian"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--feat-suffix", default="",
                    help="suffix of the feature files, e.g. g1 for the 1 s thinning")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--runtime-suffix", default="",
                    help="score runtime_<set><suffix> dumps (e.g. _fix)")
    cli = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    want = set(cli.only or ["validity", "mondrian", "shift", "bates", "controls",
                            "neighbour", "figures", "runtime"])

    scorers = all_scorers(cli.tag)
    print("scorers:", [n for n, _ in scorers])
    suf = f"_{cli.feat_suffix}" if cli.feat_suffix else ""
    cal = np.load(os.path.join(FEAT_DIR, f"cal_val{suf}.npz"))
    test = np.load(os.path.join(FEAT_DIR, f"test_id{suf}.npz"))
    report = {}
    if "validity" in want:
        report["validity"] = sec_validity(scorers, test, cal)
    if "mondrian" in want:
        report["mondrian"] = sec_mondrian(scorers, test)
        report["mondrian_resplit"] = sec_mondrian_resplit(scorers, cal, test)
    if "shift" in want:
        shifts = {}
        for s in (2.5, 0.2):
            p = os.path.join(FEAT_DIR, f"test_shift_{s}{suf}.npz")
            if os.path.exists(p):
                shifts[s] = np.load(p)
        if shifts:
            report["shift"] = sec_shift(scorers, shifts, test=test)
    if "bates" in want:
        report["bates"] = sec_bates(scorers, cal, test, K=cli.K,
                                    out_png=os.path.join(FIG_DIR, "fig_novelty_bates.png"))
    if "controls" in want:
        p = os.path.join(FEAT_DIR, "val_dense.npz")
        if os.path.exists(p):
            report["controls"] = sec_controls(np.load(p), test)
    if "neighbour" in want:
        report["neighbour"] = sec_neighbour(scorers, cal)
    if "figures" in want:
        sec_figures(scorers, test, cal, os.path.join(FIG_DIR, "fig_novelty_uniformity.png"))
    if "runtime" in want:
        sets = {n: os.path.join(FEAT_DIR, f"runtime_{n}{cli.runtime_suffix}")
                for n in ("cutin_causal", "cutin_ood", "nocutin", "cutin_tsw")}
        report["runtime"] = sec_runtime(scorers, test, sets,
                                        out_png=os.path.join(FIG_DIR, "fig_novelty_benchmarks.png"))
    out = os.path.join(CKPT, f"novelty_report{'_' + cli.tag if cli.tag else ''}.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(report)
    with open(out, "w") as fh:
        json.dump(prev, fh, indent=1, default=float)
    print("saved", out)


if __name__ == "__main__":
    main()
