"""Conformal p-values for OOD scores, with calibration-conditional corrections.

Follows Novello, Dalmau & Andeol, "Out-of-Distribution Detection Should Use
Conformal Prediction (and Vice-versa?)", ICML 2024, which in turn builds on
Bates, Candes, Lei, Romano & Sesia, "Testing for Outliers with Conformal
p-values" (2022). Pure numpy so the same file serves the offline evaluation
(hmg env) and the controller runtime (mmpreds-c0913, python 3.7 / numpy 1.21).

Why p-values instead of a threshold: an OOD score s(x) with "higher = more
novel" and a calibration set of in-distribution scores S_1..S_n give the
marginal conformal p-value

    u(x) = (1 + #{i : S_i >= s(x)}) / (n + 1)                (Novello Eq. 3)

which is a valid p-value for H0 "x is in-distribution": P(u(X) <= t) <= t,
averaged over the calibration draw. Thresholding at alpha therefore bounds the
false-alarm rate by alpha - the guarantee softmax-based CP does not offer here,
because it needs no labels and no assumption about what the model outputs.

Why a correction on top: for ONE fixed calibration set the false-positive rate
at level alpha is a random variable, FPR(alpha; D_cal) ~ Beta(l, n+1-l) with
l = floor((n+1) alpha) (Bates Prop. 1). Calibration-conditional p-values fix
that: with probability >= 1-delta over the calibration draw, P(u_cc(X) <= t) <= t
holds for ALL t simultaneously (Bates Thm. 4). The construction is a monotone
step function h applied to the marginal p-value,

    u_cc = h(u),   h(t) = b_ceil((n+1) t),   0 = b_0 <= b_1 <= ... <= b_n <= b_{n+1} = 1

where the sequence b satisfies P[U_(1) <= b_1, ..., U_(n) <= b_n] >= 1-delta for
uniform order statistics. Four such sequences are implemented (Bates Sec. 3):
Simes (exact, Prop. 2), DKWM (exact, loose), asymptotic (tight, large-n only),
and Monte-Carlo (exact by construction: min of Simes and an asymptotic sequence
whose level is tuned by simulation). Novello et al. use Monte-Carlo.

Conformal FPR / AUROC / FPR@TPR (Novello Sec. 4.4) are the same idea applied to
the empirical FPR curve: FPR^+(tau) = h(FPR_hat(tau)) is a high-probability
upper bound on the true FPR at every threshold, and the ROC / FPR@TPR95 are
recomputed against it.
"""
import os

import numpy as np

# --------------------------------------------------------------------- p-values


def marginal_pvalues(cal_scores, test_scores):
    """u(x) = (1 + #{cal >= s(x)}) / (n + 1). Higher score = more novel = small p."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    s = np.atleast_1d(np.asarray(test_scores, dtype=float))
    n = len(cal)
    # #{cal >= s} = n - #{cal < s} = n - searchsorted(cal, s, 'left')
    ge = n - np.searchsorted(cal, s, side="left")
    return (ge + 1.0) / (n + 1.0)


def randomized_pvalues(cal_scores, test_scores, rng=None):
    """u - V/(n+1), V ~ U(0,1): exactly Uniform(0,1) under exchangeability with
    continuous scores. Use THIS for goodness-of-fit tests of validity; the plain
    marginal p-values live on the grid k/(n+1) and a KS test against a continuous
    uniform flags the discreteness itself once the test set is large."""
    rng = np.random.RandomState(0) if rng is None else rng
    u = marginal_pvalues(cal_scores, test_scores)
    n = len(np.asarray(cal_scores))
    return u - rng.uniform(size=len(u)) / (n + 1.0)


# ------------------------------------------- calibration-conditional sequences


def simes_sequence(n, delta, k=None):
    """Bates Prop. 2 (generalized Simes), evaluated in log space.

    b_{n+1-i} = 1 - delta^{1/k} * (i (i-1) ... (i-k+1) / (n (n-1) ... (n-k+1)))^{1/k},
    i = 1..n; the product has k factors and is 0 (so b = 1) once i < k. Bates use
    k = n/2, which makes the small-index b_i as small as possible.
    """
    from scipy.special import gammaln

    n = int(n)
    if k is None:
        k = max(1, n // 2)
    k = int(k)
    i = np.arange(1, n + 1, dtype=float)
    # log prod_{j=0}^{k-1} (i-j) = lgamma(i+1) - lgamma(i-k+1); the falling
    # factorial in closed form. Materialising the (n, k) matrix instead needs
    # 23 GB at n = 76602, k = n/2.
    log_num = np.full(n, -np.inf)
    ok = i >= k
    if ok.any():
        num = gammaln(i[ok] + 1.0) - gammaln(i[ok] - k + 1.0)
        den = gammaln(n + 1.0) - gammaln(n - k + 1.0)
        log_num[ok] = num - den
    ratio = np.exp(log_num / k)                # (prod)^{1/k}, 0 where i < k
    b_rev = 1.0 - (delta ** (1.0 / k)) * ratio  # this is b_{n+1-i}
    b = b_rev[::-1].copy()                      # b_1..b_n
    return np.clip(b, 0.0, 1.0)


def dkwm_sequence(n, delta):
    """Bates Eq. 13: b_i = min(i/n + sqrt(log(2/delta) / (2n)), 1)."""
    i = np.arange(1, n + 1, dtype=float)
    return np.minimum(i / n + np.sqrt(np.log(2.0 / delta) / (2.0 * n)), 1.0)


def _c_n(n, delta):
    """Constant of the adaptive (Eicker/Jaeschke-type) bound, Bates Sec. 3.4."""
    lln = np.log(np.log(n))
    return (-np.log(-np.log(1.0 - delta)) + 2.0 * lln
            + 0.5 * np.log(lln) - 0.5 * np.log(np.pi)) / np.sqrt(2.0 * lln)


def asymptotic_sequence(n, delta):
    """Bates Eq. 15: b_i = min(i/n + c_n(delta) sqrt(i (n-i)) / (n sqrt n), 1)."""
    n = int(n)
    i = np.arange(1, n + 1, dtype=float)
    c = _c_n(n, delta)
    return np.minimum(i / n + c * np.sqrt(i * (n - i)) / (n * np.sqrt(n)), 1.0)


def _uniform_order_stats(n, n_sim, rng):
    """(n_sim, n) sorted uniforms via the exponential-spacings trick (fast)."""
    e = rng.exponential(size=(n_sim, n + 1))
    return np.cumsum(e[:, :-1], axis=1) / e.sum(axis=1, keepdims=True)


def coverage_probability(b, n_sim=20000, rng=None):
    """MC estimate of P[U_(1) <= b_1, ..., U_(n) <= b_n] for uniform order stats.

    This is condition (10) of Bates Thm. 4 - the only thing a sequence has to
    satisfy for h to yield calibration-conditional p-values. Used both to tune
    the Monte-Carlo sequence and as the unit test for every sequence here.
    """
    rng = np.random.RandomState(0) if rng is None else rng
    b = np.asarray(b, dtype=float)
    n = len(b)
    hits = 0
    chunk = max(1, min(n_sim, int(2e7 // max(n, 1))))
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        U = _uniform_order_stats(n, m, rng)
        hits += int(np.all(U <= b[None, :], axis=1).sum())
        done += m
    return hits / float(n_sim)


def monte_carlo_sequence(n, delta, n_sim=20000, rng=None, tol=1e-3, margin=None):
    """Bates Eq. 16: b^m = min(b^s, b^{a, delta_hat}) with delta_hat tuned by MC.

    The asymptotic sequence at nominal delta is not guaranteed at finite n, so
    its level delta_hat is found by bisection such that the min-sequence has
    coverage >= 1 - delta in simulation. Feasible because delta_hat -> 0 makes
    the asymptotic sequence loose and the min reduces to Simes, which is exact.
    Bates' released code additionally straightens the envelope above t = 1/2;
    that region only touches p-values nobody would reject, so it is omitted.
    """
    rng = np.random.RandomState(0) if rng is None else rng
    b_s = simes_sequence(n, delta)
    # The bisection keeps the LARGEST delta_hat whose simulated coverage passes,
    # so it systematically picks lucky draws (winner's curse) and an independent
    # re-check lands a little under 1 - delta. Require a margin of ~2 standard
    # errors of the MC estimate to compensate.
    if margin is None:
        margin = 2.0 * np.sqrt(delta * (1.0 - delta) / n_sim)
    lo, hi = 1e-6, min(0.5, 1.0 - 1e-6)     # delta_hat range
    best = b_s.copy()
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        b_try = np.minimum(b_s, asymptotic_sequence(n, mid))
        cov = coverage_probability(b_try, n_sim=n_sim, rng=rng)
        if cov >= 1.0 - delta + margin:
            best = b_try
            lo = mid                          # can afford a larger delta_hat
        else:
            hi = mid
        if hi - lo < tol:
            break
    return best


SEQUENCES = {
    "simes": simes_sequence,
    "dkwm": dkwm_sequence,
    "asymptotic": asymptotic_sequence,
    "monte_carlo": monte_carlo_sequence,
}


class ConditionalCorrection:
    """h(t) = b_ceil((n+1) t) with b_0 = 0, b_{n+1} = 1 (Bates Thm. 4)."""

    def __init__(self, b):
        b = np.asarray(b, dtype=float)
        self.n = len(b)
        self.b_ext = np.concatenate([[0.0], b, [1.0]])

    @classmethod
    def build(cls, n, delta, method="monte_carlo", **kw):
        return cls(SEQUENCES[method](n, delta, **kw))

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        idx = np.ceil((self.n + 1) * t).astype(int)
        idx = np.clip(idx, 0, self.n + 1)
        return self.b_ext[idx]


# ------------------------------------------------ conformal FPR / AUROC / FPR@TPR


def empirical_fpr(cal_scores, thresholds):
    """FPR_hat(tau) = (1/n) #{cal >= tau}  (higher score = alarm)."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    tau = np.atleast_1d(np.asarray(thresholds, dtype=float))
    return (len(cal) - np.searchsorted(cal, tau, side="left")) / float(len(cal))


def conformal_fpr(cal_scores, thresholds, correction):
    """FPR^+(tau) = h(FPR_hat(tau)): high-probability upper bound (Novello Eq. 10)."""
    return correction(empirical_fpr(cal_scores, thresholds))


def roc_points(cal_scores, ood_scores, correction=None):
    """(fpr, tpr, thresholds) sweeping every distinct score. If a correction is
    given, fpr is the conformal FPR instead of the empirical one."""
    cal = np.asarray(cal_scores, dtype=float)
    ood = np.asarray(ood_scores, dtype=float)
    thr = np.unique(np.concatenate([cal, ood]))[::-1]      # high -> low
    thr = np.concatenate([[np.inf], thr, [-np.inf]])
    tpr = (len(ood) - np.searchsorted(np.sort(ood), thr, side="left")) / float(len(ood))
    fpr = empirical_fpr(cal, thr)
    if correction is not None:
        fpr = correction(fpr)
    return fpr, tpr, thr


def auroc(fpr, tpr):
    order = np.argsort(fpr, kind="mergesort")
    f, t = np.asarray(fpr)[order], np.asarray(tpr)[order]
    return float(np.trapz(t, f))


def fpr_at_tpr(fpr, tpr, beta=0.95):
    """Smallest FPR among thresholds achieving TPR >= beta."""
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    ok = tpr >= beta
    return float(fpr[ok].min()) if ok.any() else 1.0


def ood_metrics(cal_scores, ood_scores, correction=None, beta=0.95):
    """Classical and (if correction given) conformal AUROC / FPR@TPR95 together."""
    out = {}
    f, t, _ = roc_points(cal_scores, ood_scores)
    out["auroc"] = auroc(f, t)
    out["fpr95tpr"] = fpr_at_tpr(f, t, beta)
    if correction is not None:
        fc, tc, _ = roc_points(cal_scores, ood_scores, correction)
        out["conf_auroc"] = auroc(fc, tc)
        out["conf_fpr95tpr"] = fpr_at_tpr(fc, tc, beta)
    return out


# ------------------------------------------------------- runtime score objects


class MahalanobisNovelty:
    """Lee et al. 2018 as implemented by oodeel: per-class means, one tied
    covariance, score = -max_c log N(z; mu_c, Sigma) (higher = more novel).

    Stored as plain arrays so the planner env can load it with np.load: the
    fitting env pins a newer scikit-learn / joblib than the planner env.
    """

    def __init__(self, mu, prec, cal_scores, delta=0.05, method="monte_carlo",
                 b_seq=None, feature=None, classes=None, mondrian=None):
        self.mu = np.asarray(mu, dtype=float)              # (C, d)
        self.prec = np.asarray(prec, dtype=float)          # (d, d)
        self.cal = np.sort(np.asarray(cal_scores, dtype=float))
        self.delta = float(delta)
        self.method = method
        self.feature = feature
        self.classes = classes
        b = (np.asarray(b_seq, dtype=float) if b_seq is not None
             else SEQUENCES[method](len(self.cal), delta))
        self.correction = ConditionalCorrection(b)
        self.mondrian = mondrian

    # oodeel Mahalanobis._gaussian_log_probs, verbatim semantics
    def class_log_probs(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        out = np.empty((len(z), len(self.mu)))
        for c in range(len(self.mu)):
            d = z - self.mu[c]
            out[:, c] = -0.5 * np.einsum("ij,jk,ik->i", d, self.prec, d)
        return out

    def score(self, z):
        return -self.class_log_probs(z).max(axis=1)

    def p_values(self, z, groups=None):
        """Marginal p-values, or group-conditional ones when `groups` is given
        and a Mondrian calibration was stored."""
        s = self.score(z)
        if groups is not None and self.mondrian is not None:
            return self.mondrian.p_values(s, groups)
        u = marginal_pvalues(self.cal, s)
        return u, self.correction(u)

    def save(self, path):
        extra = self.mondrian.to_dict() if self.mondrian is not None else {}
        np.savez(path, kind="mahalanobis", mu=self.mu, prec=self.prec,
                 cal_scores=self.cal, delta=self.delta, method=self.method,
                 b_seq=self.correction.b_ext[1:-1],
                 feature=str(self.feature), classes=np.asarray(
                     self.classes if self.classes is not None else []), **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["mu"], d["prec"], d["cal_scores"], float(d["delta"]),
                   str(d["method"]), b_seq=d["b_seq"], feature=str(d["feature"]),
                   classes=d["classes"] if "classes" in d else None,
                   mondrian=MondrianCalibration.from_dict(d))


class KnnNovelty:
    """Sun et al. 2022 as implemented by oodeel DKNN: L2-normalize, distance to
    the k-th nearest reference vector (squared L2, faiss IndexFlatL2 convention).
    Brute force numpy; fine for a bank of a few 10^4 and batch size 1."""

    def __init__(self, bank, k, cal_scores, delta=0.05, method="monte_carlo",
                 b_seq=None, feature=None, mondrian=None):
        bank = np.asarray(bank, dtype=np.float32)
        self.bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-10)
        self.k = int(k)
        self.cal = np.sort(np.asarray(cal_scores, dtype=float))
        self.delta = float(delta)
        self.method = method
        self.feature = feature
        b = (np.asarray(b_seq, dtype=float) if b_seq is not None
             else SEQUENCES[method](len(self.cal), delta))
        self.correction = ConditionalCorrection(b)
        self.mondrian = mondrian
        self._bank_sq = (self.bank ** 2).sum(axis=1)

    def score(self, z, chunk=1024):
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
        if len(z) > 4096:
            fast = self._score_torch(z)
            if fast is not None:
                return fast
        out = np.empty(len(z), dtype=float)
        # chunk over queries: a (queries x bank) distance matrix for 50k x 120k
        # would be 25 GB; 1024 x 120k is ~0.5 GB
        for a in range(0, len(z), chunk):
            zb = z[a:a + chunk]
            d2 = (zb ** 2).sum(axis=1)[:, None] + self._bank_sq[None, :] - 2.0 * zb @ self.bank.T
            d2 = np.maximum(d2, 0.0)
            out[a:a + chunk] = np.partition(d2, self.k - 1, axis=1)[:, self.k - 1]
        return out

    def _score_torch(self, z, chunk=8192):
        """Same computation on the GPU when torch is around (offline eval only;
        the planner runtime scores one vector at a time with numpy)."""
        try:
            import torch
        except Exception:
            return None
        if not torch.cuda.is_available():
            return None
        dev = torch.device("cuda")
        bank = torch.as_tensor(self.bank, device=dev)
        bank_sq = torch.as_tensor(self._bank_sq, device=dev)
        out = np.empty(len(z), dtype=float)
        with torch.no_grad():
            for a in range(0, len(z), chunk):
                zb = torch.as_tensor(z[a:a + chunk], device=dev)
                d2 = (zb * zb).sum(1)[:, None] + bank_sq[None, :] - 2.0 * zb @ bank.T
                d2 = torch.clamp(d2, min=0.0)
                kth = torch.topk(d2, self.k, dim=1, largest=False).values[:, -1]
                out[a:a + chunk] = kth.double().cpu().numpy()
        return out

    def p_values(self, z, groups=None):
        s = self.score(z)
        if groups is not None and self.mondrian is not None:
            return self.mondrian.p_values(s, groups)
        u = marginal_pvalues(self.cal, s)
        return u, self.correction(u)

    def save(self, path):
        extra = self.mondrian.to_dict() if self.mondrian is not None else {}
        np.savez(path, kind="knn", bank=self.bank, k=self.k, cal_scores=self.cal,
                 delta=self.delta, method=self.method,
                 b_seq=self.correction.b_ext[1:-1], feature=str(self.feature),
                 **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(d["bank"], int(d["k"]), d["cal_scores"], float(d["delta"]),
                   str(d["method"]), b_seq=d["b_seq"], feature=str(d["feature"]),
                   mondrian=MondrianCalibration.from_dict(d))


# Selected detector: Mahalanobis on [agent | social] (128-d), calibrated
# Mondrian by the predictor's argmax class on a 1 s-thinned ValSet. Chosen on
# three axes measured in eval_novelty.py (see REPORT_conformal_novelty.md):
#   validity   per-class alarm rate 0.049 / 0.054 / 0.053 at alpha = 0.05, the
#              tightest of the 13 candidates (n_cal 113423 / 2171 / 599)
#   real OOD   flags the 32 m/s vehicles 100.0% and the 24 m/s ones 0.0%;
#              knn_social, which wins the synthetic tensor-scaling probe
#              (0.92 vs 0.62), only reaches 23.6% on the same vehicles
#   no false alarms  0.045 / 0.041 / 0.068 on the normal-speed benchmarks
# (those numbers were measured on graduate_project's older position-output
# trunk). The file shipped here is the same detector refitted on the trunk this
# repository runs as stdan_vel (predictor/stdan_vel/checkpoints/0113_ratio211,
# md5 d52c2339): Mondrian calibration n = 71568 / 3357 / 1677. See AGENTS.md
# in this directory.
DEFAULT_NOVELTY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ckpt", "novelty_maha_concat_aug_g1.npz")


def load_novelty(path=None):
    path = DEFAULT_NOVELTY if path is None else path
    d = np.load(path, allow_pickle=False)
    kind = str(d["kind"])
    return MahalanobisNovelty.load(path) if kind == "mahalanobis" else KnnNovelty.load(path)


# ------------------------------------------------- Mondrian (group-conditional)


class MondrianCalibration:
    """One calibration array per group -> group-conditional p-values.

    Marginal calibration on driving data inherits the class imbalance of the
    road: 94% of NGSIM samples are lane keeping, so "how far is this from the
    usual" is really "how far is this from a lane-keeping vehicle". Measured on
    the reference data itself, samples the predictor calls a right lane change
    alarm 71.9% of the time at alpha = 0.05 - they are not novel, they are rare.
    That is the same marginal-vs-conditional failure the softmax gate had
    (0.912 coverage overall, 0.516 in the maneuver phase).

    Splitting the calibration set by a group that is OBSERVABLE at run time -
    here the predictor's own argmax class - restores

        P(p(X) <= alpha | group(X) = g) <= alpha   for every g

    at the cost of a smaller n per group (and hence a coarser p-value grid and a
    wider calibration fluctuation for the rare groups).
    """

    def __init__(self, cal_by_group, delta=0.05, method="monte_carlo",
                 b_by_group=None, min_n=30):
        self.groups = sorted(cal_by_group)
        self.cal = {g: np.sort(np.asarray(v, dtype=float))
                    for g, v in cal_by_group.items()}
        self.delta = float(delta)
        self.method = method
        self.min_n = int(min_n)
        # groups too small for their own quantile fall back to the pooled set
        self.pooled = np.sort(np.concatenate([self.cal[g] for g in self.groups]))
        self.corrections = {}
        for g in self.groups:
            n = len(self.cal[g])
            if n < self.min_n:
                continue
            b = (np.asarray(b_by_group[g], dtype=float)
                 if b_by_group is not None and g in b_by_group
                 else SEQUENCES[method](n, delta))
            self.corrections[g] = ConditionalCorrection(b)
        self.pooled_correction = ConditionalCorrection(
            SEQUENCES[method](len(self.pooled), delta))

    def n_of(self, g):
        return len(self.cal.get(g, ()))

    def p_values(self, scores, groups):
        """Marginal-within-group p-values and their calibration-conditional form."""
        scores = np.atleast_1d(np.asarray(scores, dtype=float))
        groups = np.atleast_1d(np.asarray(groups))
        u = np.empty(len(scores)); ucc = np.empty(len(scores))
        for g in np.unique(groups):
            m = groups == g
            usable = g in self.cal and len(self.cal[g]) >= self.min_n
            cal = self.cal[g] if usable else self.pooled
            corr = self.corrections.get(g, self.pooled_correction)
            u[m] = marginal_pvalues(cal, scores[m])
            ucc[m] = corr(u[m])
        return u, ucc

    def to_dict(self):
        out = {"mondrian_groups": np.asarray(self.groups),
               "mondrian_delta": self.delta, "mondrian_method": self.method,
               "mondrian_min_n": self.min_n}
        for g in self.groups:
            out[f"mondrian_cal_{g}"] = self.cal[g]
        return out

    @classmethod
    def from_dict(cls, d):
        if "mondrian_groups" not in d:
            return None
        groups = [int(g) for g in d["mondrian_groups"]]
        return cls({g: d[f"mondrian_cal_{g}"] for g in groups},
                   delta=float(d["mondrian_delta"]),
                   method=str(d["mondrian_method"]),
                   min_n=int(d["mondrian_min_n"]))


# ------------------------------------------------------------ validity checks


def validity_report(cal_scores, test_scores, alphas=(0.01, 0.05, 0.1)):
    """Is the calibration/test pair exchangeable, and does the FPR behave?

    A KS test of the marginal p-values against Uniform(0,1) is the WRONG check
    once the test set is large: conditional on one calibration draw the p-value
    CDF deviates from the diagonal by O(1/sqrt(n_cal)) - the Beta fluctuation of
    Bates Prop. 1 - and a big test set detects that every time (measured: 75% of
    iid draws rejected at n_cal=5000, n_test=20000). The right checks are:

      1. two-sample KS between calibration and test SCORES - exchangeability of
         the two sets is exactly "same score distribution";
      2. the alarm rate at each alpha against the Beta(l, n+1-l) 95% interval,
         l = floor((n+1) alpha) - what a single fixed calibration set can give.
    """
    from scipy.stats import ks_2samp, beta

    cal = np.asarray(cal_scores, dtype=float)
    u = marginal_pvalues(cal, test_scores)
    n = len(cal)
    out = {"n_cal": int(n), "n_test": int(len(u)),
           "ks2_p": float(ks_2samp(cal, np.asarray(test_scores, dtype=float)).pvalue),
           "mean_p": float(u.mean()), "alphas": {}}
    for a in alphas:
        l = int(np.floor((n + 1) * a))
        lo, hi = (beta.ppf([0.025, 0.975], l, n + 1 - l) if l >= 1 else (0.0, 1.0))
        rate = float(np.mean(u <= a))
        out["alphas"][str(a)] = {"alarm_rate": rate, "beta95": [float(lo), float(hi)],
                                 "inside": bool(lo <= rate <= hi)}
    return out
