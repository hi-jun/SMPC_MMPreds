"""Split-conformal p-values over the intent classes, and the CP gate.

Why p-values instead of the prediction set: with |Lambda| = 3 the set size takes
only three values, which is far too coarse to modulate a gate. The p-values
inside the set construction are continuous on [0, 1] and calibrated, and they
carry the two quantities we actually want (Vovk et al.):

    credibility = max_k p_k        "does the BEST answer even look typical?"
                                   -> small for inputs unlike anything in the
                                      calibration set: a built-in OOD signal
    confidence  = 1 - 2nd max p_k  "how firmly are the rivals excluded?"
    g = confidence * credibility

This is the property a bare evidence-magnitude gate lacks: a classifier can be
confidently WRONG off-manifold (high max-softmax on an input it has never seen)
and any monotone map of the score alone will open the gate. Credibility instead
asks whether the input conforms to the calibration data at all.

Calibration (offline, once): store the nonconformity scores of the TRUE class
over a calibration split that shares no track with training. Runtime: rank the
candidate scores against that array.
"""
import numpy as np


def lac_scores(probs):
    """LAC / THR nonconformity: s(x, k) = 1 - p_k.

    This is the score the CREDIBILITY semantics needs: an unfamiliar input drives
    the softmax toward uniform, so even the best class scores ~1 - 1/K, which
    is larger than almost every calibration score -> few calibration points are
    at least as nonconforming -> p-values collapse -> the gate closes. APS
    (below) is built for adaptive SET SIZES and behaves the opposite way here:
    a near-uniform prediction gives the top class a SMALL cumulative mass,
    which looks highly conforming and would open the gate on exactly the
    inputs we distrust. Verified empirically in the calibration report.
    """
    return 1.0 - np.asarray(probs, dtype=float)


def aps_scores(probs):
    """APS nonconformity (Romano et al. 2020) for every class.

    score(x, k) = total probability mass of the classes ranked at least as
    high as k (deterministic variant: no uniform tie-breaking, so the coverage
    is conservative rather than exact).
    """
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    out = np.empty_like(probs)
    out[order] = cumulative
    return out


SCORES = {"lac": lac_scores, "aps": aps_scores}


def true_class_scores(prob_rows, labels, score="lac"):
    """Calibration scores: the nonconformity of the observed class."""
    fn = SCORES[score]
    return np.array([fn(p)[int(y)] for p, y in zip(prob_rows, labels)])


class ConformalIntentGate:
    """Runtime p-values, gate value and context direction from softmax probs."""

    def __init__(self, cal_scores, score="lac"):
        self.cal = np.sort(np.asarray(cal_scores, dtype=float))
        self.n = len(self.cal)
        self.score_name = score
        self.score_fn = SCORES[score]

    def p_values(self, probs):
        """p_k = (#{i: s_cal_i >= score(x, k)} + 1) / (n + 1)."""
        s = self.score_fn(probs)
        # cal is ascending: #{cal >= s} = n - searchsorted(cal, s, 'left')
        ge = self.n - np.searchsorted(self.cal, s, side="left")
        return (ge + 1.0) / (self.n + 1.0)

    def gate(self, probs):
        """Returns (g, p_values, credibility, confidence)."""
        p = self.p_values(probs)
        order = np.sort(p)[::-1]
        credibility = float(order[0])
        confidence = 1.0 - float(order[1]) if len(order) > 1 else 1.0
        return float(np.clip(confidence * credibility, 0.0, 1.0)), p, \
            credibility, confidence

    def direction(self, probs):
        """Context transition target w_k = p_k / sum_j p_j."""
        p = self.p_values(probs)
        return p / max(p.sum(), 1e-12)


def msp_gate(probs, n_classes=3):
    """Baseline gate from the maximum softmax probability (MSP).

    Hard to beat for OOD detection, so the CP gate has to be compared against
    it; MSP reads only the score and is blind to how unusual the input is.
    """
    probs = np.asarray(probs, dtype=float)
    chance = 1.0 / n_classes
    return float(np.clip((probs.max() - chance) / (1.0 - chance), 0.0, 1.0))


def load_calibration(path):
    data = np.load(path)
    score = str(data["score"]) if "score" in data else "lac"
    return ConformalIntentGate(data["cal_scores"], score=score)
