"""Encoder feature taps for the 9-mode STDAN-vel trunk, and the runtime novelty p-value.

Ported 2026-10-13 from graduate_project/beliefplanning:
  StreamTap       stdan/training/isap_head.py (verbatim)
  encoder_inputs  stdan/training/train_isap_head.py (verbatim)
  StdanNovelty    the novelty path that was inline in stdan/stdan_predictor.py::_run_model
"""
import numpy as np

from predictor.uq.conformal_ood import load_novelty


class StreamTap:
    """Forward hooks extracting the two semantic streams from a GDEncoder
    without modifying the model: the target-history LSTM encoding (first of
    the two LSTM calls per forward) and the social-attention GLU output
    (pre-residual). Works for both the training model copy and the planner's
    inference copy (identical architecture)."""

    def __init__(self, gd_encoder):
        self._lstm_calls = 0
        self._hist = None
        self._social = None
        self._handles = [
            gd_encoder.lstm.register_forward_hook(self._on_lstm),
            gd_encoder.first_glu.register_forward_hook(self._on_glu),
        ]

    def _on_lstm(self, module, inputs, output):
        if self._lstm_calls % 2 == 0:      # call order: target hist, then nbrs
            self._hist = output[0]          # (T, B, D)
        self._lstm_calls += 1

    def _on_glu(self, module, inputs, output):
        self._social = output[0]            # (B, T, D)

    def features(self):
        """(agent_feat (B, D), social_feat (B, D)) from the last forward."""
        agent = self._hist.permute(1, 0, 2)[:, -1, :]
        social = self._social[:, -1, :]
        return agent, social

    def remove(self):
        for h in self._handles:
            h.remove()


def encoder_inputs(batch, device):
    (hist, nbrs, mask, lat_enc, lon_enc, _fv, _fp, _om, va, nbrsva, lane,
     nbrslane, _dis, _nbrsdis, cls_, nbrscls, _mp) = batch
    to = lambda x: x.to(device)
    return ((to(hist), to(nbrs), to(mask), to(va), to(nbrsva), to(lane),
             to(nbrslane), to(cls_), to(nbrscls)), to(lat_enc))


class StdanNovelty:
    """Conformal novelty p-value of every forward pass of a GDEncoder/Generator pair.

    z = [agent | social] (128-d) from StreamTap, scored by the detector. The
    Mondrian group is the predictor's own argmax intent class (the op_lat
    logits), the taxonomy the detector was calibrated on. Attach once, run the
    model as usual, then read result(). The model must run in eval mode
    (mc_dropout off): the calibration features were extracted with dropout off.
    """

    def __init__(self, gd_encoder, generator, detector=None):
        if detector is None or isinstance(detector, str):
            detector = load_novelty(detector)
        self.detector = detector
        self.tap = StreamTap(gd_encoder)
        self._logits = None
        self._hook = generator.op_lat.register_forward_hook(self._on_logits)

    def _on_logits(self, module, inputs, output):
        self._logits = output.detach()

    def result(self):
        """p, p_cc (calibration-conditional), score and group per batch row
        of the last forward. Small p = unfamiliar input; p <= alpha is an alarm."""
        agent, social = self.tap.features()
        z = np.concatenate([agent.cpu().numpy(), social.cpu().numpy()], axis=1)
        g = self._logits.argmax(-1).cpu().numpy().astype(int)
        u, ucc = self.detector.p_values(z, g)
        return {"p": u, "p_cc": ucc, "score": self.detector.score(z), "group": g}

    def remove(self):
        self.tap.remove()
        self._hook.remove()
