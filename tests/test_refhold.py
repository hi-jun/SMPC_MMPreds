"""refhold (2026-11-20, student through session smpc-mmpreds-82: "이거 꼭 이전 솔루션 써야되냐고 해결해보라고"): the
reference's ego speeds are the measured speed held at every step, in place of the previous solve's plan."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction, OldACCReferenceAdapter, original_nair_acc_config)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402

V_SET = 20.0
H = 15


def adapter(hold=None):
    config = original_nair_acc_config(horizon=H, dt=0.2, desired_speed=V_SET)
    config.free_cruise_no_ramp = True
    if hold is not None:
        config.reference_hold_speed = hold
    return OldACCReferenceAdapter(config)


def braking_plan():
    # a previous solve's x_nominal (scenario, step, [s, v]) that brakes from 15 to 9 m/s over the horizon
    x = np.zeros((1, H + 1, 2))
    x[0, :, 1] = np.linspace(15.0, 9.0, H + 1)
    return x


def lead(gap0=40.0, v_lead=12.0):
    # one lane-keeping lead, a gap above d_safe, at constant speed
    means = np.zeros((1, H + 1, 2))
    means[0, :, 0] = gap0 + v_lead * 0.2 * np.arange(H + 1)
    means[0, :, 1] = v_lead
    return MultimodalLeadPrediction(probabilities=np.array([1.0]), means=means,
                                    covariances=np.zeros((1, H + 1, 2, 2)))


class TestRefHold(unittest.TestCase):

    def test_default_uses_the_previous_plan(self):
        for a in (adapter(), adapter(False)):
            np.testing.assert_array_equal(a.planned_speed([0.0, 15.0]), np.full(H + 1, 15.0))     # first solve: held
            a.remember_planned_speed(braking_plan())
            np.testing.assert_array_equal(a.planned_speed([0.0, 15.0]), braking_plan()[0, :, 1])

    def test_refhold_holds_the_measured_speed(self):
        a = adapter(True)
        a.remember_planned_speed(braking_plan())
        np.testing.assert_array_equal(a.planned_speed([0.0, 15.0]), np.full(H + 1, 15.0))
        np.testing.assert_array_equal(a.planned_speed([0.0, 13.5]), np.full(H + 1, 13.5))

    def test_reference_follows_the_held_speed(self):
        # same remembered braking plan: refhold's reference equals the default's on its first solve (no plan yet);
        # the default's, once the plan is remembered, puts the ego further back and its time gap at the lower planned
        # speeds, so the blend sees more usable gap and asks for more speed from step 1 on
        x0, pred = np.array([0.0, 15.0]), lead()
        first = adapter().generate(x0, pred)
        hold = adapter(True)
        hold.remember_planned_speed(braking_plan())
        np.testing.assert_array_equal(hold.generate(x0, pred).v_ref, first.v_ref)
        planned = adapter()
        planned.remember_planned_speed(braking_plan())
        self.assertTrue(np.all(planned.generate(x0, pred).v_ref[0, 1:] >= first.v_ref[0, 1:]))
        self.assertTrue(np.any(planned.generate(x0, pred).v_ref[0, 1:] > first.v_ref[0, 1:]))

    def test_parser(self):
        self.assertTrue(ACCNairSMPCAgent._parse_reference_hold("acc_nair_smpc_stdan_3int_onerow_noramp_refhold"))
        self.assertTrue(ACCNairSMPCAgent._parse_reference_hold("acc_nair_smpc_refhold_onerow"))
        self.assertFalse(ACCNairSMPCAgent._parse_reference_hold("acc_nair_smpc_stdan_3int_onerow_noramp"))
        self.assertFalse(ACCNairSMPCAgent._parse_reference_hold("acc_nair_smpc_lkhold"))
        self.assertFalse(ACCNairSMPCAgent._parse_reference_hold("acc_nair_smpc_refholdx"))

    def test_config_default_off(self):
        self.assertFalse(original_nair_acc_config().reference_hold_speed)


if __name__ == "__main__":
    unittest.main()
