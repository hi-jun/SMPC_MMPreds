"""noramp (2026-11-19, student: "free-cruise speed ramps 빼고 실험 돌리면 결과 어떤지"): the reference's free-cruise speed
is the set speed at every step in place of the linear ramp from the current speed to it over 2 s."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import OldACCReferenceAdapter, original_nair_acc_config  # noqa: E402
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402

V_SET = 20.0


def adapter(no_ramp=None):
    config = original_nair_acc_config(horizon=15, dt=0.2, desired_speed=V_SET)
    if no_ramp is not None:
        config.free_cruise_no_ramp = no_ramp
    return OldACCReferenceAdapter(config)


class TestNoRamp(unittest.TestCase):

    def test_default_is_the_ramp(self):
        ramp = adapter()._old_speed_ramp(10.0)
        np.testing.assert_allclose(ramp[:10], np.linspace(10.0, V_SET, 10))     # v_set at step 9 = 1.8 s
        np.testing.assert_allclose(ramp[10:], V_SET)
        np.testing.assert_array_equal(ramp, adapter(False)._old_speed_ramp(10.0))

    def test_noramp_is_the_set_speed(self):
        np.testing.assert_array_equal(adapter(True)._old_speed_ramp(10.0), np.full(16, V_SET))
        # above the set speed both give the set speed
        np.testing.assert_array_equal(adapter(True)._old_speed_ramp(25.0), adapter()._old_speed_ramp(25.0))

    def test_parser(self):
        self.assertTrue(ACCNairSMPCAgent._parse_no_speed_ramp("acc_nair_smpc_stdan_3int_onerow_noramp"))
        self.assertTrue(ACCNairSMPCAgent._parse_no_speed_ramp("acc_nair_smpc_noramp_onerow"))
        self.assertFalse(ACCNairSMPCAgent._parse_no_speed_ramp("acc_nair_smpc_stdan_3int_onerow"))
        self.assertFalse(ACCNairSMPCAgent._parse_no_speed_ramp("acc_nair_smpc_norampx"))

    def test_config_default_off(self):
        self.assertFalse(original_nair_acc_config().free_cruise_no_ramp)


if __name__ == "__main__":
    unittest.main()
