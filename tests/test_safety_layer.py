import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.pcacc import VehicleObs  # noqa: E402
from utils.safety_layer import (  # noqa: E402
    PATH_FAILSAFE,
    PATH_ICS,
    PATH_NOMINAL,
    SafetyLayer,
    held_command_safe_distance,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


def plant(s, v, a, dt):
    """The replay's ideal plant: the command held over the tick, speed clipped at 0, trapezoidal position."""
    v_next = max(0.0, v + a * dt)
    return s + 0.5 * (v + v_next) * dt, v_next


def closed_loop(v0, vl0, gap0, nominal, lead_accel, dt, steps, prev0=0.0):
    """One ego behind one lead under the layer. ``nominal(k, v, gap)`` and ``lead_accel(k, vl)`` give the
    commands. Returns the smallest bumper gap and the path counts."""
    layer = SafetyLayer(a_max=2.0)
    s_e, v_e, s_l, v_l, prev = 0.0, v0, gap0, vl0, prev0
    gap_min = gap0
    paths = {PATH_NOMINAL: 0, PATH_FAILSAFE: 0, PATH_ICS: 0}
    for k in range(steps):
        gap = s_l - s_e
        dec = layer.step(k * dt, v_e, prev, nominal(k, v_e, gap), [VehicleObs(1, gap, v_l, True)], dt)
        paths[dec.path] += 1
        s_e, v_e = plant(s_e, v_e, dec.accel, dt)
        s_l, v_l = plant(s_l, v_l, lead_accel(k, v_l), dt)
        prev = dec.accel
        gap_min = min(gap_min, s_l - s_e)
    return gap_min, paths


class SafeDistanceTests(unittest.TestCase):

    def test_documented_guaranteed_gaps(self):
        cfg = SafetyLayer(2.0).cfg
        for v, expect in ((10.0, 4.4), (15.0, 7.8), (20.0, 11.9)):
            self.assertAlmostEqual(held_command_safe_distance(v, 0.0, v, cfg, 0.1), expect, delta=0.05)

    def test_held_command_follows_the_plant(self):
        # stopping inside the interval: the plant moves 0.5 * v * dt, not v dt + a dt^2 / 2; plus the stop-tick term
        cfg = SafetyLayer(2.0).cfg
        need = held_command_safe_distance(0.2, -8.0, 0.0, cfg, 0.1)
        self.assertAlmostEqual(need, 0.5 * 0.2 * 0.1 + 8.0 * 0.1 ** 2 / 8.0, places=9)


class GuaranteeTests(unittest.TestCase):

    def test_adversarial_nominal_behind_a_lead_braking_at_its_limit(self):
        # the nominal controller always accelerates; the lead brakes at -10.5 from the first tick
        for dt in (0.1, 0.05):
            for v in (5.0, 12.0, 20.0, 25.0):
                cfg = SafetyLayer(2.0).cfg
                gap0 = held_command_safe_distance(v, 0.0, v, cfg, dt) + 1.0e-3
                gap_min, paths = closed_loop(v, v, gap0, lambda k, ve, g: 2.0, lambda k, vl: -10.5, dt, int(12 / dt))
                self.assertGreaterEqual(gap_min, -1.0e-9, "dt %g v %g" % (dt, v))
                self.assertGreater(paths[PATH_FAILSAFE], 0)

    def test_random_leads_and_random_nominal_commands(self):
        rng = np.random.default_rng(20261102)
        for trial in range(400):
            dt = 0.1 if trial % 2 else 0.05
            v0, vl0 = rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0)
            cfg = SafetyLayer(2.0).cfg
            gap0 = max(0.0, held_command_safe_distance(v0, 0.0, vl0, cfg, dt)) + rng.uniform(0.0, 5.0)
            lead_seq = rng.uniform(-10.5, 2.0, size=10)          # piecewise-constant lead accelerations
            nom_seq = rng.uniform(-3.0, 2.0, size=10)            # piecewise-constant nominal commands
            steps = int(15 / dt)
            gap_min, _ = closed_loop(
                v0, vl0, gap0,
                lambda k, ve, g: float(nom_seq[min(9, k * 10 // steps)]),
                lambda k, vl: float(lead_seq[min(9, k * 10 // steps)]),
                dt, steps)
            self.assertGreaterEqual(gap_min, -1.0e-9, "trial %d" % trial)

    def test_standstill_creep_stops_before_contact(self):
        # stopped 5 cm behind a stopped lead (first seen outside its 1 cm requirement), the nominal keeps asking +0.1
        gap_min, paths = closed_loop(0.0, 0.0, 0.05, lambda k, ve, g: 0.1, lambda k, vl: 0.0, 0.1, 300)
        self.assertGreaterEqual(gap_min, 0.0)
        self.assertGreater(paths[PATH_FAILSAFE], 0)


class ModeTests(unittest.TestCase):

    def test_nominal_passes_through_when_far(self):
        layer = SafetyLayer(2.0)
        dec = layer.step(0.0, 20.0, 0.0, 1.5, [VehicleObs(1, 60.0, 20.0, True)], 0.1)
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertEqual(dec.accel, 1.5)
        self.assertEqual(dec.checked_vids, [1])
        self.assertGreater(dec.margin, 0.0)

    def test_non_members_are_ignored(self):
        layer = SafetyLayer(2.0)
        dec = layer.step(0.0, 20.0, 0.0, 2.0, [VehicleObs(1, 1.0, 0.0, False)], 0.1)
        self.assertEqual((dec.path, dec.accel, dec.checked_vids), (PATH_NOMINAL, 2.0, []))

    def test_failsafe_follows_the_profile_and_releases(self):
        layer = SafetyLayer(2.0)
        near = [VehicleObs(1, 10.0, 10.0, True)]
        layer.step(0.0, 20.0, 0.0, 0.0, [VehicleObs(1, 60.0, 10.0, True)], 0.1)     # first seen far: covered
        first = layer.step(0.1, 20.0, 0.0, 0.0, near, 0.1)
        self.assertEqual(first.path, PATH_FAILSAFE)
        self.assertAlmostEqual(first.accel, -2.0)                 # 0 - 20 m/s^3 * 0.1 s
        second = layer.step(0.2, 19.8, first.accel, 0.0, near, 0.1)
        self.assertAlmostEqual(second.accel, -4.0)                # the same profile, not restarted
        released = layer.step(0.3, 19.5, second.accel, 0.0, [VehicleObs(1, 60.0, 19.5, True)], 0.1)
        self.assertEqual((released.path, released.accel), (PATH_NOMINAL, 0.0))
        again = layer.step(0.4, 19.5, 0.0, 0.0, near, 0.1)
        self.assertAlmostEqual(again.accel, -2.0)                 # a new engagement starts a new profile

    def test_a_harder_nominal_command_is_kept(self):
        layer = SafetyLayer(2.0)
        layer.step(0.0, 20.0, 0.0, 0.0, [VehicleObs(1, 60.0, 10.0, True)], 0.1)
        dec = layer.step(0.1, 20.0, 0.0, -3.0, [VehicleObs(1, 10.0, 10.0, True)], 0.1)
        self.assertEqual(dec.path, PATH_FAILSAFE)
        self.assertEqual(dec.accel, -3.0)

    def test_entry_inside_the_distance_is_a_cut_in_and_an_ics_brakes_fully(self):
        layer = SafetyLayer(2.0)
        layer.step(0.0, 15.0, 0.0, 0.5, [VehicleObs(7, 2.0, 5.0, False)], 0.1)       # adjacent, not a member
        dec = layer.step(0.1, 15.0, 0.5, 0.5, [VehicleObs(7, 1.0, 5.0, True)], 0.1)  # enters 1 m ahead, 10 m/s slower
        self.assertEqual(dec.cutin_vids, [7])
        self.assertEqual(dec.checked_vids, [])
        self.assertEqual(dec.path, PATH_ICS)
        self.assertAlmostEqual(dec.accel, 0.5 - 2.0)

    def test_cut_in_without_ics_keeps_the_nominal_and_is_released_at_the_distance(self):
        layer = SafetyLayer(2.0)
        layer.step(0.0, 10.0, 0.0, 0.0, [VehicleObs(7, 2.0, 12.0, False)], 0.1)
        dec = layer.step(0.1, 10.0, 0.0, 0.0, [VehicleObs(7, 2.0, 12.0, True)], 0.1)   # faster than the ego
        self.assertEqual((dec.cutin_vids, dec.path, dec.accel), ([7], PATH_NOMINAL, 0.0))
        dec = layer.step(0.2, 10.0, 0.0, 0.0, [VehicleObs(7, 30.0, 12.0, True)], 0.1)
        self.assertEqual((dec.cutin_vids, dec.checked_vids), ([], [7]))

    def test_first_sight_inside_the_distance_is_a_cut_in(self):
        # e.g. a vehicle already in the path 1 m ahead at the start: outside the guarantee, no ICS at equal speeds
        layer = SafetyLayer(2.0)
        dec = layer.step(0.0, 15.0, 0.0, 0.0, [VehicleObs(7, 1.0, 15.0, True)], 0.1)
        self.assertEqual((dec.cutin_vids, dec.checked_vids, dec.path), ([7], [], PATH_NOMINAL))

    def test_first_sight_outside_the_distance_is_covered(self):
        layer = SafetyLayer(2.0)
        dec = layer.step(0.0, 15.0, 0.0, 0.0, [VehicleObs(7, 40.0, 15.0, True)], 0.1)
        self.assertEqual((dec.cutin_vids, dec.checked_vids, dec.path), ([], [7], PATH_NOMINAL))


class TokenTests(unittest.TestCase):

    def test_whole_token(self):
        parse = ACCNairSMPCAgent._parse_safety_layer
        self.assertTrue(parse("acc_nair_smpc_stdan_3int_jinput_ra0_fullfoot_l1slack1e6_safelayer"))
        self.assertTrue(parse("acc_nair_smpc_safelayer_rmove1000"))
        self.assertFalse(parse("acc_nair_smpc_stdan_3int_jinput_ra0_fullfoot_l1slack1e6"))
        self.assertFalse(parse("acc_nair_smpc_safelayers"))


if __name__ == "__main__":
    unittest.main()
