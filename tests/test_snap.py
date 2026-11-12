import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestSnap(unittest.TestCase):
    """``snap<r>`` with ``jinput``: the cost ``3 P_j r sum_{k=0}^{N-1} (j_k - j_{k-1})^2``, ``j_{-1}`` the jerk the plant
    got on the last tick from the issued commands (0 before there are two)."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
             "_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold"
             "_entrant2_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap_snap40")

    def _config(self, snap=40.0, **overrides):
        """Arm C's cost and limits (onerow w 10 + zconf0.95 + nocap) with snap<r>; one scenario."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1, controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 10.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
        config.failsafe_recovery = True
        config.failsafe_decel, config.failsafe_jerk = -8.0, 10.0
        config.one_row = True
        config.one_row_z_conf = 0.95
        config.one_row_no_cap = True
        config.snap_weight = snap
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _free_road(self):
        """No lead in the ego lane (every cell inactive)."""
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = 1000.0
        means[0, :, 1] = 20.0
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]), covariances=np.tile(np.diag([0.25, 0.09]), (1, steps, 1, 1)),
            active_mask=np.zeros((1, steps), dtype=bool), chance_confidence=np.full((1, steps), np.nan),
            sigma_confidence=np.full((1, steps), np.nan), lead_mode_lk=np.zeros((1, steps), dtype=bool),
            lead_vehicle_ids=np.full((1, steps), -1, dtype=int), clearance_scale=np.ones((1, steps)),
            lead_mode_probability=np.ones((1, steps)), mode_names=["free"])

    def _two_ticks(self, config, a_prev2=0.0, a_prev=0.1, v=12.0):
        """Two solves: the first with the issued command a_prev2, the second with a_prev (so j_{-1} = (a_prev - a_prev2)
        / command_dt at the second)."""
        controller = NairACCSMPC(config)
        first = controller.solve(np.array([0.0, v]), self._free_road(), previous_accel_cmd=a_prev2,
                                 command_dt=self.COMMAND_DT, ego_accel=a_prev2)
        second = controller.solve(np.array([v * 0.1, v]), self._free_road(), previous_accel_cmd=a_prev,
                                  command_dt=self.COMMAND_DT, ego_accel=a_prev)
        for solution in (first, second):
            self.assertNotEqual(solution.solve_path, "fallback")
        return controller, first, second

    def test_token_parse_and_validation(self):
        parse = ACCNairSMPCAgent._parse_snap
        self.assertEqual(parse(self.AGENT), 40.0)
        self.assertEqual(parse("jinput_snap160_ra0"), 160.0)
        self.assertIsNone(parse(self.AGENT.replace("_snap40", "")))
        self.assertIsNone(parse("acc_x_xsnap40"))
        self.assertIsNone(parse("acc_x_snap40x"))
        self.assertEqual(self._config().snap_weight, 40.0)
        with self.assertRaises(ValueError):            # needs jinput
            self._config(jerk_input=False)
        with self.assertRaises(ValueError):            # positive
            self._config(snap=0.0)
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=self.AGENT.replace("_jinput", ""))
        self.assertIn("snap", str(caught.exception))

    def test_previous_jerk_from_the_issued_commands(self):
        controller, _, _ = self._two_ticks(self._config(), a_prev2=0.0, a_prev=0.1)
        self.assertAlmostEqual(controller._prev_issued_jerk, 1.0, places=12)
        # the first tick has no previous command pair: j_{-1} = 0
        fresh = NairACCSMPC(self._config())
        fresh.solve(np.array([0.0, 12.0]), self._free_road(), previous_accel_cmd=0.7, command_dt=self.COMMAND_DT,
                    ego_accel=0.7)
        self.assertEqual(fresh._prev_issued_jerk, 0.0)

    def test_cost_term(self):
        # the objective equals the base terms at the solved plan plus 3 P r sum (j_k - j_{k-1})^2
        r = 40.0
        config = self._config(snap=r)
        _, _, solution = self._two_ticks(config)
        j = np.asarray(solution.j_nominal[0], dtype=float)
        x = np.asarray(solution.x_nominal[0], dtype=float)       # [s, v] per step
        a = np.asarray(solution.u_nominal[0], dtype=float)       # the planned accelerations a_1..a_N
        ref = solution.reference
        base = 3.0 * (config.q_s * np.sum((x[:, 0] - ref.x_ref[0, :, 0]) ** 2)
                      + config.q_v * np.sum((x[:, 1] - ref.x_ref[0, :, 1]) ** 2)
                      + config.r_a * np.sum((a - ref.a_ref[0]) ** 2) + config.r_jerk * np.sum(j ** 2))
        snap = 3.0 * r * np.sum(np.diff(np.concatenate(([1.0], j))) ** 2)
        self.assertAlmostEqual(solution.cost, base + snap, delta=1.0e-4 * max(1.0, abs(solution.cost)))

    def test_snap_pulls_the_first_jerk_toward_the_last(self):
        # j_{-1} = 1 m/s^3 on a free road below the set speed: the larger r, the closer j_0 to 1 (measured: j_0 0.67 without
        # the token, 0.76 at r 40, 0.82 at r 160, 0.93 at r 1e5; holding j = 1 over the horizon would pass a_max)
        gaps = []
        for snap in (None, 40.0, 160.0, 1.0e5):
            _, _, solution = self._two_ticks(self._config(snap=snap))
            gaps.append(abs(float(solution.j_nominal[0, 0]) - 1.0))
        self.assertTrue(all(earlier > later for earlier, later in zip(gaps, gaps[1:])), gaps)
        self.assertLess(gaps[3], 0.25 * gaps[0])


if __name__ == "__main__":
    unittest.main()
