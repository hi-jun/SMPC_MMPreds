import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import SAFETY_CONFIDENCE_CHANCE  # noqa: E402
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


NL_TAIL = ("_trajcutin_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer"
           "_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_pingate_floorrisk0.1_floorhyp"
           "_nopricesigma_lkonesided0.05")
HEAD = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8"


class TestChanceCells(unittest.TestCase):
    """``chancecells``: ``cutin_chance0.6`` under a name without the beta_ref that the NGSIM design no longer uses."""

    def test_reference_confidence(self):
        parse = ACCNairSMPCAgent._parse_cutin_chance_ref
        self.assertEqual(parse(HEAD + "_chancecells_dopt100lkr" + NL_TAIL), 0.6)
        self.assertEqual(parse(HEAD + "_cutin_chance0.7_dopt100lkr"), 0.7)
        self.assertEqual(parse(HEAD + "_dopt100lkr"), 0.0)
        self.assertEqual(parse(HEAD + "_xchancecells"), 0.0)
        self.assertEqual(parse(HEAD + "_chancecellsx"), 0.0)
        with self.assertRaises(ValueError):
            parse(HEAD + "_cutin_chance0.6_chancecells")

    def test_safety_constraint_mode(self):
        parse = ACCNairSMPCAgent._parse_safety_constraint_mode
        self.assertEqual(parse(HEAD + "_chancecells_dopt100lkr"), SAFETY_CONFIDENCE_CHANCE)
        self.assertEqual(parse(HEAD + "_cutin_chance0.6_dopt100lkr"), SAFETY_CONFIDENCE_CHANCE)
        self.assertNotEqual(parse(HEAD + "_dopt100lkr"), SAFETY_CONFIDENCE_CHANCE)

    def test_every_parser_reads_the_two_names_alike(self):
        # every one-argument string parser of the agent gives the same value for the NL string with either name
        new = HEAD + "_chancecells_dopt100lkr" + NL_TAIL
        old = HEAD + "_cutin_chance0.6_dopt100lkr" + NL_TAIL
        checked = 0
        for name, attr in vars(ACCNairSMPCAgent).items():
            if not (name.startswith("_parse_") and isinstance(attr, staticmethod)):
                continue
            fn = getattr(ACCNairSMPCAgent, name)
            try:
                a, b = fn(new), fn(old)
            except TypeError:
                continue
            self.assertEqual(a, b, name)
            checked += 1
        self.assertGreater(checked, 40)


if __name__ == "__main__":
    unittest.main()
