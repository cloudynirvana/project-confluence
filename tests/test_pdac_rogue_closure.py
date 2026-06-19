import unittest

from models.pdac_rogue_closure import PDACState, host_access, simulate, summarize


class PDACRogueClosureTests(unittest.TestCase):
    def test_shield_disruption_improves_host_access(self):
        shielded = PDACState(glyco_shield=0.85, stroma=0.85)
        accessible = PDACState(glyco_shield=0.25, stroma=0.25)

        self.assertGreater(host_access(accessible), host_access(shielded))

    def test_closure_stack_beats_ras_only_on_closure(self):
        ras_only = summarize(simulate("ras_only", days=180, dt=1.0))
        closure_stack = summarize(simulate("closure_stack", days=180, dt=1.0))

        self.assertLess(closure_stack["final_rogue_closure"], ras_only["final_rogue_closure"])
        self.assertLess(closure_stack["final_tumor"], ras_only["final_tumor"])

    def test_adaptive_closure_reduces_tumor_from_baseline(self):
        adaptive = summarize(simulate("adaptive_closure", days=180, dt=1.0))

        self.assertLess(adaptive["final_tumor"], adaptive["initial_tumor"])

    def test_model_bounds_core_state_variables(self):
        points = simulate("closure_stack", days=60, dt=0.5)

        for point in points:
            self.assertGreaterEqual(point.state.tumor, 0.0)
            self.assertGreaterEqual(point.state.resistance, 0.0)
            self.assertLessEqual(point.state.resistance, 1.0)
            self.assertGreaterEqual(point.state.immune, 0.0)
            self.assertLessEqual(point.state.immune, 1.0)
            self.assertGreaterEqual(point.state.glyco_shield, 0.0)
            self.assertLessEqual(point.state.glyco_shield, 1.0)


if __name__ == "__main__":
    unittest.main()

