"""Invariant tests for the 03b reactive-AIMD data path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "build_box"))
sys.path.insert(0, str(ROOT / "scripts" / "run_md"))

from generate_uks_hclo4 import (  # noqa: E402
    CHARGE,
    CLO3_ATOMS,
    ELEMENTS,
    MASSES,
    OH_ATOMS,
    analytic_demo_evaluate,
    electron_count,
    initial_geometry,
    kick_velocities,
    run_trajectory,
)


class TestUKSReaction(unittest.TestCase):
    def test_formula_and_charge(self) -> None:
        self.assertEqual(electron_count(ELEMENTS, CHARGE), 50)
        self.assertEqual(len(ELEMENTS), 6)

    def test_reference_geometry_has_bent_hydroxyl(self) -> None:
        positions = initial_geometry()
        cl_to_o = np.linalg.norm(positions[1] - positions[0])
        oh = np.linalg.norm(positions[2] - positions[1])
        u = positions[0] - positions[1]
        v = positions[2] - positions[1]
        angle = np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        self.assertAlmostEqual(float(cl_to_o), 1.64, places=2)
        self.assertAlmostEqual(float(oh), 0.98, places=2)
        self.assertAlmostEqual(float(angle), 105.0, places=1)

    def test_mass_weighted_kick_has_zero_com_velocity(self) -> None:
        velocity = kick_velocities(initial_geometry(), 0.05)
        com = (MASSES[:, None] * velocity).sum(axis=0) / MASSES.sum()
        np.testing.assert_allclose(com, 0.0, atol=1.0e-14)
        self.assertGreater(float(velocity[OH_ATOMS, 0].mean()), 0.0)
        self.assertLess(float(velocity[CLO3_ATOMS, 0].mean()), 0.0)

    def test_demo_path_stretches_reactive_bond_and_preserves_oh(self) -> None:
        def evaluator(positions, step, branch):
            return analytic_demo_evaluate(positions, step=step, branch_id=branch)

        frames = run_trajectory(evaluator, relative_speed=0.05, n_steps=40)
        r_clo = np.asarray([frame["r_clo"] for frame in frames], dtype=float)
        r_oh = np.asarray([frame["r_oh"] for frame in frames], dtype=float)
        self.assertTrue(np.all(np.diff(r_clo) >= -1.0e-10))
        self.assertGreater(r_clo[-1], r_clo[0])
        self.assertLess(float(np.max(np.abs(r_oh - r_oh[0]))), 0.05)
        self.assertGreaterEqual(float(frames[-1]["spin_square"]), float(frames[0]["spin_square"]))


if __name__ == "__main__":
    unittest.main()

