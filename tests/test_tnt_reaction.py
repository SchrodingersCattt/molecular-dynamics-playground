"""Topology and density preflight tests for the TNT C2--NO2 story."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "build_box"))
sys.path.insert(0, str(ROOT / "scripts" / "run_md"))

from generate_uks_tnt import (  # noqa: E402
    BREAKING_C,
    BREAKING_N,
    ELEMENTS,
    MASSES,
    NO2_ATOMS,
    analytic_demo_evaluate,
    density_grid,
    initial_geometry,
    kick_velocities,
)
from engine_uks import electron_count  # noqa: E402


class TestTNTReaction(unittest.TestCase):
    def test_formula_and_atom_map(self) -> None:
        self.assertEqual(electron_count(ELEMENTS, 0), 116)
        self.assertEqual(len(ELEMENTS), 21)
        self.assertEqual(tuple(ELEMENTS).count("C"), 7)
        self.assertEqual(tuple(ELEMENTS).count("N"), 3)
        self.assertEqual(tuple(ELEMENTS).count("O"), 6)
        self.assertEqual(tuple(NO2_ATOMS), (12, 15, 16))

    def test_rdkit_reference_has_no_spurious_oxygen_oxygen_bonds(self) -> None:
        positions = initial_geometry()
        oxygen = np.flatnonzero(ELEMENTS == "O")
        distances = [
            float(np.linalg.norm(positions[i] - positions[j]))
            for offset, i in enumerate(oxygen)
            for j in oxygen[offset + 1 :]
        ]
        self.assertGreater(min(distances), 1.8)
        self.assertTrue(1.2 < float(np.linalg.norm(positions[BREAKING_C] - positions[BREAKING_N])) < 1.9)

    def test_mass_weighted_kick_has_zero_com_velocity(self) -> None:
        velocity = kick_velocities(initial_geometry(), 0.05)
        com = (MASSES[:, None] * velocity).sum(axis=0) / MASSES.sum()
        np.testing.assert_allclose(com, 0.0, atol=1.0e-14)

    def test_demo_returns_alpha_beta_3d_density(self) -> None:
        grid = tuple(np.linspace(-2.0, 2.0, 5) for _ in range(3))
        result = analytic_demo_evaluate(initial_geometry(), step=0, grid=grid)
        self.assertEqual(result.rho_alpha_3d.shape, (5, 5, 5))
        self.assertEqual(result.rho_beta_3d.shape, (5, 5, 5))
        self.assertTrue(np.isfinite(result.rho_alpha_3d).all())
        self.assertTrue(np.isfinite(result.rho_beta_3d).all())


if __name__ == "__main__":
    unittest.main()
