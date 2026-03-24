"""
AlphaFold Client Tests — Project Confluence
=============================================

Tests for alphafold_client.py using mock data (no API calls).

Run:
    python -m pytest tests/test_alphafold_client.py -v
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.alphafold_client import (
    AlphaFoldClient,
    StructureData,
    ResidueInfo,
    BindingPocket,
    DISEASE_PANELS,
    GENE_KEY_RESIDUES,
    create_mock_structure,
    detect_binding_pockets,
    _estimate_secondary_structure,
)


class TestMockStructure(unittest.TestCase):
    """Test mock structure generation."""

    def test_mock_creates_correct_residues(self):
        """Mock structure should have the requested number of residues."""
        structure = create_mock_structure(n_residues=100)
        self.assertEqual(structure.sequence_length, 100)

    def test_mock_has_valid_plddt(self):
        """All pLDDT values should be in [0, 100]."""
        structure = create_mock_structure(n_residues=200)
        for r in structure.residues:
            self.assertGreaterEqual(r.plddt, 0)
            self.assertLessEqual(r.plddt, 100)

    def test_mock_has_3d_coordinates(self):
        """Each residue should have 3D coordinates."""
        structure = create_mock_structure()
        for r in structure.residues:
            self.assertEqual(r.coords.shape, (3,))
            self.assertTrue(np.all(np.isfinite(r.coords)))

    def test_mock_has_pockets(self):
        """Mock structure should detect at least one binding pocket."""
        structure = create_mock_structure(n_residues=300)
        self.assertGreater(len(structure.pockets), 0)

    def test_mock_reproducible_with_seed(self):
        """Same seed should produce identical structures."""
        s1 = create_mock_structure(seed=42)
        s2 = create_mock_structure(seed=42)
        self.assertEqual(s1.sequence_length, s2.sequence_length)
        for r1, r2 in zip(s1.residues, s2.residues):
            np.testing.assert_array_almost_equal(r1.coords, r2.coords)
            self.assertAlmostEqual(r1.plddt, r2.plddt)

    def test_mock_different_seeds_differ(self):
        """Different seeds should produce different structures."""
        s1 = create_mock_structure(seed=42)
        s2 = create_mock_structure(seed=99)
        plddts_1 = [r.plddt for r in s1.residues]
        plddts_2 = [r.plddt for r in s2.residues]
        self.assertFalse(np.allclose(plddts_1, plddts_2))


class TestStructureData(unittest.TestCase):
    """Test StructureData properties and methods."""

    def setUp(self):
        self.structure = create_mock_structure(n_residues=393, gene_name="TP53")

    def test_mean_plddt_in_range(self):
        """Mean pLDDT should be in reasonable range."""
        self.assertGreater(self.structure.mean_plddt, 0)
        self.assertLessEqual(self.structure.mean_plddt, 100)

    def test_high_confidence_fraction(self):
        """High confidence fraction should be between 0 and 1."""
        frac = self.structure.high_confidence_fraction
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_disordered_regions(self):
        """Disordered regions should be valid index ranges."""
        regions = self.structure.disordered_regions
        for start, end in regions:
            self.assertGreaterEqual(start, 1)
            self.assertLessEqual(end, self.structure.sequence_length)
            self.assertLessEqual(start, end)

    def test_get_plddt_at_valid_position(self):
        """pLDDT at valid position should be non-zero."""
        plddt = self.structure.get_plddt_at(50)
        self.assertGreater(plddt, 0)

    def test_get_plddt_at_invalid_position(self):
        """pLDDT at invalid position should return 0."""
        plddt = self.structure.get_plddt_at(99999)
        self.assertEqual(plddt, 0.0)

    def test_get_local_plddt(self):
        """Local pLDDT should average over window."""
        local = self.structure.get_local_plddt(100, window=5)
        self.assertGreater(local, 0)

    def test_to_dict(self):
        """to_dict should return valid serializable dict."""
        d = self.structure.to_dict()
        self.assertIn("uniprot_id", d)
        self.assertIn("mean_plddt", d)
        self.assertIn("n_pockets", d)
        # Should be JSON serializable
        json.dumps(d)


class TestPocketDetection(unittest.TestCase):
    """Test binding pocket detection algorithm."""

    def test_detects_pockets_in_large_structure(self):
        """Should find pockets in a 400-residue structure."""
        structure = create_mock_structure(n_residues=400)
        pockets = detect_binding_pockets(structure)
        self.assertGreater(len(pockets), 0)

    def test_no_pockets_in_tiny_structure(self):
        """Should not find pockets in very small structures."""
        structure = create_mock_structure(n_residues=5)
        pockets = detect_binding_pockets(structure, min_pocket_residues=10)
        self.assertEqual(len(pockets), 0)

    def test_pocket_druggability_in_range(self):
        """Pocket druggability scores should be in [0, 1]."""
        structure = create_mock_structure(n_residues=300)
        for pocket in structure.pockets:
            self.assertGreaterEqual(pocket.druggability_score, 0)
            self.assertLessEqual(pocket.druggability_score, 1)

    def test_pocket_volume_positive(self):
        """Pocket volumes should be positive."""
        structure = create_mock_structure(n_residues=300)
        for pocket in structure.pockets:
            self.assertGreater(pocket.volume, 0)

    def test_pockets_sorted_by_druggability(self):
        """Pockets should be sorted by druggability (best first)."""
        structure = create_mock_structure(n_residues=400)
        if len(structure.pockets) >= 2:
            for i in range(len(structure.pockets) - 1):
                self.assertGreaterEqual(
                    structure.pockets[i].druggability_score,
                    structure.pockets[i + 1].druggability_score,
                )

    def test_max_pockets_respected(self):
        """Should not return more than max_pockets."""
        structure = create_mock_structure(n_residues=400)
        pockets = detect_binding_pockets(structure, max_pockets=2)
        self.assertLessEqual(len(pockets), 2)


class TestSecondaryStructure(unittest.TestCase):
    """Test secondary structure estimation."""

    def test_assigns_structure_types(self):
        """Should assign H, E, or C to each residue."""
        structure = create_mock_structure(n_residues=100)
        valid_types = {'H', 'E', 'C'}
        for r in structure.residues:
            self.assertIn(r.secondary_structure, valid_types)


class TestDiseasePanels(unittest.TestCase):
    """Test disease panel definitions."""

    def test_all_panels_have_genes(self):
        """Each disease panel should have at least 3 genes."""
        for disease, panel in DISEASE_PANELS.items():
            self.assertGreaterEqual(len(panel), 3, f"{disease} has too few genes")

    def test_uniprot_ids_format(self):
        """UniProt IDs should match expected format (P/Q/O + 5 alphanumeric)."""
        import re
        pattern = re.compile(r'^[A-Z][0-9A-Z]{5}$')
        for disease, panel in DISEASE_PANELS.items():
            for gene, uid in panel.items():
                self.assertTrue(
                    pattern.match(uid),
                    f"{disease}/{gene}: invalid UniProt ID '{uid}'"
                )

    def test_key_residues_are_positive(self):
        """Key residue positions should be positive integers."""
        for gene, residues in GENE_KEY_RESIDUES.items():
            for r in residues:
                self.assertGreater(r, 0, f"{gene}: residue {r} is not positive")


class TestCacheMechanism(unittest.TestCase):
    """Test cache read/write."""

    def test_cache_roundtrip(self):
        """Save and load should produce identical structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = AlphaFoldClient(cache_dir=tmpdir)
            structure = create_mock_structure()

            # Save
            client._save_cache("test_key", structure)

            # Load
            loaded = client._load_cache("test_key")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.uniprot_id, structure.uniprot_id)
            self.assertEqual(loaded.sequence_length, structure.sequence_length)
            self.assertAlmostEqual(loaded.mean_plddt, structure.mean_plddt, places=1)

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = AlphaFoldClient(cache_dir=tmpdir)
            result = client._load_cache("nonexistent_key")
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
