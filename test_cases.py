"""
Test Cases for Composite Laminate Analysis

Comprehensive test suite for laminate analysis including:
- Input property configuration
- Workflow calculations
- Thermal effects
- Symmetry verification
- Edge cases
"""

import unittest
import numpy as np
from InputProperties import (
    MaterialProperties, LayupSequence, LaminateProperties,
    create_standard_QI_8ply, create_standard_QI_16ply, 
    create_standard_QI_with_temperature
)
from workflow import LaminateAnalysis, run_standard_QI_analysis
from MatrixFunctions import (
    Q_matrix, T_matrix, reuters_matrix, Q_bar, alpha_bar
)


class TestMaterialProperties(unittest.TestCase):
    """Tests for MaterialProperties class."""
    
    def test_default_carbon_epoxy_properties(self):
        """Test default carbon/epoxy material properties."""
        mat = MaterialProperties()
        self.assertEqual(mat.E_11, 130e9)
        self.assertEqual(mat.E_22, 10e9)
        self.assertEqual(mat.v_12, 0.3)
        self.assertEqual(mat.G_12, 5e9)
        self.assertEqual(mat.alpha_1, -0.5e-6)
        self.assertEqual(mat.alpha_2, 30e-6)
    
    def test_custom_material_properties(self):
        """Test custom material property initialization."""
        mat = MaterialProperties(
            name="Glass/Polyester",
            E_11=50e9,
            E_22=15e9,
            v_12=0.25,
            G_12=4e9,
            alpha_1=10e-6,
            alpha_2=50e-6
        )
        self.assertEqual(mat.name, "Glass/Polyester")
        self.assertEqual(mat.E_11, 50e9)
    
    def test_material_properties_repr(self):
        """Test material properties string representation."""
        mat = MaterialProperties()
        repr_str = repr(mat)
        self.assertIn("Carbon/Epoxy", repr_str)
        self.assertIn("E_11", repr_str)


class TestLayupSequence(unittest.TestCase):
    """Tests for LayupSequence class."""
    
    def test_qi_8ply_layup(self):
        """Test QI 8-ply layup creation."""
        layup = LayupSequence.create_QI(num_plies_per_set=8)
        self.assertEqual(layup.num_plies, 8)
        self.assertEqual(layup.angles, [0, 45, -45, 90, 90, -45, 45, 0])
        self.assertIn("QI", layup.name)
    
    def test_qi_16ply_layup(self):
        """Test QI 16-ply layup creation."""
        layup = LayupSequence.create_QI(num_plies_per_set=16)
        self.assertEqual(layup.num_plies, 16)
    
    def test_symmetric_layup(self):
        """Test symmetric layup creation."""
        base = [0, 45, -45]
        layup = LayupSequence.create_symmetric(base)
        expected = [0, 45, -45, -45, 45, 0]
        self.assertEqual(layup.angles, expected)
    
    def test_custom_layup(self):
        """Test custom layup definition."""
        angles = [0, 90, 0, 90]
        layup = LayupSequence(angles, name="Cross-ply")
        self.assertEqual(layup.num_plies, 4)
        self.assertEqual(layup.angles, angles)
    
    def test_layup_repr(self):
        """Test layup string representation."""
        layup = LayupSequence.create_QI()
        repr_str = repr(layup)
        self.assertIn("QI", repr_str)
        self.assertIn("8", repr_str)


class TestLaminateProperties(unittest.TestCase):
    """Tests for LaminateProperties class."""
    
    def test_standard_qi_8ply(self):
        """Test standard 8-ply QI laminate creation."""
        laminate = create_standard_QI_8ply()
        self.assertEqual(laminate.layup.num_plies, 8)
        self.assertAlmostEqual(laminate.ply_thickness, 0.125e-3)
        self.assertAlmostEqual(laminate.total_thickness, 8 * 0.125e-3)
    
    def test_standard_qi_16ply(self):
        """Test standard 16-ply QI laminate creation."""
        laminate = create_standard_QI_16ply()
        self.assertEqual(laminate.layup.num_plies, 16)
        self.assertAlmostEqual(laminate.total_thickness, 16 * 0.125e-3)
    
    def test_total_thickness_calculation(self):
        """Test that total thickness is correctly calculated."""
        ply_thickness = 0.1e-3
        layup = LayupSequence([0, 90], name="Simple")
        laminate = LaminateProperties(
            MaterialProperties(),
            layup,
            ply_thickness=ply_thickness
        )
        expected_thickness = 2 * ply_thickness
        self.assertAlmostEqual(laminate.total_thickness, expected_thickness)
    
    def test_temperature_property(self):
        """Test temperature change property."""
        laminate = create_standard_QI_with_temperature(delta_T=-50)
        self.assertEqual(laminate.delta_T, -50)
    
    def test_laminate_properties_repr(self):
        """Test laminate properties string representation."""
        laminate = create_standard_QI_8ply()
        repr_str = repr(laminate)
        self.assertIn("material", repr_str)
        self.assertIn("layup", repr_str)


class TestLaminateAnalysisBasics(unittest.TestCase):
    """Tests for basic LaminateAnalysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.laminate = create_standard_QI_8ply()
        self.analysis = LaminateAnalysis(self.laminate)
    
    def test_analysis_initialization(self):
        """Test that analysis is properly initialized."""
        self.assertEqual(self.analysis.layup.num_plies, 8)
        self.assertIsNone(self.analysis.A_matrix)
        self.assertIsNone(self.analysis.ABD_matrix)
    
    def test_material_matrix_calculation(self):
        """Test calculation of Q and alpha in material coordinates."""
        mat_matrices = self.analysis.calculate_material_matrices()
        
        self.assertIn("Q", mat_matrices)
        self.assertIn("alpha", mat_matrices)
        self.assertEqual(mat_matrices["Q"].shape, (3, 3))
        self.assertEqual(mat_matrices["alpha"].shape, (3, 1))
    
    def test_material_matrix_properties(self):
        """Test properties of material stiffness matrix."""
        mat_matrices = self.analysis.calculate_material_matrices()
        Q = mat_matrices["Q"]
        
        # Q matrix should be symmetric
        np.testing.assert_array_almost_equal(Q, Q.T)
        
        # Diagonal elements should be positive
        self.assertGreater(Q[0, 0], 0)  # Q_11
        self.assertGreater(Q[1, 1], 0)  # Q_22
        self.assertGreater(Q[2, 2], 0)  # Q_66


class TestTransformedMatrices(unittest.TestCase):
    """Tests for transformation of matrices to global coordinates."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.laminate = create_standard_QI_8ply()
        self.analysis = LaminateAnalysis(self.laminate)
    
    def test_transformed_matrices_calculation(self):
        """Test that transformed matrices are calculated for each ply."""
        self.analysis.calculate_transformed_matrices()
        
        self.assertEqual(len(self.analysis.Q_bar_list), 8)
        self.assertEqual(len(self.analysis.alpha_bar_list), 8)
        
        for Q_bar_ply in self.analysis.Q_bar_list:
            self.assertEqual(Q_bar_ply.shape, (3, 3))
    
    def test_zero_degree_ply_transformation(self):
        """Test that 0° ply has special behavior."""
        laminate = LaminateProperties(
            MaterialProperties(),
            LayupSequence([0], name="0-degree"),
            ply_thickness=0.125e-3
        )
        analysis = LaminateAnalysis(laminate)
        analysis.calculate_transformed_matrices()
        
        # For 0° ply, Q_bar should equal Q (approximately)
        mat_matrices = analysis.calculate_material_matrices()
        Q = mat_matrices["Q"]
        
        np.testing.assert_array_almost_equal(analysis.Q_bar_list[0], Q, decimal=10)
    
    def test_90_degree_ply_transformation(self):
        """Test that 90° ply transformation is correct."""
        laminate = LaminateProperties(
            MaterialProperties(),
            LayupSequence([90], name="90-degree"),
            ply_thickness=0.125e-3
        )
        analysis = LaminateAnalysis(laminate)
        analysis.calculate_transformed_matrices()
        
        Q_bar_90 = analysis.Q_bar_list[0]
        
        # For 90° ply, Q_bar_11 should be much less than Q_11
        mat_matrices = analysis.calculate_material_matrices()
        Q = mat_matrices["Q"]
        
        self.assertLess(Q_bar_90[0, 0], Q[0, 0])


class TestLaminateStiffnessMatrices(unittest.TestCase):
    """Tests for laminate stiffness matrix calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.laminate = create_standard_QI_8ply()
        self.analysis = LaminateAnalysis(self.laminate)
    
    def test_stiffness_matrix_calculation(self):
        """Test that A, B, D matrices are calculated."""
        self.analysis.calculate_transformed_matrices()
        self.analysis.calculate_laminate_stiffness_matrices()
        
        self.assertIsNotNone(self.analysis.A_matrix)
        self.assertIsNotNone(self.analysis.B_matrix)
        self.assertIsNotNone(self.analysis.D_matrix)
        self.assertIsNotNone(self.analysis.ABD_matrix)
    
    def test_stiffness_matrix_shapes(self):
        """Test shapes of stiffness matrices."""
        self.analysis.calculate_transformed_matrices()
        self.analysis.calculate_laminate_stiffness_matrices()
        
        self.assertEqual(self.analysis.A_matrix.shape, (3, 3))
        self.assertEqual(self.analysis.B_matrix.shape, (3, 3))
        self.assertEqual(self.analysis.D_matrix.shape, (3, 3))
        self.assertEqual(self.analysis.ABD_matrix.shape, (6, 6))
    
    def test_stiffness_matrix_symmetry(self):
        """Test that stiffness matrices are symmetric."""
        self.analysis.calculate_transformed_matrices()
        self.analysis.calculate_laminate_stiffness_matrices()
        
        # A matrix should be symmetric
        np.testing.assert_array_almost_equal(
            self.analysis.A_matrix,
            self.analysis.A_matrix.T
        )
        
        # D matrix should be symmetric
        np.testing.assert_array_almost_equal(
            self.analysis.D_matrix,
            self.analysis.D_matrix.T
        )
    
    def test_abd_matrix_structure(self):
        """Test that ABD matrix has correct block structure."""
        self.analysis.calculate_transformed_matrices()
        self.analysis.calculate_laminate_stiffness_matrices()
        
        ABD = self.analysis.ABD_matrix
        
        # Check A block
        np.testing.assert_array_almost_equal(
            ABD[0:3, 0:3],
            self.analysis.A_matrix
        )
        
        # Check D block
        np.testing.assert_array_almost_equal(
            ABD[3:6, 3:6],
            self.analysis.D_matrix
        )


class TestSymmetricLaminate(unittest.TestCase):
    """Tests for symmetric laminates."""
    
    def test_symmetric_laminate_b_matrix_zero(self):
        """Test that symmetric laminates have zero B matrix (approximately)."""
        # Create symmetric laminate
        layup = LayupSequence.create_symmetric([0, 45, -45])
        material = MaterialProperties()
        laminate = LaminateProperties(material, layup)
        analysis = LaminateAnalysis(laminate)
        
        analysis.calculate_transformed_matrices()
        analysis.calculate_laminate_stiffness_matrices()
        
        # B matrix should be very small (numerical errors only)
        max_b = np.max(np.abs(analysis.B_matrix))
        self.assertLess(max_b, 1e-6)
    
    def test_qi_laminate_isotropy(self):
        """Test that QI laminate approaches isotropy in A matrix."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.calculate_transformed_matrices()
        analysis.calculate_laminate_stiffness_matrices()
        
        A = analysis.A_matrix
        
        # For QI, A_11 and A_22 should be similar
        ratio_1122 = A[0, 0] / A[1, 1]
        self.assertAlmostEqual(ratio_1122, 1.0, places=1)
        
        # For QI, A_66 should be related to shear stiffness
        # A_12 should be less than A_11
        self.assertLess(A[0, 1], A[0, 0])


class TestThermalLoading(unittest.TestCase):
    """Tests for thermal loading calculations."""
    
    def test_thermal_loading_zero_temperature(self):
        """Test that zero temperature change gives zero thermal loads."""
        laminate = create_standard_QI_8ply()
        # delta_T = 0 by default
        analysis = LaminateAnalysis(laminate)
        
        analysis.calculate_transformed_matrices()
        analysis.calculate_thermal_loading()
        
        # Thermal forces and moments should be near zero
        np.testing.assert_array_almost_equal(analysis.N_thermal, np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(analysis.M_thermal, np.zeros(3), decimal=10)
    
    def test_thermal_loading_nonzero_temperature(self):
        """Test that nonzero temperature gives nonzero thermal loads."""
        laminate = create_standard_QI_with_temperature(delta_T=-50)
        analysis = LaminateAnalysis(laminate)
        
        analysis.calculate_transformed_matrices()
        analysis.calculate_thermal_loading()
        
        # Thermal forces should be nonzero
        self.assertFalse(np.allclose(analysis.N_thermal, np.zeros(3)))
    
    def test_thermal_loading_sign_dependency(self):
        """Test that thermal loading changes sign with temperature sign."""
        laminate_cool = create_standard_QI_with_temperature(delta_T=-50)
        analysis_cool = LaminateAnalysis(laminate_cool)
        analysis_cool.calculate_transformed_matrices()
        analysis_cool.calculate_thermal_loading()
        
        laminate_hot = create_standard_QI_with_temperature(delta_T=50)
        analysis_hot = LaminateAnalysis(laminate_hot)
        analysis_hot.calculate_transformed_matrices()
        analysis_hot.calculate_thermal_loading()
        
        # Thermal forces should have opposite signs
        np.testing.assert_array_almost_equal(
            analysis_cool.N_thermal,
            -analysis_hot.N_thermal
        )


class TestResultantStrains(unittest.TestCase):
    """Tests for resultant strain calculations."""
    
    def test_resultant_strains_calculation(self):
        """Test that resultant strains are calculated."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        self.assertIsNotNone(analysis.resultant_strains)
        self.assertEqual(analysis.resultant_strains.shape, (6,))
    
    def test_resultant_strains_zero_loading(self):
        """Test that zero loading gives zero strains."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # With delta_T = 0 and no external loads, strains should be ~0
        np.testing.assert_array_almost_equal(
            analysis.resultant_strains,
            np.zeros(6),
            decimal=10
        )
    
    def test_resultant_strains_thermal_loading(self):
        """Test that thermal loading produces strains."""
        laminate = create_standard_QI_with_temperature(delta_T=-50)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # Thermal loading should produce nonzero strains
        self.assertFalse(np.allclose(analysis.resultant_strains, np.zeros(6)))


class TestPlyStrains(unittest.TestCase):
    """Tests for individual ply strain calculations."""
    
    def test_ply_strains_calculation(self):
        """Test that ply strains are calculated for each ply."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        self.assertEqual(len(analysis.ply_strains), 8)
        
        for strain in analysis.ply_strains:
            self.assertEqual(strain.shape, (3,))
    
    def test_ply_strains_zero_loading(self):
        """Test ply strains with zero loading."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # All ply strains should be ~0
        for strain in analysis.ply_strains:
            np.testing.assert_array_almost_equal(strain, np.zeros(3), decimal=10)
    
    def test_ply_strains_variation_through_thickness(self):
        """Test that ply strains vary through thickness due to curvature."""
        # Use non-symmetric laminate that will have curvature
        layup = LayupSequence([0, 90, 90, 0], name="Asymmetric")
        material = MaterialProperties()
        laminate = LaminateProperties(material, layup, ply_thickness=0.125e-3, delta_T=-50)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # Strains should differ between plies if there's curvature
        if not np.allclose(analysis.resultant_strains[3:6], np.zeros(3)):
            # There is curvature, so strains should vary
            first_ply_strain = analysis.ply_strains[0]
            last_ply_strain = analysis.ply_strains[-1]
            
            # At least one component should differ
            self.assertFalse(np.allclose(first_ply_strain, last_ply_strain))


class TestFullWorkflow(unittest.TestCase):
    """Integration tests for complete analysis workflow."""
    
    def test_full_workflow_8ply_qi(self):
        """Test complete workflow for 8-ply QI laminate."""
        laminate = create_standard_QI_8ply()
        analysis = run_standard_QI_analysis(laminate, verbose=False)
        
        # Check that all calculations were performed
        self.assertIsNotNone(analysis.Q_bar_list)
        self.assertIsNotNone(analysis.A_matrix)
        self.assertIsNotNone(analysis.resultant_strains)
        self.assertIsNotNone(analysis.ply_strains)
    
    def test_full_workflow_16ply_qi(self):
        """Test complete workflow for 16-ply QI laminate."""
        laminate = create_standard_QI_16ply()
        analysis = run_standard_QI_analysis(laminate, verbose=False)
        
        self.assertEqual(len(analysis.ply_strains), 16)
    
    def test_full_workflow_with_thermal_loading(self):
        """Test complete workflow with thermal loading."""
        laminate = create_standard_QI_with_temperature(delta_T=-50)
        analysis = run_standard_QI_analysis(laminate, verbose=False)
        
        # Thermal loading should produce nonzero strains
        self.assertFalse(np.allclose(analysis.resultant_strains, np.zeros(6)))
    
    def test_summary_generation(self):
        """Test that analysis summary can be generated."""
        laminate = create_standard_QI_8ply()
        analysis = run_standard_QI_analysis(laminate, verbose=False)
        
        summary = analysis.get_summary()
        
        self.assertIn("LAMINATE ANALYSIS SUMMARY", summary)
        self.assertIn("MATERIAL PROPERTIES", summary)
        self.assertIn("A Matrix", summary)
        self.assertIn("PLY STRAINS", summary)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""
    
    def test_single_ply_laminate(self):
        """Test analysis of single-ply laminate."""
        layup = LayupSequence([0], name="Single Ply")
        material = MaterialProperties()
        laminate = LaminateProperties(material, layup, ply_thickness=0.125e-3)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        self.assertEqual(len(analysis.ply_strains), 1)
        self.assertEqual(laminate.total_thickness, 0.125e-3)
    
    def test_cross_ply_laminate(self):
        """Test analysis of cross-ply laminate."""
        layup = LayupSequence([0, 90, 0, 90], name="Cross-ply")
        material = MaterialProperties()
        laminate = LaminateProperties(material, layup, ply_thickness=0.125e-3)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        self.assertEqual(len(analysis.ply_strains), 4)
    
    def test_large_temperature_change(self):
        """Test with large temperature change."""
        laminate = create_standard_QI_with_temperature(delta_T=-200)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # Should still produce valid results
        self.assertIsNotNone(analysis.resultant_strains)
    
    def test_very_thin_plies(self):
        """Test with very thin plies."""
        layup = LayupSequence.create_QI()
        material = MaterialProperties()
        laminate = LaminateProperties(material, layup, ply_thickness=1e-5)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        self.assertLess(laminate.total_thickness, 1e-4)


class TestNumericalStability(unittest.TestCase):
    """Tests for numerical stability of calculations."""
    
    def test_matrix_invertibility(self):
        """Test that ABD matrix is invertible."""
        laminate = create_standard_QI_8ply()
        analysis = LaminateAnalysis(laminate)
        
        analysis.calculate_transformed_matrices()
        analysis.calculate_laminate_stiffness_matrices()
        
        # ABD matrix should be invertible
        det = np.linalg.det(analysis.ABD_matrix)
        self.assertNotAlmostEqual(det, 0, places=5)
    
    def test_no_nan_values(self):
        """Test that calculations don't produce NaN values."""
        laminate = create_standard_QI_with_temperature(delta_T=-50)
        analysis = LaminateAnalysis(laminate)
        
        analysis.run_full_analysis()
        
        # Check for NaN values
        self.assertFalse(np.any(np.isnan(analysis.A_matrix)))
        self.assertFalse(np.any(np.isnan(analysis.resultant_strains)))
        for strain in analysis.ply_strains:
            self.assertFalse(np.any(np.isnan(strain)))


def run_all_tests():
    """Run all tests and generate a report."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMaterialProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestLayupSequence))
    suite.addTests(loader.loadTestsFromTestCase(TestLaminateProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestLaminateAnalysisBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformedMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestLaminateStiffnessMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestSymmetricLaminate))
    suite.addTests(loader.loadTestsFromTestCase(TestThermalLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestResultantStrains))
    suite.addTests(loader.loadTestsFromTestCase(TestPlyStrains))
    suite.addTests(loader.loadTestsFromTestCase(TestFullWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_all_tests()
    
    print("\n" + "="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80)
