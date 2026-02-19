
import os
import sys
import io 
import contextlib
import inspect
import numpy as np
import pandas as pd
import unittest
from matplotlib import pyplot as plt


# Try to determine the base directory based on __file__
try:
    # __file__ is defined if running as a script; get its directory
    base_dir = os.path.dirname(__file__)
except NameError:
    # Fallback if __file__ is not defined (e.g., in interactive environments)
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
# Assumes that the 'src' folder is located one directory level above base_dir.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
# Insert src_path at the beginning of sys.path so that modules in 'src' can be imported.
sys.path.insert(0, src_path)

from calcverif import CalculationVerificationTool, ConvergenceType



class TestCalcVerif(unittest.TestCase):
    """
    Unit tests for the CalculationVerificationTool class.
    """
    
    
    def test_check_oscillatory_status(self):
        """
        Test the check_oscillatory_status function with different triplets to
        ensure it correctly identifies the type of convergence or divergence.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
 
        calcverif_tool = CalculationVerificationTool()       
 
        # Test monotonic convergence: f1=1, f2=2, f3=4 should yield MONOTONIC_CONVERGENCE.
        self.assertEqual(calcverif_tool._check_oscillatory_status(1, 2, 4), ConvergenceType.MONOTONIC_CONVERGENCE)
        # Test oscillatory convergence: f1=1, f2=0, f3=3 should yield OSCILLATORY_CONVERGENCE.
        self.assertEqual(calcverif_tool._check_oscillatory_status(1, 0, 3), ConvergenceType.OSCILLATORY_CONVERGENCE)
        # Test monotonic divergence: f1=4, f2=2, f3=1 should yield MONOTONIC_DIVERGENCE.
        self.assertEqual(calcverif_tool._check_oscillatory_status(4, 2, 1), ConvergenceType.MONOTONIC_DIVERGENCE)
        # Test oscillatory divergence: f1=4, f2=6, f3=5 should yield OSCILLATORY_DIVERGENCE.
        self.assertEqual(calcverif_tool._check_oscillatory_status(4, 6, 5), ConvergenceType.OSCILLATORY_DIVERGENCE)


    def test_basic_error_checking(self):
        """
        Test that analyze properly raises errors when:
         - h_values and qoi_values have mismatched lengths,
         - fewer than 3 values are provided,
         - h_values are not strictly increasing.
        """        
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        # Test case: h_values and qoi_values of different lengths.
        h_values = [0.1, 0.2]
        qoi_values = [1.0]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze(h_values, qoi_values)
        self.assertEqual(
            str(context.exception),
            "h_values and qoi_values must have the same length"
        )

        # Test case: both lists have same length but less than 3 values.
        h_values = [0.1, 0.2]
        qoi_values = [1.0, 2.0]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze(h_values, qoi_values)
        self.assertEqual(
            str(context.exception),
            "h_values and qoi_values must be of size 3 or more"
        )
        
        # Test case: h_values are not strictly increasing.
        h_values = [0.1, 0.05, 0.2]
        qoi_values = [1.0, 2.0, 4.0]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze(h_values, qoi_values)
        self.assertEqual(
            str(context.exception),
            "h_values must be strictly increasing"
        )

        # Test case: try to plot even though didn't run
        with self.assertRaises(ValueError) as context:
            calcverif_tool.plot()
        self.assertEqual(
            str(context.exception),
            "Cannot plot as did not run successfully"
        )


        # Test case: element counts increasing instead of decrease
        element_counts = [100, 200, 400]
        qoi_values = [1.0, 2.0, 4.0]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze_by_n_elems(element_counts, 2, qoi_values)
        self.assertEqual(
            str(context.exception),
            "element_counts must be strictly decreasing"
        )

        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.0, 1.0, 2]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze(h_values, qoi_values)
        self.assertEqual(
            str(context.exception),
            "Two consecutive values in qoi_values are equal - provide greater precision values"
        )

        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.0, 1.4, 1.4]
        with self.assertRaises(ValueError) as context:
            calcverif_tool.analyze(h_values, qoi_values)
        self.assertEqual(
            str(context.exception),
            "Two consecutive values in qoi_values are equal - provide greater precision values"
        )

        # check runs ok when two non-consecutive values are equal but 
        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.0, 1.4, 1.0]
        calcverif_tool.analyze(h_values, qoi_values)


    def test_when_divergent_oscillating(self):
        """
        Test analyze for cases when QOI values are diverging or oscillatory 
        with exactly three mesh values. This test captures stdout output and verifies the appropriate
        messages are printed.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        # if this is true, the output that would normally be printed to screen is collected in a string
        # instead, to check the output messages are correct. However, running this seems to mess up output afterwards,
        # so it is not run by default
        include_printed_output_test = False
        
        ## Test divergent case with 3 mesh values.
        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.0, 2.0, 1.5] # Diverging values
        

        if include_printed_output_test:
            # Capture printed output.
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
            
            sys.stdout = sys.__stdout__
            self.assertIn("QOI values diverging. Add QOI values for finer meshes", captured_output.getvalue())
        else:
            ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
        
        self.assertIsNone(ooc)
        self.assertIsNone(re)
        self.assertIsNone(gci)    
    
    
        ## Test oscillatory case with 3 mesh values.
        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.0, 0.8, 2.0] # oscilllating values
        
        if include_printed_output_test:
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
            
            sys.stdout = sys.__stdout__
            self.assertIn("QOI values oscillatory - cannot compute metrics when N=3. Repeat with N>3 meshes (could include a finer or coarser mesh).", captured_output.getvalue())
        else:
            ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
        
        self.assertIsNone(ooc)
        self.assertIsNone(re)
        self.assertIsNone(gci)

        ## Test divergent case for N=4, where the first triplet is divergent.
        h_values = [0.1, 0.2, 0.4, 0.8]
        qoi_values = [1.0, 2.0, 1.5, 2.5] # Diverging values in first triplet
        
        if include_printed_output_test:
            captured_output = io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
            
            sys.stdout = sys.__stdout__
            self.assertIn("QOI on finest meshes is diverging. Add QOI values for finer meshes", captured_output.getvalue())
        else:
            ooc, re, gci = calcverif_tool.analyze(h_values, qoi_values)
        
        self.assertIsNone(ooc)
        self.assertIsNone(re)
        self.assertIsNone(gci)    
        


    def test_N3_monotonic_const_r(self):
        """
        Test a monotonic converging case with 3 mesh values, constant r.
        This test verifies that the observed order, Richardson extrapolated value, and GCI are computed correctly.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        h_values = [0.1, 0.2, 0.4]
        qoi_values = [2.0, 3.0, 5.0] # qoi = 1 + 10h
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 1.0, 12)
        correct_gci = 1.25*(3.0-2.0)/((2**1.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "N3_mono")

        # directly call analyze_N3_case
        ooc, re, gci, _ = calcverif_tool.analyze_N3_case(h_values,qoi_values,formal_order_conv = 1.0)
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 1.0, 12)
        self.assertAlmostEqual(gci, correct_gci, 12)

        calcverif_tool = CalculationVerificationTool()       
        ooc, re, gci, _ = calcverif_tool.analyze_N3_case(h_values,qoi_values,formal_order_conv = 1.0)
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 1.0, 12)
        self.assertAlmostEqual(gci, correct_gci, 12)

        # another test
        h_values = [0.1, 0.2, 0.4]
        qoi_values = [1.50325,1.526,1.708]   # qoi = 1.5 + 3.25*h^3
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        fig, ax = plt.subplots()
        calcverif_tool.plot(ax)
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        
        correct_re = 1.50325 + (1.50325-1.526)/(2**1.0-1) # OOC > formal order so constrained OOC is used in computing RE
        correct_gci = 3.0*(1.526-1.50325)/((2**1.0-1))

        self.assertAlmostEqual(re, correct_re, 12)
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "N3_mono")

        self.assertAlmostEqual(calcverif_tool.last_ooc, 3.0, 12)
        self.assertAlmostEqual(calcverif_tool.last_re, correct_re, 12)
        self.assertAlmostEqual(calcverif_tool.last_gci, correct_gci, 12)
        self.assertAlmostEqual(calcverif_tool.last_h_values[2], 0.4, 12)
        self.assertAlmostEqual(calcverif_tool.last_qoi_values[2], 1.708, 12)



    def test_analyze_by_n_elems(self):
        """
        Test for the analyze_by_n_elems() function
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        cell_counts = [64000, 8000, 1000]
        qoi_values = [2.0, 3.0, 5.0] # qoi = 1 + 10h
        
        ooc, re, gci = calcverif_tool.analyze_by_n_elems(cell_counts,3,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 1.0, 12)
        correct_gci = 1.25*(3.0-2.0)/((2**1.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "N3_mono")
        
        self.assertEqual(len(calcverif_tool.last_h_values),3)
        self.assertAlmostEqual(calcverif_tool.last_h_values[0],1.0,8)
        self.assertAlmostEqual(calcverif_tool.last_h_values[1],2.0,8)
        self.assertAlmostEqual(calcverif_tool.last_h_values[2],4.0,8)


        ooc, re, gci = calcverif_tool.analyze_by_n_elems(cell_counts,2,qoi_values,formal_order_conv = 1.0)
        self.assertAlmostEqual(ooc, 2.0/3.0, 12)
        self.assertAlmostEqual(re, 1.0, 12)


        ooc, re, gci = calcverif_tool.analyze_by_n_elems([100,50,25],1,[1,2,4],formal_order_conv = 1.0)
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 0.0, 12)



    def test_N3_monotonic_GCI_nonconst_r(self):
        """
        Test a monotonic converging case with 3 mesh values where the mesh ratio is non-constant.
        Verifies that the computed order, Richardson extrapolated value, and GCI are as expected.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        h_values = [0.1, 0.25, 0.4]
        qoi_values = [1.50325,1.55078125,1.708]   # qoi = 1.5 + 3.25*h^3
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 3.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        self.assertAlmostEqual(re, 1.5, 12)
        correct_gci = 1.25*(1.526-1.50325)/((2**3.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "N3_mono")

        # assume formal_order_conv not known
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values)
        self.assertAlmostEqual(ooc, 3.0, 12)
        self.assertAlmostEqual(re, 1.5, 12)
        new_correct_gci = 3*(1.526-1.50325)/((2**ooc-1))
        self.assertAlmostEqual(gci, new_correct_gci, 12)


    def test_LargerN_monotonic_start_GCI_const_r(self):
        """
        Test the convergence analysis for larger N (more than 3 mesh values) when the first triplet is monotonic.
        Verifies both the computed metrics and the DataFrame of results for overlapping triplets.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        # First test with N = 4 mesh values
        h_values = [0.1, 0.2, 0.4, 0.8]
        qoi_values = [1.50325,1.526,1.708,3]   # qoi = 1.5 + 3.25*h^3, except not last one
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        correct_re = 1.50325 + (1.50325-1.526)/(2**1.0-1) # since OOC > formal order, so formal order used in RE calculation
        correct_gci = 3.0*(1.526-1.50325)/((2**1.0-1))
        self.assertAlmostEqual(re, correct_re, 12)
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "largeN_mono")
        

        # Next test with N = 6 mesh values
        h_values = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        qoi_values = [1.50325,1.526,1.708,3.164,14.812,100 ]   # qoi = 1.5 + 3.25*h^3, except not last one
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        correct_re = 1.50325 + (1.50325-1.526)/(2**1.0-1) # since OOC > formal order, so formal order used in RE calculation
        correct_gci = 3.0*(1.526-1.50325)/((2**1.0-1))
        self.assertAlmostEqual(re, correct_re, 12)
        self.assertAlmostEqual(gci, correct_gci, 12)
        self.assertEqual(calcverif_tool.method, "largeN_mono")

        # Test with N = 4 where the first triplet is monotonic but the last value causes oscillatory behavior.
        h_values = [0.1, 0.2, 0.4, 0.8]
        qoi_values = [1.0, 2.0, 4.0, 0.0]
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 0.0, 12)
        correct_gci = 1.25*(2.0-1.0)/((2**1.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)
        
        # Test with N = 6 where first three triplets are monotonic then one triplet is oscillatory.
        h_values = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        qoi_values = [1.0, 2.0, 4.0, 10, 20, -10]
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 1.0, 12)
        self.assertAlmostEqual(re, 0.0, 12)
        correct_gci = 1.25*(2.0-1.0)/((2**1.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)
        
        # Also test the detailed DataFrame output from the N>3 case.
        h_values = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        qoi_values = [11.0, 12.0, 14.0, 18, 26, -10]

        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        df = calcverif_tool.intermediate_triplet_metrics

        self.assertEqual(len(df),4)
        expected_columns = ['triplet', 'convergence', 'OOC', 'RE', 'GCI']
        actual_columns = list(df.columns)
        self.assertListEqual(actual_columns, expected_columns)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[0],10)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[1],10)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[2],10)
        self.assertTrue(pd.isna(df['OOC'].iloc[3]))

        self.assertAlmostEqual(re,  df['RE' ].iloc[0],10)
        self.assertAlmostEqual(re,  df['RE' ].iloc[1],10)
        self.assertAlmostEqual(re,  df['RE' ].iloc[2],10)
        self.assertTrue(pd.isna(df['RE'].iloc[3]))

        self.assertAlmostEqual(gci, df['GCI'].iloc[0],10)
        self.assertAlmostEqual((1.25/(2**1.0-1))*(14-12), df['GCI'].iloc[1],10)
        self.assertAlmostEqual((1.25/(2**1.0-1))*(18-14), df['GCI'].iloc[2],10)        
        self.assertTrue(pd.isna(df['GCI'].iloc[3]))

        # Repeat last test with direct call to 
        h_values = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        qoi_values = [11.0, 12.0, 14.0, 18, 26, -10]

        ooc, re, gci = calcverif_tool.analyze_largeN_mono_case(h_values,qoi_values,formal_order_conv = 1.0)
        calcverif_tool.plot()
        
        df = calcverif_tool.intermediate_triplet_metrics

        self.assertEqual(len(df),4)
        expected_columns = ['triplet', 'convergence', 'OOC', 'RE', 'GCI']
        actual_columns = list(df.columns)
        self.assertListEqual(actual_columns, expected_columns)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[0],10)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[1],10)
        self.assertAlmostEqual(ooc, df['OOC'].iloc[2],10)
        self.assertTrue(pd.isna(df['OOC'].iloc[3]))

        self.assertAlmostEqual(re,  df['RE' ].iloc[0],10)
        self.assertAlmostEqual(re,  df['RE' ].iloc[1],10)
        self.assertAlmostEqual(re,  df['RE' ].iloc[2],10)
        self.assertTrue(pd.isna(df['RE'].iloc[3]))

        self.assertAlmostEqual(gci, df['GCI'].iloc[0],10)
        self.assertAlmostEqual((1.25/(2**1.0-1))*(14-12), df['GCI'].iloc[1],10)
        self.assertAlmostEqual((1.25/(2**1.0-1))*(18-14), df['GCI'].iloc[2],10)        
        self.assertTrue(pd.isna(df['GCI'].iloc[3]))

    
    
    
    def test_LargerN_monotonic_start_GCI_nonconst_r(self):
        """
        Test convergence analysis for larger N when the mesh ratios are non-constant.
        Verifies that the computed metrics are correct under these conditions.
        """     
        calcverif_tool = CalculationVerificationTool()       

        # Test with N = 4 mesh values with non-constant ratio due to a nonstandard second h_value.
        h_values = [0.1, 0.25, 0.4, 0.8]            # r not constant, due to 0.25
        qoi_values = [1.50325,1.55078125,1.708,3]   # qoi = 1.5 + 3.25*h^3, except not last one
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 4.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        self.assertAlmostEqual(re, 1.5, 12)
        correct_gci = 3.0*(1.55078125-1.50325)/((2.5**3.0-1))
        self.assertAlmostEqual(gci, correct_gci, 12)

        # Test with N = 6 mesh values with non-constant ratio.
        h_values = [0.1, 0.25, 0.4, 0.8, 1.6, 3.2]                  # r not constant, due to 0.25
        qoi_values = [1.50325,1.55078125,1.708,3.164,14.812,100 ]   # qoi = 1.5 + 3.25*h^3, except not last one
        
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv = 5)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 12)
        self.assertAlmostEqual(re, 1.5, 12)
        correct_gci = 3.0*(1.55078125-1.50325)/((2.5**ooc-1))
        self.assertAlmostEqual(gci, correct_gci, 12)


    def test_Least_squares(self):
        """
        Test the least squares method 
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        # These inputs would normally lead to the triplets method being called, since there is no oscillation. 
        # But we can directly call analyze_using_least_squares_method to bypass the method selection logic. The least squares
        # solution should be close to the exact solution

        h_values = [0.1, 0.2, 0.4, 0.8]
        qoi_values = [1.50325,1.526,1.708,3.164]   # qoi = 1.5 + 3.25*h^3
        
        ooc, re, gci = calcverif_tool.analyze_using_least_squares_method(h_values,qoi_values,formal_order_conv=3.0)
        calcverif_tool.plot()
        
        self.assertAlmostEqual(ooc, 3.0, 4)
        self.assertAlmostEqual(re, 1.5, 4)
        correct_gci = 1.25*(1.526-1.50325)/((2**3.0-1))
        self.assertAlmostEqual(gci, correct_gci)

        ls_fit_curve = calcverif_tool.ls_fit_curve
        h = ls_fit_curve['h']
        q = ls_fit_curve['q']
        expected_q = 1.5 + 3.25*h**3
        self.assertLess(np.max(np.abs((q - expected_q))), 1e-5)

    def test_Least_squares_method_logic(self):
        """
        Test the least squares method runs when oscillatory inputs are provided
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()       

        # These inputs would normally lead to the triplets method being called, since there is no oscillation. 
        # But we can directly call analyze_using_least_squares_method to bypass the method selection logic. The least squares
        # solution should be close to the exact solution

        h_values = [0.1, 0.2, 0.4, 0.8]
        qoi_values = [1, 0.9, 2, 4]  
        ooc, re, gci = calcverif_tool.analyze(h_values,qoi_values,formal_order_conv=3.0)
        # no exact solution here, just checking the correct method was found
        self.assertEqual(calcverif_tool.method, "least_squares")


    def test_OOC_constraint(self):
        """
        Test the functionality constraining OOC before using in RE and GCI works correctly.
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        calcverif_tool = CalculationVerificationTool()
        
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.3, None), 0.5)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.49, None), 0.5)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.51, None), 0.51)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.8, None), 0.8)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(4.56, None), 4.56)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.3, 2.0), 0.5)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.49, 2.0), 0.5)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(0.51, 2.0), 0.51)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(1.6, 2.0), 1.6)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(1.81, 2.0), 2.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(1.99, 2.0), 2.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(2.4, 2.0), 2.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(5, 2.0), 2.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(4, 5.0), 4.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(4.7, 5.0), 5.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(5.1, 5.0), 5.0)
        self.assertAlmostEqual(calcverif_tool._compute_constrained_OOC(10, 5.0), 5.0)

        h = np.array([1,2,4])

        ooc, re, gci = calcverif_tool.analyze(h, h**1.0,None)
        self.assertAlmostEqual(ooc, 1.0)
        self.assertAlmostEqual(re, 0.0)
        self.assertAlmostEqual(gci, 3.0)

        ooc, re, gci = calcverif_tool.analyze(h, h**0.3,None)
        self.assertAlmostEqual(ooc, 0.3)
        self.assertAlmostEqual(re, 1 + (1-2**0.3)/(2**0.5-1))
        self.assertAlmostEqual(gci, 3*(2**0.3-1)/(2**0.5-1))

        ooc, re, gci = calcverif_tool.analyze(h, h**0.3,2.0)
        self.assertAlmostEqual(ooc, 0.3)
        self.assertAlmostEqual(re, 1 + (1-2**0.3)/(2**0.5-1))
        self.assertAlmostEqual(gci, 3*(2**0.3-1)/(2**0.5-1))

        ooc, re, gci = calcverif_tool.analyze(h, h**1.0,2.0)
        self.assertAlmostEqual(ooc, 1.0)
        self.assertAlmostEqual(re, 0.0)
        self.assertAlmostEqual(gci, 3.0)

        ooc, re, gci = calcverif_tool.analyze(h, h**1.9,2.0)
        self.assertAlmostEqual(ooc, 1.9)
        self.assertAlmostEqual(re, 1 + (1-2**1.9)/(2**2-1))
        self.assertAlmostEqual(gci, 1.25*(2**1.9-1)/(2**2-1))

        ooc, re, gci = calcverif_tool.analyze(h, h**2.15,2.0)
        self.assertAlmostEqual(ooc, 2.15)
        self.assertAlmostEqual(re, 1 + (1-2**2.15)/(2**2-1))
        self.assertAlmostEqual(gci, 1.25*(2**2.15-1)/(2**2-1))

        ooc, re, gci = calcverif_tool.analyze(h, h**2.5,2.0)
        self.assertAlmostEqual(ooc, 2.5)
        self.assertAlmostEqual(re, 1 + (1-2**2.5)/(2**2-1))
        self.assertAlmostEqual(gci, 3*(2**2.5-1)/(2**2-1))


if __name__ == '__main__':
    unittest.main()