
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
# To support reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import inspect
import unittest
import tensorflow as tf   # See project README
import deepxde as dde     # See project README   
import pandas as pd
import numpy as np

try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)




from PINN_verify_1d_elliptic import run_1d_PINN_error_convergence



class TestPinnVerify1dElliptic(unittest.TestCase):
    """
    Test the error convergence analysis code that varies the number of collocation points or the network architecture
    """    
    def test_run_1d_PINN_error_convergence_MMS1(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        
        df = run_1d_PINN_error_convergence("MMS1",    
                                            seeds = [42],                              
                                            architectures = [[2,10]],
                                            num_colloc_array = [100, 200], 
                                            num_epochs = 10000,
                                            train_distribution = "pseudo",
                                            use_best_test_loss = True,
                                            learning_rate_decay = True)
                
        # test the df is size 2 as expected, the initial columns have the correct info, and the 
        # loss was small (indicating MMS problem was solved correctly)
        self.assertEqual(len(df),2)
        self.assertEqual(df['num_colloc'].tolist(), [100,200])
        self.assertEqual(df['num_layers'].tolist(), [2,2])
        self.assertEqual(df['neurons_per_layer'].tolist(), [10,10])
        self.assertEqual(df['num_trainable_params'].tolist(), [141,141])
        self.assertTrue((df['loss_train'] < 0.1).all()) 
        

    def test_run_1d_PINN_error_convergence_MMS2(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        
        df = run_1d_PINN_error_convergence("MMS2", 
                                           seeds = [0,1],
                                           architectures = [[2,10]],
                                           num_colloc_array = [100],
                                           num_epochs = 10000,
                                           train_distribution = "pseudo",
                                           use_best_test_loss = True,
                                           learning_rate_decay = True,
                                           output_file = "test_pinn_error_convergence_fnc.csv")
                

        # test the df is size 1 as expected, the initial columns have the correct info, and the 
        # loss was small (indicating MMS problem was solved correctly)
        self.assertEqual(len(df),2)
        self.assertEqual(df['num_colloc'].tolist(), [100,100])
        self.assertEqual(df['num_layers'].tolist(), [2,2])
        self.assertEqual(df['neurons_per_layer'].tolist(), [10,10])
        self.assertEqual(df['num_trainable_params'].tolist(), [141,141])
        self.assertTrue((df['loss_train'] < 0.1).all()) 
        

        # check the dataframes saved to file
        read_df = pd.read_csv("test_pinn_error_convergence_fnc.csv")
        read_df_summary = pd.read_csv("test_pinn_error_convergence_fnc" + "_SUMMARY.csv")
        
        self.assertEqual(len(read_df),2)
        self.assertEqual(len(read_df_summary),1)

        self.assertEqual(read_df['num_colloc'].tolist(), [100,100])
        self.assertEqual(read_df['num_layers'].tolist(), [2,2])
        self.assertEqual(read_df['neurons_per_layer'].tolist(), [10,10])
        self.assertEqual(read_df['num_trainable_params'].tolist(), [141,141])
        
        self.assertEqual(read_df_summary['num_colloc'].tolist(), [100])
        self.assertEqual(read_df_summary['num_layers'].tolist(), [2])
        self.assertEqual(read_df_summary['neurons_per_layer'].tolist(), [10])
        self.assertEqual(read_df_summary['num_trainable_params'].tolist(), [141])

        # check the summary results are calculated correctly
        self.assertAlmostEqual(read_df_summary['err_l2_mean'][0], np.mean(read_df['err_l2']),14)
        self.assertAlmostEqual(read_df_summary['err_l2_min'][0], np.min(read_df['err_l2']),14)
        self.assertAlmostEqual(read_df_summary['err_l2_max'][0], np.max(read_df['err_l2']),14)
        
        # delete created file
        for output_file in ["test_pinn_error_convergence_fnc.csv", "test_pinn_error_convergence_fnc_SUMMARY.csv"]:
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == '__main__':
    unittest.main()
    