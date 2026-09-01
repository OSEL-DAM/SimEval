
import os

os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
# To support reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import inspect
import unittest
import numpy as np
import tensorflow as tf   # See project README
import deepxde as dde     # See project README   
import math

try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)



from PINN_1d_elliptic import solve_1d_elliptic_PINN



class TestPinn1DEllitpic(unittest.TestCase):    
    """
    Unit tests for the solve_1d_elliptic_PINN function that solves a 1D second order ODE using a PINN.
    """        
    
    def test_solve_simple_prob(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()   # Useful to include in new solves to avoid gradual slow-down
        tf.compat.v1.reset_default_graph() # Useful to include in new solves to avoid gradual slow-down

        pi = tf.constant(math.pi)

        # solve u'' = f with the below f(x) and u'(1)=2pi        
        def f(x):
            return -4*pi*pi*tf.math.sin(2*pi*x)
        
        bc = 2*pi
        
        # compare with the true solution
        def exact_soln(x):
            return np.sin(2*np.pi*x)
        
        x, u, soln_info = solve_1d_elliptic_PINN(f, bc, num_layers=5, num_colloc=100, seed = 0) # pass seed to ensure reproducibility for fixed hardware
        u_true = exact_soln(x)

        err = np.max(np.abs(u-u_true))
        self.assertLess(err, 0.1)
        self.assertLess(soln_info['loss_train'], 1e-2)  
        self.assertLess(soln_info['loss_test'], 1e-2)   
        self.assertEqual(len(soln_info['colloc_points']), 102) # as num_colloc = 100, and DeepXDE creates 100 internal points plus 0 and 1
        self.assertEqual(len(soln_info['resid_at_colloc']), 102)
        self.assertEqual(len(soln_info['x_dense']), 10000) 
        self.assertEqual(len(soln_info['resid_at_x_dense']), 10000)


        
        
    def test_solve_simple_prob_linear_solution(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()   # Useful to include in new solves to avoid gradual slow-down
        tf.compat.v1.reset_default_graph() # Useful to include in new solves to avoid gradual slow-down

        # solve u'' = f with the below f=0 and u'(1)=1        
        def f(x):
            return 0.0
        
        bc = 1
        
        x, u, _ = solve_1d_elliptic_PINN(f, bc,
                                         num_output_points = 101,
                                         num_colloc = 100,
                                         num_layers = 1,
                                         neurons_per_layer = 10,
                                         num_epochs = 2000, 
                                         seed = 0)  # note seed for reproducibility
                        
        u_true = x
        err = np.max(np.abs(u-u_true))
        self.assertLess(err, 0.05) # If fail try restart and re-run 
        
        
    def test_train_distribution_parameter(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()   # Useful to include in new solves to avoid gradual slow-down
        tf.compat.v1.reset_default_graph() # Useful to include in new solves to avoid gradual slow-down

        pi = tf.constant(math.pi)
    
        # solve u'' = f with the below f(x) and u'(1)=2pi        
        def f(x):
            return -4*pi*pi*tf.math.sin(2*pi*x)
        
        bc = 2*pi
        
        # Solve passing in train_distribution = 'uniform' and check the colloc points are on a uniform grid. 
        x, u, soln_info = solve_1d_elliptic_PINN(f, bc, num_layers=5, num_colloc = 10, num_epochs = 100, train_distribution = 'uniform', seed = 0)    
        true_colloc_points = np.array([0,1]+[i/11 for i in range(1,11)]).reshape(-1,1)
        diff = np.max(np.abs(true_colloc_points - soln_info['colloc_points']))
        self.assertLess(diff, 1e-6)

 
        
if __name__ == '__main__':
    unittest.main()
    