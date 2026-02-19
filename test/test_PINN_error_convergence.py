
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import sys
import inspect
import unittest
import numpy as np
import tensorflow as tf   # See project README
import deepxde as dde     # See project README   
import random
import math


## Attempt to make these tests as repeatable as possible, though the results 
## are still stochastic across sessions despite these lines.
os.environ["PYTHONHASHSEED"] = str(0)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)




from PINN_error_convergence import PINN_error_convergence



class TestPinnErrorConvergence(unittest.TestCase):
    """
    Test the error convergence analysis code that varies the number of collocation points or the network architecture
    """
    
    def test_PINN_error_convergence_MMS1(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
        dde.config.set_random_seed(42)    # note: not fully deterministic even with this seed set.
        
        df = PINN_error_convergence("MMS1", 
                                    num_colloc_array = [100, 200], 
                                    num_layers_array = [2],
                                    neurons_per_layer_array = [10],
                                    num_epochs = 10000,
                                    train_distribution = "pseudo",
                                    use_best_test_loss = True,
                                    learning_rate_decay = True,
                                    output_file = None)
        
        # test the df is size 2 as expected, the initial columns have the correct info, and the 
        # loss was small (indicating MMS problem was solved correctly)
        self.assertEqual(len(df),2)
        self.assertEqual(df['num_colloc'].tolist(), [100,200])
        self.assertEqual(df['num_layers'].tolist(), [2,2])
        self.assertEqual(df['neurons_per_layer'].tolist(), [10,10])
        self.assertEqual(df['num_trainable_params'].tolist(), [141,141])
        self.assertTrue((df['loss_train'] < 0.1).all()) # May occascially fail as results not deterministic. If fail try restart and re-run 
    

    def test_PINN_error_convergence_MMS2(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
        dde.config.set_random_seed(42)    # note: not fully deterministic even with this seed set.
        
        df = PINN_error_convergence("MMS2", 
                                    num_colloc_array = [100], 
                                    num_layers_array = [2],
                                    neurons_per_layer_array = [10],
                                    num_epochs = 10000,
                                    train_distribution = "pseudo",
                                    use_best_test_loss = True,
                                    learning_rate_decay = True,
                                    output_file = None)
        

        # test the df is size 1 as expected, the initial columns have the correct info, and the 
        # loss was small (indicating MMS problem was solved correctly)
        self.assertEqual(len(df),1)
        self.assertEqual(df['num_colloc'].tolist(), [100])
        self.assertEqual(df['num_layers'].tolist(), [2])
        self.assertEqual(df['neurons_per_layer'].tolist(), [10])
        self.assertEqual(df['num_trainable_params'].tolist(), [141])
        self.assertTrue((df['loss_train'] < 0.1).all()) # May occascially fail as results not deterministic. If fail try restart and re-run 
   




if __name__ == '__main__':
    unittest.main()
    