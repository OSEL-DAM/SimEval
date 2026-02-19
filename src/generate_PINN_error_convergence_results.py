from PINN_error_convergence import PINN_convergence_plots
from PINN_error_convergence import PINN_error_convergence
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# These are the calls used to generate the results that are saved in the saved_results folder, that are 
# discussed in the PINN notebook.

    
# Base case
PINN_error_convergence("MMS1",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 100000,
                       train_distribution="pseudo",
                       use_best_test_loss = True,
                       learning_rate_decay = True,
                       output_file = "MMS1_traindist_pseudo_usebesttest_true_lrdecay_true_epochs_100000.csv")
    
# 10000 epochs instead of 100000
PINN_error_convergence("MMS1",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 10000,
                       train_distribution="pseudo",
                       use_best_test_loss = True,
                       learning_rate_decay = True,
                       output_file = "MMS1_traindist_pseudo_usebesttest_true_lrdecay_true_epochs_10000.csv")

# Train distribution of 'uniform' instead of 'pseudo'
PINN_error_convergence("MMS1",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 100000,
                       train_distribution="uniform",
                       use_best_test_loss = True,
                       learning_rate_decay = True,
                       output_file = "MMS1_traindist_uniform_usebesttest_true_lrdecay_true_epochs_100000.csv")
    
# use_best_test_loss = False
PINN_error_convergence("MMS1",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 100000,
                       train_distribution="pseudo",
                       use_best_test_loss = False,
                       learning_rate_decay = True,
                       output_file = "MMS1_traindist_pseudo_usebesttest_false_lrdecay_true_epochs_100000.csv")

# learning_rate_decay = False
PINN_error_convergence("MMS1",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 100000,
                       train_distribution="pseudo",
                       use_best_test_loss = True,
                       learning_rate_decay = False,
                       output_file = "MMS1_traindist_pseudo_usebesttest_true_lrdecay_false_epochs_100000.csv")

# MMS2 instead of MMS1 (different test problem)
PINN_error_convergence("MMS2",
                       num_colloc_array = [10,100,1000,10000],
                       num_layers_array = [2,4],
                       neurons_per_layer_array = [5,10,20],
                       num_epochs = 100000,
                       train_distribution="pseudo",
                       use_best_test_loss = True,
                       learning_rate_decay = True,
                       output_file = "MMS2_traindist_pseudo_usebesttest_true_lrdecay_true_epochs_100000.csv")


