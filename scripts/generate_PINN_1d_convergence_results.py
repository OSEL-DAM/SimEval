
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
# To support reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf   # See project README
import deepxde as dde     # See project README   
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt

# So src folder can be seen
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/scripts"
src_path = os.path.abspath(os.path.join(base_dir, "..", "src"))
sys.path.insert(0, src_path)



from PINN_verify_1d_elliptic import run_1d_PINN_error_convergence, plot_PINN_convergence_results


##########################################################################################
#
#
#  This file contains the function calls used to generate the PINN 1D solver convergence 
#  analysis results and figures
#
#
##########################################################################################





#####################################################################################
# These are the calls used to generate the results
# Note: they are inside an if False statement to avoid accidentially 
# re-running - takes a long time to run
#####################################################################################

if False:   
    ## Baseline case
    run_1d_PINN_error_convergence("MMS1",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 100000, 
                                  train_distribution="pseudo",  
                                  use_best_test_loss = True,
                                  learning_rate_decay = True,
                                  output_file = "MMS1_traindist_pseudo_usebesttestlose_true_lrdecay_true_epochs_100000.csv")
            
    
    ## 10000 epochs instead of 100000 
    run_1d_PINN_error_convergence("MMS1",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 10000, 
                                  train_distribution="pseudo",  
                                  use_best_test_loss = True,
                                  learning_rate_decay = True,
                                  output_file = "MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_10000.csv")
            
    
    ## Uniform colloc rather than random
    run_1d_PINN_error_convergence("MMS1",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 100000, 
                                  train_distribution="uniform",  
                                  use_best_test_loss = True,
                                  learning_rate_decay = True,
                                  output_file = "MMS1_traindist_uniform_usebesttestloss_true_lrdecay_true_epochs_100000.csv")
    
    
    # use_best_test_loss = False
    run_1d_PINN_error_convergence("MMS1",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 100000, 
                                  train_distribution="pseudo",  
                                  use_best_test_loss = False,
                                  learning_rate_decay = True,
                                  output_file = "MMS1_traindist_pseudo_usebesttestloss_false_lrdecay_true_epochs_100000.csv")
    
    # learning_rate_decay = False
    run_1d_PINN_error_convergence("MMS1",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 100000, 
                                  train_distribution="pseudo",  
                                  use_best_test_loss = True,
                                  learning_rate_decay = False,
                                  output_file = "MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_false_epochs_100000.csv")
            
    
    # MMS2 instead of MMS1 (different test problem)
    run_1d_PINN_error_convergence("MMS2",
                                  seeds = [0,1,2,3,4], 
                                  architectures = [[2,20],[4,10],[4,20]],
                                  num_colloc_array = [10,20,40,80,160,320,640],
                                  num_epochs = 100000, 
                                  train_distribution="pseudo",  
                                  use_best_test_loss = True,
                                  learning_rate_decay = True,
                                  output_file = "MMS2_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_100000.csv")




#####################################################################################
# Generate figure showing all errors for baseline case
#####################################################################################

df = pd.read_csv("../saved_results/MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_100000_SUMMARY.csv")
plot_PINN_convergence_results(df)




#####################################################################################
# Generate figure comparing l2 error using different training choices
#####################################################################################

def plot_comparison_l2_convergence():
    
    # uncomment for publication version of figure
    #plt.rcParams.update({'font.size': 14,
    #                     'axes.labelsize': 14,
    #                     'xtick.labelsize': 14,
    #                     'ytick.labelsize': 14,
    #                     'legend.fontsize': 14,
    #                     'lines.linewidth': 1.8,
    #                     'lines.markersize': 6})
    
    
    fig, axes = plt.subplots(3,2,constrained_layout=True)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    files = ["../saved_results/MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_100000_SUMMARY.csv",
             "../saved_results/MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_10000_SUMMARY.csv",
             "../saved_results/MMS1_traindist_uniform_usebesttestloss_true_lrdecay_true_epochs_100000_SUMMARY.csv",
             "../saved_results/MMS1_traindist_pseudo_usebesttestloss_false_lrdecay_true_epochs_100000_SUMMARY.csv",
             "../saved_results/MMS1_traindist_pseudo_usebesttestloss_true_lrdecay_false_epochs_100000_SUMMARY.csv",
             "../saved_results/MMS2_traindist_pseudo_usebesttestloss_true_lrdecay_true_epochs_100000_SUMMARY.csv"]
    
    titles = ['Base', '10000 epochs', 'Uniform colloc pts', 'No use best loss', 'No LR decay', 'Different test problem']


    for ax_index, (file, title) in enumerate(zip(files, titles)):
        df = pd.read_csv(file)
        num_trainable_parameters_list = np.unique(
            df["num_trainable_params"].values
        )

        ax = axes[int(ax_index/2)][ax_index%2]
    
        for i, ntp in enumerate(num_trainable_parameters_list):
            df_restricted = df[df['num_trainable_params']==ntp]
            fmt = markers[i % len(markers)] + '-'
            
            nl = df_restricted['num_layers'].iloc[0]
            npl = df_restricted['neurons_per_layer'].iloc[0]
            arch_str = f"{nl} x {npl} network"
            
            ax.loglog(df_restricted['num_colloc'], df_restricted['err_l2_median'],  fmt, label = arch_str)
            ax.fill_between(df_restricted['num_colloc'], df_restricted['err_l2_min'], df_restricted['err_l2_max'], alpha=0.2)
            
            ax.set_ylim([1e-6,1e1])
            
            ax.set_title(title)

    axes[2,0].set_xlabel('Num collocation points')
    axes[2,1].set_xlabel('Num collocation points')
    axes[0,0].set_ylabel('$l_2$ error')
    axes[1,0].set_ylabel('$l_2$ error')
    axes[2,0].set_ylabel('$l_2$ error')
    
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.), loc='lower center')
        
    # uncomment for publication version of figure
    #fig.set_size_inches(7.1, 5.5)    
    #fig.savefig('Figure3.pdf', bbox_inches='tight')       # vector
    #fig.savefig('Figure3.eps', bbox_inches='tight')       # vector
    #fig.savefig('Figure3.png', dpi=300, bbox_inches='tight')  # raster backup
    #fig.savefig('Figure3.tiff', dpi=300, bbox_inches='tight')  # raster backup


# call above function
plot_comparison_l2_convergence()



