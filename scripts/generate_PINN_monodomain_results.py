
import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
# To support reproducibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import sys
import numpy as np
import tensorflow as tf   # See project README
import deepxde as dde     # See project README   
import pandas as pd
import itertools
from matplotlib import pyplot as plt

# So src folder can be seen
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/scripts"
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)


from PINN_monodomain import solve_monodomain_PINN, MonodomainMMSProblem, HeatEquationProblem   #plot_monodomain_solution, plot_monodomain_error



##########################################################################################
#
#
#  This file contains the function calls used to generate the PINN monodomain solver 
#  analysis results and figures
#
#
##########################################################################################



####################################################################
# Helper function for computing total error
####################################################################
def compute_sum_normalized_linf_errors(problem, trained_model, end_time, normalization_factors):
    """
    Computes total error, as follows. Using a uniform sample of times (every 0, 0.1T, 0.2T, .. T), compute average (over time)
    L_inf (over space) error for each of V, u1, u2, u3. Weight by normalization factors and sum.
    """
    t_values = np.linspace(0,end_time,11)
    n = 50

    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    total_error = 0.0
    
    ## Uncomment this and below lines to print component-wise total errors (note: not averaged for time, not normalized)
    #V_error = 0
    #u1_error = 0
    #u2_error = 0
    #u3_error = 0
    for t in t_values:
        points = np.column_stack(( xy, np.full(xy.shape[0], t) ))
        pred = trained_model.predict(points)
        exact = problem.exact_solution_numpy( xy[:, 0], xy[:, 1], t)
        
        for i in range(4):
            total_error += np.max(np.abs(pred[:,i] - exact[:,i]))/normalization_factors[i]
    
        #V_error += np.max(np.abs(pred[:,0] - exact[:,0]))
        #u1_error += np.max(np.abs(pred[:,1] - exact[:,1]))
        #u2_error += np.max(np.abs(pred[:,2] - exact[:,2]))
        #u3_error += np.max(np.abs(pred[:,3] - exact[:,3]))
                
   
    #print(f"{V_error} {u1_error} {u2_error} {u3_error}")
        
    return total_error/len(t_values)




####################################################################
# Main code for generating monodomain convergence results
####################################################################

# Weights are the max values of the exact solution over all x in [0,1] x [0,1], t in [0,T] where T=1 for monodomain 
# test problem and T=0.1 for heat equation test problem (except - last weight is set to be 1, exact soln for u3=0 for x,y,t).
# Max value for u1 found numerically, all others easily seen by inspection.
mono_normalization_factors = [np.sqrt(2), 3.9530, 1.0, 1.0]
heat_normalization_factors = [1.0, 0.11, 1.0, 1.0]

levels = [1,2,4,8]
seeds = [0,1,2,3,4]
results = []
output_file = 'monodomain_heat_eqn_res_2by32_10000epochs.csv'                    #10000 or 50000


total = len(["mono", "heat"]) * len(levels) * len(seeds)

# loop over all combinations of specificed inputs
for i, (problem_name, level, seed) in enumerate(itertools.product(["mono", "heat"], levels, seeds)):
    print("==========================================")
    print(f"Run {i+1}/{total}: {problem_name}, level={level}, [seed={seed}]")
    print("==========================================")
    
    tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
    tf.compat.v1.reset_default_graph()

    if problem_name == 'mono':                
        end_time = 1.0
        problem = MonodomainMMSProblem(end_time=end_time)
        factors = mono_normalization_factors
    elif problem_name == 'heat':                
        end_time = 0.1
        problem = HeatEquationProblem(end_time=end_time)
        factors = heat_normalization_factors
    else:
        assert(False)
        
    num_domain = int(level*level*level*10)
    num_boundary = int(level*level*4)
    
    trained_model = solve_monodomain_PINN(problem, 
                                          num_domain = num_domain,
                                          num_boundary = num_boundary,
                                          num_layers = 2,
                                          neurons_per_layer = 32,
                                          num_epochs = 10000,                    ##10000 or 50000
                                          seed = seed)
    #plot_monodomain_solution(trained_model, times=[0,end_time/2, end_time])
    #plot_monodomain_error(problem, trained_model, times=[0,end_time/2,end_time])

    total_error = compute_sum_normalized_linf_errors(problem, trained_model, end_time, factors)
    
    
    results.append(
        {"problem": problem_name,
         "seed":seed,
         "level":level,
         "num_domain":num_domain,
         "num_boundary":num_boundary,
         "total_error": total_error})
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)    # save without the auto-generated index
    
    
# Create summary document that calculates the mean, median, min and max over the replicates (different seeds)
output_cols = ['total_error']
group_cols = [c for c in df.columns if c not in ['seed', *output_cols]]

df_summary = df.groupby(group_cols, as_index=False).agg(
{
    col: ['median', 'mean', 'min', 'max'] 
    for col in output_cols
})

df_summary.columns = [ '_'.join(x).rstrip('_') if isinstance(x, tuple) else x for x in df_summary.columns]        

df.to_csv(output_file, index=False)    # save without the auto-generated index
summary_output_file = output_file[:-4] + "_SUMMARY.csv" 
df_summary.to_csv(summary_output_file, index=False) 
        
    
    



####################################################################
# Create the plot
####################################################################

#plt.rcParams.update({'font.size': 14,
#                     'axes.labelsize': 14,
#                     'xtick.labelsize': 14,
#                     'ytick.labelsize': 14,
#                     'legend.fontsize': 10,
#                     'lines.linewidth': 1.8,
#                     'lines.markersize': 6})


fig, ax = plt.subplots(1,1,constrained_layout=True)


problem_markers = {"mono": "o", "heat": "s"}
problem_labels = {"mono": "Monodomain", "heat": "Heat eqn"}

files = [ ("../saved_results/monodomain_heat_eqn_res_2by32_10000epochs_SUMMARY.csv", "10k epochs", "--"),
          ("../saved_results/monodomain_heat_eqn_res_2by32_50000epochs_SUMMARY.csv", "50k epochs", "-") ]

df_full = pd.read_csv("../saved_results/monodomain_heat_eqn_res_2by32_50000epochs_SUMMARY.csv")


for filename, epoch_label, linestyle in files:
    df_full = pd.read_csv(filename)
    for problem_name in ["mono", "heat"]:
        df = df_full[df_full["problem"] == problem_name]

        label = f"{problem_labels[problem_name]}, {epoch_label}"
        line, = ax.loglog( df["num_domain"], df["total_error_median"], linestyle=linestyle, marker=problem_markers[problem_name], linewidth=1.5,markersize=5, label=label)

        ax.fill_between( df["num_domain"], df["total_error_min"], df["total_error_max"], color=line.get_color(), alpha=0.15)


ax.set_xlabel("Domain collocation points")
ax.set_ylabel("Normalized total error")


ax.legend()

#fig.savefig('Figure4b.pdf', bbox_inches='tight')       # vector
#fig.savefig('Figure4b.eps', bbox_inches='tight')       # vector
#fig.savefig('Figure4b.png', dpi=300, bbox_inches='tight')  # raster backup
#fig.savefig('Figure4b.tiff', dpi=300, bbox_inches='tight')  # raster backup







