"""
PINN solver for 2D time-dependent monodomain equation

This module provides:
 - `solve_monodomain_PINN` (function): a DeepXDE-based PINN solver for the monodomain equation in 2D on the unit square, with a three-state cell model.
 - `MonodomainMMSProblem` (class): a problem class defining a MMS-based monodomain problem with known exact solution
 - `HeatEquationProblem` (class): a problem class defining a MMS-based simplified uncoupled monodomain problem - essentially the heat equation - with known exact solution
 - `plot_monodomain_solution` (function): function for plotting PINN solution
 - `plot_monodomain_error` (function): function for plotting solution error

Also defined is
 - `InMemoryBestModel` (class), a callback for retaining the model weights associated with the lowest test PDE loss.
 - `monodomain_residual_fnc` (function): function used to compute the PINN loss.
"""


import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import numpy as np
import tensorflow as tf   # See project README
import deepxde as dde     # See project README 
import matplotlib.pyplot as plt





class MonodomainMMSProblem:
    """
    Problem class defining MMS-based 2D monodomain problem from https://cdrh-rst.fda.gov/verification-test-problems-cardiac-electrophysiology-modeling-software
    (Pathmanathan and Gray, IJNMBE, 2014).
    
    An instantiation of this class can be passed to the main PINN monodomain solve function to solve this problem. It defines
    the monodomain parameters (chi, Cm, sigma), the end time, the ionic current and the cell model right-hand-side function.
    """
    def __init__(self, end_time):
        """
        Constructor - takes in simulation end time and sets up parameters
        """
        self.chi = 3.0
        self.Cm = 2.0
        self.beta = -5.9
        self.sigma_x = 1.1/np.pi**2
        self.sigma_y = 1.2/np.pi**2
        self.end_time = end_time
        
    def get_Iionic(self,Vu):
        """
        Cell model ionic current function
        
        Parameters:
            Vu (tensorflow array): Voltage and cell model state variables as array
        """
        V = Vu[:,0:1]    
        u1 = Vu[:,1:2]
        u2 = Vu[:,2:3]
        u3 = Vu[:,3:4]

        i_ion = (-0.5 * self.Cm * (u1 + u3 - V) * u2**2 * (V - u3) + self.beta * (V - u3) / self.chi)
        return i_ion
    
    def cell_model_rhs(self,Vu):
        """
        Cell model ODE right-hand-side function
        
        Parameters:
            Vu (tensorflow array): Voltage and cell model state variables as array
        Returns:
            TensorFlow array of RHS terms
        """
        V = Vu[:,0:1]    
        u1 = Vu[:,1:2]
        u2 = Vu[:,2:3]
        u3 = Vu[:,3:4]
        
        a = u1 + u3 - V
    
        f1 = a**2 * u2**2 + 0.5 * a * u2**2 * (V - u3)
        f2 = -a * u2**3
        f3 = tf.zeros_like(u3)
        
        return tf.concat((f1,f2,f3), axis=1)

    def get_initial_cond(self,xyt):
        """
        Initial condition function
        
        Parameters:
            xyt (tensorflow array): array of x values, y values, t values (ignored)
        Returns:
            TensorFlow array of initial condition of V, u1, u2, u3
        """
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]

        pi = tf.constant(np.pi, dtype=xyt.dtype)
        F = tf.cos(pi * x) * tf.cos(2.0 * pi * y)
        G = 1.0 + x * y**2

        V0 = F
        u10 = G + F
        u20 = tf.pow(G, -0.5)
        u30 = tf.zeros_like(V0)
        return tf.concat((V0, u10, u20, u30), axis=1)

    def exact_solution_numpy(self, x, y, t):
        """
        Exact solution function (uses numpy)
        
        Parameters:
            x - array of x values
            y - array of y values
            t - array of t values 
        Returns:
            NumPy array of exact solution V, u1, u2, u3
        """
        F = np.cos(np.pi * x) * np.cos(2.0 * np.pi * y)
        G = 1.0 + x * y**2

        V  = np.sqrt(1.0 + t) * F
        u1 = (1.0 + t) * G + np.sqrt(1.0 + t) * F
        u2 = 1.0 / ((1.0 + t) * np.sqrt(G))
        u3 = 0.0*V

        return np.column_stack((V, u1, u2, u3))



class HeatEquationProblem:
    """
    Problem class defining parameters such that the monodomain equation becomes an uncoupled heat equation, with simple uncoupled ODEs for the state variables. The conductivity and initial
    conditions are chosen so that the exact solution is known. An instantiation of this class can be passed to the main PINN monodomain solve function to solve this problem.
    See 
    """
    def __init__(self, end_time):
        """
        Constructor - takes in simulation end time and sets up parameters
        """
        self.chi = 1.0
        self.Cm = 1.0
        self.sigma_x = 1.1/np.pi**2
        self.sigma_y = 1.2/np.pi**2
        self.end_time = end_time
        
    def get_Iionic(self,Vu):
        """
        Cell model ionic current function
        
        Parameters:
            Vu (tensorflow array): Voltage and cell model state variables as array
        """
        return 0*Vu[:,0:1]
    
    def cell_model_rhs(self,Vu):
        """
        Cell model ODE right-hand-side function
        
        Parameters:
            Vu (tensorflow array): Voltage and cell model state variables as array
        Returns:
            TensorFlow array of RHS terms
        """
        u2 = Vu[:,2:3]
        f1 = 1.1*tf.ones_like(u2)
        f2 = -1.2*u2
        f3 = tf.zeros_like(u2)
        return tf.concat((f1,f2,f3), axis=1)

    def get_initial_cond(self,xyt):
        """
        Initial condition function
        
        Parameters:
            xyt (tensorflow array): array of x values, y values, t values (ignored)
        Returns:
            TensorFlow array of initial condition of V, u1, u2, u3
        """
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]

        pi = tf.constant(np.pi, dtype=xyt.dtype)
        V0 = tf.cos(pi * x) * tf.cos(pi * y)
        u10 = tf.zeros_like(V0)
        u20 = tf.ones_like(V0)
        u30 = 0*V0

        return tf.concat((V0, u10, u20, u30), axis=1)

    def exact_solution_numpy(self, x, y, t):
        """
        Exact solution function (uses numpy)
        
        Parameters:
            x - array of x values
            y - array of y values
            t - array of t values 
        Returns:
            NumPy array of exact solution V, u1, u2, u3
        """
        V = np.exp(-2.3*t)*np.cos(np.pi*x)*np.cos(np.pi*y)
        u1 = 1.1*t*np.ones_like(x)
        u2 = np.exp(-1.2*t)*np.ones_like(x)
        u3 = 0.0*V
        return np.column_stack((V, u1, u2, u3))



class InMemoryBestModel(dde.callbacks.Callback):
    """ 
       Callback for saving the best soln. DeepXDE provides a callback for saving the best performing model to file, which can be restored
       at the end of the solve; however repeatedly writing to file is too slow. This callback was save the best performing
       model to memory. When an instance of this class is passed into model.train() as a callback, the solver will store the best performing
       weights in memory. The metric used is total test loss, not the train loss.  
    """
    def __init__(self):
        self.best_weights = None
        self.best_value = np.inf
        self.variables = None
        #self.debug_info_printed = False
        
    def set_model(self, model):
        self.model = model
        graph_vars = self.model.sess.graph.get_collection('trainable_variables')
        self.variables = graph_vars

        """
        if not self.debug_info_printed:
            print("=== DEBUG INFO ===")
            print(f"Model session type: {type(self.model.sess)}")
            print(f"Model graph type: {type(self.model.sess.graph)}")
            
            # Print available collections
            try:
                collections = self.model.sess.graph.get_all_collection_keys()
                print(f"Available collections: {collections}")
            except:
                print("Could not get collection keys")
            
            # Try to find variables
            try:
                graph_vars = self.model.sess.graph.get_collection('trainable_variables')
                print(f"Found {len(graph_vars)} trainable variables")
                self.variables = graph_vars
            except Exception as e:
                print(f"Could not get trainable_variables: {e}")
        """ 
               
    def on_epoch_end(self): # note, loss_test is updated every display_every epochs (1000 by default)
        try:
            value = float(np.sum(self.model.train_state.loss_test))
            if value < self.best_value:
                self.best_value = value
                self.loss_train_at_best_loss_test = self.model.train_state.loss_train[0]
                self.best_weights = self.model.sess.run(self.variables)
                print(f"Saving weights for new best test loss: {value:.3e}")
        except Exception as e:
            print(f"Warning: Could not save best weights: {e}")

    def restore_best_weights(self):
        if self.best_weights is not None and self.variables is not None:
            try:
                assign_ops = [var.assign(weight) for var, weight in zip(self.variables, self.best_weights)]
                self.model.sess.run(assign_ops)
                print(f"Restoring weights corresponding to the best test loss {self.best_value:.3e}")
                print("(ignore output beginning 'Best model at step...' above, which was generated by DeepXDE)\n")
            except Exception as e:
                print(f"Warning: Could not restore weights: {e}")
                


def monodomain_residual_fnc(problem, xt, vu):
    """
    Residual function to be used by DeepXDE in computing the loss.
    This function would more naturally be defined within the main PINN_monodomain_solve function, however it is defined outside that 
    so can be externally tested.
    
    Parameters:
        problem - an instantiated problem class (see MonodomainMMSProblem). Must provide 
                  chi, Cm, sigma_x, sigma_y, end_time, get_Iionic(), cell_model_rhs(), and get_initial_cond()
        xt - array [x,y,t]
        Vu - array [V, u1, u2, u3]
    
    """
    V_t  = dde.grad.jacobian(vu, xt, i=0, j=2)  #dV/dt
    V_xx = dde.grad.hessian(vu, xt, component=0, i=0, j=0) #d2V/dxx
    V_yy = dde.grad.hessian(vu, xt, component=0, i=1, j=1)

    u1_t = dde.grad.jacobian(vu, xt, i=1, j=2)
    u2_t = dde.grad.jacobian(vu, xt, i=2, j=2)
    u3_t = dde.grad.jacobian(vu, xt, i=3, j=2)
    
    i_ion = problem.get_Iionic(vu) 

    pde_residual = problem.chi * (problem.Cm * V_t + i_ion) - problem.sigma_x * V_xx - problem.sigma_y * V_yy
    
    cell_model_rhs = problem.cell_model_rhs(vu)
    
    ode1_residual = u1_t - cell_model_rhs[:,0:1]
    ode2_residual = u2_t - cell_model_rhs[:,1:2]
    ode3_residual = u3_t - cell_model_rhs[:,2:3]
        
    return [pde_residual, ode1_residual, ode2_residual, ode3_residual]        
          



def solve_monodomain_PINN(problem,
                          num_domain = 1000,
                          num_boundary = 200,
                          num_test = 1000,
                          num_layers = 2,
                          neurons_per_layer = 32,
                          num_epochs = 3000,
                          train_distribution = "Hammersley",
                          use_best_test_loss = True,
                          learning_rate_decay = True,
                          seed = None):
    """
    Main PINN monodomain solve function
    
    Solves the monodomain equation on a 2D unit square geometry
    
    The monodomain equation parameters and cell model are provided as input, through the "problem" input parameter.
    
    Initial condition is enforced as a hard constraint, not as a soft constrain in the loss function.
    
    Parameters
        problem - an instantiated problem class (see MonodomainMMSProblem). Must provide 
                  chi, Cm, sigma_x, sigma_y, end_time, get_Iionic(), cell_model_rhs(), and get_initial_cond()
        num_domain - number of collocation points in the space-time domain
        num_boundary - number of collocation points on the spatial boundary
        num_test - number of independent test points used to compute test loss
        num_layers - number of hidden layers in the neural network
        neurons_per_layer - number of neurons in each hidden layer
        num_epochs - number of training epochs
        train_distribution - DeepXDE sampling distribution used for training points - defaults to Hammersley
        use_best_test_loss - if True, restore the model weights corresponding to the lowest test loss during training
        learning_rate_decay - if True, use a decaying learning rate during training
        seed - (int)  If set, will be used as seed for randon number generation. Set to enforce reproducibility (results should be identical for fixed hardware). 
            See comments in code for how this is implemented - have to do more than just deepxde.config.set_random_seed(seed)          

    Returns:
        model - trained DeepXDE model
    """    
    # The following is needed to enforce reproducibility. First, we call deepxde.config.set_random_seed() which will pass the seed to
    # the relevant functions for tensorflow, numpy, random. 
    # However, there is known DeepXDE initializer-seeding bug issue (May 2026) where Keras initializers are created upon initial import 
    # and therefore before the user seed is set. This leads to the user seed being neglected in weight initialization if the following line is called below 
    #   net = dde.nn.FNN(layer_size, activation, "Glorot uniform")
    # Different sessions would end up using different initial weights and therefore different results. 
    # Using the following variant ensures reproducibility across sessions.
    #   glorot_initializer = tf.keras.initializers.glorot_uniform(seed=seed)
    #   net = dde.nn.FNN(layer_size, activation, glorot_initializer)
    # https://github.com/lululxvi/deepxde/issues/2086
    # (Note: another fix to ensure reproducibility across sessions is to call "import random; random.seed(0)" before importing tensorflow but then user can't control the seed)
    if seed is not None:
        dde.config.set_random_seed(seed)  


    # define domain in space and time
    geometry = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    time_domain = dde.geometry.TimeDomain(0.0, problem.end_time)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)

    ## Boundary conditions    
    # don't need to define separate boundary regions, as same condition can be applied to all sides
    # Because sigma is diagonal and positive and the square is axis aligned,
    # dV/dn = 0 is equivalent to n dot sigma grad(V) = 0 on every edge.
    def spatial_boundary(_, on_boundary):
        return on_boundary

    def zero_flux(_):
        return np.zeros((len(_), 1), dtype=np.float32)

    voltage_bc = dde.icbc.NeumannBC(geometry_time, zero_flux, spatial_boundary, component=0)


    # define the PDE to to be solved
    data = dde.data.TimePDE(geometry_time,
                            lambda xt, vu: monodomain_residual_fnc(problem, xt, vu),
                            [voltage_bc],  # note no initial condition constraints, enforced as a hard constraint below
                            num_domain=num_domain,
                            num_boundary=num_boundary,
                            train_distribution=train_distribution,
                            num_test=num_test)
    

    # set up network
    layer_size = [3] + [neurons_per_layer] * num_layers + [4]
    activation = "tanh"
    # See comments at top near dde.config.set_random_seed(seed)  
    glorot_initializer = tf.keras.initializers.glorot_uniform(seed=seed)
    net = dde.nn.FNN(layer_size, activation, glorot_initializer)

    # function that takes the raw network output and transforms as below, in effect enforcing the 
    # initial condition as a hard constraint. 
    def initcond_output_transform(xt, raw_output):
        t = xt[:, 2:3]
        return problem.get_initial_cond(xt) + t * raw_output

    net.apply_output_transform(initcond_output_transform)
       
    # set up PINN model object
    model = dde.Model(data, net)
    
    # weights for the five loss terms (V-pde, u1-ODE, u2-ODE, u3-ODE, boundary-conditions). Note
    # if initial conditions were added to the loss, it would need heavy weights just like the BC.
    loss_weights = [1.0, 1.0, 1.0, 1.0, 10]
    
    # Compile model.
    # if learning_rate_decay is true, set up for learning rate to decrease
    if learning_rate_decay:        
        model.compile("adam", lr=0.001, decay=("inverse time", 1000, 0.5), loss_weights=loss_weights)
    else:
        model.compile("adam", lr=0.001, loss_weights = loss_weights)


    # Train model    
    # If use_best_test_loss is true, set up so model weights are saved each time the test loss is minimized
    if use_best_test_loss:
        best_model_cb = InMemoryBestModel()
        model.train(iterations=num_epochs,callbacks=[best_model_cb])  
        best_model_cb.restore_best_weights()
    else:
        model.train(iterations=num_epochs)
        
    
    return model






def plot_monodomain_solution(trained_model, times, output_N=50):
    """
    Plot the solution at the specified times. 
    
    Parameters:
        trained_model - DeepXDE model object, after training
        times - array of times (be careful last time is not greater than the end time used for training)
        output_N - number of evaluation points per dimension
    """
    x = np.linspace(0.0, 1.0, output_N)
    y = np.linspace(0.0, 1.0, output_N)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))


    fig, axes = plt.subplots(4,len(times),figsize=(12, 10),constrained_layout=True)

    # Predict solution at each time
    predictions = {}

    for t in times:
        points = np.column_stack((xy, np.full(xy.shape[0], t) ))
        predictions[t] = trained_model.predict(points)

    variable_names = ["V", "$u_1$", "$u_2$", "$u_3$"]
    for row, name in enumerate(variable_names):

        # Same scale across time for each variable
        all_values = [predictions[t][:, row].reshape(output_N,output_N) for t in times]

        vmin = min(v.min() for v in all_values)
        vmax = max(v.max() for v in all_values)

        for col, (t, Z) in enumerate(zip(times, all_values)):

            ax = axes[row, col]

            pcm = ax.pcolormesh(X, Y, Z, shading="auto", vmin=vmin, vmax=vmax)

            if row == 0:
                ax.set_title(f"t = {t:g}")
            if col == 0:
                ax.set_ylabel(f"{name}")
            #if row == 2:
            #    ax.set_xlabel("x")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # One colorbar per row
        fig.colorbar(pcm,ax=axes[row, :],label=name,shrink=0.9)

    plt.show()



def plot_monodomain_error(problem, trained_model, times, output_N=50):
    """
    Plot the error at the specified times. 
    
    Parameters:
        problem - problem class, must have a exact_solution_numpy function
        trained_model - DeepXDE model object, after training
        times - array of times (be careful last time is not greater than the end time used for training)
        output_N - number of evaluation points per dimension
    """

    x = np.linspace(0.0, 1.0, output_N)
    y = np.linspace(0.0, 1.0, output_N)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    errors = {}
    for t in times:
        points = np.column_stack(( xy, np.full(xy.shape[0], t) ))
        pred = trained_model.predict(points)
        exact = problem.exact_solution_numpy( xy[:, 0], xy[:, 1], t )
        errors[t] = pred - exact

    fig, axes = plt.subplots(4,len(times),figsize=(12, 10),constrained_layout=True)

    variable_names = ["V", "$u_1$", "$u_2$", "$u_3$"]
    for row, name in enumerate(variable_names):

        # Same scale across time for each variable
        all_values = [errors[t][:, row].reshape(output_N,output_N) for t in times]

        # Symmetric scale about zero for errors
        max_abs = max(np.max(np.abs(v)) for v in all_values)

        for col, (t, Z) in enumerate(zip(times, all_values)):

            ax = axes[row, col]

            pcm = ax.pcolormesh(X, Y, Z, shading="auto", vmin=-max_abs, vmax=max_abs, cmap="RdBu_r")

            if row == 0:
                ax.set_title(f"t = {t:g}")
            if col == 0:
                ax.set_ylabel(f"{name} error")
            #if row == 2:
            #    ax.set_xlabel("x")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])


        # One colorbar per row
        fig.colorbar(pcm,ax=axes[row, :],label=f"{name} error",shrink=0.9)

    plt.show()


