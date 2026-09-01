
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


try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)

from PINN_monodomain import solve_monodomain_PINN, MonodomainMMSProblem, HeatEquationProblem, plot_monodomain_solution, plot_monodomain_error, monodomain_residual_fnc



class TestPinnMonodomain(unittest.TestCase):    
    """
    Unit tests for the solve_monodomain_PINN function
    """ 

    def __compute_linf_errors(self, problem, trained_model, t, n=50):
        """
        Helper function which computes the Linf error for each state vaiable using the exact solution
        """
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(x, y)
        xy = np.column_stack((X.ravel(), Y.ravel()))
    
        points = np.column_stack(( xy, np.full(xy.shape[0], t) ))
        pred = trained_model.predict(points)
        exact = problem.exact_solution_numpy( xy[:, 0], xy[:, 1], t)
        V_error = np.max(np.abs(pred[:,0] - exact[:,0]))
        u1_error = np.max(np.abs(pred[:,1] - exact[:,1]))
        u2_error = np.max(np.abs(pred[:,2] - exact[:,2]))
        u3_error = np.max(np.abs(pred[:,3] - exact[:,3]))
        return [V_error, u1_error, u2_error, u3_error]



    def test_with_heat_equation(self):
        """
        Check against exact solution when parameters and cell model chosen so that the PDE is just the heat equation
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
        tf.compat.v1.reset_default_graph()

        end_time = 0.1
        problem = HeatEquationProblem(end_time=end_time)
        trained_model = solve_monodomain_PINN(problem, num_epochs=4000, seed=0)
        plot_monodomain_solution(trained_model, times=[0,end_time/2, end_time])
        plot_monodomain_error(problem, trained_model, times=[0,end_time/2,end_time])
        linf_errors = self.__compute_linf_errors(problem, trained_model, end_time)
        #print(linf_errors)
        
        for err in linf_errors:
            self.assertLess(err, 2e-1)
        


    def test_with_monodomain_MMS_equation(self):
        """
        Check against Pathmanathan and Gray 2014 monodomain 2D MMS-derived analytic solution
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
        tf.compat.v1.reset_default_graph()

        end_time = 1.0
        problem = MonodomainMMSProblem(end_time=end_time)
        trained_model = solve_monodomain_PINN(problem, num_epochs=10000, seed=0)
        plot_monodomain_solution(trained_model, times=[0,end_time/2, end_time])
        plot_monodomain_error(problem, trained_model, times=[0,end_time/2,end_time])
        linf_errors = self.__compute_linf_errors(problem, trained_model, end_time)
        #print(linf_errors)

        self.assertLess(linf_errors[0], 0.4)



    def test_residuals_given_true_soln(self):
        """
        Check the main residual function gives near-zero values when the exact solution is passed in
        """
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        tf.keras.backend.clear_session()  # clear tf session to avoid gradual slow-down
        tf.compat.v1.reset_default_graph()

        problem = MonodomainMMSProblem(end_time=1.0)

        # three points in x,y,t space
        points = np.array([[0.1, 0.2, 0.1],
                           [0.3, 0.7, 0.4],
                           [0.6, 0.4, 0.7]], dtype=np.float32)
        xyt = tf.constant(points)

        x = xyt[:,0:1]
        y = xyt[:,1:2]
        t = xyt[:,2:3]

        # can't call the following because not a tensorflow version: 
        # exact_solution = problem.exact_solution_numpy(xyt)
        # so explicitly computing here:
        pi = tf.constant(np.pi, dtype=xyt.dtype)

        F = tf.cos(pi * x) * tf.cos(2.0 * pi * y)
        G = 1.0 + x * y**2

        V  = tf.sqrt(1.0 + t) * F
        u1 = (1.0 + t) * G + tf.sqrt(1.0 + t) * F
        u2 = 1.0 / ((1.0 + t) * tf.sqrt(G))
        u3 = tf.zeros_like(V)

        # compute residuals
        residuals = monodomain_residual_fnc(problem, xyt, tf.concat((V,u1,u2,u3),axis=1))

        with tf.compat.v1.Session() as sess:
            residual_values = sess.run(residuals)
            #print(residual_values)
    
        for residual in residual_values:
            self.assertLess(np.max(np.abs(residual)), 1e-5)


 
        
if __name__ == '__main__':
    unittest.main()
