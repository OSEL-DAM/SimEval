
import os
import sys
import io 
import contextlib
import inspect
import unittest
import numpy as np

try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd() + "/test"

# Get the absolute path to the 'src' directory relative to this file.
src_path = os.path.abspath(os.path.join(base_dir, '..', 'src'))
sys.path.insert(0, src_path)

from fe_solver_for_mms import solve_simple_prob, error_in_L2_norm, error_in_H1_norm



class TestFeSolverForMMS(unittest.TestCase):
    """
    Unit tests for the defined in fe_solver_for_mms, that are used in the MMS notebook. 
    """
    
    def test_solve_simple_prob(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        
        # solve u'' = f with the below f(x) and u'(1)=2pi        
        def f(x):
            return -4*np.pi*np.pi*np.sin(2*np.pi*x)
        
        bc = 2*np.pi
        h = 0.02
        x, u = solve_simple_prob(force_term=f,boundary_condition=bc,h=h)

        # compare with the true solution
        def exact_soln(x):
            return np.sin(2*np.pi*x)
        u_true_at_nodes = exact_soln(x)
        err = np.linalg.norm(u-u_true_at_nodes)/np.max(np.abs(u))
        self.assertLess(err, 1e-5)
        
        
    def test_error_in_L2_norm(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        
        ###########################################################
        # check the L2 norm of u_h-u is calcutated correctly when
        # u*=0, u_h is linear from 0 to 1
        ###########################################################
        def zero_function(x):
            return 0
        
        x = np.array([0,1.0])
        u = np.array([0,1.0])       
        error_L2 = error_in_L2_norm(x,u,zero_function)
        exact_error = np.sqrt(1.0/3.0)
        self.assertAlmostEqual(error_L2, exact_error, 3)

        
        ###########################################################
        # check the L2 norm of u_h-u is calcutated correctly when
        # u*=x, u_h is zero 
        ###########################################################
        def linear_function(x):
            return x

        x = np.array([0,1.0])
        u = np.array([0,0.0])       
        error_L2 = error_in_L2_norm(x,u,linear_function)
        exact_error = np.sqrt(1.0/3.0)
        self.assertAlmostEqual(error_L2, exact_error, 3)


        ###########################################################
        # now check with a more complicated case, u*=x^2, uh linear from
        # 10 to 11 (at x=0.5) then down to 8 at x=1
        ###########################################################
        def quadratic_function(x):
            return x*x
        
        x = np.array([0,0.5,1.0])
        u = np.array([10, 11, 8])       
        error_L2 = error_in_L2_norm(x,u,quadratic_function)
        
        # should be    integral from 0   to 0.5 of [10+2x        -x^2]^2 dx 
        #           +  integral from 0.5 to 1   of [11-1.5(x-0.5)-x^2]^2 dx 
        
        x=0.5
        exact_intgl1 = x**5/5 - x**4 - 16*x**3/3 + 20*x**2 + 100*x

        exact_intgl2_fn = lambda x: x**5/5 + 3*x**4 + 8*x**3/3 -84*x**2 + 196*x
        exact_intgl2 =  exact_intgl2_fn(1) - exact_intgl2_fn(0.5)
        exact_error = np.sqrt(exact_intgl1 + exact_intgl2)

        self.assertAlmostEqual(error_L2, exact_error, 3)



    def test_error_in_H1_norm(self):
        print("\n==========================\nEntering " + inspect.stack()[0][3] + "\n==========================")
        
        ###########################################################
        # check the H1 norm of u_h-u is calcutated correctly when
        # u*=0, u_h is linear from 0 to 1
        ###########################################################
        def zero_function(x):
            return 0
        
        x = np.array([0,1.0])
        u = np.array([0,1.0])       
        error_H1 = error_in_H1_norm(x,u,zero_function,zero_function)
        exact_error = np.sqrt(1.0/3.0  + 1.0)
        self.assertAlmostEqual(error_H1, exact_error, 3)

        
        ###########################################################
        # check the L2 norm of u_h-u is calcutated correctly when
        # u*=x, u_h is zero 
        ###########################################################
        def linear_function(x):
            return x

        x = np.array([0,1.0])
        u = np.array([0,0.0])       
        error_H1 = error_in_H1_norm(x,u,linear_function,lambda x:1.0)
        exact_error = np.sqrt(1.0/3.0 + 1.0)
        self.assertAlmostEqual(error_H1, exact_error, 3)

        
        ###########################################################
        # now check with a more complicated case, u*=x^2, uh linear from
        # 10 to 11 (at x=0.5) then down to 8 at x=1
        ###########################################################
        def quadratic_function(x):
            return x*x

        def quadratic_function_deriv(x):
            return 2*x

        x = np.array([0,0.5,1])
        u = np.array([10,11,8])       
        error_H1 = error_in_H1_norm(x,u,quadratic_function,quadratic_function_deriv)
        

        ## L2 component
        # should be    integral from 0   to 0.5 of [10+2x        -x^2]^2 dx 
        #           +  integral from 0.5 to 1   of [11-1.5(x-0.5)-x^2]^2 dx 
        
        x=0.5
        exact_intgl1 = x**5/5 - x**4 - 16*x**3/3 + 20*x**2 + 100*x

        exact_intgl2_fn = lambda x: x**5/5 + 3*x**4 + 8*x**3/3 -84*x**2 + 196*x
        L2_component =  exact_intgl1 + exact_intgl2_fn(1) - exact_intgl2_fn(0.5)

        ## H1 seminorm component (i.e. derivative part of the integral) 
        # should be    integral from 0   to 0.5 of [2-2x]^2 dx 
        #           +  integral from 0.5 to 1   of [-6-2x]^2 dx 
        
        intgl_fn = lambda x,alpha:4*x**3/3 - 2*alpha*x**2 + x*alpha**2
        H1_seminorm_component = intgl_fn(0.5,2)+intgl_fn(1,-6)-intgl_fn(0.5,-6)
        exact_error = np.sqrt(L2_component + H1_seminorm_component)

        self.assertAlmostEqual(error_H1, exact_error, 3)
        
if __name__ == '__main__':
    unittest.main()