import math
import numpy as np


def solve_simple_prob(force_term, boundary_condition, h, intentional_error = False):
    """
    Solves u''(x) = f(x) on [0,1], with u(0)=0 and u'(1)=g using linear finite elements.

    Parameters:
        force_term (function):
            Function that takes a scalar x in [0,1] and returns f(x).
        boundary_condition (float): 
            Value of g at x=1 for the boundary condition.
        h (float)
            Spatial step size. A regular grid with elements of size h will be created. Should divide [0,1] evenly.
        intentional_error (bool):
            Defaults to False. If True, a coding error will be introduced (failure to set the first 
            entry of the right-hand side vector to 0).
                            
    Returns: A tuple (x, u) where:
        - x: The nodes of the mesh, an array of points in [0,1].
        - u: The nodal values of the solution, corresponding to the approximate solution at each mesh point.
    """    

    # Set up regular mesh given chosen step-size h
    N_elem = int(np.round(1/h))
    x = np.linspace(0,1,N_elem+1)

    # Initiate stiffness matrix K and load vector F
    K = np.zeros((N_elem+1, N_elem+1))
    F = np.zeros(N_elem+1)

    # loop over elements and create local stiffness matrices and load vectors
    for i in range(N_elem):
        # local stiffness matrix known analytically
        K_local = np.array([[1, -1], [-1, 1]]) / h
        
        # Local load vector computed using quadrature
        quad_point1 = (1 - math.sqrt(1.0/3.0))/2
        quad_point2 = (1 + math.sqrt(1.0/3.0))/2
        weight1 = 0.5
        weight2 = 0.5
        
        x1 = x[i] + quad_point1*h
        x2 = x[i] + quad_point2*h
        f1 = force_term(x1)
        f2 = force_term(x2)
        
        F1 = weight1*f1*(1-quad_point1) + weight2*f2*(1-quad_point2)
        F2 = weight1*f1*quad_point1 + weight2*f2*quad_point2        
        F_local = -np.array([F1, F2])*h
            
        # Assemble local stiffness and load into global matrix and vector
        K[i:i+2, i:i+2] += K_local        
        F[i:i+2] += F_local
        
    # Boundary condition component of the integral
    F[-1] += boundary_condition  # Neumann condition at x = 1
        
    # Alter linear system to apply Dirichlet boundary condition
    K[0, :] = 0
    K[0, 0] = 1  # Dirichlet condition at x = 0 (u = 0)
    # if the intentional_error flag is passed, we intentionally 'forget' to zero the first entry
    if not intentional_error:
        F[0]  = 0
    
    # Solve the system
    u = np.linalg.solve(K, F)
    return x, u    



def error_in_L2_norm(x,uh,true_soln):
    """
    Compute the error between the finite element solution and the true solution in the L2 (square integral) norm. 
    Computes the square root of the integral of (u_h - u*)^2, where u_h is the finite element solution across the
    domain and u* is the true solution. Uses quadrature to compute this integral.

    Parameters: 
        x (array): The nodes of the mesh, an array of points in [0,1].
        uh (array): The nodal values of the solution, corresponding to the approximate solution at each mesh point.
        true_soln(array): param true_soln: A scalar-valued function providing the true solution
    
    Returns:
        scalar: L2 norm of u_h-u*

    """
    N_elem = len(x)-1
    ret = 0
    for i in range(0,N_elem):
        quad_points = np.array([(1 - math.sqrt(1.0/3.0))/2, (1 + math.sqrt(1.0/3.0))/2])
        weights = np.array([0.5,0.5])
        h = x[i+1]-x[i]
        for q,w in zip(quad_points, weights):
            x_quadpoint = x[i] + q*h
            s = true_soln(x_quadpoint)
            uu = uh[i]*(1-q) + uh[i+1]*q
            ret += h*w*(s-uu)**2
    return np.sqrt(ret)


def error_in_H1_norm(x,uh,true_soln,true_soln_derivative):
    """
    Compute the error between the finite element solution and the true solution in the H^1 norm. 
    Computes the square root of the integral of (u_h - u*)^2 + (du_h/dx - du*/dx)^2 + , where u_h is the finite element solution across the
    domain and u* is the true solution. Uses quadrature to compute this integral.


    Parameters: 
        x (array): The nodes of the mesh, an array of points in [0,1].
        uh (array): The nodal values of the solution, corresponding to the approximate solution at each mesh point.
        true_soln(array): param true_soln: A scalar-valued function providing the true solution
        true_soln_derivative(array): A scalar-valued function providing the derivative of the true solution
    Returns:
        scalar: H1 norm of u_h-u*
    """
    N_elem = len(x)-1
    derivative_term_integral = 0
    for i in range(0,N_elem):        
        quad_points = np.array([(1 - math.sqrt(1.0/3.0))/2, (1 + math.sqrt(1.0/3.0))/2])
        weights = np.array([0.5,0.5])
        h = x[i+1]-x[i]

        for q,w in zip(quad_points, weights):
            x_quadpoint = x[i] + q*h
            ds = true_soln_derivative(x_quadpoint)
            du = (uh[i+1]-uh[i])/h
            derivative_term_integral += h*w*(ds-du)**2
    return np.sqrt(derivative_term_integral + error_in_L2_norm(x,uh,true_soln)**2)