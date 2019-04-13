# The Brusselator version 4.0
# Version 1.0: Implementation of odeint from scipy.integrate
# Version 2.0: Implementation of ode instead of odeint, limit cycles and stability analysis
# Version 3.0: Reaction-diffusion system in 1D
# Version 4.0: Reaction-diffusion system in 2D

import time
start_time = time.time()

# Importing the package for computations in Python
import numpy as np
# Importing the odesolver for solving the system of ODE's
from scipy.integrate import ode
# Importing the package for plotting in Python
import matplotlib.pyplot as plt
# Importing random
import random 

def main():
    """
    In this function, which represents the overall Brusselator model in 2D,
    the following actions are implemented:
    1. Setting the characteristic parameters a and b
    2. Discretization of time and space 
    3. Solving the system with the ODEsolver
    4. Plotting the system in the 2D grid
    """ 
    ## 1. Setting the characteristic parameters a and b
    a = 4.5
    b = 7.5
    # Setting the range for the colormaps
    vmin_x = 2.0
    vmax_x = 6.0
    vmin_y = 1.0
    vmax_y = 2.0
    
    ## 2. Discretization of time and space
    # Discretization of space in the 2D grid
    m = 32                      # Size of x-coordinate side
    n = 32                      # Size of y-coordinate side
    # Transforming the 2D coordinates m and n to k 
    gridpoints_k = m * n
    # Discretization of time
    time = 10.0                   # Total timeduration 
    timesteps = 10
    t = np.linspace(0, time, timesteps+1)
    
    ## 3. Solving the system with the ODEsolver 
    # Creating initial concentrationprofile which is a matrix which has the size
    # of the defined 2D grid
    concentration_x_initial = np.zeros((m, n))
    concentration_y_initial = np.zeros((m, n))
    # Initialization of the concentration profiles with random numbers around
    # the equilibrium values
    delta = 0.3
    for i in range(0, m):
        for j in range(0, n):
            concentration_x_initial[i,j] = a + (random.random()-0.5) * delta
            concentration_y_initial[i,j] = b/a + (random.random()-0.5) * delta
    # Stacking the initial concentrationprofile of x and y in one vector,
    # since the ODE solver can only handle one vector. 
    c0 = np.hstack([concentration_x_initial.ravel(), 
                    concentration_y_initial.ravel()])
    # Calculating the concentration profiles over time
    c = ODEsolver(t, c0, a, b)

    ## 4. Plotting the system in the 2D grid
    for i in range(timesteps):
        fig = plt.subplots(1,2)
        fig = plt.subplots_adjust(right=0.8, wspace=0.3)
        # Dividing the concentration profile in a concentration profile for
        # concentration x and a concentration profile for concentration y
        concentration_x= c[i,0:gridpoints_k]
        concentration_y = c[i,gridpoints_k:gridpoints_k*2]
        concentration_x = np.reshape(concentration_x,(m,n))
        concentration_y = np.reshape(concentration_y,(m,n))
        plt.suptitle('Brusselator model in 2D')
        plt.subplot(1,2,1)
        plt.imshow(concentration_x, vmin=vmin_x, vmax=vmax_x, origin='bottom')
        plt.xlabel('X-direction')
        plt.ylabel('Y-direction')
        plt.title('Concentration X')
        plt.subplot(1,2,2)
        plt.imshow(concentration_y, vmin=vmin_y, vmax=vmax_y, origin='bottom', cmap='YlGnBu')
        plt.xlabel('X-direction')
        plt.ylabel('Y-direction')
        plt.title('Concentration Y')
        # Storing the figure in a file with name 00i.png
        filename = '%03i.png' % i
        plt.savefig(filename, dpi=300)
        # Plotting the name of the filename 
        print('Plotting image to %s' % filename)
        plt.show()
        plt.close()
        
def ODEsolver(t, c0, a, b):
    """
    In this function, the system is integrated with the odesolver
    Input: 
        - t: time points
        - c0: initial concentrations of x and y
        - a,b: characteristic parameters
    The method used is the Real-valued Variable-coefficeint ODE solver 
    based on backward differentiation formulas for stiff problems.
        - atol: absolute tolerance for solution
        - rtol: relative tolerance for solution
    """
    solver = ode(func).set_integrator('vode', method='bdf', atol=1e-8, rtol=1e-8, nsteps=5000 )
    solver.set_initial_value(c0, t[0]).set_f_params(a, b)
    y = c0
    idx = 1
    while solver.successful() and solver.t < t[-1]:
        solver.integrate(t[idx])
        y = np.vstack([y, solver.y])
        print('Storing solution at t = %f, (%4.2f %%)' % (solver.t, solver.t/t[-1]*100))
        idx += 1
    return y

def func(t, c, a, b):
    """ 
    In this function, the function that defines the system of ODE's is defined.
    Input:
        - t: time point
        - c: concentrations of x and y at particular point in time
        - a,b: characteristic parameters
    Output:
        - h: system of ODE's (dx/dt and dy/dt stacked into one array)
    
    In this function, first dx/dt and dy/dt is initialized. The periodic 
    boundary conditions are implemented such that the values for dx/dt and dy/dt 
    at the boundaries are calculated with the modulus operator. 
    
    There are two functions within this function, reaction and diffusion. These
    functions calculate the reaction and diffusion term separate.
    """
    # Calculating the shape of the input vector
    size = int(np.sqrt(len(c)/2))
    
    # Since the concentration profile contains a stacking of the concentration
    # profiles of concentration x and concentration y, the concentration profile
    # needs to be divided
    x, y = np.split(c,2)
    
    # Reshaping the concentration profiles of x and y to matrices
    x = x.reshape((size,size))
    y = y.reshape((size,size))
    
    # Initialization of dx/dt and dy/dt
    dxdt = np.zeros((size,size))
    dydt = np.zeros((size,size))
    
    # Calculation of the reaction term
    reaction_x = np.zeros((size,size))
    reaction_y = np.zeros((size,size))
    for i in range(0, size):
        for j in range(0, size):
            reaction_x[i,j], reaction_y[i,j] = reaction_term(a, b, x[i,j], y[i,j])
    
    # Calculation of the diffusion term
    D1 = 2.0
    D2 = 16.0
    diffusion_x = D1 * diffusion_term(x)
    diffusion_y = D2 * diffusion_term(y)
    
    # Adding the diffusion term to existing reaction term
    dxdt = np.add(reaction_x, diffusion_x)
    dydt = np.add(reaction_y, diffusion_y)
    
    # Reshaping the matrix back into a vector, and putting it into one
    # concentration profile
    h = np.hstack([dxdt.ravel(),dydt.ravel()])
    
    return h  
        
def reaction_term(a, b, x, y):
    """
    In this function, the reaction term is calculated with the concentration
    of x and y and the characteristic parameters.
    """
    reaction_x = a - (b+1.0)*x + (x**2 * y)
    reaction_y = (b*x) - (x**2 * y)
    return reaction_x, reaction_y

def diffusion_term(c):
    """
    In this function, the diffusion term is calculated with the Laplacian. As input, concentration
    c can be concentration x or concentration y. WIth this concentration profile,
    the diffusion term is calculated. ALso, the periodic boundary conditions
    are implemented with the use of modulus. 
    """
    # Initialization of the diffusion term
    diffusion_c = np.zeros(c.shape)
    
    for i in range(0, diffusion_c.shape[0]):
    # Calculating the indices in the x-direction
        i1 = (i%(diffusion_c.shape[0]-1)) -1
        i2 = (i%(diffusion_c.shape[0]-1)) + 1
        
        for j in range(0, diffusion_c.shape[1]):
            # Calculating the indices in the y-direction
            j1 = (j%(diffusion_c.shape[1]-1)) -1
            j2 = (j%(diffusion_c.shape[1]-1)) +1
            
            # Calculating the diffusion term
            diffusion_c[i,j] = c[i1,j] + c[i2,j] + c[i,j1] + c[i,j2] - 4.0 * c[i,j]
    
    return diffusion_c 

       
if __name__ == '__main__':
    main()