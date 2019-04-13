# The Brusselator version 3.0
# Version 1.0: Implementation of odeint from scipy.integrate
# Version 2.0: Implementation of ode instead of odeint, limit cycles and stability analysis
# Version 3.0: Reaction-diffusion system in 1D

# Importing the package for computations in Python
import numpy as np
# Importing the odesolver for solving the system of ODE's
from scipy.integrate import ode
# Importing the package for plotting in Python
import matplotlib.pyplot as plt

def main():
    ""
    "" 
    ## Setting the characteristic parameters a and b
    choice = input("What kind of behaviour do you want to see? Option 1 is stable focus, option 2 is unstable focus, option 3 is stable point, option 4 is unstable point and option 5 is hopf bifurcation. Only insert the number of your chosen option.")
    choice = int(choice)
    if choice == 1: 
        a = 1.0
        b = 1.9
    if choice == 2:
        a = 1.0
        b = 2.1
    if choice == 3:
        a = 0.2
        b = 0.4
    if choice == 4:
        a = 0.01
        b = 1.15
    if choice == 5:
        a = 1.0
        b = 2.0
    
    ## Discretization of time and space
    m = 1.0                   # Size of 1D grid
    gridpoints_m = 100        # Number of grid points
    dm = m/gridpoints_m
    x_coordinate = np.linspace(0,m,gridpoints_m)
    
    time = 25           # Total timeduration 
    t = np.linspace(0, time, 100)
    
    ## Solving the system with the ODEsolver 
    # Creating initial concentrationprofile
    concentration_x_initial = np.zeros(gridpoints_m)
    concentration_y_initial = np.zeros(gridpoints_m)
    c0 = np.hstack([concentration_x_initial, concentration_y_initial])
    middlepoint_x = gridpoints_m//2
    middlepoint_y = middlepoint_x+gridpoints_m
    c0[middlepoint_x] = 1.0
    c0[middlepoint_y] = 1.0
    # Calculating the values of x and y 
    c = ODEsolver(t, c0, a, b, dm, gridpoints_m)
    concentration_x = c[:,0:gridpoints_m]
    concentration_y = c[:,gridpoints_m:(gridpoints_m*2)]
    
    ## Making plots for each timestep
        # Terminal command: avconv -r 10 -i Brusselatorsolution%d_parameters_a_b.png title,mp4
    for i in range(gridpoints_m):
        plt.figure()
        plt.plot(x_coordinate, concentration_x[i,:], 'b', label='Concentration x')
        plt.plot(x_coordinate, concentration_y[i,:], 'r', label='Concentration y')
        plt.axis([0,1,0,3])
        plt.xlabel('Position in x-direction')
        plt.ylabel('Concentration')
        plt.title('Brusselator model in 1D at timestep ' + str(i+1) + ' with a=' + str(a) + ' and b=' + str(b))
        plt.legend()
        plt.savefig("Brusselatorsolution" + str(i) + "_parameters_" + str(a) + "_" + str(b) + ".png", dpi=240, format= "PNG")
    
def ODEsolver(t, c0, a, b, dm, gridpoints_m):
    """
    In this function, the system is integrated with the odesolver
    Input: 
        - t: time points
        - c0: initial concentrations of x and y
        - a,b: characteristic parameters
        - dm: size of 1D grid divided by the number of gridpoints
        - gridpoints_m: number of gridpoints
    The method used is the Real-valued Variable-coefficeint ODE solver 
    based on backward differentiation formulas for stiff problems.
        - atol: absolute tolerance for solution
        - rtol: relative tolerance for solution
    """
    solver = ode(func).set_integrator('vode', method='bdf', atol=1e-8, rtol=1e-8, nsteps=5000)
    solver.set_initial_value(c0, t[0]).set_f_params(a, b, dm, gridpoints_m)
    y = c0
    idx = 1
    while solver.successful() and solver.t < t[-1]:
        solver.integrate(t[idx])
        y = np.vstack([y, solver.y])
        idx += 1
    return y

        
def func(t, c, a, b, dm, gridpoints_m):
    """ 
    In this function, the function that defines the system of ODE's is defined.
    Input:
        - t: time point
        - c: concentrations of x and y at particular point in time
        - a,b: characteristic parametsrs
        - dm: size of 1D grid divided by the number of gridpoints
        - gridpoints_m: number of gridpoints
    Output:
        - h: system of ODE's (dx/dt and dy/dt stacked into one array)
    
    In this function, first dx/dt and dy/dt is initialized. The periodic 
    boundary conditions are implemented such that the values for dx/dt and dy/dt 
    at the boundaries are calculated separated from the center points. 
    Extended explaination is provided in the report. 
    
    There are two functions within this function, reaction and diffusion. These
    functions calculate the reaction and diffusion term separate.
    """
    # Initialization of system of ODE's 
    index = gridpoints_m*2
    dxdt = np.zeros(gridpoints_m)
    dydt = np.zeros(gridpoints_m)

    # Periodic boundary conditions: value at the left boundary (i=0)
    reaction = reaction_term(a, b, c[0],c[gridpoints_m])
    diffusion = diffusion_term(dm, c[gridpoints_m-1], c[0], c[1], c[index-1], c[gridpoints_m], c[gridpoints_m+1])
    dxdt[0] = diffusion[0] + reaction[0]
    dydt[0] = diffusion[1] + reaction[1]

    # Periodic boundary conditions: value at right boundary (i=gridpoints_m-1)
    reaction = reaction_term(a, b, c[gridpoints_m-1], c[index-1])
    diffusion = diffusion_term(dm, c[gridpoints_m-2], c[gridpoints_m-1], c[0], c[index-2], c[index-1], c[gridpoints_m])
    dxdt[gridpoints_m-1] = diffusion[0] + reaction[0]
    dydt[gridpoints_m-1] = diffusion[1] + reaction[1]
    
    # Loop over the gridpoints_m in order to define the system of ODE's 
        # range(1,gridpoints_m-1) means from 1 till gridpoints_m-1 (so not 
        # including gridpoints_m-1)
    for i in range(1,gridpoints_m-1):
        reaction = reaction_term(a, b, c[i], c[gridpoints_m+i])
        diffusion = diffusion_term(dm, c[i-1], c[i], c[i+1], c[gridpoints_m+i-1], c[gridpoints_m+i], c[gridpoints_m+i+1])
        dxdt[i] = diffusion[0] + reaction[0]
        dydt[i] = diffusion[1] + reaction[1]
    
    # Stacking dx/dt and dy/dt into one array
    h = np.hstack([dxdt,dydt])
    return h  
        
def reaction_term(a, b, concentration_x, concentration_y):
    """
    In this function, the reaction term is calculated with the concentration
    of x and y and the characteristic parameters.
        - reaction[0] = reaction term of dx/dt
        - reaction[1] = reaction term of dy/dt
    """
    reaction = [0,0]
    reaction[0] = a - (b*concentration_x) + ((concentration_x)**2 * concentration_y) - concentration_x
    reaction[1] = (b*concentration_x) - ((concentration_x)**2 * concentration_y)
    return reaction

def diffusion_term(dm, x_previous, x_current, x_next, y_previous, y_current, y_next):
    """
    In this function, the diffusion term is calculated with the concentrations
    of x and y at the previous, current and next step. Also the diffusion
    coefficients are defined within this function and stored into a constant.
        - diffusion[0] = diffusion term of dx/dt
        - diffusion[1] = diffusion term of dy/dt
    """
    D1 = 1e-5
    D2 = 1e-5
    constant_x = D1/(dm**2)
    constant_y = D2/(dm**2)
    diffusion = [0,0]
    diffusion[0] = constant_x * (x_previous - (2*x_current) + x_next)
    diffusion[1] = constant_y * (y_previous - (2*y_current) + y_next)
    return diffusion
           
if __name__ == '__main__':
    main()