# The Brusselator version 2.0 
# Version 1.0: Implementation of odeint from scipy.integrate
# Version 2.0: Implementation of ode instead of odeint, limit cycles and stability analysis

# Importing the package for computations in Python
import numpy as np
# Importing the odesolver for solving the system of ODE's
from scipy.integrate import ode
# Importing the package for plotting in Python
import matplotlib.pyplot as plt

def main():
    """
    In this function, which represents the overall Brusselator model,
    the following actions are implemented:
    1. Setting the characteristic parameters a and b
    2. Solving the system with the ODEsolver mentioned below this function
    3. Plotting the system as function of time
    4. Plotting the limit cycles with vector field
    5. Performing the stability analysis
    """
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
    
    ## Solving the system with the ODEsolver 
    # Creating the points in time as a vector
    t = np.linspace(0, 1000, 10000)
    # Calculating the values of x and y with different initial concentrations
    c1 = ODEsolver(t, [1.0, 1.0], a, b)
    c2 = ODEsolver(t, [0.9, 2.2], a, b)
    
    ## Plotting the system as function of time 
    # Create figure and axes 
    fig = plt.figure(dpi=240)
    ax = fig.axes
    # Create logarithmic plot
        # x concentration is represented by c1[:,0]
        # y concentration is represented by c1[:,1]
    plt.semilogx(t, c1[:,0], 'b', label='Concentration x')
    plt.semilogx(t, c1[:,1], 'r', label='Concentration y')
    plt.legend()
    plt.xlabel('Time in logarithmic scale')
    plt.ylabel('Concentration')
    plt.title('Brusselator model in 0D' + ' with a=' + str(a) + ' and b=' + str(b))
    plt.show()
    
    ## Plotting the limit cycles with vector field
    # 1. Plotting the limit cycles
    # Create figure and axes 
    fig, ax = plt.subplots(dpi=240)
    # Create logarithmic plot
    plt.plot(c1[:,0], c1[:,1], label='$x_{0}$,$y_{0}$ = 1.0, 1.0')
    plt.plot(c2[:,0], c2[:,1], label='$x_{0}$,$y_{0}$ = 0.9, 2.2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Limit cycles' + ' with a=' + str(a) + ' and b=' + str(b))
    
    # 2. Plotting the vector field
    xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1],50)
    yy = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1],50)
    # Defining the location of the arrows with X and Y 
    X, Y = np.meshgrid(xx, yy)
    # Defining the data for the arrows with u (corresponds with X) and v (corresponds with Y)
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    ni, nj = u.shape
    for i in range(ni):
        for j in range(nj):
            cprime = func(t, [X[i,j], Y[i,j]], a, b)
            u[i,j] = cprime[0]
            v[i,j] = cprime[1]
    # Plotting the 2D vector field
    Q = plt.quiver(X, Y, u, v, color='black')
    plt.show()

    
    ## Performing the stability analysis   
    # 1. Calculate trace , determinant, discriminant and eigenvalues of Jacobian Matrix
    J_ss = np.zeros((2,2))
    J_ss[0,0] = b-1
    J_ss[0,1] = a**2
    J_ss[1,0] = -b
    J_ss[1,1] = -a**2
    Trace_J_ss = np.trace(J_ss)
    Determinant_J_ss = np.linalg.det(J_ss)
    Discriminant_J_ss = (Trace_J_ss**2.0) - (4.0*Determinant_J_ss)
    Eigenvalues_J_ss,Eigenvectors_J_ss = np.linalg.eig(J_ss)
    
    # 2. Printing the stability analysis
    print('Trace of Jacobian Matrix = %f' % Trace_J_ss)
    print('Determinant of Jacobian Matrix = %f' % Determinant_J_ss)
    print('Discriminant of Jacobian Matrix = %f' % Discriminant_J_ss)
    print('Eigenvalue 1 of Jacobian Matrix = ', Eigenvalues_J_ss[0])
    print('Eigenvalue 2 of Jacobian Matrix = ', Eigenvalues_J_ss[1])
    
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
    solver = ode(func,jac).set_integrator('vode', method='bdf', atol=1e-14, rtol=1e-16, nsteps=5000)
    solver.set_initial_value(c0, t[0]).set_f_params(a,b).set_jac_params(a,b)
    y = c0
    idx = 1
    while solver.successful() and solver.t < t[-1]:
        solver.integrate(t[idx])
        y = np.vstack([y, solver.y])
        idx += 1
    return y

def func(t,c,a,b):
    """ 
    This function is used by the integration function mentioned above and
    defines the system of ODE's.
        Input: 
        - t: point in time
        - c: values of x and y at defined point in time t
        - a,b: characteristic parameters
    Note: The system of ODE's is first prelocated which makes the function faster
    """
    f = [0,0]
    f[0] = a - b * c[0] + c[0]**2 * c[1] - c[0]
    f[1] = b * c[0] - c[0]**2 * c[1]              
    return f

def jac(t,c,a,b):
    """
    This function is used by the integration function mentioned above and
    defines the Jacobian matrix of the system of ODE's.
        Input: 
        - t: point in time
        - c: values of x and y at defined point in time t
        - a,b: characteristic parameters
    Note: The Jacobian Matrix is first prelocated which makes the function faster
    """
    J = np.zeros((2,2))
    J[0,0] = -b + 2.0 * c[0] * c[1] - 1
    J[0,1] = c[0]**2
    J[1,0] = b - 2.0 * c[0] * c[1]
    J[1,1] = - c[0]**2
    return J

if __name__ == '__main__':
    main()