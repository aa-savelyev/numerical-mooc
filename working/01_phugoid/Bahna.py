import numpy
import math
from matplotlib import pyplot

g = 9.8 
Po = 1.091
r = 0.5
A = r**2*math.pi
Ve = 325
Cd = 0.15
Mp0 = 100 
Vm = 20
h0 = 0
Ms = 50
v0 = 0

def f(u):
    h = u[0]
    v = u[1]
    Mp = u[2]
    return numpy.array([v, -g+(Vm*Ve)/(Mp+Ms)-Po*v*A*Cd*math.fabs(v)/(2*(Mp+Ms)),-Vm])

def f2(u):
    h = u[0]
    v = u[1]
    Mp = u[2]
    return numpy.array([v, -g-Po*v*A*Cd*math.fabs(v)/(2*Ms), 0])
    
def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)
    
T = 40                       # final time
dt = 0.1                           # time increment
N = int(T/dt) + 1                  # number of time-steps
t = numpy.linspace(0, T, N)      # time discretization

# initialize the array containing the solution for each time-step
u = numpy.empty((N, 3))
u[0] = numpy.array([h0, v0, Mp0])# fill 1st element with initial values

# time loop - Euler method
for n in range(N-1):
    if t[n] < 5:
        u[n+1] = euler_step(u[n], f, dt)
    elif t[n] >= 5:
        u[n+1] = euler_step(u[n], f2, dt)