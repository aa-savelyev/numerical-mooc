import numpy                       
from matplotlib import pyplot                 
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Basic initial conditions
L = 20
nx = 81
dx = L/(nx-1)
#dt = 2e-4
dt = 2e-4*81/nx
gamma = 1.4


def display(x, u, index=0, fun='rho', color='#003366', ls='-'):
    '''Display current state'''
    
    f = numpy.zeros_like(u)
    if (fun == 'rho'):
        f = u[:,0]
    elif (fun == 'u'):
        f = u[:,1]/u[:,0]
    elif (fun == 'p'):
        f = (gamma-1)*(u[:,2] - 0.5*u[:,1]**2/u[:,0])
    
    pyplot.plot(x, f, color=color, ls=ls, lw=2)
#    pyplot.ylim(60, 80)
    print(f[index])
    
def conserv_var(rho, u, p):
    """Computes conservative variables"""
    
    U = []
    U.append(rho)   # density
    U.append(rho*u) # momentum
    eT = p/(gamma-1)/rho + 0.5*u**2
    U.append(rho*eT)
    
    return U
    
def init_conditions():
    """Computes initial conditions
    
    Parameters
    ----------
    None
        
    Returns
    -------
    u : 2d array of floats
        Array with conserved variables at at every point x
    """
    
    u = numpy.zeros((nx, 3))
    u_l = conserv_var(1.0, 0., 1e5)    # right
    u_r = conserv_var(0.125, 0., 1e4)   # left
    u[:] = u_r
    u[:nx//2,:] = u_l
    
    return u
    
    
def computeF(u):
    """Computes Euler's flux

    Parameters
    ----------
    u : array of floats
        Array with conserved variables at one point x
        
    Returns
    -------
    F : array of floats
        Array with flux at one point x
    """
    
    u1 = u[:,0]
    u2 = u[:,1]
    u3 = u[:,2] 
    A = u2**2/u1
    F = numpy.zeros_like(u)
    F[:,0] = u2
    F[:,1] = A + (gamma-1)*(u3 - 0.5*A)
    F[:,2] = (u3 + (gamma-1)*(u3 - 0.5*A))*u2/u1
    
    return F

def richtmyer(u, T):
    """ Computes the solution with the Richtmyer scheme
    
    Parameters
    ----------
    u : array of floats
        Array with conserved variables at every point x
    T : float
        End time
        
    Returns
    -------
    u : array of floats
        Array with conserved variables at the moment T at every point x
    """          
    
    #initialize results array
    u_n = u.copy()
    
    # setup some temporary arrays
    F = numpy.zeros_like(u) 
    ustar = numpy.zeros((nx-1, 3))
    Fstar = numpy.zeros((nx-1, 3))
    
    for t in range(int(T/dt)):
        F = computeF(u)
        ustar[:] = 0.5*(u[1:] + u[:-1]) - 0.5*dt/dx*(F[1:] - F[:-1])     
        Fstar = computeF(ustar)
        u_n[1:-1] = u[1:-1] - dt/dx*(Fstar[1:] - Fstar[:-1])
#        u_n[0] = u[0]
#        u_n[-1] = u[-1]
        u = u_n.copy()
        
    return u_n

def godunov(u, T):
    """ Computes the solution with the Godunov scheme
    
    Parameters
    ----------
    u : array of floats
        Array with conserved variables at every point x
    T : float
        End time
        
    Returns
    -------
    u : array of floats
        Array with conserved variables at the moment T at every point x
    """            
    
    #initialize results array
    u_n = u.copy()
    
    # setup some temporary arrays
    F = numpy.zeros_like(u)
    u_plus = numpy.zeros_like(u)
    u_minus = numpy.zeros_like(u)
    
    for t in range(int(T/dt)):
        u_plus[:-1] = u[1:] # Can't do i+1/2 indices, so cell boundary
        u_minus = u.copy()  # arrays at index i are at location i+1/2
        F = 0.5 * (computeF(u_minus) + computeF(u_plus) - 
                   dx/dt * (u_plus - u_minus))
        u_n[1:-1] = u[1:-1] + dt/dx*(F[:-2] - F[1:-1])
        u_n[0] = u[0]
        u_n[-1] = u[-1]
        u = u_n.copy()
        
    return u_n
    
def minmod(e, dx):
    """
    Compute the minmod approximation to the slope
    
    Parameters
    ----------
    e : array of float 
        input data
    dx : float 
        spacestep
    
    Returns
    -------
    sigma : array of float 
            minmod slope
    """
    
    sigma = numpy.zeros_like(e)
    de_minus = numpy.ones_like(e)
    de_plus = numpy.ones_like(e)
    
    de_minus[1:] = (e[1:] - e[:-1])/dx
    de_plus[:-1] = (e[1:] - e[:-1])/dx
#    print('de_minus = ', de_minus)
#    print('de_plus = ', de_plus)
    
    # The following is inefficient but easy to read
    for j in range(len(e[0])):
        for i in range(1, len(e[:,0])):
            
            if (de_minus[i,j] * de_plus[i,j] < 0.0):
                sigma[i,j] = 0.0
            elif (numpy.abs(de_minus[i,j]) < numpy.abs(de_plus[i,j])):
                sigma[i,j] = de_minus[i,j]
            else:
                sigma[i,j] = de_plus[i,j]
    
    print('sigma = ', sigma)
    return sigma
    
def muscl(u, T):
    """ Computes the solution with the Godunov scheme
    
    Parameters
    ----------
    u : array of floats
        Array with conserved variables at every point x
    T : float
        End time
        
    Returns
    -------
    u : array of floats
        Array with conserved variables at the moment T at every point x
    """   
    
    #initialize results array
    u_n = u.copy()
    
    # setup some temporary arrays
    flux = numpy.zeros_like(u)
    u_star = numpy.zeros_like(u)
    
    for t in range(int(T/dt)):
               
        sigma = minmod(u, dx) #calculate minmod slope

        #reconstruct values at cell boundaries
        u_left  = u + sigma*dx/2.
        u_right = u - sigma*dx/2.     
        
        flux_left  = computeF(u_left) 
        flux_right = computeF(u_right)
        
        #flux i = i + 1/2
        flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] - dx/dt *\
                          (u_right[1:] - u_left[:-1] ))
        
        #rk2 step 1
        u_star[1:-1] = u[1:-1] + dt/dx * (flux[:-2] - flux[1:-1])
        
        u_star[0]  = u[0]
        u_star[-1] = u[-1]
        
        
        sigma = minmod(u_star,dx) #calculate minmod slope
    
        #reconstruct values at cell boundaries
        u_left  = u_star + sigma*dx/2.
        u_right = u_star - sigma*dx/2.
        
        flux_left = computeF(u_left) 
        flux_right = computeF(u_right)
        
        flux[:-1] = 0.5 * (flux_right[1:] + flux_left[:-1] - dx/dt *\
                          (u_right[1:] - u_left[:-1] ))
        
        u_n[1:-1] = 0.5 * (u[1:-1] + u_star[1:-1] + dt/dx * (flux[:-2] - flux[1:-1]))
        
        u_n[0] = u[0]
        u_n[-1] = u[-1]
        u = u_n.copy()
        
    return u_n
    

x = numpy.linspace(-L/2,L/2,nx)
i = int(numpy.where(x==2.5)[0])

fun_str = 'rho'
u = init_conditions()
display(x, u, i, fun_str, ls='--')
#u_richtmyer = richtmyer(u, 0.01)
u_godunov = godunov(u, 0.01)
u_muscl = muscl(u, 0.01)
#display(x, u_richtmyer, i, fun_str, ls='-')
display(x, u_godunov, i, fun_str, ls='-', color='green')
display(x, u_muscl, i, fun_str, ls='-', color='red')

#print(u_n[i])


