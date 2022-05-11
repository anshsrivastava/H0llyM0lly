# vectorized version
import numpy as np
from scipy.constants import c

def integrand(z,omega_k,omega_m,omega_a):
    '''
    Takes in:
        z, omega_m, omega_k: floats
    Returns the integrand
    '''
    return 1/np.sqrt(omega_m*(1+z)**3 + omega_k*(1+z)**2 + omega_a)
    
def trapezoidal(upper_limit,omega_m,omega_a,number_steps=10000,lower_limit=0):
    '''
    Takes in:
        upper_limit, omega_m, omega_a: floats
    Returns the result of integrating the integrand using the trapezoidal rule
    '''
    omega_k=1-omega_m-omega_a
    
    summ = .5*integrand(lower_limit,omega_k,omega_m,omega_a) + .5*integrand(upper_limit,omega_k,omega_m,omega_a)
    step_size = (upper_limit-lower_limit)/number_steps
    #k_points = arange(lower_limit,upper_limit,h)
    #equivalent to 
    k_points = np.linspace(lower_limit,upper_limit,int(number_steps),endpoint=False)

    for _ in k_points:
        summ += integrand(_,omega_k,omega_m,omega_a)
    return summ*step_size

def simpsons(upper_limit,omega_m,omega_a,number_steps=1000,lower_limit=0):
    '''
    Takes in:
        upper_limit, omega_m, omega_a: floats
    Returns the result of integrating the integrand using the simpson's rule
    '''
    number_steps=int(number_steps)
    #must have even number of steps
    if number_steps %2!=0:
        number_steps+=1
        print("Number of steps modifed:",number_steps)
    omega_k=1-omega_m-omega_a
    summ = integrand(lower_limit,omega_k,omega_m,omega_a) + integrand(upper_limit,omega_k,omega_m,omega_a)
    step_size = (upper_limit-lower_limit)/number_steps
    #sum odd values of i
    for i in range(1,number_steps,2):
        summ+=4*integrand(lower_limit+i*step_size,omega_k,omega_m,omega_a)
    #sum even values of i
    for i in range(2,number_steps,2):
        summ+=2*integrand(lower_limit+i*step_size,omega_k,omega_m,omega_a)
    return summ*step_size/3

def luminosity_dist(z,omega_m,omega_a,H_0,number_steps=10000,method='s'):
    '''
    Takes in:
        z, omega_m, omega_a, H_0: floats
    Returns the luminosity distance
    '''
    omega_k=1-omega_m-omega_a
    d_h = c*1E-3/H_0
    if method == "t":
        d_c = d_h*trapezoidal(z,omega_m,omega_a,number_steps)
    #Use Simpson's method by default 
    else:
        d_c = d_h*simpsons(z,omega_m,omega_a,number_steps)
    if omega_k>0:
        return d_h*(1+z) * np.sinh( np.sqrt(omega_k) * d_c/d_h ) / np.sqrt(omega_k)
    elif omega_k==0:
        return d_c*(1+z)
    else:
        return d_h*(1+z) * np.sin( np.sqrt(-omega_k) * d_c/d_h ) / np.sqrt(-omega_k)

def distance_modulus(z,omega_m,omega_a,H_0, number_steps=10000,Mpc=True):
    '''
    Takes in:
        z, omega_m, omega_a, H_0: floats
    Returns the distance modulus
    '''
    luminosity_distance = luminosity_dist(z,omega_m,omega_a,H_0,number_steps)
    # to-do: check consistency of units
    if Mpc:
        return 5*np.log10(luminosity_distance) + 25  
    else: 
        return 5*np.log10(luminosity_distance/10) # Assumed pc unit