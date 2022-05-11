from distance import *
from MCMC import *
import random 
import numpy as np

#### TESTS FOR distance.py

def test_simpsons():
    '''
    Tests the simpsons() function from distance.py
    '''
    lower_limit=0
    upper_limit=3
    number_of_points=10
    h = (upper_limit-lower_limit)/number_of_points
    omega_m=.286
    omega_a=.714
    omega_k=1-omega_m-omega_a
    k_points = np.arange(lower_limit,upper_limit+.1*h,h)
    f_k=np.zeros(len(k_points),float)
    for k in range(len(k_points)):
        f_k[k]=integrand(k_points[k],omega_k,omega_m,omega_a)
    assert (1.504993571457544-simpsons(upper_limit,omega_m,omega_a,number_of_points)) < 1E-4 # Comparing against a 10^-4 tolerance
    print('simpsons looks good')

def test_trapezoidal_vectorization():
    '''
    Tests the trapezoidal() function from distance.py
    '''
    zs = np.array([1,2,3])
    for z, output in zip(zs, trapezoidal(zs,1,1,5)):
        assert output == trapezoidal(z,1,1,5)
    print('trapezoidal vectorization looks good')

def test_luminosity_dist_3():
    '''
    Tests the luminosity_dist() function from distance.py
    '''
    assert np.abs(luminosity_dist(3,.286,.714,69.6,100,"s") - 25924.3) < 100
    print('luminosity distance looks good')

def test_integrand():
    '''
    Tests integrand from distance.py
    '''
    for i in range(0, 5):
        z = random.uniform(0, 10)
        omega_k = random.uniform(0, 10)
        omega_m = random.uniform(0, 10)
        omega_a = random.uniform(0, 10)
        v = omega_m*(1+z)**3 + omega_k*(1+z)**2 + omega_a # parameter that must be greater than 0
        if v > 0.001: # Avoiding division by zero and imaginary integrand eventough it is almost impossible
            assert np.abs(np.sqrt(1/v) - integrand(z, omega_k, omega_m, omega_a))<0.001 # Checking against a 0.001 tolerance
    print("Passed 5 out of 5 tests for integrand")
    
def test_distance_modulus():
    '''
    Tests distance_modulus from distance.py
    '''
    z = 42
    omega_m = 42
    omega_a = 42
    H_0 = 42
    calculated_distance_modulus = 5*np.log10(luminosity_dist(z, omega_m, omega_a, H_0)) + 25
    assert np.abs(distance_modulus(z, omega_m, omega_a, H_0) - calculated_distance_modulus) < 0.0001 # Comparing with a 0.0001 tolerance
    print("Passed 1 out of 1 test cases for distance_modulus")
    
    
#### TESTS FOR MCMC.py


def test_priors():
    '''
    Tests the log_priors(pars) function in MCMC.py
    '''
    assert log_priors([1 * random.randint(1, 231213), 1 * random.randint(1, 231213), 1 * random.randint(1, 231213)])== 0 
    assert log_priors([-1 * random.randint(1, 231213), 1 * random.randint(1, 231213), 1 * random.randint(1, 231213)])== float('-inf')
    assert log_priors([1 * random.randint(1, 231213), -1 * random.randint(1, 231213), 1 * random.randint(1, 231213)])== float('-inf')
    assert log_priors([1 * random.randint(1, 231213), 1 * random.randint(1, 231213), -1 * random.randint(1, 231213)])== float('-inf')
    assert log_priors([-1 * random.randint(1, 231213), -1 * random.randint(1, 231213), 1 * random.randint(1, 231213)])== float('-inf')
    assert log_priors([1 * random.randint(1, 231213), -1 * random.randint(1, 231213), -1 * random.randint(1, 231213)])== float('-inf')
    assert log_priors([-1 * random.randint(1, 231213), 1 * random.randint(1, 231213), -1 * random.randint(1, 231213)])== float('-inf')
    print('7 out of 7 test cases passed for log_priors') 
