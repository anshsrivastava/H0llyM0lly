from distance import *
from MCMC import *
import random 

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