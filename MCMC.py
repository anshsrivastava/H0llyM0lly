# MCMC code

import numpy as np
from distance import *

def log_priors(pars):
    omega_m,omega_a,H0 = pars
    logp = -np.inf
    logp = 0 if (omega_m > 0 and omega_a >0 and H0>0) else logp
    return logp    

def log_likelihood(pars,z_data,m_b,inv_cov,number_steps=100, M =-19.2):
    
    omega_m, omega_a ,H_0 = pars
    
    mu_model = distance_modulus(z_data,omega_m,omega_a,H_0,number_steps)
    
    if np.isnan(mu_model).any():
        return -np.inf 

    m_model = mu_model+M
    
    residual = m_b-m_model
    result = -(np.dot( np.dot(residual, inv_cov) ,residual ))/2.
    
    return result

def log_posterior(pars,x,y,inv_cov):

    return log_likelihood(pars,x,y,inv_cov) + log_priors(pars)


def generator(Omega, cov = 0.1**2. * np.identity(3)):

    return np.random.multivariate_normal(Omega,cov)


def MCMC(seed,x,y,inv_cov,cov_gen=1):
    '''seed is the previous value in parameter space.
    y is the vector of observed values {y_i}.
    x is the vector of observed predictors {x_i}
    cov is the covariance matrix
    cov_gen is the meta-parameter controlling the width of the gaussian in the generator function.'''

    new = generator(seed,cov_gen)
    Pnew = log_posterior(new,x,y,inv_cov)
    Pold = log_posterior(seed,x,y,inv_cov)
    r = Pnew-Pold
    r = np.min([0,r])
    u = np.log(np.random.uniform(0,1))
    if r<u:
        return seed
    else:
        return new

def run_MCMC(x,y,cov,p0,cov_gen=1,nsteps = 1000):
 
    inv_cov=np.linalg.inv(cov)

    chain = np.zeros((nsteps,len(p0)))
    seed = p0
    for i in range(nsteps):

       sprout =  MCMC(seed,x,y,inv_cov,cov_gen)
       chain[i] =  sprout
       seed = sprout

    return chain
    




