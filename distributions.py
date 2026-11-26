import numpy as np
import math

def predict_poisson(claims_in_class, k):
    
    parameter = claims_in_class.mean()
    n = len(claims_in_class)

    probability = np.exp(-parameter) * parameter**k / math.factorial(k)
    prediction = n * probability

    return prediction

def fit_gamma(x, severity, no_of_claims):

    n = len(severity)

    mu = (no_of_claims*severity).sum() / no_of_claims.sum()
    phi = (no_of_claims * (severity - mu)**2).sum() / ((n-1) * mu**2)

    alpha = 1 / phi
    beta = alpha / mu

    pdf = 0
    for w in no_of_claims.unique():
        shape = w * alpha
        rate = w * beta
        share = (no_of_claims == w).sum() / n
        
        density = rate**shape / math.gamma(shape) * x**(shape-1) * np.exp(-rate*x)
        pdf = pdf + share*density

    return pdf

