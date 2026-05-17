import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc # type: ignore

c = 3e8

# Function to calculate the temporal spread of the width of the gaussian due to modal dispersion
def modal_dispersion(n1, n2, L):
    NA = np.sqrt(n1**2 - n2**2)
    delta_tau_per_L = NA**2 / (2*n1*c)
    sigma_modal = (delta_tau_per_L * L) / (2 * np.sqrt(3))
    return sigma_modal
    

def chromatic_dispersion(D_material, delta_lambda, L):
    sigma_lambda = delta_lambda / (2 * np.sqrt(2 * np.log(2))) # spread due to spectral bandwidth since LED has assumed difference in wavelengths
    sigma_chrom = np.abs(D_material) * L * sigma_lambda
    return sigma_chrom


def total_dispersion(sigma_modal, sigma_chrom): # returns the total dispersion.
    total_dispersion = np.sqrt(sigma_modal**2 + sigma_chrom**2)
    return total_dispersion
        



    