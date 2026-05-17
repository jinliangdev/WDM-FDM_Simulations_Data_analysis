"""

WDM in a MMF

Single LED channel, step-index MM, Single pulse

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc # type: ignore


c = 3e8

n1 = 1.46
n2 = 1.44
NA = np.sqrt(n1**2 - n2**2)

delta = (n1 - n2) / n1

lambda0 = 850e-9
delta_lambda = 40e-9

D_material = -80e-12 / (1e-9 * 1e3)

delta_tau_per_L = (NA**2) / (2 * n1 * c)

L = np.linspace(1, 2000, 2000)

sigma_modal = (delta_tau_per_L * L) / (2 * np.sqrt(3))

sigma_lambda = delta_lambda / (2 * np.sqrt(2 * np.log(2)))

sigma_chrom = np.abs(D_material) * L * sigma_lambda

sigma_total = np.sqrt(sigma_modal**2 + sigma_chrom**2)


f_3dB = 0.187 / sigma_total
BW_distance = f_3dB * (L / 1000)



t = np.linspace(-50e-9, 50e-9, 5000)

sigma_in = 15e-9

lengths_for_pulse = [10, 100, 500, 1000, 2000]

def gaussian_pulse(t, sigma):
     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
Lv = lengths_for_pulse[1]

sig_m = (delta_tau_per_L * Lv) / (2 * np.sqrt(3))
sig_c = np.abs(D_material) * Lv * sigma_lambda
sig_t = np.sqrt(sigma_in**2 + sig_m**2 + sig_c**2)

pulse_in  = gaussian_pulse(t, sigma_in)
pulse_out = gaussian_pulse(t, sig_t)
print (sig_t)

# plt.plot(t * 1e9, pulse_in  / pulse_in.max(),  label="input")
# plt.plot(t * 1e9, pulse_out / pulse_out.max(), label="detected")
plt.plot(t * 1e9, pulse_in,  label="input")
plt.plot(t * 1e9, pulse_out, label="detected")  
plt.xlabel("Time (ns)")
plt.ylabel("Probability density (s^-1)")
plt.legend

plt.grid()
plt.show()


