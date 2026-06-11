"""

WDM in a MMF

Single LED channel, step-index MM

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
import random
import Dispersion as spd # custom module to calculate the modal and chromatic dispersion


c = 3e8

#defining cable materials
n1 = 1.46
n2 = 1.44
D_material = -80e-12 / (1e-9 * 1e3)
L = 7500

#defining source
lambda0 = 625e-9
delta_lambda = 40e-9
sigma_input = 15e-9

#defining bit sequence
Nbits = 50
ppb = 100
data = [random.randint(0,1) for i in range(Nbits)]


#calculating the bit error rate
def compute_BER(t, signal, data, T, sigma_output):
    #this function estimates the bit error rate by sampling the signal at centre of each bit
    #then it is compared against a threshold which is half the peak amplitude of a broadened pulse
    #each sample is compared to transmitted bit, mismatch is an error bit

    #parameters include the the time array, signal which is sum of broadened gaussians, data which is our original sequence, T which is period of bits, sigma_output which
    #is the broadened pulse widths.

    #function returns bit error rate
    #threshold used
    threshold = 0.5 * gaussian_pulse(0, sigma_output)

    errors = 0

    for bit_index, transmitted_bit in enumerate(data):

        # finds the middle (in time) of the k-th bit 
        sample_time = (bit_index + 0.5) * T
        sample_index = np.argmin(np.abs(t - sample_time))

        received_value = signal[sample_index]
        detected_bit = int(received_value > threshold)

        if detected_bit != transmitted_bit:
            errors += 1

    return errors / len(data), threshold

#defining a single gaussian pulse.
def gaussian_pulse(t, sigma, centre = 0):
     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(t - centre)**2 / (2 * sigma**2))

# total pulse broadening calculation, calculted by addition in quadrature
# modal dispersion - due to different modes arriving at different times
# chromatic dispersion - due to different wavelengths travelling at different speeds
# input - finite input pulse width from LED driver
sigma_output = np.sqrt(spd.modal_dispersion(n1, n2, L)**2 +
                       spd.chromatic_dispersion(D_material,delta_lambda, L)**2 +
                       sigma_input**2)


#----------------------------------------------
# a 2D sweep of modulation frequency, and fibre length
f_sweep = np.logspace(4, 8, 100)# 10khz - 100Mhz
L_sweep = np.logspace(1, 4, 50)# 10m to 10km
BER_grid = np.zeros((len(L_sweep), len(f_sweep)))

for i, L in enumerate(L_sweep):
    # recompute sigma_output for this length
    sig_out = np.sqrt(spd.modal_dispersion(n1, n2, L)**2 + 
                      spd.chromatic_dispersion(D_material, delta_lambda, L)**2 + 
                      sigma_input**2)
    
    for j, f in enumerate(f_sweep):
        T = 1 / f # bit period
        t = np.linspace(0, Nbits * T, Nbits * ppb) # creates a time array across all bits
        SIGNAL = np.zeros_like(t)

        #adds one broadened gaussian per bit that is 1, centred in its time slot
        for k, bit in enumerate(data):
            if bit == 1:
                t_centre = (k + 0.5) * T
                SIGNAL += gaussian_pulse(t, sig_out, t_centre)

        BER, _ = compute_BER(t, SIGNAL, data, T, sig_out)
        BER_grid[i, j] = BER
    #analytic calculation of the BER threshold curve
    #this is the maximum useable bit rate, above this frequency, pulses start to overlap enough to start increasing BER
    #this represents the good and bad regioins
    f_error_curve = np.array([
        0.6 / np.sqrt(spd.modal_dispersion(n1, n2, Lv)**2 +
                    spd.chromatic_dispersion(D_material, delta_lambda, Lv)**2 +
                    sigma_input**2)
        for Lv in L_sweep])

fig, ax = plt.subplots(figsize=(10, 7))
#simple color mesh to represent BER
im = ax.pcolormesh(f_sweep / 1e6, L_sweep / 1e3, BER_grid, cmap = "RdYlGn_r", vmin = 0, vmax = 0.5)
plt.colorbar(im, ax=ax, label = "BER")


ax.set_xscale('log')
ax.set_xlabel('Modulation frequency (MHz)')
ax.set_ylabel('Fibre length (km)')
ax.set_title('BER heatmap — LED in Step-Index MMF')
#the BER threshold curve is added to compare our sweep/simulation data to an analytic calculationi
ax.plot(f_error_curve / 1e6, L_sweep / 1e3, 
        color='cyan', lw=2, linestyle='--', label='BER threshold')

#----------------------------------------------



plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
