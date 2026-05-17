"""

WDM in a MMF

Single LED channel, step-index MM

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc # type: ignore
import random
import Dispersion as spd

c = 3e8

#Defining cable materials
n1 = 1.46
n2 = 1.44
D_material = -80e-12 / (1e-9 * 1e3)
L = 7500

#Defining source
lambda0 = 850e-9
delta_lambda = 40e-9
sigma_input = 15e-9

#Defining data stream
Nbits = 50
ppb = 100
data = [random.randint(0,1) for i in range(Nbits)]


#Calculating the bit error rate
def compute_BER(t, signal, data, T, sigma_output):
    threshold = 0.5 * gaussian_pulse(0, sigma_output, 0)
    
    errors = 0
    for i, transmitted_bit in enumerate(data):
        idx = np.argmin(np.abs(t - (i + 0.5) * T))
        sample = signal[idx]
        detected_bit = 1 if sample > threshold else 0
        
        if detected_bit != transmitted_bit:
            errors += 1
            #print(f"ERROR at bit {i}: transmitted={transmitted_bit}, detected={detected_bit}, sample={sample:.4e}")
    
    BER = errors / len(data)
    #print(f"Total errors: {errors}, BER: {BER:.3f}")
    return BER, threshold

#Defining a single pulse.
def gaussian_pulse(t, sigma, centre = 0):
     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(t - centre)**2 / (2 * sigma**2))

#Calculating the final width of the pulse due to dispersion.
sigma_output = np.sqrt(spd.modal_dispersion(n1, n2, L)**2 + spd.chromatic_dispersion(D_material,delta_lambda, L)**2 + sigma_input**2)
f_3dB = 0.187/sigma_output


#Three regimes, low modulation frequency, bits are seen clearly, rising, BER rises slightly as signals smear. BER higher, even more smearing causing higher BER
#----------------------------------------------
# T_clean = 6 * sigma_output
# T_rising = 1 * sigma_output
# T_high = 0.3 * sigma_output

# regimes = [
#      (T_clean, 'T = 6 sigma_input'),
#      (T_rising, 'T = 2 sigma'),
#      (T_high, 'T = 0.5 sigma')
# ]


# t = np.linspace(0, Nbits * T_clean, Nbits * ppb)
# SIGNAL = np.zeros_like(t)
# INPUT = np.zeros_like(t)


# fig, axes = plt.subplots(3, 1, figsize = (12, 8))

# for ax, (T, label) in zip(axes, regimes):
#     t = np.linspace(0, Nbits * T, Nbits * ppb)
#     SIGNAL = np.zeros_like(t)
#     INPUT = np.zeros_like(t)
    

#     for i, bit in enumerate(data):
#         t_centre = (i + 0.5) * T
#         if bit == 1:
#             SIGNAL += gaussian_pulse(t, sigma_output, t_centre)
#             INPUT += gaussian_pulse(t, sigma_input, t_centre)

#     BER, threshold = compute_BER(t, SIGNAL, data, T, sigma_output)
#     ax.axhline(threshold, color='red', linestyle='--', linewidth=1, label='Threshold')
#     threshold = 0.5 * gaussian_pulse(0, sigma_output, 0)


    
#     f_mod = 1/T
#     for i in range(Nbits):
#         ax.axvline(x=(i + 0.5) * T, color='gray', alpha=0.3, linewidth=0.5)
#     ax.plot(t, SIGNAL, label = "Output")
#     ax.plot(t, INPUT, label = "Input")
#     ax.text(0.98,0.95, f"f = {f_mod/1e6:.2f} MHz\nf_3dB = {f_3dB/1e6:.2f} MHz\nBER = {BER:.3f}", transform=ax.transAxes, ha='right', va='top', fontsize = 9)
#     ax.set_title(label)
#     ax.set_xlabel("Time (s)")
#     #ax.grid(True)
#     ax.legend()
#----------------------------------------------



# Produces a heatmap of frequency and length against the BER.
#----------------------------------------------
f_sweep = np.logspace(4, 8, 100)
L_sweep = np.logspace(1, 4, 50)
BER_grid = np.zeros((len(L_sweep), len(f_sweep)))

f_error_curve = np.array([
    0.6 / np.sqrt(spd.modal_dispersion(n1, n2, Lv)**2 +
                  spd.chromatic_dispersion(D_material, delta_lambda, Lv)**2 +
                  sigma_input**2)
    for Lv in L_sweep])

for i, L in enumerate(L_sweep):
    # recompute sigma_output for this length
    sig_out = np.sqrt(spd.modal_dispersion(n1, n2, L)**2 + 
                      spd.chromatic_dispersion(D_material, delta_lambda, L)**2 + 
                      sigma_input**2)
    
    for j, f in enumerate(f_sweep):
        T = 1 / f
        t = np.linspace(0, Nbits * T, Nbits * ppb)
        SIGNAL = np.zeros_like(t)

        for k, bit in enumerate(data):
            if bit == 1:
                t_centre = (k + 0.5) * T
                SIGNAL += gaussian_pulse(t, sig_out, t_centre)

        BER, _ = compute_BER(t, SIGNAL, data, T, sig_out)
        BER_grid[i, j] = BER

    f_3dB_curve = np.array([
    0.187 / np.sqrt(spd.modal_dispersion(n1, n2, Lv)**2 +
                    spd.chromatic_dispersion(D_material, delta_lambda, Lv)**2 +
                    sigma_input**2)
    for Lv in L_sweep])

    f_error_curve = np.array([
        0.6 / np.sqrt(spd.modal_dispersion(n1, n2, Lv)**2 +
                    spd.chromatic_dispersion(D_material, delta_lambda, Lv)**2 +
                    sigma_input**2)
        for Lv in L_sweep])

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.pcolormesh(f_sweep / 1e6, L_sweep / 1e3, BER_grid, cmap = "RdYlGn_r", vmin = 0, vmax = 0.5)
plt.colorbar(im, ax=ax, label = "BER")


ax.set_xscale('log')
ax.set_xlabel('Modulation frequency (MHz)')
ax.set_ylabel('Fibre length (km)')
ax.set_title('BER heatmap — LED in Step-Index MMF')

ax.plot(f_3dB_curve / 1e6, L_sweep / 1e3, 
        color='white', lw=2, linestyle='--', label='f_3dB')
ax.plot(f_error_curve / 1e6, L_sweep / 1e3, 
        color='cyan', lw=2, linestyle='--', label='BER threshold')

#----------------------------------------------



plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()