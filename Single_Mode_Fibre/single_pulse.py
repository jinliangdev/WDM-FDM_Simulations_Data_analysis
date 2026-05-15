import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Sellmeier equation for fused silica
# ============================================================

def sellmeier_fused_silica(lambda_um):
    """
    Calculate refractive index n(lambda) for fused silica.

    Uses the three-term Sellmeier equation from Malitson.

    Parameters
    ----------
    lambda_um : float or array
        Vacuum wavelength in micrometres.

    Returns
    -------
    n : float or array
        Refractive index of fused silica.
    """

    # Malitson fused silica Sellmeier coefficients
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794

    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2

    lambda_squared = lambda_um**2

    n_squared = (
        1
        + (B1 * lambda_squared) / (lambda_squared - C1)
        + (B2 * lambda_squared) / (lambda_squared - C2)
        + (B3 * lambda_squared) / (lambda_squared - C3)
    )

    return np.sqrt(n_squared)


def material_dispersion_from_sellmeier(lambda0_nm):
    """
    Calculate the material dispersion parameter D_m from the Sellmeier equation.

    Parameters
    ----------
    lambda0_nm : float
        Central wavelength in nm.

    Returns
    -------
    D_s_per_m2 : float
        Material dispersion parameter in SI units: s / m^2.

    D_ps_per_nm_km : float
        Material dispersion parameter in common units: ps / (nm km).
    """

    c = 299792458

    # Convert nm to micrometres for the Sellmeier equation
    lambda0_um = lambda0_nm * 1e-3

    # Step size for finite difference in micrometres
    h_um = 1e-5

    n_plus = sellmeier_fused_silica(lambda0_um + h_um)
    n_0 = sellmeier_fused_silica(lambda0_um)
    n_minus = sellmeier_fused_silica(lambda0_um - h_um)

    # Second derivative d²n/dλ² with λ measured in micrometres
    d2n_dlambda2_per_um2 = (n_plus - 2 * n_0 + n_minus) / h_um**2

    # Convert from 1/um² to 1/m²
    d2n_dlambda2_per_m2 = d2n_dlambda2_per_um2 * 1e12

    # Convert central wavelength to metres
    lambda0_m = lambda0_nm * 1e-9

    # Material dispersion parameter
    D_s_per_m2 = -(lambda0_m / c) * d2n_dlambda2_per_m2

    # Convert s/m² to ps/(nm km)
    # 1 ps/(nm km) = 1e-6 s/m²
    D_ps_per_nm_km = D_s_per_m2 / 1e-6

    return D_s_per_m2, D_ps_per_nm_km


# ============================================================
# Gaussian pulse functions
# ============================================================

def gaussian_pulse(t, t0, sigma_t, amplitude=1.0):
    """
    Generate a Gaussian intensity pulse.

    I(t) = I0 exp[-(t - t0)^2 / (2 sigma_t^2)]

    Parameters
    ----------
    t : array
        Time array in seconds.

    t0 : float
        Pulse centre time in seconds.

    sigma_t : float
        RMS temporal width of the intensity pulse in seconds.

    amplitude : float
        Peak intensity.

    Returns
    -------
    pulse : array
        Gaussian intensity pulse.
    """

    return amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma_t**2))


def apply_material_dispersion_single_pulse(
    t,
    input_pulse,
    lambda0_nm,
    fibre_length_km,
    spectral_sigma_nm
):
    """
    Apply material dispersion broadening to a single Gaussian pulse.

    This is still the simple broadening approximation:
        sigma_out = sqrt(sigma_in^2 + sigma_disp^2)

    but D_m is now calculated from the Sellmeier equation.

    Parameters
    ----------
    t : array
        Time array in seconds.

    input_pulse : array
        Input intensity pulse.

    lambda0_nm : float
        Central wavelength of the pulse in nm.

    fibre_length_km : float
        Fibre length in km.

    spectral_sigma_nm : float
        RMS spectral width of the source in nm.

    Returns
    -------
    output_pulse : array
        Output intensity pulse after material dispersion.

    results : dict
        Useful calculated parameters.
    """

    # Calculate material dispersion from Sellmeier equation
    D_s_per_m2, D_ps_per_nm_km = material_dispersion_from_sellmeier(lambda0_nm)

    # Convert units
    L_m = fibre_length_km * 1e3
    spectral_sigma_m = spectral_sigma_nm * 1e-9

    # Estimate original pulse centre and RMS width from the input pulse
    area = np.trapezoid(input_pulse, t)
    mean_t = np.trapezoid(t * input_pulse, t) / area
    sigma_in = np.sqrt(
        np.trapezoid((t - mean_t) ** 2 * input_pulse, t) / area
    )

    input_peak = np.max(input_pulse)

    # Dispersion-induced RMS broadening
    sigma_disp = abs(D_s_per_m2) * L_m * spectral_sigma_m

    # Output RMS width
    sigma_out = np.sqrt(sigma_in**2 + sigma_disp**2)

    # Conserve pulse area, so wider pulse has lower peak intensity
    output_peak = input_peak * sigma_in / sigma_out

    output_pulse = gaussian_pulse(
        t=t,
        t0=mean_t,
        sigma_t=sigma_out,
        amplitude=output_peak
    )

    results = {
        "lambda0_nm": lambda0_nm,
        "fibre_length_km": fibre_length_km,
        "spectral_sigma_nm": spectral_sigma_nm,
        "D_s_per_m2": D_s_per_m2,
        "D_ps_per_nm_km": D_ps_per_nm_km,
        "sigma_in_ns": sigma_in * 1e9,
        "sigma_disp_ns": sigma_disp * 1e9,
        "sigma_out_ns": sigma_out * 1e9,
        "input_peak": input_peak,
        "output_peak": output_peak,
    }

    return output_pulse, results


# ============================================================
# Example simulation
# ============================================================

# Pulse parameters
lambda0_nm = 850          # central wavelength
fibre_length_km = 50       # fibre length
spectral_sigma_nm = 0.1    # RMS spectral width of source

pulse_sigma_ns = 0.2       # RMS temporal width of input pulse
amplitude = 1.0

# Time array
t = np.linspace(0, 10e-9, 5000)
t0 = 5e-9

input_pulse = gaussian_pulse(
    t=t,
    t0=t0,
    sigma_t=pulse_sigma_ns * 1e-9,
    amplitude=amplitude
)

output_pulse, results = apply_material_dispersion_single_pulse(
    t=t,
    input_pulse=input_pulse,
    lambda0_nm=lambda0_nm,
    fibre_length_km=fibre_length_km,
    spectral_sigma_nm=spectral_sigma_nm
)

if __name__ == "__main__":

# ============================================================
# Print results
# ============================================================

    print("Single-pulse Sellmeier material dispersion results")
    print("--------------------------------------------------")
    print(f"Material assumed: fused silica / SiO2, not silicon")
    print(f"Sellmeier terms: 3")
    print(f"Central wavelength: {results['lambda0_nm']:.1f} nm")
    print(f"Fibre length: {results['fibre_length_km']:.1f} km")
    print(f"Spectral sigma: {results['spectral_sigma_nm']:.3f} nm")
    print(f"D_m: {results['D_ps_per_nm_km']:.3f} ps/(nm km)")
    print(f"Input sigma: {results['sigma_in_ns']:.4f} ns")
    print(f"Dispersion broadening sigma: {results['sigma_disp_ns']:.4f} ns")
    print(f"Output sigma: {results['sigma_out_ns']:.4f} ns")
    print(f"Input peak intensity: {results['input_peak']:.4f}")
    print(f"Output peak intensity: {results['output_peak']:.4f}")


    # ============================================================
    # Plot
    # ============================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        t * 1e9,
        input_pulse,
        linewidth=2,
        label="Input pulse"
    )

    plt.plot(
        t * 1e9,
        output_pulse,
        linewidth=2,
        label="Output pulse after material dispersion"
    )

    plt.xlabel("Time / ns")
    plt.ylabel("Intensity / arbitrary units")
    plt.title(
        f"Single Gaussian Pulse with Sellmeier Material Dispersion\n"
        f"λ = {lambda0_nm} nm, L = {fibre_length_km} km"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
