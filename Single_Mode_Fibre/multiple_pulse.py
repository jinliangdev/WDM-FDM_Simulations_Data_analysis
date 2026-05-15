import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Sellmeier equation for fused silica
# ============================================================

def sellmeier_fused_silica(lambda_um):
    """
    Refractive index of fused silica using the Sellmeier equation.

    lambda_um is wavelength in micrometres.
    """

    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794

    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2

    lam2 = lambda_um**2

    n_squared = (
        1
        + (B1 * lam2) / (lam2 - C1)
        + (B2 * lam2) / (lam2 - C2)
        + (B3 * lam2) / (lam2 - C3)
    )

    return np.sqrt(n_squared)


def beta_from_omega(omega):
    """
    Calculate propagation constant beta(omega) using Sellmeier.

    beta(omega) = n(omega) omega / c

    omega is absolute angular frequency in rad/s.
    """

    c = 299792458

    omega = np.asarray(omega)

    # Convert angular frequency to wavelength:
    # omega = 2 pi c / lambda
    lambda_m = 2 * np.pi * c / omega
    lambda_um = lambda_m * 1e6

    n = sellmeier_fused_silica(lambda_um)

    beta = n * omega / c

    return beta


def calculate_beta1(omega0):
    """
    Numerically calculate beta1 = d beta / d omega at omega0.

    beta1 is the inverse group velocity.
    """

    domega = omega0 * 1e-6

    beta_plus = beta_from_omega(omega0 + domega)
    beta_minus = beta_from_omega(omega0 - domega)

    beta1 = (beta_plus - beta_minus) / (2 * domega)

    return beta1


# ============================================================
# Pulse-train generation
# ============================================================

def generate_random_bits(num_bits, seed=1):
    """
    Generate random 0s and 1s.
    """

    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, num_bits)


def gaussian_field_pulse(t, centre, intensity_sigma, peak_intensity=1.0):
    """
    Gaussian electric-field envelope.

    The intensity is:

        I(t) = |E(t)|²

    If we want the intensity to have standard deviation sigma, then
    the field envelope must have:

        E(t) = sqrt(I0) exp[-(t - t0)² / (4 sigma²)]

    because squaring the field gives:

        I(t) = I0 exp[-(t - t0)² / (2 sigma²)]
    """

    return np.sqrt(peak_intensity) * np.exp(
        -((t - centre) ** 2) / (4 * intensity_sigma**2)
    )


# ============================================================
# FFT-based material dispersion simulation
# ============================================================

def simulate_fft_material_dispersion_pulse_train(
    cable_length_km=50,
    wavelength_nm=1550,
    pulse_sigma_ns=0.1,
    duration_ns=20,
    bit_rate_Gbps=1,
    peak_intensity=1.0,
    seed=1,
    samples_per_bit=256,
    pad_bits=20,
    remove_group_delay=True
):
    """
    Simulate material dispersion of a Gaussian pulse train using FFT propagation.

    Parameters
    ----------
    cable_length_km : float
        Fibre length in km.

    wavelength_nm : float
        Central optical wavelength in nm.

    pulse_sigma_ns : float
        Standard deviation of the INTENSITY Gaussian pulse in ns.

    duration_ns : float
        Duration of the data stream in ns.

    bit_rate_Gbps : float
        Bit rate in Gbit/s.

    peak_intensity : float
        Peak intensity of each input pulse.

    seed : int
        Random seed for bit generation.

    samples_per_bit : int
        Number of time samples per bit.

    pad_bits : int
        Extra empty bit periods added before and after the signal.
        This reduces FFT wrap-around artefacts.

    remove_group_delay : bool
        If True, removes the bulk group delay so the output stays in the
        same time window. This keeps the plot readable.

    Returns
    -------
    t_plot : array
        Time array in ns.

    input_intensity : array
        Input intensity signal.

    output_intensity : array
        Output intensity after FFT propagation.

    bits : array
        Transmitted bit sequence.

    results : dict
        Useful calculated parameters.
    """

    # ----------------------------
    # Unit conversions
    # ----------------------------

    c = 299792458

    L = cable_length_km * 1e3
    lambda0 = wavelength_nm * 1e-9
    pulse_sigma = pulse_sigma_ns * 1e-9
    duration = duration_ns * 1e-9
    bit_rate = bit_rate_Gbps * 1e9

    bit_period = 1 / bit_rate

    num_bits = int(np.floor(duration / bit_period))

    if num_bits < 1:
        raise ValueError("Duration is too short for the chosen bit rate.")

    # ----------------------------
    # Time grid with padding
    # ----------------------------

    dt = bit_period / samples_per_bit

    pad_time = pad_bits * bit_period
    total_time = duration + 2 * pad_time

    t = np.arange(0, total_time, dt)
    N = len(t)

    # ----------------------------
    # Generate random bits
    # ----------------------------

    bits = generate_random_bits(num_bits, seed=seed)

    pulse_centres = pad_time + (np.arange(num_bits) + 0.5) * bit_period

    # ----------------------------
    # Build full input intensity pulse train
    # ----------------------------

    input_intensity = np.zeros_like(t)

    for bit, centre in zip(bits, pulse_centres):
        if bit == 1:
            input_intensity += peak_intensity * np.exp(
                -((t - centre) ** 2) / (2 * pulse_sigma**2)
            )

    # Convert the intensity envelope into a field envelope for propagation
    E_in = np.sqrt(input_intensity).astype(np.complex128)

    # ----------------------------
    # FFT of full pulse train
    # ----------------------------

    E_w = np.fft.fft(E_in)

    # Envelope angular frequency offsets
    freq_offsets = np.fft.fftfreq(N, d=dt)
    Omega = 2 * np.pi * freq_offsets

    # Optical carrier angular frequency
    omega0 = 2 * np.pi * c / lambda0

    # Absolute optical angular frequencies for each FFT component
    omega_abs = omega0 + Omega

    if np.any(omega_abs <= 0):
        raise ValueError(
            "Some FFT frequency components produce nonphysical negative optical frequencies. "
            "Increase samples_per_bit or use a lower bit rate."
        )

    # ----------------------------
    # Sellmeier propagation phase
    # ----------------------------

    beta_abs = beta_from_omega(omega_abs)
    beta0 = beta_from_omega(omega0)

    if remove_group_delay:
        # Remove beta0 and beta1 terms.
        #
        # beta0 is just a global phase.
        # beta1 shifts the whole pulse train in time by the group delay.
        #
        # Removing beta1 puts us in the moving frame of the pulse,
        # so we mainly see the broadening/distortion.
        beta1 = calculate_beta1(omega0)
        phase = (beta_abs - beta0 - beta1 * Omega) * L
    else:
        # Keep group delay.
        # Warning: for long fibres, the signal may shift far outside the window.
        phase = (beta_abs - beta0) * L

    H = np.exp(1j * phase)

    # Apply fibre dispersion
    E_w_out = E_w * H

    # ----------------------------
    # Inverse FFT back to time
    # ----------------------------

    E_out = np.fft.ifft(E_w_out)

    output_intensity = np.abs(E_out) ** 2

    # ----------------------------
    # Plotting time axis
    # ----------------------------

    # Shift time so the actual data stream starts at 0 ns
    t_plot = (t - pad_time) * 1e9

    # ----------------------------
    # Useful diagnostics
    # ----------------------------

    results = {
        "num_bits": num_bits,
        "bits": bits,
        "cable_length_km": cable_length_km,
        "wavelength_nm": wavelength_nm,
        "pulse_sigma_ns": pulse_sigma_ns,
        "duration_ns": duration_ns,
        "bit_rate_Gbps": bit_rate_Gbps,
        "bit_period_ns": bit_period * 1e9,
        "samples_per_bit": samples_per_bit,
        "dt_ps": dt * 1e12,
        "input_peak_intensity": np.max(input_intensity),
        "output_peak_intensity": np.max(output_intensity),
        "remove_group_delay": remove_group_delay,
    }

    return t_plot, input_intensity, output_intensity, bits, results


## Plotting Everything

# ============================================================
# Plotting functions
# ============================================================

def plot_full_signal(
    t_ns,
    input_signal,
    output_signal,
    cable_length_km=None,
    wavelength_nm=None,
    bit_rate_Gbps=None,
    figsize=(11, 5)
):
    """
    Plot the full simulated time window, including padding.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        t_ns,
        input_signal,
        linewidth=2,
        label="Input pulse train"
    )

    plt.plot(
        t_ns,
        output_signal,
        linewidth=1.6,
        label="Output after FFT Sellmeier dispersion"
    )

    plt.xlabel("Time / ns")
    plt.ylabel("Intensity / arbitrary units")

    if cable_length_km is not None and wavelength_nm is not None and bit_rate_Gbps is not None:
        plt.title(
            f"FFT-Based Material Dispersion of Gaussian Pulse Train\n"
            f"L = {cable_length_km} km, λ = {wavelength_nm} nm, "
            f"bit rate = {bit_rate_Gbps} Gbit/s"
        )
    else:
        plt.title("FFT-Based Material Dispersion of Gaussian Pulse Train")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_data_window(
    t_ns,
    input_signal,
    output_signal,
    duration_ns,
    figsize=(11, 5)
):
    """
    Plot only the actual data window, excluding the padding region.
    """

    mask = (t_ns >= 0) & (t_ns <= duration_ns)

    plt.figure(figsize=figsize)

    plt.plot(
        t_ns[mask],
        input_signal[mask],
        linewidth=2,
        label="Input pulse train"
    )

    plt.plot(
        t_ns[mask],
        output_signal[mask],
        linewidth=1.6,
        label="Output after FFT Sellmeier dispersion"
    )

    plt.xlabel("Time / ns")
    plt.ylabel("Intensity / arbitrary units")
    plt.title("Zoomed View of Data Window")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_input_only(
    t_ns,
    input_signal,
    duration_ns=None,
    figsize=(11, 4)
):
    """
    Plot only the original input pulse train.
    If duration_ns is given, only the data window is plotted.
    """

    if duration_ns is not None:
        mask = (t_ns >= 0) & (t_ns <= duration_ns)
        t_plot = t_ns[mask]
        signal_plot = input_signal[mask]
    else:
        t_plot = t_ns
        signal_plot = input_signal

    plt.figure(figsize=figsize)

    plt.plot(
        t_plot,
        signal_plot,
        linewidth=2,
        label="Input pulse train"
    )

    plt.xlabel("Time / ns")
    plt.ylabel("Intensity / arbitrary units")
    plt.title("Original Input Pulse Train")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_output_only(
    t_ns,
    output_signal,
    duration_ns=None,
    figsize=(11, 4)
):
    """
    Plot only the output pulse train after dispersion.
    If duration_ns is given, only the data window is plotted.
    """

    if duration_ns is not None:
        mask = (t_ns >= 0) & (t_ns <= duration_ns)
        t_plot = t_ns[mask]
        signal_plot = output_signal[mask]
    else:
        t_plot = t_ns
        signal_plot = output_signal

    plt.figure(figsize=figsize)

    plt.plot(
        t_plot,
        signal_plot,
        linewidth=2,
        label="Output after FFT Sellmeier dispersion"
    )

    plt.xlabel("Time / ns")
    plt.ylabel("Intensity / arbitrary units")
    plt.title("Output Pulse Train After Material Dispersion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_three_panel_comparison(
    t_ns,
    input_signal,
    output_signal,
    duration_ns,
    figsize=(11, 9)
):
    """
    Plot input only, output only, and overlay in three stacked panels.
    This is usually the clearest comparison plot.
    """

    mask = (t_ns >= 0) & (t_ns <= duration_ns)

    t_plot = t_ns[mask]
    input_plot = input_signal[mask]
    output_plot = output_signal[mask]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Input only
    axes[0].plot(t_plot, input_plot, linewidth=2)
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Original Input Pulse Train")
    axes[0].grid(True)

    # Output only
    axes[1].plot(t_plot, output_plot, linewidth=2)
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("Output Pulse Train After FFT Sellmeier Dispersion")
    axes[1].grid(True)

    # Overlay
    axes[2].plot(
        t_plot,
        input_plot,
        linewidth=2,
        label="Input"
    )

    axes[2].plot(
        t_plot,
        output_plot,
        linewidth=1.6,
        label="Output"
    )

    axes[2].set_xlabel("Time / ns")
    axes[2].set_ylabel("Intensity")
    axes[2].set_title("Input and Output Overlaid")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def print_simulation_results(results):
    """
    Print useful simulation diagnostics.
    """

    print("FFT material dispersion simulation results")
    print("------------------------------------------")
    print(f"Bits transmitted: {results['bits']}")
    print(f"Number of bits: {results['num_bits']}")
    print(f"Cable length: {results['cable_length_km']:.2f} km")
    print(f"Wavelength: {results['wavelength_nm']:.2f} nm")
    print(f"Pulse intensity sigma: {results['pulse_sigma_ns']:.4f} ns")
    print(f"Bit rate: {results['bit_rate_Gbps']:.3f} Gbit/s")
    print(f"Bit period: {results['bit_period_ns']:.4f} ns")
    print(f"Time step: {results['dt_ps']:.4f} ps")
    print(f"Input peak intensity: {results['input_peak_intensity']:.4f}")
    print(f"Output peak intensity: {results['output_peak_intensity']:.4f}")
    print(f"Group delay removed: {results['remove_group_delay']}")


if __name__ == "__main__":
# Example run

    cable_length_km = 100
    wavelength_nm = 450
    pulse_sigma_ns = 0.1
    duration_ns = 10
    bit_rate_Gbps = 2

    t_ns, input_signal, output_signal, bits, results = simulate_fft_material_dispersion_pulse_train(
        cable_length_km=cable_length_km,
        wavelength_nm=wavelength_nm,
        pulse_sigma_ns=pulse_sigma_ns,
        duration_ns=duration_ns,
        bit_rate_Gbps=bit_rate_Gbps,
        peak_intensity=1.0,
        seed=4,
        samples_per_bit=512,
        pad_bits=30,
        remove_group_delay=True
    )

    plot_full_signal(
        t_ns,
        input_signal,
        output_signal,
        cable_length_km=cable_length_km,
        wavelength_nm=wavelength_nm,
        bit_rate_Gbps=bit_rate_Gbps
    )

    plot_data_window(
        t_ns,
        input_signal,
        output_signal,
        duration_ns=duration_ns
    )

    plot_three_panel_comparison(
        t_ns,
        input_signal,
        output_signal,
        duration_ns=duration_ns
    )

    print_simulation_results(results)