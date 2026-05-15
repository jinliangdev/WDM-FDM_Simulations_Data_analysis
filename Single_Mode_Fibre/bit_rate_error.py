import numpy as np
import matplotlib.pyplot as plt

import multiple_pulse


# ============================================================
# Bit decoding functions
# ============================================================

def decode_bits_from_signal(
    t_ns,
    output_signal,
    num_bits,
    bit_rate_Gbps,
    threshold=None
):
    """
    Decode bits by sampling the output signal at the centre of each bit slot.

    Parameters
    ----------
    t_ns : array
        Time array in ns.

    output_signal : array
        Received/output intensity signal.

    num_bits : int
        Number of transmitted bits.

    bit_rate_Gbps : float
        Bit rate in Gbit/s.
        Since 1 Gbit/s = 1 bit/ns, the bit period in ns is:
            T_b = 1 / bit_rate_Gbps

    threshold : float or None
        Decision threshold.
        If None, the threshold is chosen as half of the maximum output signal.

    Returns
    -------
    decoded_bits : array
        Decoded 0/1 bit sequence.

    sample_times_ns : array
        Times where the signal was sampled.

    sampled_values : array
        Output signal values at the sampling times.

    threshold : float
        Threshold used for decoding.
    """

    bit_period_ns = 1 / bit_rate_Gbps

    # Sample at the centre of each bit slot
    sample_times_ns = (np.arange(num_bits) + 0.5) * bit_period_ns

    # Interpolate the received signal at the sample times
    sampled_values = np.interp(sample_times_ns, t_ns, output_signal)

    # Default threshold
    if threshold is None:
        threshold = 0.5 * np.max(output_signal)

    # Decode rule:
    # above threshold -> 1
    # below threshold -> 0
    decoded_bits = (sampled_values > threshold).astype(int)

    return decoded_bits, sample_times_ns, sampled_values, threshold


# ============================================================
# BER analysis function
# ============================================================

def calculate_bit_error_rate(
    cable_length_km=50,
    wavelength_nm=1550,
    pulse_sigma_ns=0.05,
    duration_ns=20,
    bit_rate_Gbps=2,
    peak_intensity=1.0,
    seed=1,
    samples_per_bit=512,
    pad_bits=30,
    remove_group_delay=True,
    threshold=None,
    plot=True
):
    """
    Run the FFT material-dispersion pulse-train simulation,
    decode the output, and calculate the bit error rate.

    This function calls the simulation function from multiple_pulse.py.

    Parameters
    ----------
    cable_length_km : float
        Fibre length in km.

    wavelength_nm : float
        Central wavelength in nm.

    pulse_sigma_ns : float
        Standard deviation of each Gaussian intensity pulse in ns.

    duration_ns : float
        Total data duration in ns.

    bit_rate_Gbps : float
        Bit rate in Gbit/s.

    peak_intensity : float
        Peak intensity of each input pulse.

    seed : int
        Random seed for bit generation.

    samples_per_bit : int
        Time resolution.

    pad_bits : int
        Padding before and after the data stream.
        This helps avoid FFT wrap-around artefacts.

    remove_group_delay : bool
        Whether to remove the bulk propagation delay.

    threshold : float or None
        Receiver decision threshold.
        If None, threshold = 0.5 * max(output_signal).

    plot : bool
        If True, plots the input signal, output signal, sample points,
        and threshold line.

    Returns
    -------
    ber : float
        Bit error rate.

    results : dict
        Dictionary containing transmitted bits, decoded bits,
        sampled values, threshold, and simulation outputs.
    """

    # ------------------------------------------------------------
    # Run the FFT-based material dispersion simulation
    # ------------------------------------------------------------

    t_ns, input_signal, output_signal, transmitted_bits, sim_results = (
        multiple_pulse.simulate_fft_material_dispersion_pulse_train(
            cable_length_km=cable_length_km,
            wavelength_nm=wavelength_nm,
            pulse_sigma_ns=pulse_sigma_ns,
            duration_ns=duration_ns,
            bit_rate_Gbps=bit_rate_Gbps,
            peak_intensity=peak_intensity,
            seed=seed,
            samples_per_bit=samples_per_bit,
            pad_bits=pad_bits,
            remove_group_delay=remove_group_delay
        )
    )

    num_bits = len(transmitted_bits)

    # ------------------------------------------------------------
    # Decode the received signal
    # ------------------------------------------------------------

    decoded_bits, sample_times_ns, sampled_values, used_threshold = (
        decode_bits_from_signal(
            t_ns=t_ns,
            output_signal=output_signal,
            num_bits=num_bits,
            bit_rate_Gbps=bit_rate_Gbps,
            threshold=threshold
        )
    )

    # ------------------------------------------------------------
    # Calculate BER
    # ------------------------------------------------------------

    errors = decoded_bits != transmitted_bits
    num_errors = np.sum(errors)
    ber = num_errors / num_bits

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------

    results = {
        "ber": ber,
        "num_errors": int(num_errors),
        "num_bits": int(num_bits),

        "transmitted_bits": transmitted_bits,
        "decoded_bits": decoded_bits,
        "errors": errors,

        "sample_times_ns": sample_times_ns,
        "sampled_values": sampled_values,
        "threshold": used_threshold,

        "t_ns": t_ns,
        "input_signal": input_signal,
        "output_signal": output_signal,

        "simulation_results": sim_results,

        "cable_length_km": cable_length_km,
        "wavelength_nm": wavelength_nm,
        "pulse_sigma_ns": pulse_sigma_ns,
        "duration_ns": duration_ns,
        "bit_rate_Gbps": bit_rate_Gbps,
        "peak_intensity": peak_intensity,
        "seed": seed,
        "samples_per_bit": samples_per_bit,
        "pad_bits": pad_bits,
        "remove_group_delay": remove_group_delay,
    }

    # ------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------

    if plot:
        plot_decoding_result(
            t_ns=t_ns,
            input_signal=input_signal,
            output_signal=output_signal,
            threshold=used_threshold,
            duration_ns=duration_ns,
            ber=ber,
            num_errors=int(num_errors),
            num_bits=int(num_bits)
        )

    return ber, results


# ============================================================
# Plotting function
# ============================================================

def plot_decoding_result(
    t_ns,
    input_signal,
    output_signal,
    threshold,
    duration_ns,
    ber,
    num_errors,
    num_bits
):
    """
    Plot the original input signal, dispersed output signal,
    decision threshold, and BER.
    """

    mask = (t_ns >= 0) & (t_ns <= duration_ns)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Original transmitted signal
    ax.plot(
        t_ns[mask],
        input_signal[mask],
        linewidth=2,
        label="Original input signal"
    )

    # Output signal after dispersion
    ax.plot(
        t_ns[mask],
        output_signal[mask],
        linewidth=1.8,
        label="Output signal after dispersion"
    )

    # Horizontal threshold line
    ax.axhline(
        y=threshold,
        linestyle=":",
        linewidth=2,
        label=f"Decision threshold = {threshold:.3f}"
    )

    # BER text box
    ber_text = (
        f"Bit Error Rate = {ber:.6f}\n"
        f"Errors = {num_errors}/{num_bits}"
    )

    ax.text(
        0.02,
        0.95,
        ber_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.9
        )
    )

    ax.set_xlabel("Time / ns")
    ax.set_ylabel("Intensity / arbitrary units")
    ax.set_title("Bit Error Rate Analysis")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Results printing function
# ============================================================

def print_ber_results(results):
    """
    Print BER analysis results in a readable format.
    """

    print("Bit error rate analysis")
    print("-----------------------")
    print(f"Transmitted bits: {results['transmitted_bits']}")
    print(f"Decoded bits:     {results['decoded_bits']}")
    print(f"Errors:           {results['errors']}")
    print()
    print(f"Number of bits:   {results['num_bits']}")
    print(f"Number of errors: {results['num_errors']}")
    print(f"BER:              {results['ber']:.6f}")
    print(f"Threshold used:   {results['threshold']:.4f}")
    print()
    print("Simulation parameters")
    print("---------------------")
    print(f"Cable length:     {results['cable_length_km']} km")
    print(f"Wavelength:       {results['wavelength_nm']} nm")
    print(f"Pulse sigma:      {results['pulse_sigma_ns']} ns")
    print(f"Duration:         {results['duration_ns']} ns")
    print(f"Bit rate:         {results['bit_rate_Gbps']} Gbit/s")
    print(f"Samples per bit:  {results['samples_per_bit']}")
    print(f"Pad bits:         {results['pad_bits']}")
    print(f"Group delay removed: {results['remove_group_delay']}")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    ber, results = calculate_bit_error_rate(
        cable_length_km=50,
        wavelength_nm=850,
        pulse_sigma_ns=0.01,
        duration_ns=100,
        bit_rate_Gbps=1,
        peak_intensity=1.0,
        seed=67,
        samples_per_bit=512,
        pad_bits=30,
        remove_group_delay=True,
        threshold=None,
        plot=True
    )

    print_ber_results(results)