import numpy as np
import matplotlib.pyplot as plt

import bit_rate_error


def generate_ber_heatmap_data(
    cable_lengths_km,
    bit_rates_Gbps,
    wavelength_nm=1550,
    pulse_sigma_ns=0.05,
    duration_ns=20,
    peak_intensity=1.0,
    seed=1,
    samples_per_bit=512,
    pad_bits=30,
    remove_group_delay=True,
    threshold=None
):
    """
    Generate a 2D array of bit error rates for different cable lengths
    and bit rates.
    """

    ber_grid = np.zeros((len(bit_rates_Gbps), len(cable_lengths_km)))

    total_iterations = len(bit_rates_Gbps) * len(cable_lengths_km)
    iteration = 0

    for i, bit_rate in enumerate(bit_rates_Gbps):
        for j, cable_length in enumerate(cable_lengths_km):

            ber, results = bit_rate_error.calculate_bit_error_rate(
                cable_length_km=cable_length,
                wavelength_nm=wavelength_nm,
                pulse_sigma_ns=pulse_sigma_ns,
                duration_ns=duration_ns,
                bit_rate_Gbps=bit_rate,
                peak_intensity=peak_intensity,
                seed=seed,
                samples_per_bit=samples_per_bit,
                pad_bits=pad_bits,
                remove_group_delay=remove_group_delay,
                threshold=threshold,
                plot=False
            )

            ber_grid[i, j] = ber

            iteration += 1

            print(
                f"[{iteration}/{total_iterations}] Completed: "
                f"L = {cable_length:.2f} km, "
                f"Bit rate = {bit_rate:.2f} Gbit/s, "
                f"BER = {ber:.6f}"
            )

    return ber_grid

def plot_ber_heatmap(
    cable_lengths_km,
    bit_rates_Gbps,
    ber_grid,
    wavelength_nm=1550,
    pulse_sigma_ns=0.05,
    duration_ns=20,
    figsize=(9, 6)
):
    """
    Plot BER as a heatmap against cable length and bit rate.
    """

    plt.figure(figsize=figsize)

    extent = [
        cable_lengths_km[0],
        cable_lengths_km[-1],
        bit_rates_Gbps[0],
        bit_rates_Gbps[-1]
    ]

    im = plt.imshow(
        ber_grid,
        origin="lower",
        aspect="auto",
        extent=extent
    )

    cbar = plt.colorbar(im)
    cbar.set_label("Bit Error Rate")

    plt.xlabel("Cable length / km")
    plt.ylabel("Bit rate / Gbit/s")
    plt.title(
        f"BER Heatmap for Material Dispersion\n"
        f"λ = {wavelength_nm} nm, pulse σ = {pulse_sigma_ns} ns, duration = {duration_ns} ns"
    )

    plt.tight_layout()
    plt.show()


def plot_ber_heatmap_with_values(
    cable_lengths_km,
    bit_rates_Gbps,
    ber_grid,
    wavelength_nm=1550,
    pulse_sigma_ns=0.05,
    duration_ns=20,
    figsize=(10, 6)
):
    """
    Plot BER heatmap and write the BER value inside each square.
    """

    fig, ax = plt.subplots(figsize=figsize)

    extent = [
        cable_lengths_km[0],
        cable_lengths_km[-1],
        bit_rates_Gbps[0],
        bit_rates_Gbps[-1]
    ]

    im = ax.imshow(
        ber_grid,
        origin="lower",
        aspect="auto",
        extent=extent
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Bit Error Rate")

    ax.set_xlabel("Cable length / km")
    ax.set_ylabel("Bit rate / Gbit/s")
    ax.set_title(
        f"BER Heatmap for Material Dispersion\n"
        f"λ = {wavelength_nm} nm, pulse σ = {pulse_sigma_ns} ns, duration = {duration_ns} ns"
    )

    for i, bit_rate in enumerate(bit_rates_Gbps):
        for j, cable_length in enumerate(cable_lengths_km):
            ax.text(
                cable_length,
                bit_rate,
                f"{ber_grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    cable_lengths_km = np.linspace(10, 200, 20)
    bit_rates_Gbps = np.linspace(1, 10, 10)

    wavelength_nm = 850
    pulse_sigma_ns = 0.03
    duration_ns = 100
    peak_intensity = 1.0
    seed = 67
    samples_per_bit = 512
    pad_bits = 30
    remove_group_delay = True
    threshold = None

    ber_grid = generate_ber_heatmap_data(
        cable_lengths_km=cable_lengths_km,
        bit_rates_Gbps=bit_rates_Gbps,
        wavelength_nm=wavelength_nm,
        pulse_sigma_ns=pulse_sigma_ns,
        duration_ns=duration_ns,
        peak_intensity=peak_intensity,
        seed=seed,
        samples_per_bit=samples_per_bit,
        pad_bits=pad_bits,
        remove_group_delay=remove_group_delay,
        threshold=threshold
    )

    plot_ber_heatmap(
        cable_lengths_km=cable_lengths_km,
        bit_rates_Gbps=bit_rates_Gbps,
        ber_grid=ber_grid,
        wavelength_nm=wavelength_nm,
        pulse_sigma_ns=pulse_sigma_ns,
        duration_ns=duration_ns
    )

    # plot_ber_heatmap_with_values(
    #     cable_lengths_km=cable_lengths_km,
    #     bit_rates_Gbps=bit_rates_Gbps,
    #     ber_grid=ber_grid,
    #     wavelength_nm=wavelength_nm,
    #     pulse_sigma_ns=pulse_sigma_ns,
    #     duration_ns=duration_ns
    # )