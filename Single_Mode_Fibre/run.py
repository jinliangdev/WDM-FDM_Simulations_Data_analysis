import numpy as np
import multiple_pulse
import bit_rate_error
import ber_heatmap

# Example run for Multiple Pulses

cable_length_km = 100
wavelength_nm = 850
pulse_sigma_ns = 0.1
duration_ns = 10
bit_rate_Gbps = 1

def plot_multiple_pulse():

    t_ns, input_signal, output_signal, bits, results = multiple_pulse.simulate_fft_material_dispersion_pulse_train(
        cable_length_km=cable_length_km,
        wavelength_nm=wavelength_nm,
        pulse_sigma_ns=pulse_sigma_ns,
        duration_ns=duration_ns,
        bit_rate_Gbps=bit_rate_Gbps,
        peak_intensity=1.0,
        seed=67,
        samples_per_bit=512,
        pad_bits=30,
        remove_group_delay=True
    )

    multiple_pulse.plot_data_window(
        t_ns,
        input_signal,
        output_signal,
        duration_ns=duration_ns
    )

def plot_single_bit_error_rate():
    ber, results = bit_rate_error.calculate_bit_error_rate(
        cable_length_km=100,
        wavelength_nm=850,
        pulse_sigma_ns=0.015,
        duration_ns=50,
        bit_rate_Gbps=3,
        peak_intensity=1.0,
        seed=67,
        samples_per_bit=512,
        pad_bits=30,
        remove_group_delay=True,
        threshold=None,
        plot=True
    )

def plot_heatmap():
        cable_lengths_km = np.linspace(10, 400, 20)
        bit_rates_Gbps = np.linspace(1, 10, 20)

        wavelength_nm = 850
        pulse_sigma_ns = 0.01
        duration_ns = 50
        peak_intensity = 1.0
        seed = 2
        samples_per_bit = 512
        pad_bits = 30
        remove_group_delay = True
        threshold = None

        ber_grid = ber_heatmap.generate_ber_heatmap_data(
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

        ber_heatmap.plot_ber_heatmap(
            cable_lengths_km=cable_lengths_km,
            bit_rates_Gbps=bit_rates_Gbps,
            ber_grid=ber_grid,
            wavelength_nm=wavelength_nm,
            pulse_sigma_ns=pulse_sigma_ns,
            duration_ns=duration_ns
        )

        # ber_heatmap.plot_ber_heatmap_with_values(
        #     cable_lengths_km=cable_lengths_km,
        #     bit_rates_Gbps=bit_rates_Gbps,
        #     ber_grid=ber_grid,
        #     wavelength_nm=wavelength_nm,
        #     pulse_sigma_ns=pulse_sigma_ns,
        #     duration_ns=duration_ns
        # )

#plot_heatmap()

#plot_single_bit_error_rate()

#plot_multiple_pulse()