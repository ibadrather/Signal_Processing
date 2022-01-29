import numpy as np
import matplotlib.pyplot as plt
import os

os.system("clear")

## Create a signal with multiple frequencies
dt =  1/ 200    # 2000 Hz Signal
total_t = 3
total_samples = int(total_t / dt)
t = np.linspace(0, total_t, total_samples )  # Timestamps

# Angular Frequencies in the signal
omega_1 = 50
omega_2 = 120

# generate a signal with these two frequencies
clean_signal = np.sin(2 * np.pi * omega_1 * t) + np.sin(2 * np.pi * omega_2 * t)

# Adding Gaussian noise to the signal with a variance
variance = 0.5
noisy_signal = np.random.normal(clean_signal, variance)

# Let's also visualise only the noise
noise = noisy_signal - clean_signal

# Let's Visualise these signals
#plt.plot(t, clean_signal, "r", label = "Clean Signal")
#plt.plot(t, noisy_signal, "b",  label = "Noisy Signal")
#plt.plot(t, noise, "c", label  = "Noise")
#plt.legend()
#plt.show()

## Now that we have Our Signals setup, we will try to filter out this noise
#noisy_signal = clean_signal
# Compute discrete fourier transform of the noisy signal
len_signal = len(noisy_signal)
signal_hat = np.fft.fft(noisy_signal, len_signal)

# Calculate Power Spectral Density
PSD = signal_hat * np.conj(signal_hat) / len_signal

# Create a frequency array for plotting
frequencies = (1 / dt * len_signal) * np.arange(len_signal)

# Let's plot everything now

fig, axs = plt.subplots(2, 1)
plt.sca(axs[0])
plt.plot(t, clean_signal, "r", LineWidth = 2,  label = "Clean Signal")
plt.plot(t, noisy_signal, "b",LineWidth = 1, label = "Noisy Signal")
plt.legend()

plt.sca(axs[1])
plt.plot(frequencies, PSD, "r", LineWidth = 2,  label = "PSD")
plt.legend()
plt.show()

## Now that we have the PSD, we can see some frequencies have the most power whilst the
## most of the other frequencies have very low power. So, based on this power we will
## filter out the other frequencies

# PSD threshold: below this power we will not pickup any signal

psd_threshold = 25  # based on current PSD plot

indices = PSD > psd_threshold

# Frequencies which have power above threshold will stay. Others will be zero
PSD_clean = PSD * indices
signal_hat = indices * signal_hat

# Taking inverse fourier transform now
signal_filtered = np.fft.ifft(signal_hat)

# Let's Visualise these signals
#plt.plot(t, clean_signal, "r", label = "Clean Signal")
plt.plot(t, noisy_signal, "b",  label = "Noisy Signal")
plt.plot(t, signal_filtered, "c", label  = "Filtered Signal")
plt.legend()
plt.show()