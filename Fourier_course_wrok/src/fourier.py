import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import detrend


def load_data():
    data = pd.read_csv('../data/Noale_Italy_26_7_dBm.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Value'] = pd.to_numeric(data['Value'])

    return data


def plot_data(data):
    plt.plot(data['Date'], data['Value'])
    plt.title('Data from Noale Italy 26 7')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Datetime [MM-DD HH]')
    plt.ylabel('dBm')
    plt.tight_layout()
    plt.savefig(f'../artifacts/date_to_value_plot_FOURIER.png', format='png')
    plt.show()


def plot_histogram(data):
    timediff = data['Date'].diff()[1:]
    timediff_minutes = timediff.dt.total_seconds() / 60

    bin_count = 8
    # Plot histogram

    plt.xlabel('Time Difference (Minutes)')
    plt.xlim(min(timediff_minutes), max(timediff_minutes))
    # Setting better alignment for x ticks
    plt.xticks(np.arange(min(timediff_minutes), max(timediff_minutes),
                         (max(timediff_minutes) - min(timediff_minutes)) / bin_count))
    hist_info = plt.hist(timediff_minutes, bins=bin_count, edgecolor='black')  # You can adjust the number of bins
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences in Days')
    plt.savefig(f'../artifacts/timediff_histo_FOURIER.png', format='png')
    plt.show()

    return hist_info


def plot_fft(data):
    data = np.asarray(data)
    data = detrend(data)
    N = len(data)
    # Perform FFT
    yf = fft(data)

    # Generate frequency bins (with appropriate sampling rate if applicable)
    sampling_rate = avg_sample_time / 60  # 5 minutes per sample
    xf = fftfreq(int(N), 1 / sampling_rate)  # Frequency bins

    # Keep only the positive half of the spectrum
    xf = xf[:int(N // 2)]
    yf = yf[:int(N // 2)]

    # Plot FFT
    plt.figure(figsize=(12, 6))
    plt.plot(xf * 1000, 2.0 / N * np.abs(yf))  # Convert xf to mHz
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency (mHz)")
    plt.ylabel("Amplitude")
    # plt.yscale('log')
    plt.xticks([xf[i] * 1000 for i in range(len(xf)) if i % 3 == 0], rotation=90)  # Adjust tick step
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'../artifacts/fft_plot.png', format='png')
    plt.show()
    dominant_freqs = xf[np.argsort(np.abs(xf) ** 2)[-15:]]  # Top 5 frequencies
    print("Dominant Frequencies:", dominant_freqs)


def normalize(data):
    x_max = np.max(data)
    x_min = np.min(data)
    normalized = ((data - x_min) / (x_max - x_min)) * 255
    return normalized


if __name__ == '__main__':
    signal = load_data()
    plot_data(signal)

    plot_data(signal)
    plt.imshow(mpimg.imread('../data/solar-activity.png'))
    plt.axis('off')  # Turn off axis labels
    plt.title("Solar activity data from www.spaceweatherlive.com")
    plt.show()

    hist_info = plot_histogram(signal)

    frequencies = hist_info[0]
    bins = hist_info[1]
    total_samples = sum(frequencies)
    avg_sample_time = sum(
        [frequencies[i] * ((bins[i] + bins[i + 1]) * 0.5) for i in range(len(frequencies))]) / total_samples
    missing_samples = 24 * 60 / 5 - total_samples - 1

    print({"minutes in a day": 24 * 60, "total": total_samples, "missing": missing_samples,
           "average sample time [min]": avg_sample_time})

    plot_fft(signal['Value'])

    normalized_signal = normalize(signal['Value'])

    # Plot the normalized signal
    plt.figure(figsize=(12, 6))
    plt.plot(normalized_signal)
    plt.title("Normalized Signal (0 to 255)")
    plt.xlabel("Time")
    plt.ylabel("Normalized Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../artifacts/normalized_signal.png")
    plt.show()