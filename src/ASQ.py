import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from typing import Dict


def ceil_to_power_of_two(num):
    return 2 ** math.ceil(math.log(num, 2))


def build_norm_hist(signal, thresholds):
    # ----- STEP 1: CREATING THE NORMALIZED HISTOGRAM -> DOTS CONNECTED
    # Use Counter to count occurrences of each value
    h = Counter(signal)

    # Find the closest bigger power of 2 and use it to determine the set of 'k'
    k_range = {x + 1 for x in range(ceil_to_power_of_two(signal.max()))}
    for x in k_range.difference(h.keys()):
        h[x] = 0

    n = len(signal)
    # Sort the keys (unique values in data) and corresponding counts
    sorted_keys = sorted(h.keys())
    sorted_values = [h[k] / n for k in sorted_keys]

    # Plot the histogram as dots
    plt.scatter(sorted_keys, sorted_values)
    plt.plot(sorted_keys, sorted_values, color='red', linestyle='-', label='Normalized histogram')

    # Highlight sections of the histogram based on thresholds
    colors = ['blue', 'green', 'yellow', 'orange', 'purple', 'red', 'blue',
              'green', 'yellow', 'orange', 'purple', 'red', 'blue']  # For different sections
    colors = colors[:len(thresholds)-1]  # Adjust colors based on number of thresholds
    for i in range(len(thresholds) - 1):
        # Define the current range
        left, right = thresholds[i], thresholds[i+1]
        plt.fill_between(
            sorted_keys, sorted_values, where=((np.array(sorted_keys) >= left) & (np.array(sorted_keys) <= right)),
            color=colors[i % len(colors)], alpha=0.3, label=f'q{i+1}'
        )

    # Set x and y scales to start from 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig('norm_distribution.pdf', format='pdf', bbox_inches='tight')
    # Show the plot
    plt.show()

    return dict(zip(sorted_keys, sorted_values))


def determine_thresholds(h: Dict[int, int], s: int, p: float):
    keys = list(h.keys())
    thresholds = [keys[0]]  # Start from the first value

    lp = rp = 1  # Initialize left and right pointers
    agg_sum = 0
    while rp <= len(keys) and len(thresholds) < s:
        agg_sum += h[rp]
        if agg_sum >= p:
            lp = rp
            agg_sum = h[rp]
            thresholds.append(lp)
        rp += 1

    thresholds.append(keys[-1])  # Always end with the max value as a threshold
    return thresholds


def calculate_output_levels(h):
    qs = []
    threshold_pointer = 1
    numerator = denominator = 0
    for l in h.keys():
        numerator += l*h[l]
        denominator += h[l]
        if l == thresholds[threshold_pointer]:
            qs.append(numerator / denominator)
            numerator = l*h[l]
            denominator = h[l]
            threshold_pointer += 1

    # Rounding the output levels
    qs[0] = round(qs[0])
    for i in range(1, len(qs)):
        if qs[i-1] == round(qs[i]):
            qs[i] = qs[i-1] + 1
        else:
            qs[i] = round(qs[i])

    return qs


def non_uniform_quantization_scale(thresholds, output_levels):
    quan_scale = defaultdict(int)

    # Assign the quantized levels to each threshold range
    for i in range(1, len(thresholds)):
        for x in range(thresholds[i - 1], thresholds[i] + 1):
            if x == thresholds[i]:
                x -= 0.00001
            quan_scale[x] = output_levels[i - 1]

    # Plot the non-uniform quantization scale
    plt.plot(quan_scale.keys(), quan_scale.values(), label='Non-uniform quantization scale')

    # Adding opaque grey lines for projection on x-axis and y-axis
    plt.vlines(x=range(thresholds[0], thresholds[-1]), ymin=0, ymax=output_levels[-1],
               linestyles=(0, (5, 5)), colors='grey', alpha=0.4)
    plt.hlines(y=output_levels, xmin=0, xmax=thresholds[-1],
               linestyles=(0, (5, 5)), colors='grey', alpha=0.4)
    plt.xlim(left=thresholds[0])
    plt.ylim(bottom=0)
    plt.xlabel('Input Levels')
    plt.ylabel('Quantized Levels')
    plt.title('Non-uniform Quantization Scale with Projected Edges')
    plt.legend()
    plt.savefig('NQU-scale.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    return quan_scale


def compute_errors(signal, z_hat):
    mse = np.square(np.subtract(signal, z_hat)).mean()  # mean squared error
    nmse = (mse * len(signal) / sum(np.square(z_hat))) * 100  # normalized mean squared error
    snr = 10 * math.log((sum(np.square(signal))/len(signal))/mse, 10)  # signal-to-noise ratio
    psnr = 10 * math.log((sum(np.square(np.full(len(signal), max(z_hat)))) / len(signal)) / mse, 10)  # peak signal-to-noise ratio
    compress_ratio = b / c

    error_values = {
        'mse': mse,
        'nmse': nmse,
        'snr': snr,
        'psnr': psnr,
        'K': compress_ratio
    }

    return error_values


if __name__ == '__main__':
    # Example data
    s = 4  # Number of sections (thresholds)
    a = input('Enter your signal data separated by space ->')
    if not a:
        a = '5 5 6 7 6 5 4 3 2 2'
    # input from the exercise -> 5 4 3 6 7 5 4 2 1 2
    data = np.array([*map(int, a.split())])

    # STEP 1: Build the histogram
    h = build_norm_hist(data, thresholds=[])

    # STEP 2: Compute c and p
    c = math.log(s, 2)
    b = math.log(max(h.keys()), 2)
    p = 1 / s  # Proportion threshold

    # STEP 3: Determine thresholds
    thresholds = determine_thresholds(h, s, p)
    # STEP 4: Plot the divided sections
    build_norm_hist(data, thresholds)
    # STEP 5: Calculating the output levels qr
    output_levels = calculate_output_levels(h)  # qs
    # STEP 6: Non-uniform quantization scale
    quan_scale = non_uniform_quantization_scale(thresholds, output_levels)
    # STEP 7: Calculating new values
    z_hat = np.array([quan_scale[i] for i in data])
    # STEP 8: Calculating the quality measures
    error_values = compute_errors(data, z_hat)

    for key, value in error_values.items():
        print(f'{key} = {value}')