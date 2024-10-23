import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(data):
    plt.plot(data['Date'], data['Value'])

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Datetime [MM-DD-HH]')
    plt.ylabel('W/m²')
    plt.tight_layout()
    plt.savefig(f'../artifacts/date_to_value_plot_FOURIER.pdf', format='pdf')
    plt.show()


def plot_histogram(data):
    timediff = data['Date'].diff()
    timediff_days = timediff.dt.total_seconds() / 60

    # Plot histogram
    plt.hist(timediff_days, bins=20, edgecolor='black')  # You can adjust the number of bins
    plt.xlabel('Time Difference (Minutes)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences in Days')
    plt.savefig(f'../artifacts/timediff_histo_FOURIER.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../data/Noale_Italy_26_7_dBm.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Value'] = pd.to_numeric(data['Value'])

    plot_data(data)
    plot_histogram(data)