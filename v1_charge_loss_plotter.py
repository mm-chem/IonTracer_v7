import math
import pickle
import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter


def generate_filelist(termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelist = []
    folder = fd.askdirectory(title="Choose top folder")
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(termString):
                filelist.append(os.path.join(root, file))
                print("Loaded: " + filelist[-1])
    return filelist


def gauss(x, A, mu, sigma, offset):
    return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    plot_color = 'dodgerblue'
    fit_color = 'red'

    SMALL_SIZE = 18
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    freqComputedChargeLoss = []

    files = generate_filelist('_freq_computed_charge_loss.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for freq in db:
            freqComputedChargeLoss.append(freq)
        dbfile.close()

    dropsChargeChange = []

    files = generate_filelist('_amp_computed_charge_loss.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for amp in db:
            dropsChargeChange.append(amp)
        dbfile.close()

    # Plot frequency computed charge loss alongside directly computed charge loss
    fig, ax = plt.subplots(layout='tight', figsize=(7, 5))
    hist_out = ax.hist(dropsChargeChange, 100, range=[-15, 10], color=plot_color)

    bins = hist_out[1][0:-1]
    counts = hist_out[0]

    A_constraints = [-np.inf, np.inf]
    mu_constraints = [-np.inf, np.inf]
    sigma_constraints = [0, np.inf]
    offset_constraints = [0, 1]
    lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
    upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
    param, param_cov = curve_fit(gauss, bins, np.array(counts), bounds=(lower_bounds, upper_bounds))

    peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])

    ax.plot(bins, peak_contrib_to_slice, linewidth=3, linestyle="solid", color=fit_color)
    text_string = f'{param[1]:.2f}'
    ax.text(param[1] + 5, param[0] - 1.5, text_string, fontsize=16)
    print("Amplitude Computed Peak Center: ", str(param[1]))

    ax.set_title("")
    ax.set_xlabel('Charge', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    ax.set_xticks([-15, -10, -5, 0, 5, 10])
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    save_path = "/Users/mmcpartlan/Desktop/"
    plt.savefig(save_path + 'exported_amp_computed.png', bbox_inches='tight', dpi=300.0,
                transparent='true')

    fig, ax = plt.subplots(layout='tight', figsize=(7, 5))
    hist_out = ax.hist(freqComputedChargeLoss, 100, range=[-4, 0], color=plot_color)
    ax.set_title("")
    ax.set_xlabel('Charge', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    ax.set_xticks([-4, -3, -2, -1, 0])
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    plt.axvline(-1, color=fit_color, linestyle='solid', linewidth=3)
    plt.axvline(-2, color=fit_color, linestyle='solid', linewidth=3)

    # Comment out for SINGLE gaussian fitting
    bin_spacing = abs(bins[1] - bins[0])
    peak_indices, properties = sig.find_peaks(counts, width=2, distance=100 * 0.05, prominence=max(counts) * 0.25)

    save_path = "/Users/mmcpartlan/Desktop/"
    plt.savefig(save_path + 'exported_f_computed.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')
