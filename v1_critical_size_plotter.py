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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


if __name__ == "__main__":
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

    fit_to_gaussian = 0

    z2_n = []

    files = generate_filelist('_z2_n.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        for ratio in db:
            z2_n.append(ratio)
        dbfile.close()

    fig, ax = plt.subplots(layout='tight', figsize=(7, 5))
    hist_out = ax.hist(z2_n, 100, range=[0, 0.5], color='black')

    bins = hist_out[1][0:-1]
    counts = hist_out[0]

    A_constraints = [-np.inf, np.inf]
    mu_constraints = [-np.inf, np.inf]
    sigma_constraints = [0, np.inf]
    offset_constraints = [0, 100]
    lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
    upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]
    try:
        # param, param_cov = curve_fit(gauss, bins, np.array(counts), p0=[1000, 0.2, 0.5, 0])
        param, param_cov = curve_fit(gauss, bins, np.array(counts))
    except Exception:
        print("Curve fit failed due to retardation.")

    peak_contrib_to_slice = gauss(bins, param[0], param[1], param[2], param[3])

    if fit_to_gaussian:
        plt.plot(bins, peak_contrib_to_slice, linestyle="solid", color='red', linewidth=3)
        print("z^2/n center", str(param[1]))
    else:
        half_max = (max(counts) - min(counts))/2
        find_nearest(bins, half_max)


    ax.set_title("")
    ax.set_xlabel('Critical Value', fontsize=24, weight='bold')
    ax.set_ylabel('Counts', fontsize=24, weight='bold')
    ax.set_xticks([0, 0.2, 0.4])
    ax.tick_params(axis='x', which='major', labelsize=26, width=4, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=4, length=8)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=3, length=4)
    ax.tick_params(axis='y', which='minor', width=3, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    plt.axvline(0.375, color='red', linestyle='solid', linewidth=3)

    save_path = "/Users/mmcpartlan/Desktop/"
    plt.savefig(save_path + 'exported_z2_n.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')
