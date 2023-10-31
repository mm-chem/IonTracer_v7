import pickle
import numpy as np
from scipy import signal as sig
from scipy import stats
import matplotlib.pyplot as plt
import os
from tkinter import filedialog as fd


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

    file_order = [0, 2, 1, 3]
    slope_max_small = 0.06
    slope_max_large = 15
    smooth_window = 10
    smooth_polyorder = 2

    files = generate_filelist('.pickle')
    slope_distributions = []
    for file in files:
        dbfile = open(file, 'rb')
        db = pickle.load(dbfile)
        slope_distributions.append(db)
        dbfile.close()

    # DEFINE ax to be the small and ax2 to be the large numbers
    colors = ['blue', 'green', 'orange', 'purple']
    color_counter = 0
    f, (ax, ax2) = plt.subplots(1, 2, facecolor='w', figsize=(14, 7))
    smoothed_output = False
    bins = 50

    small_plot_dists = []
    small_plot_dists_truncated = []
    small_weights = []
    large_plot_dists = []
    large_plot_dists_truncated = []
    large_weights = []

    sorted_slope_distributions = []
    for n in range(len(slope_distributions)):
        sorted_slope_distributions.append(slope_distributions[file_order[n]])

    for dist in sorted_slope_distributions:
        mode, counts = stats.mode(dist)
        print(mode, counts)
        if mode < 1:
            counts_small, bins_small = np.histogram(dist, bins=bins, range=[0, slope_max_small])
            small_plot_dists_truncated.append(bins_small[:-1])
            small_weights.append(counts_small/(np.max(counts_small)))
            small_plot_dists.append(bins_small)
        else:
            counts_big, bins_big = np.histogram(dist, bins=bins, range=[1, slope_max_large])
            large_plot_dists_truncated.append(bins_big[:-1])
            large_plot_dists.append(bins_big)
            large_weights.append(counts_big/(np.max(counts_big)))

    ax.hist(small_plot_dists_truncated, small_plot_dists[0], weights=small_weights, color=['orange', 'dodgerblue', 'black'],
            histtype="stepfilled")
    labels = ["0", "0.02", "0.04", "0.06"]
    x = [0.000, 0.02, 0.04, 0.06]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax2.hist(large_plot_dists_truncated, large_plot_dists[0], weights=large_weights, color='teal')
    labels = ["5", "10", "15"]
    x = [5, 10, 15]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    ax.set_xlim(0, slope_max_small)
    ax2.set_xlim(1, slope_max_large)

    # hide the spines between ax and ax2
    ax.yaxis.tick_left()
    ax.tick_params(axis='x', which='major', labelsize=26, width=3, length=8)
    ax.tick_params(axis='y', which='major', labelsize=26, width=3, length=8)
    ax2.tick_params(axis='x', which='major', labelsize=26, width=3, length=8)
    # ax2.tick_params(axis='y', which='major', )
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', width=2, length=4)
    ax.tick_params(axis='y', which='minor', width=2, length=4)
    ax2.minorticks_on()
    ax2.tick_params(axis='x', which='minor', width=2, length=4)

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax2.spines['bottom'].set_linewidth(3)

    ax2.axes.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    save_path = "/Users/mmcpartlan/Desktop/"
    plt.savefig(save_path + 'exported_stacked_rel_hydration.png', bbox_inches='tight', dpi=300.0, pad_inches=0.5,
                transparent='true')


    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    # ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    #
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    # ax2.plot((-d, +d), (-d, +d), **kwargs)

    # plt.suptitle("Cluster Hydration Distribution")
    # plt.xlabel('Slope: Drift (Hz) per STFT Step (5ms)')
    # plt.ylabel('Counts')
    # plt.tight_layout(pad=0)
    # plt.show()
