import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog as fd
from scipy import signal as sig
import v6_IonClass as ionTracer
import warnings

warnings.filterwarnings('ignore')


def choose_save_folder():
    win = tk.Tk()
    win.focus_force()
    win.withdraw()
    folder = fd.askdirectory(title="Choose save folder")
    return folder


def choose_top_folders(termString=None):
    if termString is not None:
        folders = []
        win = tk.Tk()
        win.focus_force()
        win.withdraw()
        folder = fd.askdirectory(title="Choose top folder")
        walkResults = os.walk(folder)
        for root, dirs, files in walkResults:
            for dir in dirs:
                if dir.endswith(termString):
                    folders.append(os.path.join(root, dir))
        return folders
    else:
        win = tk.Tk()
        win.focus_force()
        win.withdraw()
        folder = fd.askdirectory(title="Choose top folder")
        return [folder]


def generate_filelist(folders, termString):
    # NOTE: folders variable must be a list, even if it is a list of one
    filelists = []
    for folder in folders:
        filelist = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(termString):
                    filelist.append(os.path.join(root, file))
        filelists.append(filelist)
    return filelists


# WARNING: EXPORTING ZXX FILES FOR BACKGROUND GENERATION IS SLOW AND TAKES A TON OF SPACE
export_Zxx_files = False

# Parameters for STFT, indexed for each directory to be analyzed
voltage_scale = 0.2  # Input range for 16-bit digitizer card
fs = 1000000  # Sampling frequency in Hz
segment_length = 25  # Segment length in ms
step_length = 5  # How many ms to advance the window
zerofill = 250  # How many ms worth of data to zerofill

low_freq = 8000  # Lower bound on region of interest
high_freq = 40000  # Upper bound on region of interest
min_trace_charge = 40  # Minimum amplitude to trace (default 25)
min_trace_length = 5
time_correlation_tolerance = 25  # In time bin_count on the x-axis. + or - direction
freq_correlation_tolerance = 1000  # In Hz on the y-axis. How close do trace fragments need to be - direction ONLY
max_positive_slope = 50  # In Hz
max_negative_slope = -200  # In Hz, how much can points in ONE FRAGMENT differ

harm_pairing_threshold = 75  # In Hz (maximum deviation for a trace to be considered a harmonic)
check_start = 100  # Start check at x ms
check_length = 100  # ms length of checking transform
magnitude_to_charge_factor = 682500  # Magic number to convert amplitude to approximate charge
pts_per_seg = fs * (segment_length / 1000)  # Points per FT segment
zerofill_pts = fs * (zerofill / 1000)  # Zerofill points added per segment
pts_per_step = fs * (step_length / 1000)  # Points to move the window ahead by
f_reso = fs / zerofill_pts  # Zero-filled resolution, not true resolution
min_trace_spacing = 4 * f_reso  # Hz (allowed between peaks in checkFFT and STFT without culling the smaller one)
freq_correlation_tolerance = freq_correlation_tolerance / f_reso  # Converted to bins now
pts_per_check = fs * (check_length / 1000)  # Points in the spot check section
f_reso_check = fs / pts_per_check
low_freq_pts = int(low_freq / f_reso)
high_freq_pts = int(high_freq / f_reso)
harm_low_freq = low_freq * 2
harm_high_freq = high_freq * 2
harm_low_freq_pts = int(harm_low_freq / f_reso)
harm_high_freq_pts = int(harm_high_freq / f_reso)
culling_dist_samps = min_trace_spacing / f_reso


def one_file(file, save_dir):
    file = Path(file)
    save_dir = Path(save_dir)
    folder = file.parts[-3].split(".data")[0] + ".traces"
    trace_save_directory = save_dir / folder
    print(file)

    data_bits = np.fromfile(file, dtype=np.uint16)  # Import file
    data_volts = ((voltage_scale / 66536) * 2 * data_bits) - voltage_scale  # Convert from uint16 to volts
    data_volts_nopulse = data_volts[step_length * int(fs / 1000) + 702:]  # Cut out trap pulse at beginning
    max_time = (len(data_volts_nopulse) / fs) * 1000  # Signal length after cutting out pulse

    ################################################################################################################
    # Calculate the check section to estimate ion presence
    ################################################################################################################
    check_part = sig.detrend(data_volts_nopulse[int(fs * check_start / 1000):
                                                int(fs * (check_start + check_length) / 1000)], type='constant')

    check_spec = np.fft.fft(check_part, n=int(pts_per_check))
    check_magnitude = np.abs(check_spec) * magnitude_to_charge_factor
    # Only look at freqs of interest. Apply 1/N scaling to amplitude (not in DFT equation)
    magnitude_slice = check_magnitude[int(low_freq / f_reso_check):int(high_freq / f_reso_check)] / pts_per_check

    # Pick peaks out, then analyze the whole file if there is anything there
    sep_distance = min_trace_spacing / f_reso_check
    if sep_distance < 1:
        sep_distance = 1

    peak_indexes, check_prop = sig.find_peaks(magnitude_slice, height=min_trace_charge,
                                              distance=sep_distance)
    check_peaks = peak_indexes * f_reso_check + low_freq
    print("Resolution: " + str(f_reso))

    if check_peaks.size:

        ################################################################################################################
        # Calculate STFT and trace the full file....
        ################################################################################################################
        f, t, Zxx = sig.stft(data_volts_nopulse, window='hamming', detrend='constant',
                             fs=fs, nperseg=int(pts_per_seg), nfft=int(zerofill_pts),
                             noverlap=int(pts_per_seg - pts_per_step))

        STFT = np.abs(Zxx) * magnitude_to_charge_factor
        STFT_phase = np.angle(Zxx, deg=True)
        STFT_cut = STFT[low_freq_pts:high_freq_pts]
        STFT_cut_phase = STFT_phase[low_freq_pts:high_freq_pts]
        pick_time_full = np.arange(0, max_time / step_length)
        pick_time_pts = pick_time_full[int(0.6 + (segment_length / 2) / step_length):-int(
            0.6 + (segment_length / 2) / step_length)]  # cut one half seg +1 from each edge

        starting_t = pick_time_pts[0]
        traces = ionTracer.IonField(low_freq, starting_t, min_trace_length,
                                    time_correlation_tolerance, freq_correlation_tolerance)

        ################################################################################################################
        # Ion Tracing / Post-Processing
        ################################################################################################################
        t_range_offset = pick_time_pts[0]
        for t in pick_time_pts:
            print("Calculating slice: ", t)
            current_slice = STFT_cut[:, int(t)]
            current_slice_phase = STFT_cut_phase[:, int(t)]
            traces.ion_hunter(current_slice, current_slice_phase, int(t), min_trace_charge, min_trace_spacing, f_reso,
                              max_positive_slope, max_negative_slope)

        traces.post_process_traces()
        traces.pair_harmonics(harm_pairing_threshold)

        ################################################################################################################
        # Trace Visualization Calls (for debugging)
        ################################################################################################################
        traces.plot_paired_traces(STFT_cut, segment_length, step_length)
        tracesHeader = str(low_freq) + "|" + str(f_reso) + "|" + str(t_range_offset)
        traces.write_ions_to_files(trace_save_directory, file, tracesHeader, export_Zxx_files)


if __name__ == "__main__":
    folders = choose_top_folders()
    # file_ending = ".B.txt"  # For SPAMM 2
    file_ending = ".txt"  # For SPAMM 3
    filelist = generate_filelist(folders, file_ending)
    save_dir = choose_save_folder()
    print(save_dir)
    for file in filelist[0]:
        if file.endswith(file_ending):
            one_file(file, save_dir)
