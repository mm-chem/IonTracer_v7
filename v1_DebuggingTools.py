from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog as fd
import v1_DropShot as DropShot
import v6_IonClass as IonTools
import v6_STFT_analysis as STFT


class DebuggingTools:
    def __init__(self, fs):
        self.fs = fs
        self.STFT_signals_Zxx = []
        self.STFT_signals_f = []
        self.STFT_signals_t = []
        self.FFT_signals = []
        self.FFT_freqs = []
        self.folders = STFT.choose_top_folders()
        self.files = STFT.generate_filelist(self.folders, ".bin")

        self.segment_length = 25  # Segment length in ms
        self.step_length = 5  # How many ms to advance the window
        self.zerofill = 100  # How much zerofill time to add (in ms)
        self.ptsperseg = self.fs * (self.segment_length / 1000)  # Points per FT segment
        self.zerofillpts = self.fs * (self.zerofill / 1000)  # Zerofill points added per segment
        self.ptsperstep = self.fs * (self.step_length / 1000)  # Points to move the window ahead by
        self.f_reso = self.fs / self.zerofillpts  # Zero-filled resolution, not true resolution

        self.signals = []
        for file in self.files:
            new_file = np.fromfile(file[0], dtype=np.float)
            self.signals.append(new_file)
            f, t, Zxx = self.compute_STFT(new_file)
            self.STFT_signals_Zxx.append(Zxx)
            self.STFT_signals_f.append(f)
            self.STFT_signals_t.append(t)

            computed_fft, freqs = self.compute_FFT(new_file)
            self.FFT_signals.append(computed_fft)
            self.FFT_freqs.append(freqs)

        self.points_in_transient = len(self.signals[0])
        self.transient_time = self.fs * self.points_in_transient

        init_complete = 1

    def compute_STFT(self, data_array):
        f, t, Zxx = sig.stft(data_array, window='boxcar', detrend='constant',
                             fs=self.fs, nperseg=int(self.ptsperseg), nfft=int(self.zerofillpts),
                             noverlap=int(self.ptsperseg - self.ptsperstep))
        Zxx = np.abs(Zxx)
        self.plot_STFT(Zxx, 15000, 45000)
        plt.plot(f, Zxx[:, 3], color='red')
        return f, t, Zxx

    def compute_FFT(self, data_array):
        N = len(data_array)  # Number of sample points
        T = 1.0 / self.fs  # sample spacing
        computed_fft = fft(data_array)
        freqs = fftfreq(N, T)[:N // 2]
        self.plot_FFT(freqs, computed_fft, N)
        return computed_fft, freqs

    def plot_FFT(self, freqs, computed_fft, N):
        plt.plot(freqs, 2.0 / N * np.abs(computed_fft[0:N // 2]))
        plt.grid()
        plt.show()

    def plot_STFT(self, Zxx, lowfreq, highfreq):
        lowfreqpts = int(lowfreq / self.f_reso)
        highfreqpts = int(highfreq / self.f_reso)
        STFT_cut = np.abs(Zxx[lowfreqpts:highfreqpts])
        plt.pcolormesh(STFT_cut[:, int(0.6 + (self.segment_length / 2) / self.step_length):-int(
            0.6 + (self.segment_length / 2) / self.step_length)])
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    debugger = DebuggingTools(20000000)
