import math
import numpy as np
import copy as cpy
import scipy.stats as sp
from scipy import signal as sig
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import traceback


class Ion:
    def __init__(self):
        self.center = None
        self.magnitude = None
        self.phase = None
        self.index = None
        self.trace = []
        self.magnitude_trace = []
        self.phase_trace = []
        self.trace_indices = []
        self.interpolated_indices = []

        self.harmonic_trace = []
        self.harmonic_magnitude = []
        self.harmonic_phase = []

        self.paired = 0
        self.linfitEquation = None
        self.fit_coeff_of_determination = None
        self.interpolation_counter = 0
        self.avg_deviation = None
        self.avg_freq_bin = None
        self.linfit_slope = None
        self.linfit_intercept = None

    def reset_pairings(self):
        self.paired = 0

    def return_trace_at_time(self, desired_time):
        return self.trace[self.trace_indices.index(desired_time)]

    def return_magnitude_trace_at_time(self, desired_time):
        return self.magnitude_trace[self.trace_indices.index(desired_time)]

    def return_harm_trace_at_time(self, desired_time):
        return self.harmonic_trace[self.trace_indices.index(desired_time)]

    def return_harm_magnitude_at_time(self, desired_time):
        return self.harmonic_magnitude[self.trace_indices.index(desired_time)]

    def return_harm_phase_at_time(self, desired_time):
        return self.harmonic_phase[self.trace_indices.index(desired_time)]

    def return_phase_trace_at_time(self, desired_time):
        return self.phase_trace[self.trace_indices.index(desired_time)]

    def delete_step(self, index):
        self.trace.pop(index)
        self.magnitude_trace.pop(index)
        self.phase_trace.pop(index)
        self.trace_indices.pop(index)

    def merge_as_step(self, step):
        self.trace.append(self.center)
        self.magnitude_trace.append(self.magnitude)
        self.phase_trace.append(self.phase)
        self.trace_indices.append(self.index)

        self.index = step.index
        self.magnitude = step.magnitude
        self.phase = step.phase
        self.center = step.center
        self.update_lin_fit()

        self.paired = 1
        step.paired = 1

    def merge_as_trace(self, trace):
        trace_start_index = trace.trace_indices[0]
        trace_end_index = trace.trace_indices[-1]
        if trace_start_index > self.trace_indices[-1]:
            # If new trace starts AFTER current trace
            # print("Merging traces (append)")
            for step in range(len(trace.trace)):
                self.trace.append(trace.trace[step])
                self.magnitude_trace.append(trace.magnitude_trace[step])
                self.phase_trace.append(trace.phase_trace[step])
                self.trace_indices.append(trace.trace_indices[step])
        elif trace_end_index < self.trace_indices[0]:
            # If new trace starts BEFORE current trace
            # print("Merging traces (prepend)")
            for step in reversed(range(len(trace.trace))):
                # Insert them in reverse order since we are always writing to the front of the list
                # NOTE: Prepending to lists is slow (consider double-linked list "deque")
                self.trace.insert(0, trace.trace[step])
                self.magnitude_trace.insert(0, trace.magnitude_trace[step])
                self.phase_trace.insert(0, trace.phase_trace[step])
                self.trace_indices.insert(0, trace.trace_indices[step])
        else:
            # Insert the trace into the 'middle' of the existing trace. Overwrite later datapoints
            # Find the value closest (low side) to trace_start_index
            counter = 0
            for element in range(len(self.trace_indices)):
                if self.trace_indices[element] < trace_start_index:
                    counter = counter + 1
            # print("Merging traces (insertion)")
            for step in reversed(range(len(trace.trace))):
                # Insert them in reverse order since we are always writing to the front of the list
                # NOTE: Prepending to lists is slow (consider double-linked list "deque")
                self.trace.insert(counter, trace.trace[step])
                self.magnitude_trace.insert(counter, trace.magnitude_trace[step])
                self.phase_trace.insert(counter, trace.phase_trace[step])
                self.trace_indices.insert(counter, trace.trace_indices[step])
            for element in range(len(self.trace_indices) - 1):
                if self.trace_indices[element] > self.trace_indices[element + 1]:
                    # Traces are out of order / duplicate traces were introduced in merge
                    excise_index = element
            try:
                for i in range(len(self.trace_indices), excise_index, -1):
                    self.delete_step(i - 1)
            except Exception:
                print("Excise-index not defined. Merged null trace?")

        self.update_lin_fit()

    def merge_with_self(self):
        self.trace.append(self.center)
        self.magnitude_trace.append(self.magnitude)
        self.phase_trace.append(self.phase)
        self.trace_indices.append(self.index)
        self.update_lin_fit()
        self.paired = 1

    def update_lin_fit(self):
        # This option uses the ENTIRE trace to calculate a linfit
        if len(self.trace) > 1:
            fit = sp.linregress(self.trace_indices, self.trace)
            self.linfitEquation = np.poly1d([fit.slope, fit.intercept])
            self.linfit_slope = fit.slope
            self.linfit_intercept = fit.intercept
            self.fit_coeff_of_determination = fit.rvalue ** 2

            self.avg_deviation = np.average(np.abs(np.subtract(self.trace, self.linfitEquation(self.trace_indices))))
            self.avg_freq_bin = np.average(self.trace)
            # This option uses the last 5 points to calculate a linfit
        # if len(self.trace) > 5:
        #     fit = np.polyfit(self.trace_indices[-6:-1], self.trace[-6:-1], 1)
        #     self.linfitEquation = np.poly1d(fit)

    def merge_as_harmonic(self, harmonic_trace):
        # Build overlap vector to check that traces share time-domain indices
        common_trace_indices = np.intersect1d(self.trace_indices, harmonic_trace.trace_indices)
        common_trace_indices = [int(i) for i in common_trace_indices]

        # Shorten fundamental ion to include only common indices
        self.trace = [self.return_trace_at_time(i) for i in common_trace_indices]
        self.magnitude_trace = [self.return_magnitude_trace_at_time(i) for i in common_trace_indices]
        self.phase_trace = [self.return_phase_trace_at_time(i) for i in common_trace_indices]
        self.trace_indices = common_trace_indices

        # Shorten harmonic ion to include only common indices
        harmonic_trace.trace = [harmonic_trace.return_trace_at_time(i) for i in common_trace_indices]
        harmonic_trace.magnitude_trace = [harmonic_trace.return_magnitude_trace_at_time(i) for i in
                                          common_trace_indices]
        harmonic_trace.phase_trace = [harmonic_trace.return_phase_trace_at_time(i) for i in common_trace_indices]
        harmonic_trace.trace_indices = common_trace_indices

        # Merge the two ions together
        self.harmonic_trace = harmonic_trace.trace
        self.harmonic_magnitude = harmonic_trace.magnitude_trace
        self.harmonic_phase = harmonic_trace.phase_trace

        return len(common_trace_indices)

    def count_drops(self, drop_height, separation):
        derivative = np.diff(self.trace)
        peak_indices, properties = sig.find_peaks(np.abs(derivative), height=drop_height, distance=separation)
        peak_indices = peak_indices.tolist()
        return len(peak_indices)


class IonField:
    def __init__(self, f_offset, t_offset, min_trace_length, time_correlation_tolerance, freq_correlation_tolerance):
        self.f_range_offset = f_offset
        self.t_range_offset = t_offset
        self.time_correlation_tolerance = time_correlation_tolerance
        self.freq_correlation_tolerance = freq_correlation_tolerance
        self.resolution = None
        self.min_trace_length = min_trace_length
        self.ions = []
        self.paired_ions = []
        self.dead_ions = []
        self.difference_matrix = []
        self.sliceRecord = []
        self.phaseSliceRecord = []
        self.mu_cache = []
        self.A_cache = []
        self.phase_cache = []

    def clear_peak_cache(self):
        self.mu_cache = []
        self.A_cache = []
        self.phase_cache = []

    def round_to_resolution(self, number):
        return self.resolution * np.round(number / self.resolution)

    def reset_ion_pairs(self):
        for ion in self.ions:
            ion.paired = 0

    @staticmethod
    def fit_function(x, A, mu, sigma, offset):
        return A * abs(np.sinc((2 / sigma) * (x - mu))) + offset

    @staticmethod
    def gauss(x, A, mu, sigma, offset):
        return offset + A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def crazy_snake_fit(x, A, k, m, b):
        return A * np.sin(k * x) + m * x + b

    def fitted_peak_picking(self, slice, phaseSlice, height=None, distance=None):
        peak_indices, properties = sig.find_peaks(slice, height=height, distance=distance)
        peak_indices = peak_indices.tolist()
        x = range(len(slice))
        fitting = False

        while len(peak_indices) > 0:
            peak_index = peak_indices[0]
            peak_amp = slice[peak_index]
            peak_indices.pop(0)

            centerline_deviation = 5
            A_deviation = 2

            A_constraints = [peak_amp - A_deviation, peak_amp + A_deviation]
            mu_constraints = [peak_index - centerline_deviation, peak_index + centerline_deviation]
            sigma_constraints = [0, 5]
            offset_constraints = [-20, 20]

            lower_bounds = [A_constraints[0], mu_constraints[0], sigma_constraints[0], offset_constraints[0]]
            upper_bounds = [A_constraints[1], mu_constraints[1], sigma_constraints[1], offset_constraints[1]]

            try:
                if fitting:
                    param, param_cov = curve_fit(self.fit_function, np.array(x), np.array(slice),
                                                 bounds=(lower_bounds, upper_bounds))

                    if param[1] < len(slice):
                        self.A_cache.append(param[0])
                        self.mu_cache.append(int(np.round(param[1])))

                        # peak_contrib_to_slice = self.fit_function(x, param[0], param[1], param[2], param[3])
                        # plt.plot(x, slice)
                        # plt.plot(x, peak_contrib_to_slice)
                        # plt.show()
                else:
                    self.A_cache.append(peak_amp)
                    self.mu_cache.append(peak_index)
                    self.phase_cache.append(phaseSlice[peak_index])
            except Exception:
                print("Error: Sinc fitting failed at " + str(peak_index) + ". Using unfitted peak.")
                self.A_cache.append(peak_amp)
                self.mu_cache.append(peak_index)
                self.phase_cache.append(phaseSlice[peak_index])

    def update_diff_matrix(self, new_ions):
        numberOfRows = len(self.ions)
        numberOfCols = len(new_ions)
        self.difference_matrix = np.empty([numberOfRows, numberOfCols])
        for row in range(numberOfRows):
            for col in range(numberOfCols):
                # Rows correspond to existing ions
                # Cols correspond to newly identified peaks
                newCenter = new_ions[col].center
                oldCenter = self.ions[row].center
                self.difference_matrix[row, col] = newCenter - oldCenter

    def ion_hunter(self, STFT_slice, STFT_slice_phase, STFT_index, min_height, min_separation, resolution, max_positive_slope, max_negative_slope):
        self.resolution = resolution
        self.sliceRecord.append(STFT_slice)
        self.phaseSliceRecord.append(STFT_slice_phase)
        max_interpolated_points = 3
        max_negative_slope = max_negative_slope / self.resolution
        new_ions = []
        self.fitted_peak_picking(STFT_slice, STFT_slice_phase, height=min_height,
                                 distance=min_separation / resolution)

        # Plot results of each peak picking (per STFT slice)
        # plt.plot(STFT_slice)
        # plt.scatter(self.mu_cache, self.A_cache, c='red')
        # plt.show()

        try:
            for peakIndex in range(len(self.mu_cache)):
                newIon = Ion()
                newIon.index = STFT_index
                newIon.center = self.mu_cache[peakIndex]
                newIon.magnitude = self.A_cache[peakIndex]
                newIon.phase = self.phase_cache[peakIndex]
                new_ions.append(newIon)
        except Exception:
            print("Error reading recursive indices.")

        if len(self.ions) != 0 and len(new_ions) != 0:
            self.update_diff_matrix(new_ions)

            # Sort diff matrix rows in order of minimum values... should pair 'easy' ions off first
            col_of_mins = np.argmin(abs(self.difference_matrix), axis=1)
            flattened_mins = np.zeros(len(self.ions))
            for i in range(len(self.ions)):
                flattened_mins[i] = abs(self.difference_matrix[i, col_of_mins[i]])

            row_sorted_indices = np.argsort(flattened_mins)
            self.difference_matrix = self.difference_matrix[row_sorted_indices, :]

            ionsCopy = cpy.deepcopy(self.ions)
            for i in range(len(self.ions)):
                self.ions[i] = ionsCopy[row_sorted_indices[i]]

            # Begin pairing off peaks
            for index in range(len(self.ions)):
                # Update each ion's linfit equation
                self.ions[index].update_lin_fit()
                # Find the closest new peak to existing ion
                min_diff_index = np.argmin(abs(self.difference_matrix[index, :]))

                # Merge peak and ion if:
                drift_allowance = max_positive_slope / self.resolution  # In frequency bin_count
                difference = self.difference_matrix[index, min_diff_index]
                if drift_allowance >= difference >= max_negative_slope:
                    # RECALL: ion pairing is INCLUDED in the merge function (for the ION, not the merged peak)
                    self.ions[index].merge_as_step(new_ions[min_diff_index])
                    new_ions[min_diff_index].paired = 1
                    self.ions[index].interpolation_counter = 0
                    # Arbitrarily large number, should never be picked...
                    self.difference_matrix[:, min_diff_index] = 1000000

                # If an existing ion is NOT paired at this point:
                #           Extrapolate next frequency point based on linfit equation, append that to ion trace
                #           Set paired flag to 1
                #           Count how many segments this occurs for. Reset this counter if pairing is successful
                if self.ions[index].paired == 0 and self.ions[index].interpolation_counter <= max_interpolated_points \
                        and self.ions[index].linfitEquation is not None:
                    try:
                        extrapolated_point = self.ions[index].linfitEquation(self.ions[index].trace_indices[-1] + 1)
                        extrapolated_point = np.round(extrapolated_point)
                        newIon = Ion()
                        newIon.center = extrapolated_point
                        newIon.index = STFT_index
                        newIon.magnitude = STFT_slice[int(np.round(newIon.center))]
                        newIon.phase = 0

                        # Recall: Ion is marked as paired inside the merge function
                        self.ions[index].merge_as_step(newIon)
                        self.ions[index].interpolation_counter = self.ions[index].interpolation_counter + 1
                    except Exception:
                        print("Extrapolation out of bounds.")

            # # For the NEW ions that are (sadly) unpaired for now... give them one free pass
            for index in range(len(new_ions)):
                if new_ions[index].paired == 0:
                    new_ions[index].paired = 1
                    self.ions.append(new_ions[index])

            deleteFlag = 1
            while deleteFlag:
                counter = 0
                deleteFlag = 0
                while counter < len(self.ions):
                    if self.ions[counter].paired == 0:
                        if len(self.ions[counter].trace) >= self.min_trace_length:
                            # Toss ions for garbage collection
                            removed_element = self.ions.pop(counter)
                            self.dead_ions.append(removed_element)
                            deleteFlag = 1
                        else:
                            # Toss ions without garbage collection
                            self.ions.pop(counter)
                            deleteFlag = 1

                    counter = counter + 1
            self.reset_ion_pairs()

            # if divmod(STFT_index, 80)[1] == 0:
            #     self.plot_ion_traces(np.flipud(np.rot90(np.array(self.sliceRecord), k=1)))

        else:
            # Spin up ions traces the first time that a new file is loaded
            self.ions = new_ions
            # for ion in self.ions:
            #     ion.merge_with_self()
            self.difference_matrix = np.empty([len(new_ions), len(new_ions)])

        # Reset the peaks that were fitted and stored in the IonField object
        self.clear_peak_cache()

    def check_for_nan(self):
        for element in self.ions:
            if math.isnan(np.average(element.trace)):
                raise ValueError('Trace average is NaN!!')

            for i in element.trace:
                if math.isnan(i):
                    raise ValueError('NaN value detected in trace!!')

    def ion_sort_key(self, ion):
        return np.average(ion.trace) * self.resolution + self.f_range_offset

    # Sort traces from low to high for easier pairing. Run AFTER post-processing function
    def sort_traces(self):
        # Sorted ions are still in the bin space
        self.ions.sort(key=self.ion_sort_key)

    def pair_harmonics(self, harmonic_pairing_threshold):
        harmonic_traces = []
        method = 1
        enforce_slope_polarity = 1
        self.sort_traces()

        # Method 1: find traces using average trace frequency.
        if method == 1:
            for ion in self.ions:
                ionAvgFreq = np.average(ion.trace) * self.resolution + self.f_range_offset
                ion_polarity = ion.linfit_slope > -0.0001
                ion_drops = ion.count_drops(drop_height=(np.average(ion.trace) / 2000) + 10 / self.resolution,
                                            separation=5)
                counter = 0
                for testIon in self.ions:
                    # This loop structure can only delete one trace per cycle. This variable below keeps track of that
                    index_modified = 0
                    testIonAvgFreq = np.average(testIon.trace) * self.resolution + self.f_range_offset
                    test_ion_polarity = testIon.linfit_slope > -0.0001
                    test_ion_drops = testIon.count_drops(drop_height=(np.average(testIon.trace) / 2000)
                                                                     + 10 / self.resolution, separation=5)
                    if abs(ionAvgFreq - testIonAvgFreq / 2) <= harmonic_pairing_threshold:
                        if enforce_slope_polarity:
                            if test_ion_polarity == ion_polarity:
                                # If we find a harmonic trace, delete it and append to trace as a harmonic
                                removed_element = self.ions.pop(counter)
                                merged_points = ion.merge_as_harmonic(removed_element)
                                harmonic_traces.append(removed_element)
                                self.paired_ions.append(ion)
                                print("Harmonic pair found (method 1 w/ polarity): " + str(
                                    np.average(ion.trace) * self.resolution + self.f_range_offset) + ", "
                                      + str(np.average(
                                    removed_element.trace) * self.resolution + self.f_range_offset) + ", " + str(
                                    ion_polarity)
                                      + " with slope (ion, harm) " + str(ion.linfit_slope) + ", "
                                      + str(removed_element.linfit_slope) + " and " + str(ion_drops)
                                      + " drops" + " (points of overlap: " + str(merged_points) + ")")
                                index_modified = 1
                        else:
                            # If we find a harmonic trace, delete it and append to trace as a harmonic
                            removed_element = self.ions.pop(counter)
                            merged_points = ion.merge_as_harmonic(removed_element)
                            harmonic_traces.append(removed_element)
                            self.paired_ions.append(ion)
                            print("Harmonic pair found (method 1 w/o polarity): " + str(
                                np.average(ion.trace)) + ", " + str(
                                np.average(removed_element.trace)))
                            index_modified = 1


                    # Delete higher order harmonics so they do not get double counted (ex: 3rd and 4th harmonic)
                    for N in range(3, 10):
                        if N - 0.002 < abs(testIonAvgFreq / ionAvgFreq) < N + 0.002 and index_modified != 1:
                            removed_element = self.ions.pop(counter)
                            print("Higher order (" + str(N) + ") found and deleted.")
                            index_modified = 1
                    counter = counter + 1



        # Method 2: find traces using intercept of fit equation
        if method == 2:
            for ion in self.ions:
                freq_intercept = ion.linfit_intercept * self.resolution + self.f_range_offset
                counter = 0
                for testIon in self.ions:
                    test_freq_intercept = testIon.linfit_intercept * self.resolution + self.f_range_offset
                    if abs(freq_intercept - test_freq_intercept / 2) <= harmonic_pairing_threshold:
                        # If we find a harmonic trace, delete it and append to trace as a harmonic
                        removed_element = self.ions.pop(counter)
                        merged_points = ion.merge_as_harmonic(removed_element)
                        harmonic_traces.append(removed_element)
                        self.paired_ions.append(ion)
                        print("Harmonic pair found (method 2): " + str(ion.linfit_intercept) + ", "
                              + str(removed_element.linfit_intercept))
                    counter = counter + 1

        self.ions = self.ions + harmonic_traces
        self.sort_traces()

    def post_process_traces(self):
        self.reset_ion_pairs()  # NOTE: Using .paired property differently than before (1 is bad, 0 is good)
        # self.plot_ion_traces(np.flipud(np.rot90(np.array(self.sliceRecord), k=1)))

        # Append the 'dead' ions to the ion list
        for ion in self.dead_ions:
            self.ions.append(ion)

        # Strip interpolated points off of the end of each trace to make time-correlation easier
        for ion in self.ions:
            for i in range(ion.interpolation_counter):
                ion.delete_step(-i)

            # Delete any ions that are obviously not valid
            deleteFlag = 1
            while deleteFlag:
                counter = 0
                deleteFlag = 0
                while counter < len(self.ions):
                    if len(self.ions[counter].trace) < self.min_trace_length:
                        print("Deleting trace at " + str(
                            np.average(self.ions[counter].trace) * self.resolution + self.f_range_offset)
                              + " : Insufficient length.")
                        self.ions.pop(counter)
                        deleteFlag = 1
                    counter = counter + 1

        # self.plot_ion_traces(np.flipud(np.rot90(np.array(self.sliceRecord), k=1)))
        self.check_for_nan()

        plot_residuals_vs_freq = 0
        if plot_residuals_vs_freq:
            deviations = []
            avg_freqs = []
            for ion in self.ions:
                deviations.append(ion.avg_deviation)
                avg_freqs.append(ion.avg_freq_bin)

            plt.scatter(avg_freqs, deviations)
            plt.show()

        plot_regression_lines = 0
        fit_regression_lines = 0
        if plot_regression_lines:
            for ion in self.ions:
                fit_line_y = ion.linfitEquation(ion.trace_indices)
                plt.plot(ion.trace_indices, fit_line_y, color='red')

                if fit_regression_lines:
                    A_constraints = [0, np.inf]
                    k_constraints = [0, np.inf]
                    m_constraints = [-np.abs(ion.linfit_intercept),
                                     np.abs(ion.linfit_intercept)]
                    b_constraints = [-np.abs(2 * ion.linfit_intercept),
                                     np.abs(2 * ion.linfit_intercept)]

                    lower_bounds = [A_constraints[0], k_constraints[0], m_constraints[0], b_constraints[0]]
                    upper_bounds = [A_constraints[1], k_constraints[1], m_constraints[1], b_constraints[1]]

                    param, param_cov = curve_fit(self.crazy_snake_fit, np.array(ion.trace_indices), np.array(ion.trace),
                                                 bounds=(lower_bounds, upper_bounds))
                    fit_output_plot = self.crazy_snake_fit(np.array(ion.trace_indices), param[0], param[1], param[2],
                                                           param[3])
                    plt.plot(ion.trace_indices, fit_output_plot)

                plt.scatter(ion.trace_indices, ion.trace)
                print(param)
                plt.show()

        # Create a new space so that merges made here do not affect the main ion chain
        self.sort_traces()
        merge_playground = cpy.deepcopy(self.ions)

        full_duration_traces = []
        t_0_traces = []
        floating_traces = []
        for ion in merge_playground:
            # Pull out ions that last the entire trapping interval
            if len(ion.trace) >= len(self.sliceRecord) - self.min_trace_length:
                if True:
                    ion.reset_pairings()
                    full_duration_traces.append(ion)
                    print("==> Full length trace accepted at " + str(
                        round(np.average(ion.trace)) * self.resolution + self.f_range_offset) +
                          ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) +
                          " with slope ( " + str(ion.linfit_slope) + ")")
                # else:
                #     print("Full length trace rejected at " + str(round(np.average(ion.trace))) +
                #           ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) + " )" +
                #           " with slope ( " + str(ion.linfit_slope) + ")")

            # Pull out ions that start before t = 20 steps but do not last the entire trapping interval
            elif ion.trace_indices[0] < 20 and len(ion.trace) != len(self.sliceRecord):
                if True:  # Insert some condition here that does not rely on the linear fit
                    ion.reset_pairings()
                    t_0_traces.append(ion)
                    print("==> T0 trace accepted at " + str(
                        round(np.average(ion.trace)) * self.resolution + self.f_range_offset) +
                          ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) + " )" +
                          " with slope ( " + str(ion.linfit_slope) + ")")
                # else:
                #     print("T0 trace rejected at " + str(round(np.average(ion.trace))) +
                #           ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) + " )" +
                #           " with slope ( " + str(ion.linfit_slope) + ")")

            # Pull out ions that are 'floating' in the middle of the STFT
            elif ion.trace_indices[0] >= 20 and len(ion.trace) != len(self.sliceRecord):
                if True:
                    ion.reset_pairings()
                    floating_traces.append(ion)
                    print("==> Floating trace accepted at " + str(
                        round(np.average(ion.trace)) * self.resolution + self.f_range_offset) +
                          ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) + " )" +
                          " with slope ( " + str(ion.linfit_slope) + ")")
                # else:
                #     print("Floating trace rejected at " + str(round(np.average(ion.trace))) +
                #           ": Residuals checkpoint ( " + str(ion.fit_coeff_of_determination) + " )" +
                #           " with slope ( " + str(ion.linfit_slope) + ")")

        # Stitching things together:
        # Start with one T0 trace

        T0_appended = []
        while len(t_0_traces) > 0:
            T0 = t_0_traces.pop(0)
            append_floating = True
            while append_floating:
                append_floating = False
                possible_merges = []
                for floating in floating_traces:
                    # If trace end and start points roughly match up...
                    if abs(floating.trace_indices[0] - T0.trace_indices[-1]) < self.time_correlation_tolerance:
                        # Check to see if they have a frequency correlation
                        if self.freq_correlation_tolerance >= (T0.trace[-1] - floating.trace[0]) >= 0:
                            # Add the floating trace to the end of the trace that starts at T0
                            if floating.paired != 1:
                                possible_merges.append(floating)

                # self.ions = possible_merges
                # self.plot_ion_traces(np.flipud(np.rot90(np.array(self.sliceRecord), k=1)))
                # Find the closest possible frequency jump and assume that is the correct trace. Doing it this way
                # allows for much larger frequency jumps to be allowed without confounding the results.
                if len(possible_merges) > 0:
                    keeper = possible_merges[0]
                    for element in possible_merges:
                        min_floating = abs(keeper.trace[0] - T0.trace[-1])
                        check_floating = abs(element.trace[0] - T0.trace[-1])
                        if check_floating < min_floating:
                            keeper = element

                    T0.merge_as_trace(keeper)
                    keeper.paired = 1
                    append_floating = True

            T0_appended.append(T0)

        # FOR DEBUGGING PURPOSES:
        # self.ions = full_duration_traces
        # self.ions = T0_appended
        # self.ions = full_duration_traces + T0_appended + floating_traces  # Intend LIST CONCATENATION
        self.ions = full_duration_traces + T0_appended  # Intend LIST CONCATENATION ** NORMAL PROGRAM LINE **

    def plot_ion_traces(self, Zxx):

        for i in range(len(self.ions)):
            try:
                plt.plot(np.array(self.ions[i].trace_indices),
                         np.array(self.ions[i].trace) * self.resolution + self.f_range_offset)
                # plt.plot(np.array(self.ions[i].trace_indices), self.ions[i].linfitEquation(
                #     self.ions[i].trace_indices) * self.resolution + self.f_range_offset, linestyle='dashed')
            except Exception as e:
                print('Trace plotting error:', e)
        for i in range(len(self.dead_ions)):
            try:
                plt.plot(np.array(self.dead_ions[i].trace_indices),
                         np.array(self.dead_ions[i].trace) * self.resolution + self.f_range_offset,
                         color="green", linestyle='dashdot')
            except Exception as e:
                print('Trace plotting error:', e)

        plot_steps = len(Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_range_offset, plot_height * self.resolution + self.f_range_offset, self.resolution),
            slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
        plt.pcolormesh(x, y, Zxx, cmap='hot')
        plt.colorbar()
        plt.show()

    def plot_paired_traces(self, Zxx, segment_length, step_length):
        plt.subplot(1, 2, 1)

        for i in range(len(self.ions)):
            try:
                plt.plot(np.array(self.ions[i].trace_indices),
                         np.array(self.ions[i].trace) * self.resolution + self.f_range_offset)
                plt.plot(np.array(self.ions[i].trace_indices), self.ions[i].linfitEquation(
                    self.ions[i].trace_indices) * self.resolution + self.f_range_offset,
                         linestyle='dashed')
            except Exception:
                print("Can't plot traces.")

        plot_steps = len(Zxx[0])  # Assuming Zxx is of form [[], []] (list of lists)
        plot_height = len(Zxx)

        # generate 2 2d grids for the x & y bounds
        y, x = np.mgrid[
            slice(self.f_range_offset, plot_height * self.resolution + self.f_range_offset, self.resolution),
            slice(self.t_range_offset, plot_steps + self.t_range_offset, 1)]
        plt.pcolormesh(x, y, Zxx, cmap='hot')
        plt.colorbar()
        plt.title("Total Traces")
        plt.xlabel('Time Segments')
        plt.ylabel('Frequency Bins')

        plt.subplot(1, 2, 2)
        for i in range(len(self.paired_ions)):
            plt.plot(np.array(self.paired_ions[i].trace_indices),
                     np.array(self.paired_ions[i].trace) * self.resolution + self.f_range_offset, color='green')
            plt.plot(np.array(self.paired_ions[i].trace_indices),
                     np.array(self.paired_ions[i].harmonic_trace) * self.resolution + self.f_range_offset,
                     linestyle='dashed', color='green')

        plt.pcolormesh(x, y, Zxx, cmap='hot')
        plt.colorbar()
        plt.title("Paired Traces")
        plt.xlabel('Time Segments')
        plt.ylabel('Frequency Bins')
        plt.show()

    def write_ions_to_files(self, save_file_folder, file, tracesHeader, export_Zxx_files):
        # Assumes files are named with xxxxx.B.txt convention
        if len(self.paired_ions) > 0:
            source_file_name = file.parts[-1]
            source_file_name = source_file_name.split(".B.txt")[0]
            save_file_folder = save_file_folder / source_file_name
            save_file_folder.mkdir(exist_ok=True, parents=True)
            savedIons = 0
            save_file_name = source_file_name + "_Zxx" + ".Zxx"
            save_file_path_full = save_file_folder / save_file_name
            save_file_folder.touch(save_file_name)
            fileDivider = "//"
            ##################################################################
            if export_Zxx_files:
                Zxx = np.array2string(np.array(self.sliceRecord), threshold=np.inf)
                writeList = [tracesHeader] + [fileDivider] + [Zxx]
                writeString = ", ".join(str(x) for x in writeList)
                save_file_path_full.write_text(writeString)
            ##################################################################
            for counter in range(len(self.paired_ions)):
                if len(self.paired_ions[counter].trace) > 0.05 * len(self.sliceRecord):
                    if len(self.paired_ions[counter].magnitude_trace) > 0.05 * len(self.sliceRecord):
                        save_file_name = source_file_name + "_trace_" + str(counter) + ".trace"
                        save_file_path_full = save_file_folder / save_file_name
                        save_file_folder.touch(save_file_name)
                        fileDivider = "//"
                        traceFloat = [float(i) for i in self.paired_ions[counter].trace]
                        magFloat = [float(i) for i in self.paired_ions[counter].magnitude_trace]
                        phaseFloat = [float(i) for i in self.paired_ions[counter].phase_trace]

                        traceFloatHarm = [float(i) for i in self.paired_ions[counter].harmonic_trace]
                        magFloatHarm = [float(i) for i in self.paired_ions[counter].harmonic_magnitude]
                        phaseFloatHarm = [float(i) for i in self.paired_ions[counter].harmonic_phase]

                        indicesFloat = [float(i) for i in self.paired_ions[counter].trace_indices]
                        writeList = [tracesHeader] + [fileDivider] + traceFloat + [fileDivider] + magFloat + \
                                    [fileDivider] + phaseFloat + [fileDivider] + traceFloatHarm + [fileDivider] + \
                                    magFloatHarm + [fileDivider] + phaseFloatHarm + [fileDivider] + indicesFloat
                        writeString = ", ".join(str(x) for x in writeList)
                        savedIons = savedIons + 1
                        save_file_path_full.write_text(writeString)
            print("Saved " + str(savedIons) + "/" + str(len(self.ions)) + " paired traces!")
