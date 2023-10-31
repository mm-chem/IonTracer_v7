import numpy as np
import traceback
import matplotlib.pyplot as plt
import csv
import math
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import scipy.signal as sig
import os
import itertools
import tkinter as tk
from tkinter import filedialog as fd
import v6_STFT_analysis as STFT
from scipy import signal as sig
from scipy import optimize as opt
import v1_DropShot as Rayleigh


def calculate_lost_mass(initial_charge, charge_lost, initial_mass, apparent_charge_lost):
    mass_lost = []
    try:
        initial_charge += 1.0
        initial_charge -= 1.0
        initial_charge_array = [initial_charge]
    except TypeError:
        initial_charge_array = initial_charge
        print(initial_charge)

    try:
        initial_mass += 1.0
        initial_mass -= 1.0
        initial_mass_array = [initial_mass]
    except TypeError:
        initial_mass_array = initial_mass
        print(initial_mass)

    for i_charges in initial_charge_array:
        for i_mass in initial_mass_array:
            final_charge = i_charges - charge_lost
            final_mass = - i_mass * (1 / ((i_charges / final_charge) * (apparent_charge_lost / i_charges - 1)))
            mass_lost.append(initial_mass - final_mass)
    return mass_lost


if __name__ == "__main__":
    N = 1000
    min_mass = 1000000
    max_mass = 10000000
    step = (max_mass - min_mass) / N
    followRayleigh = True

    if followRayleigh:
        simulation_size, simulation_charge = Rayleigh.returnRayleighLine(min_mass, max_mass, N)
    else:
        simulation_size = np.linspace(min_mass, max_mass, step)
        simulation_charge = np.linspace(50, 300, len(simulation_size))

    row = 0
    col = 0
    outputArray = np.zeros([len(simulation_size), len(simulation_charge)])
    for charge in simulation_charge:
        for size in simulation_size:
            result = calculate_lost_mass(initial_charge=charge, charge_lost=2, initial_mass=size, apparent_charge_lost=1.70)
            outputArray[row, col] = result[0] / 1000
            col = col + 1
        row = row + 1
        col = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = simulation_size
    y = simulation_charge
    X, Y = np.meshgrid(x, y)
    zs = np.array(calculate_lost_mass(initial_charge=np.ravel(Y), charge_lost=2, initial_mass=np.ravel(X), apparent_charge_lost=1.70)[0] / 1000)
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    x_axis_ticks = np.multiply(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1000000)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    ax.set_xticks(x_axis_ticks, labels)

    ax.set_xlabel('Precursor Size (MDa)')
    ax.set_ylabel('Precursor Charge')
    ax.set_zlabel('Progeny Size (kDa)')
    plt.show()

    progeny_mass_const_size = calculate_lost_mass(initial_charge=simulation_charge, charge_lost=2, initial_mass=300000000, apparent_charge_lost=1.00)

    plt.plot(simulation_charge, progeny_mass_const_size)
    plt.title("Precursor (fixed m = 300MDa)")
    plt.xlabel('Charge of Precursor')
    plt.ylabel('Progeny Mass (dz = -2 appears as -1)')
    plt.show()

    print(calculate_lost_mass(initial_charge=300, charge_lost=2, initial_mass=1000000, apparent_charge_lost=1.7))