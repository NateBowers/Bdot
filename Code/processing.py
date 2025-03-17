import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 
import json
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'serif'
try:
    plt.rcParams['font.serif'] = 'cm'
except:
    pass



# Constants (all in base units)
# N = 1 #number of turns
# g = 1 #amplifier gain
# mu = 1 #permiability
# r = 1 #Helmholtz radius
# Rp = 1 #resister measured across

a = 3
b = 4
c = 1
w_test = np.arange(1,10,0.5)
re_test = (a * np.sin(w_test) + b) + np.random.normal(0, 0.2, size=len(w_test))
im_test = (c * np.square(w_test)) - (b * w_test) + np.random.normal(0, 3, size=len(w_test))

# h = np.array([5.0, 6.1, 7.2, 8.3, 9.4])
# y1 = np.array([ 16.00,  18.42,  20.84,  23.26,  25.68])
# y2 = np.array([-20.00, -25.50, -31.00, -36.50, -42.00])



class ProbeCalibration():

    def __init__(self):
        "Creates probe object, either import calibration data with 'import data' or load calibration values with 'load'"
        pass

    def __str__(self):
        "prints calibrated values with uncertainties at 4 sig figs"
        try:
            a, tau, tau_s = self.a, self.tau, self.tau_s
            return f'a={a[0]:.4g}±{a[1]:.4g}, \ntau={tau[0]:.4g}±{tau[1]:.4g}, \ntau_s={tau_s[0]:.4g}±{tau_s[1]:.4g}'
        except:
            return f'probeCalibration object stored at <{hex(id(self))}>'
    
    def load_data(self, omega_data: np.array, re_data: np.array, im_data: np.array, r: float, Rp: float, N: int=10, g: int = 1):
        """Ensure the data passed into ProbeCalibration class is of the right type and size

        Args:
            omega_data (nd.array): Frequency in Hz
            re_data (np.array): Real part of V_meas / V_ref
            im_data (nd.array): Imaginary part of V_meas / V_ref
            r (float): Helmholz radius in meters
            Rp (float): Resistance of resister measured across in ohms 
            N (int, optional): Number loops, counting both twin-twisted wires. Defaults to 10
            g (int, optional): Amplifier gain. Defaults to 1

        Raises:
            TypeError: Checks to see if data is an ndarray
            ValueError: Check to make sure w, re, im data the same length
            ValueError: Check to make sure the data is all 1 dim arrays
        """

        if not (isinstance(omega_data, np.ndarray) or isinstance(re_data, np.ndarray) or isinstance(im_data, np.ndarray)):
            raise TypeError('Data is not a ndarray')
        if not (omega_data.shape == re_data.shape == im_data.shape):
            raise ValueError('Data are not of the same length')
        if not (omega_data.ndim == re_data.ndim == im_data.ndim == 1):
            raise ValueError('Data is not a 1-dimensional array')
        

        self.omega_data = omega_data
        self.re_data = re_data
        self.im_data = im_data

        mu_0 = 1.25663706127e-6 # Permeability of free space in N/A^2
        self.C = ((16 * N * g * mu_0) / ((5 ** 1.5) * r * Rp))

    def load_calibrated_parameters(self, path, probe_name, axis):
        if axis not in ['x axis', 'y axis', 'z axis']:
            raise ValueError(f'Axis {axis} does not exist, axis must be either "x axis", "y axis", or "z axis"')

        try:
            with open(path, 'r') as file:
                data = json.load(file)
                try:
                    probe_index = [probe['Probe name'] for probe in data].index(probe_name) 
                    probe = data[probe_index]
                except:
                    raise ValueError(f'Probe "{probe_name}" does not exist')
        except:
            raise FileNotFoundError
        
        self.a = probe['Calibration data'][axis]['a']
        self.tau = probe['Calibration data'][axis]['tau']
        self.tau_s = probe['Calibration data'][axis]['tau_s']

        # print(probe)

            # print(probe_index)

            # probe_names = [probe['Probe name'] for probe in data]
            # if probe_name in probe_names:
            #     probe_index = probe_names[probe_name]
            #     print("True")
            # if 'Probe name' in [probe['Probe name'] for probe in data]:
            #     print("foo")
            #     # self.a = probe['Calibration data'][axis]['a']
            #     # self.tau = probe['Calibration data'][axis]['tau']
            #     # self.tau_s = probe['Calibration data'][axis]['tau_s']
            # else:
            #     raise ValueError(f'No saved parameters for axis "{axis}" of probe "{probe_name}"')
        pass

    def save(self, probe_name, path, axis):
        # 
        try:
            a, tau, tau_s = self.a, self.tau, self.tau_s
        except:
            raise ValueError('No fit parameters to save')
        
        # generate info for saving
        
        try:
            with open(path, 'r+') as file:
                foo
        except:
            pass
        
        pass

    
    def re_curve(self, w, a, tau, tau_s):
        """ Real component of Vmeas/Vref """
        # y = (a * self.C * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
        # return y
        return a * np.sin(w) + tau

    def im_curve(self, w, a, tau, tau_s):
        """ Imaginary component of Vmeas/Vref"""
        # y = (a * self.C ((w ** 3) * tau * tau_s + w)) / (1 + (tau_s * w) ** 2)
        # return y
        return (tau_s * np.square(w)) - (tau * w)
    
    def combined_curves(self, combo_w, a, tau, tau_s):
        """ Combine real and imaginary parts into one function"""

        extract_re = combo_w[:len(self.re_data)]
        extract_im = combo_w[len(self.im_data):]

        result_re = self.re_curve(extract_re, a, tau, tau_s)
        result_im = self.im_curve(extract_im, a, tau, tau_s)

        return np.append(result_re, result_im)
    
    def fit_curve(self, initial_guess: np.array = [1,1,1], sigma: float = None ):
        """ Fits the model and saves the parameters

        Args:
            initial_guess (np.array, optional): Initial guess for a, tau, and tau_s. Defaults to [1,1,1].
            sigma (float, optional): Standard deviation of errors in y_data. Defults to None
        """

        x_combo = np.hstack((self.omega_data, self.omega_data))
        y_combo = np.hstack((self.re_data, self.im_data))

        popt, pcov = curve_fit(self.combined_curves, x_combo, y_combo, initial_guess, sigma = sigma)
        self.a, self.tau, self.tau_s = zip(popt, np.square(np.diag(pcov)))
        # print(fittedParameters)
        # self.covariance = pcov
        # print()
        return

    def graph(self, data_set: str = 'both', add_fit: bool = False):
        """ Plotting tool for calibration data. Can include 

        Args:
            data_set (str, optional): Plot either the real component 're', the imaginary component 'im', or both 'both'. Defaults to 'both'.
            add_fit (bool, optional): Whether to plot the fitted curve. Defaults to False.

        Raises:
            ValueError: Data_set value not 're', 'im' or 'both'
            ValueError: Trying to plot the fit before parameters generated
        """
        if data_set not in ('re', 'im', 'both'):
            raise ValueError("Data set must be one of 're', or 'im'")
        
        x = self.omega_data
        y_re = self.re_data
        y_im = self.im_data

        if data_set == 're':
            plt.scatter(x, y_re)
            plt.ylabel(r'Re$ (V_{\text{meas}} / V_{\text{ref}})$')
            plt.xlabel(r'$\omega$ (Hz)')
        elif data_set == 'im':
            plt.scatter(x, y_im)
            plt.ylabel(r'Im$ (V_{\text{meas}} / V_{\text{ref}})$')
            plt.xlabel(r'$\omega$ (Hz)')
        elif data_set == 'both':
            plt.subplot(2,1,1)
            plt.scatter(x, y_re)
            plt.ylabel(r'Re$ (V_{\text{meas}} / V_{\text{ref}})$')
            plt.xlabel(r'$\omega$ (Hz)')
            plt.subplot(2,1,2)
            plt.scatter(x, y_im)
            plt.ylabel(r'Im$ (V_{\text{meas}} / V_{\text{ref}})$')
            plt.xlabel(r'$\omega$ (Hz)')

        if add_fit:
            try:
                a = self.a
                tau = self.tau
                tau_s = self.tau_s
            except:
                raise ValueError('Fit parameters not generated, run calibration method first')

            y_re_fit = self.re_curve(x, a[0], tau[0], tau_s[0])
            y_im_fit = self.im_curve(x, a[0], tau[0], tau_s[0])

            if data_set == 're':
                plt.plot(x, y_re_fit, label="Fit")
            elif data_set == 'im':
                plt.plot(x, y_im_fit, label="Fit")
            elif data_set == 'both':
                plt.subplot(2,1,1)
                plt.plot(x, y_re_fit, label="Fit")
                plt.legend()
                plt.subplot(2,1,2)
                plt.plot(x, y_im_fit, label="Fit")
                plt.legend()

        plt.show()
        return
    

if __name__ == '__main__':
    bar = ProbeCalibration()
    bar.load_calibrated_parameters('calibration.json', "test2", "x axis")
    print(bar)

    foo = ProbeCalibration()
    foo.load_data(w_test, re_test, im_test, r=1, Rp=1)
    foo.fit_curve(initial_guess=[2, 5, 2])
    print(foo)
    foo.graph(add_fit=True)


# foo = ProbeCalibration(w_test, re_test, im_test, r=1, Rp=1)
# foo.fit_curve(initial_guess=[2, 5, 2])
# # foo.graph(add_fit=True)
# print(foo)
