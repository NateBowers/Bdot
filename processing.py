import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 

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



class Probe_calibration():

    def __init__(self, omega_data: np.array, re_data: np.array, im_data: np.array, r: float, Rp: float, N: int=10, g: int = 1):

        """Ensure the data passed into Probe_calibration class is of the right type and size

        Args:
            omega_data (nd.array): Frequency in Hz
            re_data (np.array): Real part of V_meas / V_ref
            im_data (nd.array): Imaginary part of V_meas / V_ref
            r (float): Helmholz radius in meters
            Rp (float): Resistance of resister measured across in ohms 
            N (int, optional): Number loops, counting both twin-twisted wires
            g (int, optional): Amplifier gain

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
    
    def calibrate(self, initial_guess: np.array = [1,1,1]):

        x_combo = np.hstack((self.omega_data, self.omega_data))
        y_combo = np.hstack((self.re_data, self.im_data))

        fittedParameters, pcov = curve_fit(self.combined_curves, x_combo, y_combo, initial_guess)

        self.a, self.tau, self.tau_s = fittedParameters
        print(fittedParameters)
        self.covariance = pcov
        return

    def graph(self, data_set: str = 'both', add_fit: bool = False):

        
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

            y_re_fit = self.re_curve(x, a, tau, tau_s)
            y_im_fit = self.im_curve(x, a, tau, tau_s)

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
    
    

foo = Probe_calibration(w_test, re_test, im_test, r=1, Rp=1)
foo.calibrate(initial_guess=[2, 5, 2])
foo.graph(add_fit=True)
