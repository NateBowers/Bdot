import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize
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

w_test = np.array([1,2,3,4])
re_test = np.array([2,2,2,2])
im_test = np.array([1,2,3,4])



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
        y = (a * self.C * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
        return y

    def im_curve(self, w, a, tau, tau_s):
        """ Imaginary component of Vmeas/Vref"""
        y = (a * self.C ((w ** 3) * tau * tau_s + w)) / (1 + (tau_s * w) ** 2)
        return y
    
    def calibrate(self):
        self.a = 1
        self.tau = 1
        self.tau_s = 1
        return

    def graph(self, data_set: str, add_fit: bool = False):

        if add_fit:
            try:
                a = self.a
                tau = self.tau
                tau_s = self.tau_s
            except:
                raise ValueError('Fit parameters not generated, run calibration method first')
        
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

        plt.show()


        
        return
    

foo = Probe_calibration(w_test, re_test, im_test)

foo.graph(data_set = 'both')





      
# def model_func(x, a, b):
#     return a * np.sin(b * x)
# np.random.seed(0)
# xdata = np.linspace(0, 15, 100)
# y = model_func(xdata, 2.5, 0.8)
# ydata = y + 0.2 * np.random.normal(size=len(xdata))
# plt.figure(figsize=(8, 6))
# plt.scatter(xdata, ydata, label='Data with noise')

# # Fitting the function to the data using curve_fit
# popt, pcov = scipy.optimize.curve_fit(model_func, xdata, ydata, method='dogbox')
# # Getting the optimized parameters
# a_opt, b_opt = popt
# print(f'Optimized parameters: a = {a_opt}, b = {b_opt}')
# plt.plot(xdata, model_func(xdata, *popt), 'r-', label='Fitted curve')

# plt.xlabel('X data')
# plt.ylabel('Y data')
# plt.title('Curve Fitting Example')
# plt.legend()
# plt.grid(True)
# plt.show()