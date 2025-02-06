import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk) 


# Constants (all in base units)
N = 1 #number of turns
g = 1 #amplifier gain
mu = 1 #permiability
r = 1 #Helmholtz radius
Rp = 1 #resister measured across


class Probe_calibration():
    def __init__(self, omega_data: np.array, re_data: np.array, im_data: np.array):

        """Ensure the data passed into Probe_calibration class is of the right type and size

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
        return
    





def calibrate_probe(omega_data, re_data, im_data):


    

    # Real component

    v_ratio_real = a * 

    # Imaginary component

    v_ratio = a * C 

    return





      
def model_func(x, a, b):
    return a * np.sin(b * x)
np.random.seed(0)
xdata = np.linspace(0, 15, 100)
y = model_func(xdata, 2.5, 0.8)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
plt.figure(figsize=(8, 6))
plt.scatter(xdata, ydata, label='Data with noise')

# Fitting the function to the data using curve_fit
popt, pcov = scipy.optimize.curve_fit(model_func, xdata, ydata, method='dogbox')
# Getting the optimized parameters
a_opt, b_opt = popt
print(f'Optimized parameters: a = {a_opt}, b = {b_opt}')
plt.plot(xdata, model_func(xdata, *popt), 'r-', label='Fitted curve')

plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('Curve Fitting Example')
plt.legend()
plt.grid(True)
plt.show()