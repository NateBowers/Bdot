import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert
import unittest
from lmfit import Model
import lmfit
from main import load_data


N = 16
g = 1
R_p = 0.22
r = 0.008
mu_0 = 0.000001256

factor = (N * g * 8 * mu_0) / (r * R_p * (5**1.5))




def re_curve_meinecke(w, a, tau, tau_s):
    """ Real component of Vmeas/Vref """
    y = factor * (a * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
    return y

def im_curve_meinecke(w, a, tau, tau_s):
    """ Imaginary component of Vmeas/Vref"""
    y =  factor * (a * tau * tau_s * (w**3) + a * w) / (1 + (tau_s * w) ** 2)
    return y

def re_curve_Everson(w, a, tau, tau_s):
    pass
def im_curve_Everson(w, a, tau, tau_s):
    pass




print(factor)


path_dict = {
    'freq' : ('test_data.csv', 0, 3),
    'v_meas': ('test_data.csv', 8, 3),
    'v_ref': ('test_data.csv', 6, 3),
    'v_meas_im': ('test_data.csv', 9, 3),
    'v_ref_im': ('test_data.csv', 7, 3),
}

v_ratio_re, v_ratio_im, freq = load_data(path_dict, True)
lower = 0
upper = 250
v_ratio_re = v_ratio_re[lower:upper]
v_ratio_im = v_ratio_im[lower:upper]
freq = freq[lower:upper]


def objective(params, x, y1, y2):
    a = params['a']
    tau = params['tau']
    tau_s = params['tau_s']
    
    resid1 = y1 - re_curve_meinecke(x, a, tau, tau_s)
    resid2 = y2 - im_curve_meinecke(x, a, tau, tau_s)
    
    return np.concatenate((resid1, resid2))


params = lmfit.Parameters()
params.add('a', value=0.00001026)
params.add('tau', value=-1.67E-08)
params.add('tau_s', value=3.92E-08)
result = lmfit.minimize(objective, params, args=(freq, v_ratio_re, v_ratio_im))
print(lmfit.fit_report(result))

a_calib, tau_calib, tau_s_calib = result.params.valuesdict().values()


fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
fig.set_size_inches(15, 5)
for ax in axes:
    ax.set_xlabel('omega')
    ax.set(adjustable='datalim')

ax1, ax2 = axes
ax1.plot(freq, v_ratio_re, 'b-', label='Re(Vm/Vr)')
ax1.plot(freq, re_curve_meinecke(freq, a_calib, tau_calib, tau_s_calib), 'r-', label='Re(Best Fit)')
ax1.set_ylabel('Re(V_meas/V_ref)')

ax2.plot(freq, v_ratio_im, 'b-', label='Im(Vm/Vr)')
ax2.plot(freq, im_curve_meinecke(freq, a_calib, tau_calib, tau_s_calib), 'r-', label='Im(Best Fit)')
ax2.set_ylabel('Im(V_meas/V_ref)')

plt.figtext(
    0.05,        # x
    0.2,       # y
    lmfit.fit_report(result), 
    horizontalalignment='left', 
    fontsize=8, 
    fontfamily='monospace'
)
    



fig.subplots_adjust(left=0.4)    
plt.figlegend(loc='upper left')
plt.show()

