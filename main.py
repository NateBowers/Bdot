import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert
import unittest
from lmfit import Model
import lmfit




# def calc_fit_params(v_ratio_re, v_ratio_im, omega, sys_params):
#     C = 200
#     v_ratio_re = v_ratio_re[0:C]
#     v_ratio_im = v_ratio_im[0:C]
#     omega = omega[0:C]
#     mu_0 = 1
#     # const = (16 * sys_params['N'] * sys_params["g"] * mu_0) / (sys_params["r"] * sys_params["Rp"] * (5**(1.5)))
#     const = 1.67652e-07
#     omega_combo = np.hstack((omega, omega))
#     v_ratio_combo_scaled = np.hstack((v_ratio_re, v_ratio_im)) / const

#     def combined_curves(combo_w, a, tau, tau_s):
#         """ Combine real and imaginary parts into one function"""

#         extract_re = combo_w[:len(v_ratio_re)]
#         extract_im = combo_w[len(v_ratio_im):]

#         result_re = re_curve_meinecke(extract_re, a, tau, tau_s)
#         result_im = im_curve_meinecke(extract_im, a, tau, tau_s)

#         return np.append(result_re, result_im)
    
#     guess = [1.026e-5,-1.67e-8,3.92e-8]     # a, tau, tau_s
#     # popt, pcov = curve_fit(combined_curves, omega_combo, v_ratio_combo_scaled, p0=guess)

#     # print(popt)

#     re_predict = re_curve_meinecke(omega, guess[0], guess[1], guess[2])

#     plt.plot(omega, v_ratio_re)
#     plt.plot(omega, re_predict)
#     plt.show()

#     return const









# Helper Functions



def convert_to_volt(arr: np.array, init_unit:str) -> np.array:
    """Convert array to volts"""
    if init_unit.lower() == 'volt':
        return arr
    elif init_unit.lower() == 'dbmv':
        return 10 ** ((arr - 60) / 20)
    else:
        raise NotImplementedError('Support for units besides "volt" and "dBmV" are not yet implimented')
    
# def re_curve_meinecke(w, a, tau, tau_s):
#     """ Real component of Vmeas/Vref """
#     y = (a * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
#     return y

# def im_curve_meinecke(w, a, tau, tau_s):
#     """ Imaginary component of Vmeas/Vref"""
#     y =  (a * tau * tau_s * (w**3) + a * w) / (1 + (tau_s * w) ** 2)
#     return y


def real_component(w, a, tau, tau_s):
    """Real component of the complex model."""
    return 1.67652E-07 * (a * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)

def imaginary_component(w, a, tau, tau_s):
    """Imaginary component of the complex model."""
    return 1.67652E-07 * (a * tau * tau_s * (w ** 3) + a * w) / (1 + (tau_s * w) ** 2)

def complex_model(w, a, tau, tau_s):
    """Combined complex model returning real and imaginary parts."""
    return real_component(w, a, tau, tau_s) + 1j * imaginary_component(w, a, tau, tau_s)

# Main Functions

def load_data(path_dict: dict, 
              analytic: bool = False,
              re_offset: float = 0,
              im_offset: float = 0,
              units: str = 'dbmv', 
              graph: bool = False,
              save_graph: bool = False, 
              show_graph: bool = False,
              save_path: str = None) -> list[np.array, np.array, np.array]:
    """Load calibration data and calculate the real and imaginary parts of
        v_meas/v_ref in terms of volts. Also converts frequency to angular
        frequency

    Args:
        path_dict (dict): Info on where freq, v_meas, v_ref, and (optional)
                        v_meas_im, v_ref_im is stored and which columns to use
        analytic (bool, optional): Does the data contain imaginary information. 
                        Defaults to False.
        re_offset (float, optional):
        im_offset (float, optional):
    Raises:
        TypeError: Check save_path to ensure it is a string

    Returns:
        list[np.array, np.array, np.array]: Re(V_meas/V_ref), Im(V_meas/V_ref), omega. 
    """

    if not analytic:
        freq = np.genfromtxt(path_dict["freq"][0], usecols=path_dict["freq"][1], skip_header=path_dict["freq"][2], delimiter=',')
        v_meas = np.genfromtxt(path_dict["v_meas"][0], usecols=path_dict['v_meas'][1], skip_header=path_dict["v_meas"][2], delimiter=',')
        v_ref = np.genfromtxt(path_dict["v_ref"][0], usecols=path_dict["v_ref"][1], skip_header=path_dict["v_ref"][2], delimiter=',')

        v_m_analytic = hilbert(v_meas)
        v_r_analytic = hilbert(v_ref)

        v_ratio = v_m_analytic / v_r_analytic
        v_ratio_re, v_ratio_im = np.real(v_ratio), np.imag(v_ratio)

    else:
        freq = np.genfromtxt(path_dict["freq"][0], usecols=path_dict["freq"][1], skip_header=path_dict["freq"][2], delimiter=',')
        v_meas = np.genfromtxt(path_dict["v_meas"][0], usecols=path_dict["v_meas"][1], skip_header=path_dict["v_meas"][2], delimiter=',')
        v_ref = np.genfromtxt(path_dict["v_ref"][0], usecols=path_dict["v_ref"][1], skip_header=path_dict["v_ref"][2], delimiter=',')
        v_meas_im = np.genfromtxt(path_dict["v_meas_im"][0], usecols=path_dict["v_meas_im"][1], skip_header=path_dict["v_meas_im"][2], delimiter=',')
        v_ref_im = np.genfromtxt(path_dict["v_ref_im"][0], usecols=path_dict["v_ref_im"][1], skip_header=path_dict["v_ref_im"][2], delimiter=',')

        v_m_analytic = v_meas + 1j*v_meas_im
        v_r_analytic = v_ref +1j*v_ref_im

    v_ratio = v_m_analytic / v_r_analytic
    v_ratio_re, v_ratio_im = np.real(v_ratio) + re_offset, np.imag(v_ratio) + im_offset


    if np.sum(v_ratio_re < 0) > len(v_ratio_re) / 2:
        v_ratio_re = -1 * v_ratio_re
    if np.sum(v_ratio_im < 0) > len(v_ratio_im) / 2:
        v_ratio_im = -1 * v_ratio_im

    

    return v_ratio_re, v_ratio_im, freq * 2 * np.pi




def fit(frequency, re_data, im_data):
    complex_data = re_data + 1j*im_data
    # Prepare the model
    model = Model(complex_model)
    
    # Fit the model to the data
    params = model.make_params(a=1, tau=1, tau_s=1)
    result = model.fit(complex_data, params, w=frequency)

    # Print the fitting report
    print(result.fit_report())

    # Plot the results
    plt.figure()
    plt.plot(frequency, re_data, 'bo', label='re_data')
    plt.plot(frequency, im_data, 'bo', label='im_data')
    plt.plot(frequency, result.best_fit, 'r-', label='Best Fit')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Complex Measurement')
    plt.legend()
    plt.show()


def fit_single(frequency, data):

    # Prepare the model
    model = Model(real_component)
    
    # Fit the model to the data
    params = model.make_params(a=0.00001026, tau=-1.67E-08, tau_s=3.92E-08)
    result = model.fit(data, params, w=frequency)

    # Print the fitting report
    print(result.fit_report())

    # Plot the results
    plt.figure()
    plt.plot(frequency, data, 'bo', label='Data')
    plt.plot(frequency, result.best_fit, 'r-', label='Best Fit')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Complex Measurement')
    plt.legend()
    plt.show()







# From measured voltages, values for a, g, N, tau_s, reconstruct the field

def reconstruct(v: unumpy.uarray,
                a: ufloat,
                tau_s: ufloat,
                t_step: float,
                g: ufloat = ufloat(1,0),
                N: int = 10,
                b_0: float = 0) -> unumpy.uarray:
    """Reconstruct magnetic field from a probe with uncertainties. Error is 
       automatically propogated with ufloats and the uncertainty package.

    Args:
        v (unumpy.uarray): Measured voltages with uncertainties for each time, in units of Volts
        a (ufloat): Relaxation time with uncertainty, in units of Seconds
        tau_s (ufloat): Prob tip area with uncertaintin, in units of Meters^2
        t_step (float): Time step between measurments, in units of Seconds
        g (ufloat, optional): Amplifier gain. Defaults to ufloat(1,0).
        N (int, optional): Number of turns. Defaults to 10.
        b_0 (float, optional): Initial field strength. Defaults to 0.

    Returns:
        unumpy.uarray:  Array of numerically integrated B values with associated
                        uncertainties. 
    """    
    b = unumpy.uarray(np.empty(len(v)), 1)
    b[0] = ufloat(b_0,1)

    for i in range(len(b)):        
        b[i] = (v[0] / a * g) + (v[i] * t_step / a * g * 2 * N) + (v[i] * tau_s / a * g * N) + (t_step / a * g * N) * np.sum(v[1:i])

    return b


def plot_with_error(ax: matplotlib.axes.Axes,
                    x: np.array,
                    y: np.array,
                    y_err: np.array,
                    curvelabel: str=None,
                    xlabel: str=None,
                    ylabel: str=None):
    """_summary_

    Args:
        ax (matplotlib.axes.Axes): axis to plot data on
        x (np.array): x values to be graphed
        y (np.array): y values
        y_err (np.array): uncertainties in y values
        curvelabel (str, optional): Label for legend in plot. Defaults to None.
        xlabel (str, optional): y label. Defaults to None.
        ylabel (str, optional): y label. Defaults to None.
    """
    out = ax.plot(x, y, label=curvelabel)
    out = ax.fill_between(x, y-y_err, y+y_err, alpha=0.2, label='Error Band')
    out = ax.set_xlabel(xlabel)
    out = ax.set_ylabel(ylabel)
    out = ax.legend()
    return out


class CustomTestCase(unittest.TestCase):
    def npAssertAlmostEqual(self, first, second, rtol=1e-06, atol=1e-08):
        np.testing.assert_allclose(first, second, rtol=rtol, atol=atol)

class DataLoading(CustomTestCase):

    def test_ratio(self):
        path_dict = {
            'freq' : ('test_data.csv', 0, 3),
            'v_meas': ('test_data.csv', 8, 3),
            'v_ref': ('test_data.csv', 6, 3),
            'v_meas_im': ('test_data.csv', 9, 3),
            'v_ref_im': ('test_data.csv', 7, 3),
        }
        v_ratio_re, v_ratio_im, _ = load_data(path_dict, analytic=True, re_offset=-0.0007, im_offset=0.002)
        v_ratio_re_xlsx = np.genfromtxt('test_data.csv', usecols=13, skip_header=3, delimiter=',')
        v_ratio_im_xlsx = np.genfromtxt('test_data.csv', usecols=12, skip_header=3, delimiter=',')



        self.npAssertAlmostEqual(v_ratio_re, v_ratio_re_xlsx)
        self.npAssertAlmostEqual(v_ratio_im, v_ratio_im_xlsx)











# def objective(params, x1, y1, x2, y2):
#     amp1 = params['amp1']
#     cen1 = params['cen1']
#     wid1 = params['wid1']
#     amp2 = params['amp2']
#     cen2 = params['cen2']
#     wid2 = params['wid2']
    
#     resid1 = y1 - function1(x1, amp1, cen1, wid1)
#     resid2 = y2 - function2(x2, amp2, cen2, wid2)
    
#     return np.concatenate((resid1, resid2))

# x1 = np.linspace(-10, 10, 100)
# y1 = function1(x1, 5, 0, 2) + np.random.normal(0, 0.1, 100)
# x2 = np.linspace(-10, 10, 100)
# y2 = function2(x2, 3, 1, 3) + np.random.normal(0, 0.1, 100)

# params = lmfit.Parameters()
# params.add('amp1', value=4, min=0)
# params.add('cen1', value=0)
# params.add('wid1', value=2, min=0)
# params.add('amp2', value=2, min=0)
# params.add('cen2', value=1)
# params.add('wid2', value=3, min=0)

# result = lmfit.minimize(objective, params, args=(x1, y1, x2, y2))

# print(lmfit.fit_report(result))

if __name__ == '__main__':

    # unittest.main()



    path_dict = {
        'freq' : ('test_data.csv', 0, 3),
        'v_meas': ('test_data.csv', 8, 3),
        'v_ref': ('test_data.csv', 6, 3),
        'v_meas_im': ('test_data.csv', 9, 3),
        'v_ref_im': ('test_data.csv', 7, 3),
    }



    test1 = True
    test2 = False

    if test1:

        v_ratio_re, v_ratio_im, freq = load_data(path_dict, True)
        lower = 50
        upper = 250
        v_ratio_re = v_ratio_re[lower:upper]
        v_ratio_im = v_ratio_im[lower:upper]
        freq = freq[lower:upper]

        plt.plot(freq, v_ratio_re)
        plt.show()

        fit_single(freq, v_ratio_im)
        # fit(freq, v_ratio_re, v_ratio_im)

        # v_ratio_xlsx = np.genfromtxt('test_data.csv', usecols=13, skip_header=3, delimiter=',')
        # v_ratio_im_xlsx = np.genfromtxt('test_data.csv', usecols=12, skip_header=3, delimiter=',')




        # freq = np.genfromtxt('test_data.csv', usecols=0, skip_header=3, delimiter=',')
        # v_meas = np.genfromtxt('test_data.csv', usecols=8, skip_header=3, delimiter=',')
        # v_ref = np.genfromtxt('test_data.csv', usecols=6, skip_header=3, delimiter=',')
        # v_meas_im = np.genfromtxt('test_data.csv', usecols=9, skip_header=3, delimiter=',')
        # v_ref_im = np.genfromtxt('test_data.csv', usecols=7, skip_header=3, delimiter=',')



        # v_meas_complex = v_meas + 1j * v_meas_im
        # v_ref_complex = v_ref + 1j * v_ref_im

        # v_ratio_complex = v_meas_complex / v_ref_complex
        # v_ratio_re = -1 * (v_ratio_complex.real - 0.0007)
        # v_ratio_im = -1 * (v_ratio_complex.imag + 0.002)


        # print(v_ratio_im_xlsx[:20])
        # print(v_ratio_im[:20])

        # plt.plot(freq[:100], v_ratio_im[:100])
        # plt.show()


        # print('v_ref')
        # print(v_ref[:20])
        # print('v-r-im')
        # print(v_ref_im[:20])
        # print('v-m')
        # print(v_meas[:20])
        # print('v-m-im')
        # print(v_meas_im[:20])

        # path_dict = {
        #         "freq": ["test_data.csv", 0, 4],
        #         "v_meas": ["test_data.csv", 11, 4],
        #         "v_ref": ["test_data.csv", 9, 4],
        #         "v_meas_im": ["test_data.csv", 12, 4],
        #         "v_ref_im": ["test_data.csv", 10, 4]
        #     }

        # test = load_data(path_dict, analytic=True)
        # print(test)


    if test2:
    
        #  TESTING RECONSTRUCTION  #
        v_arr = np.array([1,2,3,3,3,3,3,4,4,4,4,4,3,2,1,1,2,2,2,1,1,1,2,2,1])
        x_val = np.arange(len(v_arr))
        v_unc = 0.01
        uv_arr = unumpy.uarray(v_arr, v_unc)

        b = reconstruct(uv_arr, a=ufloat(1,0.1), tau_s=ufloat(1,0.1), t_step=0.1)

        b_val = unumpy.nominal_values(b)
        b_err = unumpy.std_devs(b)

        fig, ax = plt.subplots(1, 1)
        ax = plot_with_error(ax, x_val,b_val, b_err, xlabel='foo')
        plt.show()
