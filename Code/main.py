import numpy as np
from scipy.signal import hilbert
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Helper Functions

def convert_to_volt(arr: np.array, init_unit:str) -> np.array:
    """Convert array to volts"""
    if init_unit.lower() == 'volt':
        return arr
    elif init_unit.lower() == 'dbmv':
        return 10 ** ((arr - 60) / 20)
    else:
        raise NotImplementedError('Support for units besides "volt" and "dBmV" are not yet implimented')
    
def re_curve(w, a, tau, tau_s):
    const = 1.67652E-07
    """ Real component of Vmeas/Vref """
    y = const * (a * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
    return y

def im_curve(w, a, tau, tau_s):
    const = 1.67652E-07
    """ Imaginary component of Vmeas/Vref"""
    y = const * (a * ((w ** 3) * tau * tau_s + w)) / (1 + (tau_s * w) ** 2)
    return y




# Main Functions

def load_data(path_dict: dict, 
              analytic: bool = False, 
              units: str = 'BbmV', 
              graph: bool = False, 
              save_graph: bool = False, 
              save_path: str = None) -> list[np.array, np.array, np.array]:
    """Load calibration data and calculate the real and imaginary parts of
        v_meas/v_ref in terms of volts. Also converts frequency to angular
        frequency

    Args:
        path_dict (dict): Info on where freq, v_meas, v_ref, and (optional)
                        v_meas_im, v_ref_im is stored and which columns to use
        analytic (bool, optional): Does the data contain imaginary information. 
                        Defaults to False.
        units (str, optional): Units of the data, either 'dBmV' or 'volt. 
                        Defaults to 'dBmV'.
        graph (bool, optional): Defaults to False.
        save_graph (bool, optional): Defaults to False.
        save_path (str, optional): Path to save the graph to. Defaults to None.

    Raises:
        TypeError: Check save_path to ensure it is a string

    Returns:
        list[np.array, np.array, np.array]: Re(V_meas/V_ref), Im(V_meas/V_ref), omega
    """

    if not analytic:
        freq = np.genfromtxt(path_dict["freq"][0], usecols=path_dict["freq"][1], skip_header=path_dict["freq"][2], delimiter=',')
        v_meas = np.genfromtxt(path_dict["v_meas"][0], usecols=path_dict['v_meas'][1], skip_header=path_dict["v_meas"][2], delimiter=',')
        v_ref = np.genfromtxt(path_dict["v_ref"][0], usecols=path_dict["v_ref"][1], skip_header=path_dict["v_ref"][2], delimiter=',')

        v_meas = convert_to_volt(v_meas, units)
        v_ref = convert_to_volt(v_ref, units)

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
    v_ratio_re, v_ratio_im = np.real(v_ratio), np.imag(v_ratio)
    
    if graph:
        re_offset, im_offset = v_ratio_re[0], v_ratio_im[0]

        C = 200
        fig, ax = plt.subplots(1,2, figsize=(10,10))
        ax[0].plot(freq[0:C], v_ratio_im[0:C]-im_offset)
        ax[0].set_ylabel("Im(V_meas/V_ref)")
        ax[0].set_box_aspect(1)
        ax[1].plot(freq[0:C], v_ratio_re[0:C]-re_offset)
        ax[1].set_ylabel("Re(V_meas/V_ref)")
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_box_aspect(1)

        if save_graph:
            if type(save_path) is not str:
                raise TypeError('save_path must be a string')
            path = save_path+'.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')

        plt.show()

    return v_ratio_re, v_ratio_im, freq * 2 * np.pi


def calc_fit_params(v_ratio_re, v_ratio_im, omega, sys_params):
    mu_0 = 1
    # const = (16 * sys_params['N'] * sys_params["g"] * mu_0) / (sys_params["r"] * sys_params["Rp"] * (5**(1.5)))
    const = 1.67652E-07
    omega_combo = np.hstack((omega, omega))
    v_ratio_combo = np.hstack((v_ratio_re, v_ratio_im))

    def combined_curves(combo_w, a, tau, tau_s):
        """ Combine real and imaginary parts into one function"""

        extract_re = combo_w[:len(v_ratio_re)]
        extract_im = combo_w[len(v_ratio_im):]

        result_re = re_curve(extract_re, a, tau, tau_s)
        result_im = im_curve(extract_im, a, tau, tau_s)

        return np.append(result_re, result_im)
    
    guess = [1,1,1]
    popt, pcov = curve_fit(combined_curves, omega_combo, v_ratio_combo, p0=guess)

    print(popt)

    return const


if __name__ == "__main__":

    path_dict_1 = {
        "freq": ["Data/Test1/V_meas.csv", 0, 1],
        "v_meas": ["Data/Test1/V_meas.csv", 1, 1],
        "v_ref": ["Data/Test1/V_ref.csv", 1, 1]
    }

    path_dict_2 = {
        "freq": ["Data/Test2/fullData.csv", 0, 3],
        "v_meas": ["Data/Test2/fullData.csv", 3, 3],
        "v_meas_im": ["Data/Test2/fullData.csv", 4, 3],
        "v_ref": ["Data/Test2/fullData.csv", 1, 3],
        "v_ref_im": ["Data/Test2/fullData.csv", 2, 3]
    }

    path_dict_3 = {
        "freq": ["Data/111219_Bdot1_Bx_calibration_9kHz_to_500MHz_0dBm_7.52V/fullData.csv", 0, 3],
        "v_meas": ["Data/111219_Bdot1_Bx_calibration_9kHz_to_500MHz_0dBm_7.52V/fullData.csv", 3, 3],
        "v_meas_im": ["Data/111219_Bdot1_Bx_calibration_9kHz_to_500MHz_0dBm_7.52V/fullData.csv", 4, 3],
        "v_ref": ["Data/111219_Bdot1_Bx_calibration_9kHz_to_500MHz_0dBm_7.52V/fullData.csv", 1, 3],
        "v_ref_im": ["Data/111219_Bdot1_Bx_calibration_9kHz_to_500MHz_0dBm_7.52V/fullData.csv", 2, 3]
    }

    setup_params = {"N":16, "g": 1, "r":1, "Rp":0.22}

    # load_data(path_dict_2, analytic=True, units='volt', graph=True, save_graph=True, save_path='Data/Test3/plot3')
    # load_data(path_dict_1, graph=True, save_graph=True, save_path='Data/Test1/plot')

    re, im, freq = load_data(path_dict_2, analytic=True, units='volt')
    calc_fit_params(re, im, freq, setup_params)