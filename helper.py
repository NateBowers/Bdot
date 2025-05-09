import numpy as np
from scipy.signal import hilbert
import math
from pprint import pprint


def load_data(path_dict: dict, 
              analytic: bool = False,
              re_offset: float = 0,
              im_offset: float = 0,
             ) -> list[np.array, np.array, np.array]:
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



def load_oxford_data(path, layout):
    data = np.genfromtxt(path, delimiter=',', usecols=np.arange(0,19), skip_header=0)
    
    N = data[2,17]
    tau_true = data[3,17]
    g = data[4,17]
    if math.isnan(g):
        g = 0.5
    R_p = data[5,17]
    mu_0 = data[6,17]
    r = data[7,17]
    tau_s_true = data[8,17]
    a_true = data[9,17]
    re_offset = data[13,17]
    im_offset = data[14,17]

    factor = (N * 16 * mu_0) / (r * R_p * (5**1.5))


    experiment_params = {
        'tau_true':tau_true,
        'factor': factor,
        'tau_s_true' : tau_s_true,
        'a_true' : a_true,
    }
    

    if layout == 9:
        freq   = data[3:,0]
        v_r_re = data[3:,4]
        v_r_im = data[3:,5]
        v_m_re = data[3:,6]
        v_m_im = data[3:,7]

        v_r_co = v_r_re + 1j* v_r_im
        v_m_co = v_m_re + 1j* v_m_im
        v_ratio = v_m_co / v_r_co
        v_ratio_re, v_ratio_im = np.real(v_ratio)+re_offset, np.imag(v_ratio)+im_offset

    elif layout == 1:

        freq   = data[3:,0]
        v_r_re = data[3:,6]
        v_r_im = data[3:,7]
        v_m_re = data[3:,8]
        v_m_im = data[3:,9]

        v_r_co = v_r_re + 1j* v_r_im
        v_m_co = v_m_re + 1j* v_m_im
        v_ratio = v_m_co / v_r_co
        v_ratio_re, v_ratio_im = -(np.real(v_ratio)+re_offset), -(np.imag(v_ratio)+im_offset)
        

    return freq, v_ratio_re, v_ratio_im, experiment_params


if __name__ == '__main__':

    path = 'test_data/test_data.csv'
    # print(load_oxford_data(path,9))
    params = load_oxford_data(path,1)[3]

    print(params['a_true']*params['factor'])
    # pprint(load_oxford_data(path,9)[3], indent=4)

