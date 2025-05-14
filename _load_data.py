"""
    Two different methods for loading bdot calibration data. All output
    a tuple in the form (freq, v_ratio_re, v_ratio_im, factor) where freq
    is angular frequency and factor is in units of s * m^-2. 

    implimented load functions:
        load_data_UCLA
        load_data_Oxford

    load_data_UCLA was designed for the Meinecke Group's 2025 experiment
    at UCLA's PHOENIX laser facility
"""

import numpy as np
from scipy.signal import hilbert
import math
import json

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
    raise NotImplementedError('deprecated')
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


def load_data_Oxford(path, layout):
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
        

    return freq, v_ratio_re, v_ratio_im, factor



def load_data_UCLA(data_path: str, 
                   config_path: str='config.json',
                   mag_unit: str='m', 
                   phase_unit: str='deg') -> tuple:
    """Load data from oscilliscope readout given from UCLA's system.
       It's assumed the system has already divided v_measured and
       v_reference to generate v_ratio. The path should point to a
       txt file with the first 15 lines being header, the first
       column being frequency, the second magnitude, and the third
       phase. Also calculates the constant factor from calibration
       setup configuration.

    Args:
        data_path (str): Path to .txt file where data is stored
        config_path (str): Path to .json file with configuration information
        mag_unit (str, optional): Units for magnitude data. Defaults to 'm'.
        phase_unit (str, optional): Units for phase data. Defaults to 'deg'.

    Raises:
        NotImplementedError: No mag units besides milli ('m') are currently supported
        ValueError: The only supported phase units are deg and rad

    Returns:
        tuple: (angular frequence, real(v_ratio), imaginary(v_ratio), factor) 
    """
    freq, mag, phase = np.genfromtxt(data_path, skip_header=15).T

    freq_ang = freq * 2 * np.pi

    if mag_unit == 'm':
        mag_for_calc = mag / 1000
    elif mag_unit != 'm':
        raise NotImplementedError('Magnitude units besides milli are not yet supported')
    
    if phase_unit == 'deg':
        phase_for_calc = -phase * np.pi / 180
    elif phase_unit == 'rad':
        phase_for_calc = phase
    else:
        raise ValueError('Phase unit must be "deg" or "rad"')

    v_re = mag_for_calc * np.cos(phase_for_calc)
    v_im = mag_for_calc * np.sin(phase_for_calc)

    with open(config_path, 'r') as file:
        data = json.load(file)
    mu_0 = 4 * np.pi * 10e-7    # vacuum permeability, kg * m * s^-2 * A^-2
    g = data['g']               # amp gain, unitless
    N = data['N']               # num probe loops, unitless
    R_p = data['R_p']           # resistor measured across, kg * m^2 * s^-3 * A^-2
    r = data['r']               # helmholtz coil radius, m

    factor = (g * N * mu_0 * 16) / (R_p * r * (5**1.5))     # s * m^-2

    return freq_ang, v_re, v_im, factor
