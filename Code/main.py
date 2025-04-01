import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


# Helper Functions

def convert_to_volt(arr: np.array, init_unit:str):
    if init_unit.lower() == 'volt':
        return arr
    elif init_unit.lower() == 'dbmv':
        return 10 ** ((arr - 60) / 20)
    else:
        raise NotImplementedError('Support for units besides "volt" and "dBmV" are not yet implimented')

def dBmV_to_volt(arr: np.array):
    """Convert an array of values from dBmV to volts."""
    return 10 ** ((arr - 60) / 20)


# for freq, v_meas, v_ref (either magnitude or complex), include path and column

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

def load_data(path_dict: dict, 
              analytic: bool = False, 
              units: str = 'dbmV', 
              graph: bool = False, 
              save_graph: bool = False, 
              save_path: str = None) -> list[np.array, np.array]:

    if not analytic:
        """Loads data, takes Hilbert transfor to get re, im parts, options for unit conversion"""
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
        """Loads data, options for unit conversion"""
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

    return v_ratio_re, v_ratio_im


# load_data(path_dict_1, graph=True, save_graph=True, save_path='Data/Test1/plot')

load_data(path_dict_2, analytic=True, units='volt', graph=True, save_graph=True, save_path='Data/Test3/plot3')

