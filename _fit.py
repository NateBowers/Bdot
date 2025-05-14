"""
    FitProbe object
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from _load_data import load_data_UCLA


class FitProbe:
    def __init__(self, freq, v_ratio_re, v_ratio_im, factor, name):
        self.freq = freq
        self.re = v_ratio_re
        self.im = v_ratio_im
        self.factor = factor
        self.name = name

    def __re_curve_meinecke(self, w, a, tau, tau_s):
        """ Real component of Vmeas/Vref """
        y = self.factor * (a * (w ** 2) * (tau_s - tau)) / (1 + (tau_s * w) ** 2)
        return y

    def __im_curve_meinecke(self, w, a, tau, tau_s):
        """ Imaginary component of Vmeas/Vref"""
        y =  self.factor * (a * tau * tau_s * (w**3) + a * w) / (1 + (tau_s * w) ** 2)
        return y

    def __re_curve_everson(w, a, tau, tau_s):
        raise NotImplementedError

    def __im_curve_everson(w, a, tau, tau_s):
        raise NotImplementedError


    def __objective(self, params, x, y1, y2, y1_var, y2_var):
        """Objective function to be minimized"""
        a = params['a']
        tau = params['tau']
        tau_s = params['tau_s']


        resid1 = (y1 - self.curve_func_re(x, a, tau, tau_s)) / y1_var
        resid2 = (y2 - self.curve_func_im(x, a, tau, tau_s)) / y2_var
        
        return np.concatenate((resid1, resid2))
    
    def calibrate(self,
                  clip_low: int=100,
                  clip_high: int=-1,
                  curve_func: str='meinecke',
                  verbose: bool=False):
        if curve_func == 'meinecke':
            self.curve_func_re = self.__re_curve_meinecke
            self.curve_func_im = self.__im_curve_meinecke

        elif curve_func == 'everson':
            self.curve_func_re = self.__re_curve_everson
            self.curve_func_im = self.__im_curve_everson
        
        
        self.re = self.re[clip_low:clip_high]
        self.im = self.im[clip_low:clip_high]
        self.freq = self.freq[clip_low:clip_high]

        re = self.re
        im = self.im
        freq = self.freq

        re_var = np.ones(len(re))
        im_var = np.ones(len(re))
        exp_dict = {
            'tau_true': 6e-08,
            'tau_s_true' : -1e-08,
            'a_true' : 3e-06,
        }
        params = lmfit.Parameters()
        params.add('a', value=exp_dict['a_true'])
        params.add('tau', value=exp_dict['tau_true'])
        params.add('tau_s', value=exp_dict['tau_s_true'])
        result = lmfit.minimize(self.__objective, params, args=(freq, re, im, re_var, im_var))

        self.result = result

        if verbose:
            print(lmfit.fit_report(result))

        a_calib, tau_calib, tau_s_calib = result.params.valuesdict().values()

        self.a = a_calib
        self.tau = tau_calib
        self.tau_s = tau_s_calib

        return a_calib, tau_calib, tau_s_calib


    def graph(self, show: bool=True, save: bool=False, save_path = None):
        fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
        fig.set_size_inches(15, 5)
        for ax in axes:
            ax.set_xlabel('omega')
            ax.set(adjustable='datalim')


        ax1, ax2 = axes
        ax1.plot(self.freq, self.re, 'b-', label='Re(Vm/Vr)')
        ax1.plot(self.freq, self.curve_func_re(self.freq, self.a, self.tau, self.tau_s), 'r-', label='Re(Best Fit)')
        ax1.set_ylabel('Re(V_meas/V_ref)')

        ax2.plot(self.freq, self.im, 'b-', label='Im(Vm/Vr)')
        ax2.plot(self.freq, self.curve_func_im(self.freq, self.a, self.tau, self.tau_s), 'r-', label='Im(Best Fit)')
        ax2.set_ylabel('Im(V_meas/V_ref)')

        plt.figtext(
            0.05,        # x
            0.2,       # y
            lmfit.fit_report(self.result), 
            horizontalalignment='left', 
            fontsize=8, 
            fontfamily='monospace'
        )

        fig.subplots_adjust(left=0.4)    
        plt.figlegend(loc='upper left')
        
        if save and not save_path:
            plt.savefig(self.name, dpi=300)
        elif save and save_path:
            plt.savefig(save_path, dpi=300)

        if show:
            plt.show()


if __name__ == '__main__':
    freq, v_ratio_re, v_ratio_im, factor = load_data_UCLA('test_data_3/BXPX_2.TXT')
    probe = FitProbe(freq, v_ratio_re, v_ratio_im, factor, 'test')
    params = probe.calibrate(verbose=True)
    probe.graph()
    print(params)












