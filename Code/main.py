import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


if True:
    f, v_ref = np.genfromtxt("Data/Test1/V_ref.csv", dtype=float, skip_header=1, delimiter=",").T
    f_meas, v_meas = np.genfromtxt("Data/Test1/V_meas.csv", dtype=float, skip_header=1, delimiter=",").T

    try:
        np.array_equal(f, f_meas)
    except:
        raise ValueError('Measured and reference data must be measured at the same frequency points')

    v_ref_analytic = hilbert(v_ref)
    v_ref_re = np.real(v_ref_analytic)
    v_ref_im = np.imag(v_ref_analytic)

    v_meas_analytic = hilbert(v_meas)
    v_meas_re = np.real(v_meas_analytic)
    v_meas_im = np.imag(v_meas_analytic)


else:
    f, v_ref_re, v_ref_im, v_meas_re, v_meas_im = np.genfromtxt('Data/Test2/fullData.csv', delimiter=',', skip_header=3, usecols=[0,1,2,3,4]).T
    # f, v_ref_im = np.genfromtxt("Data/Test2/v_ref_im.csv", delimiter=',', skip_header=1).T

    v_ref_analytic = v_ref_re + 1j*v_ref_im
    v_meas_analytic = v_meas_re + 1j*v_meas_im



v_ratio_analytic = v_meas_analytic / v_ref_analytic
v_ratio_re = np.real(v_ratio_analytic)
v_ratio_im = np.imag(v_ratio_analytic)

im_offset = np.imag(v_ratio_analytic[0])
re_offset = np.real(v_ratio_analytic[0])


C = 200

fig, ax = plt.subplots(1,2)
ax[0].plot(f[0:C], v_ratio_im[0:C]-im_offset)
ax[0].set_ylabel("Im(V_meas/V_ref)")
ax[0].set_box_aspect(1)
ax[1].plot(f[0:C], v_ratio_re[0:C]-re_offset)
ax[1].set_ylabel("Re(V_meas/V_ref)")
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_box_aspect(1)
plt.show()


