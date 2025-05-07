import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import matplotlib

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


if __name__ == '__main__':
 
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
