import tkinter
from tkinter.messagebox import showerror
import numpy as np
import scipy

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure


'''
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

'''

def bad():
        raise Exception("I'm Bad!")

# any name as accepted but not signature
def report_callback_exception(self, exc, val, tb):
    showerror("Error", message=str(val))

tkinter.Tk.report_callback_exception = report_callback_exception

def model_func(x, a, b):
    return a * np.sin(b * x)
np.random.seed(0)
xdata = np.linspace(0, 15, 100)
y = model_func(xdata, 2.5, 0.8)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
popt, pcov = scipy.optimize.curve_fit(model_func, xdata, ydata, method='dogbox')
# Getting the optimized parameters
a_opt, b_opt = popt
print(f'Optimized parameters: a = {a_opt}, b = {b_opt}')



root = tkinter.Tk()
root.wm_title("Embedded in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot()


t = np.arange(0, 3, .01)
ax = fig.add_subplot()
line, = ax.plot(t, 2 * np.sin(2 * np.pi * t))
ax.set_xlabel("time [s]")
ax.set_ylabel("f(t)")

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)


def update_frequency(new_val):
    # retrieve frequency
    f = float(new_val)

    # update data
    y = 2 * np.sin(2 * np.pi * f * t)
    line.set_data(t, y)

    # required to update canvas and attached toolbar!
    canvas.draw()


slider_update = tkinter.Scale(root, from_=1, to=5, orient=tkinter.HORIZONTAL,
                              command=update_frequency, label="Frequency [Hz]")

# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tkinter.BOTTOM)
slider_update.pack(side=tkinter.BOTTOM)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

tkinter.Button(master=root, text="bad", command=bad).pack()


tkinter.mainloop()