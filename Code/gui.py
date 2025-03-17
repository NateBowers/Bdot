# import tkinter as tk

# window = tk.Tk()
# window.geometry("300x200") 

# label = tk.Label(
#     text="Hello, Tkinter",
#     foreground="black",  # Set the text color to white
#     background="white"  # Set the background color to black
# )
# label.pack()
# window.mainloop()

import tkinter as tk

def foo():
    print("Hello world")


class App(tk.Tk):  # define a class that inherits from Tk
    def __init__(self):  # this method runs when you instantiate your App class
        super().__init__()  # initialize Tk (similar to 'root = tk.Tk()' above)
        # create a label widget that's a member of this App class
        # self.label = tk.Label(self, text='Hello, World!')
        # self.label.pack()
        self.geometry("300x200")
        self.tk_setPalette('light gray')
        # self.button = tk.Button(text='foo', command=foo())
        # self.button.pack()

        self.menu = tk.Menu(self)
        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="New")
        file_menu.add_command(label="Open")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        self.menu.add_cascade(label="File", menu=file_menu)

        self.config(menu=self.menu)


if __name__ == '__main__':
    root = App()  # instantiate your class - this runs your __init__ method
    root.mainloop()  # run the application 
    # instances of App have access to 'mainloop' because App inherits from Tk,
    # just like 'root' in the first example!
