import tkinter as tk
from tkinter import filedialog, Text

# Function to open a file dialog and display the contents if it's a text file
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        print(f"Selected file: {file_path}")

# Create the main window
root = tk.Tk()
root.title("Tkinter Window")

# Set the window size
root.geometry("400x300")

# Add a frame to contain the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# Add a button to open the file dialog
open_button = tk.Button(button_frame, text="Open File", command=open_file)
open_button.pack(side=tk.LEFT, padx=10)

# Add a button to close the application
close_button = tk.Button(button_frame, text="Close", command=root.destroy)
close_button.pack(side=tk.LEFT, padx=10)

# Add a Text widget to display the file contents
text_widget = Text(root, wrap='word')
text_widget.pack(expand=True, fill='both')

# Run the application
root.mainloop()