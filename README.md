# About

This repo has code for calibrating bdot probes. 

This was written and tested with Python 3.13.1, but it will likely work in other current python versions. 

# Install

1. Make sure you have python and github installed with brew (instructions to install brew are [here](https://brew.sh/)
   1. In a terminal window, run `brew install git`
2. Clone the git repo (MAC)
   1. Navigate to the directory you want to clone the repo to in a terminal window
   2. Run `git clone https://github.com/NateBowers/Bdot.git`
3. Run `pip install -r requirements.txt` 
4. Run `Code/main.py`
   1. It is recommended you run this in an IDE rather than straight in a terminal. I like VSCode
5. Add data to `Data/`
   1. Create a new test file for each test
   2. Change the file locations in `main.py` as needed (do **not** push those changes to the repo)
   3. Check `Test1/` and `Test2/` for examples of how to format data


