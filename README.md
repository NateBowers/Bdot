# Bdot Calibration Code

`calibrate.py` provides a command line method for calculating the calibration parameters—a, τ, and τ<sub>s</sub>—for a bdot probe. For each probe dataset, this program generates an image with calibration parameters and the fit graphed against the data. 

It has two required arguments: 
* `data_path` gives the data file or directory containing data to run the calibration code on. 
* `output_dir` gives the directory to store the generated images
  
They can be either relative paths to directory `calibrate.py` is executed in or absolute paths.
  
There are also two optional flags:
* `-o` overwrite any data in `outout_dir`
* `-v` verbose output



## Install and setup

1. Make sure you have python installed on your computer. 
2. Install git on your computer. Instructions can be found [here](https://github.com/git-guides/install-git)
3. In the command line to the directory you want to install and run the calibration code in
4. Run the following code in the command line to clone the git repo and install the required python packages
```
    git clone https://github.com/NateBowers/bdot-calibration.git

    pip install -r requirements.txt
```

## Test calibration

To run a test calibration, run the following command in terminal
```
    python calibrate.py test_data/ output_directory/ 
```
If that does not work, replace `python` with `python3`. If that does not work, check what the alias for python on your system is. 


## Actual calibration

To run the calibration code, edit `config.json` to match the actual configuration of the system. Make sure the data is formatted in the same way as in the `test_data/` folder. In particular, this program the following for each data set:
* It is stored in a .TXT file,
* The first 15 rows are header information, 
* The first column gives frequency in Hz,
* The second column gives the magnitude of (v_meas/v_ref) in mU where U indicates unitless,
* The third column gives the phase of (v_meas/v_ref_) in degrees, and
* The setup defined in `config.json` is the same for each data set in the folder.

