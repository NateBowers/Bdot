import argparse
import os
from pathlib import Path
import numpy as np
from _load_data import load_data_UCLA
from _fit import FitProbe


# Take command line arguments
parser = argparse.ArgumentParser(description='calculates bdot calibration parameters a, tau, and tau_s')
parser.add_argument('data_path', help='path to file or dir where data is stored')
parser.add_argument('output_dir', help='directory to store output data')
parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite data in output_dir (default: False)')
parser.add_argument('-v', '--verbose', action='store_true', help='print fit info for each probe (default: False)')

args = parser.parse_args()
data_path = args.data_path
output_dir = args.output_dir

print('Beginning calibration routine...')



# Checks if output_dir exists and if overwrite flag is set
if os.path.isdir(output_dir):
    if len(os.listdir(output_dir)) == 0:
        pass
    elif args.overwrite:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir,file))
    else:
        raise OSError(f'{output_dir} is not empty, set -o flag to overwrite its contents')
else:
    os.mkdir(output_dir)


# Make list of each data file to calibrate
paths = []
if os.path.isfile(data_path):
    paths.append(data_path)
elif os.path.isdir(data_path):
    for file in os.listdir(data_path):
        if file.endswith('TXT'):
            paths.append(os.path.join(data_path, file))

# Runs calibration code on each path and saves it to output_dir
for path in paths:
    name, _ = os.path.splitext(os.path.basename(path))
    save_path = output_dir + '/' + name + '.png'
    freq, v_ratio_re, v_ratio_im, factor = load_data_UCLA(path)
    probe = FitProbe(freq, v_ratio_re, v_ratio_im, factor, os.path.basename(path))
    probe.calibrate(verbose=args.verbose)
    probe.graph(show=False, save=True, save_path=save_path)

print(f'Successfully calibrated all probes in {data_path} and stored results in {output_dir}')
