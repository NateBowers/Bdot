import json


name = 'test2'
with open('calibration.json', 'r') as file:
    data = json.load(file)
    for probe in data:
        if probe['Probe name'] == name:
            foo = probe['Calibration data']['x axis']
            bar = probe['Calibration data']['y axis']
            baz = probe['Calibration data']['z axis']
            # x_vals = probe['Calibration Data']['x axis']
            # y_vals = probe['Calibration data']['y axis']
            # z_vals = probe['Calibration data']['z axis']
        else:
            pass


print(foo)
print(bar)
print(baz)

    # probe_names = [probe['Probe name'] for probe in data
    # print(probe_names)