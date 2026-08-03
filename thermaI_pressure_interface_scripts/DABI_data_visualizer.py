import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
import matplotlib.dates
from pathlib import Path
import io
import re
import gc

def DABI_time_plot(param, selectarray, plot_ax, time, DABIarray, style='.'):
    if param == 'dimple_pressure':
        plot_ax.set(ylabel="Pressure (PSIG)")
        for dimple in selectarray:
            pressure_dimple_key = dimple + ": Press.(PSIG)"
            plot_ax.plot(time, DABIarray.loc[:, pressure_dimple_key], style, label=pressure_dimple_key)
            
    elif param == 'dimple_temp':
        plot_ax.set(ylabel="IR Temperature (C)")
        for dimple in selectarray:
            temp_dimple_key = dimple + ": Temp.(C)"
            plot_ax.plot(time, DABIarray.loc[:, temp_dimple_key], style, label=temp_dimple_key)
            
    elif param == 'line_pressure':
        plot_ax.set(ylabel="Line Pressure (PSIG)")
        plot_ax.plot(time, DABIarray.loc[:, "High Acc. Transducer (PSIG)"], style, label='Line Pressure')
        
    elif param == 'chamber_pressure':
        plot_ax.set(ylabel="Chamber Pressure (Torr)")
        plot_ax.plot(time, DABIarray.loc[:, "Chamber Press(Torr)"], style, label='Chamber Pressure')
        
    elif param == 'RF-Tank(V)':
        plot_ax.set(ylabel='Voltage')
        plot_ax.plot(time, DABIarray.loc[:, "RF-Tank(V)"], style, label='RF generator voltage')
        
    elif param == 'TC_temp':
        plot_ax.set(ylabel="Thermocouple Temperatue (C)")
        for thermocouple in selectarray:
            TC_dimple_key = 'Thermocouple' + str(thermocouple)
            plot_ax.plot(time, DABIarray.loc[:, TC_dimple_key], style, label=('Thermocouple ' + str(thermocouple)))
    elif param == '':
        # if blank argument is passed, do nothing
        pass
    else:
        raise KeyError("Error! Invalid data output type!")

# datetime format used for high-temp DABI entries
DABI_datetime_format = r'%m/%d/%Y  %H:%M:%S.%f'

# Base RADICAL directory
RADICAL_data_path = r'D:\OneDrive\Northwestern University\DARPA RADICAL - RADICAL Documents'

# Specimen, experiment
specimen = r'RAD-BD-005'
experiment = r'(250818) Creep + Static Experiment'
# if True, will merge all csv files and save them in one directory.
save_merged_file = True

# Find all .csv files. NOTE, must be in local OS style alphabetical order.
base_dir = os.path.join(RADICAL_data_path, r'Data', r'Specimen Directory', r'Bonded_DABI', specimen, experiment, r'Test Data')
all_dir_names = os.listdir(base_dir)
csvnames = []
first_timestamp_entry = []
for dir_filename in all_dir_names:
    if dir_filename.endswith("_record.csv"):
        # Add list to entry
        csvnames.append(dir_filename)
        
        # Read first timestamp if valid csv file
        path_itr = os.path.join(base_dir, dir_filename)
        first_timestamp_entry.append(datetime.strptime(pd.read_csv(path_itr, skiprows=1, nrows=1, usecols=['time']).iloc[0, 0], DABI_datetime_format))

# Sort the list of filenames by the timestep of the first entry in each array
csvnames = np.array(csvnames)[np.argsort(first_timestamp_entry)]

data_dfs = []
for itr, csvname_itr in enumerate(csvnames):
    
    # Location of DABI capture data
    path_to_DABI_data = os.path.join(base_dir, csvname_itr)

    # Load data and add to dataframe
    data_dfs.append(pd.read_csv(path_to_DABI_data, skiprows=1, low_memory=False))

# Merge all dataframes and merge indices
DABI_data = pd.concat(data_dfs, ignore_index=True)

# remove original list of dataframes from memory
del data_dfs
gc.collect

# Which dimples or thermocoup1es?
left_select_array = ['D1', 'D2', 'D3', 'D4', 'D5']
'''
left_select_array = ['A1', 'A2', 'A3', 'A4', 'A5',
                     'B1', 'B2', 'B3', 'A4', 'B5',
                     'C1', 'C2', 'C3', 'C4', 'C5',
                     'D1', 'D2', 'D3', 'D4', 'D5',
                     'E1', 'E2', 'E3', 'E4', 'E5']
'''
right_select_array = [5]

# Parameters - select from: local parameters:
#   dimple_pressure
#   dimple_temp
# anThermocouplel parameters:
#   TC_temp,
#   line_pressure,
#   chamber_pressure

# Note that 'Dimple parameters are plotted for all dimples in dimple_select_array
leftparam = "dimple_pressure"
rightparam = ""

# Times - in local time, depending on the time of day, including the date
plot_start_str = '8/18/2025  18:50:0.0'
plot_end_str = '8/18/2025  19:20:00.0'

# time format - options local, GMT 0, or elapsed.
time_format = 'local'

# offset mode for elapsed mode - either 'rezero' or 'shift'. If 'rezero', change to elapsed with
# this at the start time. If 'shift', move all times by this number.
offset_mode = 'rezero'
timestamp_offset = '09:49:27.243767'

## IDENTIFY TIME STAMPS
# convert main array to datetime
DABI_timestamps_datetime = []

# Start and end time in datetime format
for itr, timestr in enumerate(DABI_data.time):
    try:
        DABI_timestamps_datetime.append(datetime.strptime(timestr, DABI_datetime_format))
    except TypeError:
        ## todo: write this block more elegantly
        ## for some reason the excel file is getting loaded with a bunch of NaNs 
        ## ideally fix this
        pass
        

# First and last timesteps in array
DABI_timestamps_first = DABI_timestamps_datetime[0]
DABI_timestamps_last = DABI_timestamps_datetime[-1]

# Convert requested start & stop times to datetime format
plot_start_datetime =  datetime.strptime(plot_start_str, DABI_datetime_format)
plot_end_datetime = datetime.strptime(plot_end_str, DABI_datetime_format)

# Find start and end indices
start_idx = np.searchsorted(DABI_timestamps_datetime, plot_start_datetime, side='right')
end_idx = np.searchsorted(DABI_timestamps_datetime, plot_end_datetime, side='right')

# Empty data error
if end_idx - start_idx == 0:
    raise ValueError("No data to plot!")

# Save merged csv file
if save_merged_file:
    try: DABI_data.to_csv(os.path.join(base_dir, "merged_file", "merged.csv"), index=False)
    except:
        os.mkdir(os.path.join(base_dir, "merged_file"))
        DABI_data.to_csv(os.path.join(base_dir, "merged_file", "merged.csv"), index=False)
        
# Reduce size of array to selected time interval
DABI_data_reduced = DABI_data[start_idx:end_idx]
time_reduced_datetime = DABI_timestamps_datetime[start_idx:end_idx]

# This is where the changes would be made to the time array
## PLOTTING
timefig, left_ax = plt.subplots()
left_ax.grid(which='both')
right_ax = left_ax.twinx()

# Right hand plot
DABI_time_plot(rightparam, right_select_array, right_ax, time_reduced_datetime, DABI_data_reduced, style='k*')

# Left hand plot
DABI_time_plot(leftparam, left_select_array, left_ax, time_reduced_datetime, DABI_data_reduced)

# X Axis formatting
left_ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) 
left_ax.set(xlabel="Local Time")

# legend
left_ax.legend(loc='best')
right_ax.legend(loc='best')
plt.show()