import os
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import matplotlib.pyplot as plt

# 1) Rotate coordinates
# Create helper function to rotate from DABI coordinates to specimen coordinates based on bond orientation, bond side, and chamber orientation
def rotate_coordinates(print_coord, bond_angle, chamfer_side, chamber_orientation):
    
    # Bond orientation takes one of four values: '0', '90', '180', '270' - rotation in degrees clockwise relative to the plenum.
    # Chamfer side takes one of two values: 'CL' or 'CR' - to determine, rotate until chamfer-round side is up.
    #   'CL' means the chamfer is on the left, round right. 'CR' means chamfer is on the right, round left.
    # Chamber orientation takes one of two values: 'U' or 'D' - 'U' means the chamfer-round side is up (facing OUT towards edge of table),
    #   'D' means the chamfer-round side is down (facing IN towards center of table)
    # Coordinates are given as strings, e.g. 'I1', 'J3', etc.
    
    # Create mapping of coordinates to (x, y) 0-indexed positions in 5x5 array centered on middle of DABI
    coord_map = {
        'I1': (-2, 2), 'I2': (-1, 2), 'I3': (0, 2), 'I4': (1, 2), 'I5': (2, 2),
        'J1': (-2, 1), 'J2': (-1, 1), 'J3': (0, 1), 'J4': (1, 1), 'J5': (2, 1),
        'K1': (-2, 0), 'K2': (-1, 0), 'K3': (0, 0), 'K4': (1, 0), 'K5': (2, 0),
        'L1': (-2, -1), 'L2': (-1, -1), 'L3': (0, -1), 'L4': (1, -1), 'L5': (2, -1),
        'M1': (-2, -2), 'M2': (-1, -2), 'M3': (0, -2), 'M4': (1, -2), 'M5': (2, -2)
    }
    
    # Get initial (x, y) position
    x, y = coord_map[print_coord]
    
    # Flip if chamfer side is 'CR'
    if chamfer_side == 'CR':
        x = -x
    else:
    # otherwise don't change x
        x = x
    
    # Rotate based on bond orientation
    angle = int(bond_angle)
    
    # Rotate using rotation matrix based on bond angle
    if angle == 0:
        x_rot, y_rot = x, y # No rotation
    elif angle == 90:
        x_rot, y_rot = y, -x
    elif angle == 180:
        x_rot, y_rot = -x, -y
    elif angle == 270:
        x_rot, y_rot = -y, x
    else:
        raise ValueError("Invalid bond orientation")
    
    # Rotate based on chamber orientation
    if chamber_orientation == 'OUT':
        x_rot, y_rot = x_rot, y_rot # No change
    elif chamber_orientation == 'IN':
        x_rot, y_rot = -x_rot, -y_rot # Rotate 180 degrees
    else:
        raise ValueError("Invalid chamber orientation")
    
    # Reflect coordinates across y=-x line to convert from specimen to DABI coordinate order (letters become numbers and vice versa)
    x_rot, y_rot = -y_rot, -x_rot
    
    # map back to array coordinates with transposed letter-number system
    inv_coord_map = {v: k for k, v in coord_map.items()}
    new_coord_IM = inv_coord_map[(x_rot, y_rot)]

    # Convert from I-M to A-E
    letter = new_coord_IM[0]
    number = new_coord_IM[1]
    new_letter = chr(ord(letter) - (ord('I') - ord('A')))
    new_coord_IM = f"{new_letter}{number}"

    return new_coord_IM

def process_dimple_locations(file_path):
    # Load the entire excel file
    df_all = pd.read_excel(file_path, sheet_name="55", skiprows=2)  # Load all sheets into a dictionary of DataFrames

    # Iterate through each row and apply rotation to non-empty label columns
    for idx, row in df_all.iterrows():
        bond_angle = row['Bond Angle']
        chamfer_side = row['Chamfer Side']
        chamber_orientation = row['DABI Orientation']
        
        print_coord = row['Printed Coordinate']
        
        if pd.notna(print_coord):  # Check if the cell is not empty
            try:
                new_coord = rotate_coordinates(print_coord, bond_angle, chamfer_side, chamber_orientation)
                df_all.at[idx, 'DABI Coordinate'] = new_coord  # Update the DataFrame with the new coordinate
            except ValueError as e:
                print(f"Error processing row {idx}: {e}")
                
    # Return .csv with updated coordinates
    
    # Exclude all columns except "sample index" through "DABI Coordinate"
    df_all_output = df_all.loc[:, 'Printed Coordinate':'DABI Coordinate']
    output_csv_path = file_path.replace('.xlsx', '__copy_and_paste_into_main_csv.csv')
    df_all_output.to_csv(output_csv_path, index=False)
    print(f"Updated coordinates saved to {output_csv_path}")

def merge_DABI_data(specimen, experiment, RADICAL_data_path=r"V:\\"):
    # Find all .csv files. NOTE, must be in local OS style alphabetical order.
    base_dir = os.path.join(RADICAL_data_path, r'Data', r'Specimen Directory', r'Bonded_DABI', specimen, experiment, r'Test Data')
    all_dir_names = os.listdir(base_dir)
    csvnames = []
    DABI_datetime_format = '%m/%d/%Y %H:%M:%S.%f'  # Example: '08/25/2023 14:30:15.123456'
    first_timestamp_entry = []
    for dir_filename in all_dir_names:
        if dir_filename.endswith("_record.csv"):
            # Add list to entry
            csvnames.append(dir_filename)
            
            # Read first timestamp if valid csv file
            path_itr = os.path.join(base_dir, dir_filename)
            try:
                first_timestamp_entry.append(datetime.strptime(pd.read_csv(path_itr, skiprows=1, nrows=1, usecols=['time']).iloc[0, 0], DABI_datetime_format))
            except Exception as e:
                print(f"Error reading timestamp from {path_itr}: {e}")

    # Sort the list of filenames by the timestep of the first entry in each array
    csvnames = np.array(csvnames)[np.argsort(first_timestamp_entry)]

    data_dfs = []
    for itr, csvname_itr in enumerate(csvnames):
        
        # Location of DABI capture data
        path_to_DABI_data = os.path.join(base_dir, csvname_itr)

        # Load data and add to dataframe
        try:
            data_dfs.append(pd.read_csv(path_to_DABI_data, skiprows=1, low_memory=False))
        except Exception as e:
            ## todo: add header row if missing
            print(f"Error loading {path_to_DABI_data}: {e}")

    # Merge all dataframes and merge indices
    DABI_data = pd.concat(data_dfs, ignore_index=True)
    
    # Save processsed data to new .csv file
    output_csv_path = os.path.join(base_dir, f"{specimen}_{experiment}_merged_DABI_data.csv")
    DABI_data.to_csv(output_csv_path, index=False)

    # remove original list of dataframes from memory
    del data_dfs
    gc.collect()

def ident_pressure_steps(specimen, experiment, thresh=100, RADICAL_data_path=r"V:\\"):
    # function which does the following:
    # 1) loads merged DABI data
    # 2) Identifies start and end times where high-accuracy transducer channel is above a certain threshold (say, 100 psi)
    # 3) Identifies active and inactive dimple locations based on individual dimple pressures
    # 4) Identifies average temperature of each dimple location during active periods
    # 5) informs user if temperature changes more than than a certain threshold (say, 25 degrees C) during active periods
    DABI_datetime_format = '%m/%d/%Y %H:%M:%S.%f'
    # saves information to summary .csv file
    
    # Load merged DABI data
    DABI_data = pd.read_csv(os.path.join(RADICAL_data_path, r'Data', r'Specimen Directory', r'Bonded_DABI', specimen, experiment, r'Test Data', f"{specimen}_{experiment}_merged_DABI_data.csv"))
    
    DABI_timestamps = pd.to_datetime(DABI_data['time'], format=DABI_datetime_format)
    
    # Identify indices where the high-accuracy pressure changes above threshold (say, 100 psi)
    press_data = DABI_data[r'High Acc. Transducer (PSIG)']
    
    # apply 5-point moving average filter to remove incidental drops below threshold
    # press_data = press_data.rolling(window=5, center=True).mean()
    above_thresh = press_data > thresh
    
    # find locations where above_thresh goes from False to True or True to False
    change_points = np.where(above_thresh.astype(int).diff().fillna(0) != 0)[0]
    
    # Find times corresponding to change points
    DABI_change_times = DABI_timestamps.iloc[change_points]
    
    # Preallocate list of zeros the length of change points to indicate if the change point is a start or end and if it was adjusted
    change_point_states = np.zeros(len(change_points), dtype=int)

    # Iterate through change points
    for itr in range(len(change_points)):
        idx = change_points[itr]
        time = DABI_change_times.iloc[itr]
        
        # Determine if change point is start or end
        if above_thresh.iloc[idx]:
            change_point_states[itr] = 0  # Start
        else:
            change_point_states[itr] = 1  # End
        
        # Find difference between change point and second-previous DATA point - this fixes situations where there is a large gap in time between data points
        prev_idx = change_points[itr] - 1
        prev_time = DABI_timestamps.iloc[prev_idx]
        time_diff = (time - prev_time).total_seconds()
        
        # If time difference is greater than 2 seconds, change the change point to the second-previous data point
        if time_diff > 2.0:
            change_points[itr] = idx - 1  # Move change point to previous index
   
    # Sort change point array into start-end pairs
    
    # Collect list of start and end points using change_point_states
    start_points = change_points[change_point_states == 0]
    end_points = change_points[change_point_states == 1]
    
    # Check if both are the same length
    if len(start_points) != len(end_points):
        # report error
        print("Error: Mismatched start and end points detected.")
        # if there are more start points than end points, the last point in DABI_timestamps is an end point
        if len(start_points) > len(end_points):
            end_points = np.append(end_points, len(DABI_timestamps) - 1)
            print("Note: Added end point at end of data at final timestamp in DABI_timestamps.")
            
        # if there are more end points than start points, the first point in DABI_timestamps is a start point
        elif len(end_points) > len(start_points):
            start_points = np.insert(start_points, 0, 0)
            print("Note: Added start point at beginning of data at initial timestamp in DABI_timestamps.")

    # Create list of tuples of start and end points
    raw_start_end_pairs = list(zip(start_points, end_points))
    
    # and create list of corresponding timestamps
    raw_start_end_times = [(DABI_timestamps.iloc[start], DABI_timestamps.iloc[end]) for start, end in raw_start_end_pairs]
    
    # Check list of start-end pairs for order:
    # 1) start must be before end
    # 2) start-end pairs must not overlap
    # 3) Each start-end pair must be at least 1 second long
    
    cropped_start_end_pairs = [] # to hold valid start-end pairs after checks
    cropped_start_end_times = [] # to hold valid start-end times after checks
    for itr, pair in enumerate(raw_start_end_pairs):
        start, end = pair
        if start >= end:
            print(f"Error: Start point {start} is not before end point {end}.")
        if itr > 0:
            prev_start, prev_end = raw_start_end_pairs[itr - 1]
            if start <= prev_end:
                print(f"Error: Start point {start} overlaps with previous end point {prev_end}.")
        
        # Print time difference between start and end
        time_diff = (DABI_timestamps.iloc[end] - DABI_timestamps.iloc[start]).total_seconds()
        print(f"Start-end pair {start}-{end} is {time_diff:.2f} seconds long.")

        if time_diff < 1.0:
            # delete the start-end pair from the list
            print(f"Warning: Start-end pair {start}-{end} is less than 1 second long. Removing from list.")
        else:
            cropped_start_end_pairs.append(pair)
            cropped_start_end_times.append(raw_start_end_times[itr])

    # loop through start-end pairs and plot the high-accuracy pressure data with vertical lines indicating start and end points
    # plt.figure(figsize=(12, 6))
    # plt.plot(DABI_timestamps, press_data, label='High Acc. Transducer (PSIG)')
    # for start, end in start_end_pairs:
    #     plt.axvline(DABI_timestamps.iloc[start], color='g', linestyle='--', label='Start' if start == start_end_pairs[0][0] else "")
    #     plt.axvline(DABI_timestamps.iloc[end], color='r', linestyle='--', label='End' if end == start_end_pairs[0][1] else "")
    
    # plt.show()

    #### Identify active and inactive dimple locations based on individual dimple pressures
    # iterate through start-end pairs
    
    # Generate list of dimple IDs, iterating from A1 to E5
    letters = ['A', 'B', 'C', 'D', 'E']
    numbers = ['1', '2', '3', '4', '5']
    dimple_ids = [f"{letter}{number}" for letter in letters for number in numbers]
    test_segments = []
    
    # Fill out test_segment class for each start-end pair
    for start, end in cropped_start_end_pairs:
        # create instance of class test_segment to hold information about this test segment
        test_segments.append(test_segment(start, end, DABI_timestamps.iloc[start], DABI_timestamps.iloc[end]))
        segment = test_segments[-1]
        
        # select data between start and end points
        segment_data = DABI_data.iloc[start:end]
        
        # Identify active dimples as those with pressure above threshold
        # Identify "creep" or "static" test based on difference between the maximum pressure and the average pressure during the test segment
        test_type = "N/A"
    
        # Identify test type based on pressure profile
        # creep experiments have a long hold with nearly constant pressure
        # static tests have several pressure jumps with very short holds
        # Therefore - to determine if it is a 'creep' test, determine if there is a hold of at least 1 minute where the pressure does not change by more than 5% of the max pressure during that hold

        # use high-accuracy pressure data as individual pressure channels are not very reliable
        high_acc_press_data = segment_data[r'High Acc. Transducer (PSIG)']
        max_press = high_acc_press_data.max()
        segment.high_acc_transducer_max = max_press

        min_hold_time = 60  # seconds
        hold_threshold = 0.05 * max_press  # 5% of max pressure
        test_type = "Static"  # default to static unless creep criteria are met
        hold_start_idx = None
        for i in range(len(high_acc_press_data)):
            if hold_start_idx is None:
                hold_start_idx = i
            # Check if pressure has changed by more than hold_threshold from the start of the hold
            if abs(high_acc_press_data.iloc[i] - high_acc_press_data.iloc[hold_start_idx]) > hold_threshold:
                # Pressure has changed, reset hold start index
                hold_start_idx = i
            else:
                # Pressure has not changed, check if hold time is greater than min_hold_time
                hold_time = (pd.to_datetime(segment_data['time'].iloc[i], format=DABI_datetime_format) - pd.to_datetime(segment_data['time'].iloc[hold_start_idx], format=DABI_datetime_format)).total_seconds()
                if hold_time >= min_hold_time:
                    test_type = "Creep"
                    break  # No need to check further, we found a creep hold
        # Add test type to segment
        segment.test_type = test_type

        # iterate through pressure columns to determine active columns and identify temperature statistics
        for dimple_id in dimple_ids:
            pres_column = f"{dimple_id}: Press.(PSIG)" 
            dimple_press_data = segment_data[pres_column]
            
            # Identify average pressure during active periods
            avg_pressure = dimple_press_data[dimple_press_data > thresh].mean()

            if (dimple_press_data > thresh).any():
                # Active dimple
                segment.active_dimples.append(dimple_id)
            else:
                # Inactive dimple
                segment.inactive_dimples.append(dimple_id)

            # Identify average temperature of each dimple location during active periods
            temp_column = f"{dimple_id}: Temp.(C)"
            dimple_temp_data = segment_data[temp_column]
            segment.dimple_temps_avg[dimple_id] = dimple_temp_data.mean()
            segment.dimple_temps_delta[dimple_id] = dimple_temp_data.max() - dimple_temp_data.min()
            segment.dimple_temps_std[dimple_id] = dimple_temp_data.std()
            
            # individual dimple pressure statistics
            segment.max_individual_pressure[dimple_id] = dimple_press_data.max()
            segment.min_individual_pressure[dimple_id] = dimple_press_data.min()
            segment.average_individual_pressure[dimple_id] = avg_pressure
            
            # high accuracy transducer statistics
            segment.high_acc_transducer_min = high_acc_press_data.min()
            segment.high_acc_transducer_max = high_acc_press_data.max()
            segment.high_acc_transducer_avg = high_acc_press_data.mean()

            # Inform user if temperature changes more than a certain threshold (say, 25 degrees C) during active periods
            # if segment.dimple_temps_delta[dimple_id] > 25:
            #    print(f"Warning: Dimple {dimple_id} temperature changed by more than 25 C during active period from {segment.start_time} to {segment.end_time}. Change: {segment.dimple_temps_delta[dimple_id]:.2f} C")
            # return segments summary dataframe
    return test_segments

def save_test_segments_summary(test_segments, specimen, experiment, RADICAL_data_path=r"V:\\"):
    letters = ['A', 'B', 'C', 'D', 'E']
    numbers = ['1', '2', '3', '4', '5']
    dimple_ids = [f"{letter}{number}" for letter in letters for number in numbers]
    
    # take experiment date from experiment string
    # the experiment string is in the format '(DDMMYY) Experiment Name'
    date_str = experiment.split(')')[0].strip('(')
    
    # Preallocate segments summary dataframe
    # Columns: segment number, start time, end time, and if A1-A5 are active (1) or inactive (0).
    summary_columns = ['Segment Number', 'Start Time', 'End Time', 'Duration (s)', 'Test Type', 'A1 Active', 'A2 Active', 'A3 Active', 'A4 Active', 'A5 Active',
                       'B1 Active', 'B2 Active', 'B3 Active', 'B4 Active', 'B5 Active',
                       'C1 Active', 'C2 Active', 'C3 Active', 'C4 Active', 'C5 Active',
                       'D1 Active', 'D2 Active', 'D3 Active', 'D4 Active', 'D5 Active',
                       'E1 Active', 'E2 Active', 'E3 Active', 'E4 Active', 'E5 Active']
    
    # Fill dataframe
    segments_summary = pd.DataFrame(columns=summary_columns, index=range(len(test_segments)))
    for itr, segment in enumerate(test_segments):
        row_data = {
            'Segment Number': itr + 1,
            'Start Time': segment.start_time,
            'End Time': segment.end_time,
            'Duration (s)': segment.duration,
            'Test Type': segment.test_type
        }
        
        for dimple_id in dimple_ids:
            active_key = f"{dimple_id} Active"
            row_data[active_key] = 1 if dimple_id in segment.active_dimples else 0
            
        segments_summary.loc[itr] = row_data
    
    # Save segments summary to .csv file
    output_csv_path = os.path.join(RADICAL_data_path, r'Data', r'Specimen Directory', r'Bonded_DABI', specimen, experiment, r'Test Data', f"DABI_test_segments_summary.csv")
    segments_summary.to_csv(output_csv_path, index=False)
    print(f"Test segments summary saved to {output_csv_path}")

def pastable_summary_format(experiment_steps, specimen, experiment_id, bond_angle, chamfer_side, DABI_orientation, RADICAL_data_path=r"V:\\"):
    letters = ['A', 'B', 'C', 'D', 'E']
    numbers = ['1', '2', '3', '4', '5']
    chamber_dimple_ids_unsorted = [f"{letter}{number}" for letter in letters for number in numbers]

    print_letters = ['I', 'J', 'K', 'L', 'M'][::-1]
    print_dimple_ids = [f"{letter}{number}" for letter in print_letters for number in numbers]
    
    # preallocate dataframe
    # rows: DABI dimple ids and the plate dimple ids
    # Identify plate dimple IDs using rotate_coordinates function
    chamber_coords = [] # to hold converted plate dimple ids
    for dimple_id in print_dimple_ids:
        converted_id = rotate_coordinates(dimple_id, bond_angle, chamfer_side, DABI_orientation)
        chamber_coords.append(converted_id)
    
    # columns: for each creep experiment step, report the pressure, temperature, and duration if the dimple is active
    # columns: for each static experiment step, report the pressure and temperature if the dimple is active
    summary_columns = []
    summary_columns.extend(['Plate ID'])
    for itr, step in enumerate(experiment_steps):
        if step.test_type == 'Creep':
            summary_columns.extend([f"Step {itr+1} Pressure Average (PSIG)", f"Step {itr+1} Temperature Average (C)", f"Step {itr+1} Duration (s)"])
        elif step.test_type == 'Static':
            summary_columns.extend([f"Step {itr+1} Pressure Maximum (PSIG)", f"Step {itr+1} Temperature Average (C)"])

    summary_df = pd.DataFrame(index=chamber_coords, columns=summary_columns)
    
    # fill dataframe using list of test_segment instances
    for print_dimple_id in print_dimple_ids:
        dimple_id = rotate_coordinates(print_dimple_id, bond_angle, chamfer_side, DABI_orientation)
        for itr, step in enumerate(experiment_steps):
            if dimple_id in step.active_dimples:
                temperature = step.dimple_temps_avg[dimple_id]
                summary_df.at[dimple_id, 'Plate ID'] = print_dimple_id
                summary_df.at[dimple_id, f"Step {itr+1} Temperature Average (C)"] = f"{temperature:.2f}"
                if step.test_type == 'Creep':
                    summary_df.at[dimple_id, f"Step {itr+1} Duration (s)"] = f"{step.duration:.2f}"

                    pressure = step.high_acc_transducer_avg
                    summary_df.at[dimple_id, f"Step {itr+1} Pressure Average (PSIG)"] = f"{pressure:.2f}"
                else:
                    pressure = step.high_acc_transducer_max
                    summary_df.at[dimple_id, f"Step {itr+1} Pressure Maximum (PSIG)"] = f"{pressure:.2f}"
            else:
                summary_df.at[dimple_id, 'Plate ID'] = print_dimple_id
                if step.test_type == 'Static':
                    summary_df.at[dimple_id, f"Step {itr+1} Pressure Maximum (PSIG)"] = "Inactive"
                else:
                    summary_df.at[dimple_id, f"Step {itr+1} Pressure Average (PSIG)"] = "Inactive"
                summary_df.at[dimple_id, f"Step {itr+1} Temperature Average (C)"] = "Inactive"
                if step.test_type == 'Creep':
                    summary_df.at[dimple_id, f"Step {itr+1} Duration (s)"] = "Inactive"
    
    # swap index to first column and vice versa
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'Chamber ID'}, inplace=True)
    
    # save summary df to .csv file in main file tree
    output_csv_path = os.path.join(RADICAL_data_path, r'Data', r'Specimen Directory', f"{specimen}_DABI_test_segments.csv")
    summary_df.to_csv(output_csv_path, index=False)
    return summary_df

def check_pressure_transducer_health(file_path, RADICAL_data_path=r"V:\\"):
    # function to compare pressure transducer readings with high-accuracy transducer readings over course of a test
    # Load data from file_path
    DABI_data = pd.read_csv(os.path.join(RADICAL_data_path, file_path), skiprows=1, low_memory=False)
    # Extract relevant columns
    high_acc_press = DABI_data[r'High Acc. Transducer (PSIG)']
    
    # Generate iterator from A1 to E5
    letters = ['A', 'B', 'C', 'D', 'E']
    numbers = ['1', '2', '3', '4', '5']
    dimple_ids = [f"{letter}{number}" for letter in letters for number in numbers]
    
    # Create figure with subplots showing dimple pressure vs high-accuracy pressure
    comp_fig, comp_ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 6), sharex={'col'}, sharey={'row'}, layout='tight')
    
    # Create figure showing difference between dimple pressure and high-accuracy pressure vs time
    diff_fig, diff_ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 6), sharex={'col'}, sharey={'row'}, layout='tight')
    
    # Figure with select dimples only
    dimple_ids = ['A1', 'B2', 'C3', 'D4', 'E5']
    for dimple_id in dimple_ids:
        # plot against time
        pres_column = f"{dimple_id}: Press.(PSIG)" 
        dimple_press_data = DABI_data[pres_column]
        plt.plot(DABI_data['time'], dimple_press_data, label=f"Dimple {dimple_id}")

    plt.show()
    # crop time to only times where high-accuracy pressure is within 100 psi of the highest recorded pressure
    # high_acc_above_10 = high_acc_press > 10
    # high_acc_press = high_acc_press[high_acc_above_10]
    # DABI_data = DABI_data[high_acc_above_10]

    # for dimple_id in dimple_ids:
    #     pres_column = f"{dimple_id}: Press.(PSIG)" 
    #     dimple_press_data = DABI_data[pres_column]
        
    #     # Determine subplot location
    #     row = ord(dimple_id[0]) - ord('A')
    #     col = int(dimple_id[1]) - 1

    #     # Plot pressure data
    #     comp_ax[row, col].plot(high_acc_press, dimple_press_data)
    #     comp_ax[row, col].set_title(f"Dimple {dimple_id}")
    #     comp_ax[row, col].set_xlabel("High Accuracy Pressure (PSIG)")
    #     comp_ax[row, col].set_ylabel("Dimple Pressure (PSIG)")
        
    #     # plot time series of difference between dimple pressure and high-accuracy pressure
    #     diff_data = dimple_press_data[high_acc_above_10] - high_acc_press[high_acc_above_10]
    #     diff_ax[row, col].plot(DABI_data['time'][high_acc_above_10], diff_data)
    #     diff_ax[row, col].set_title(f"Dimple {dimple_id}")
    #     diff_ax[row, col].set_xlabel("Time")
    #     diff_ax[row, col].set_ylabel("Pressure Difference (PSIG)")
        
    #     # hide axes labels in diff_data
    #     diff_ax[row, col].tick_params(labelbottom=False, labelleft=False)
    
    # save figures as .svg files
    # locate root of base .csv file
    base_dir = os.path.dirname(os.path.join(RADICAL_data_path, file_path))
    
    plt.show()
    
    # save figures
    comp_fig.savefig(os.path.join(base_dir, "pressure_transducer_comparison.png"))
    diff_fig.savefig(os.path.join(base_dir, "pressure_transducer_difference.png"))
    
    # close figures
    plt.close(comp_fig)
    plt.close(diff_fig)

class test_segment:
    # class to hold information about each test segment
    def __init__(self, start_idx, end_idx, start_time, end_time):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_time = start_time
        self.end_time = end_time
        
        self.duration = (end_time - start_time).total_seconds()
        
        # list of active and inactive dimples
        self.active_dimples = []
        self.inactive_dimples = []
        
        # dimple temperature statistics, from FLIR
        self.dimple_temps_avg = {}
        self.dimple_temps_delta = {}
        self.dimple_temps_std = {}
        
        # individual dimple pressures from crummy transducers 
        self.max_individual_pressure = {}
        self.min_individual_pressure = {}
        self.average_individual_pressure = {}
        
        # high-accuracy transducer max pressure during segment
        self.high_acc_transducer_max = 0.0
        self.high_acc_transducer_min = 0.0
        self.high_acc_transducer_avg = 0.0

# # Specimen, experiment
# active_experiment_id = r'(251022) 3 Stage Creep'

# # RADICAL base directory
# RADICAL_data_path = 'V:\\'

# # process_dimple_locations(excel_file_path)
# default_specimen = r'RAD-55-03-BD-01'

# # pressure threshold in psi
# pressure_threshold = 100

# # merge DABI data
# # DB.merge_DABI_data(default_specimen, experiment=active_experiment_id, RADICAL_data_path="V:\\")

# # Identify pressure steps
# experiment_steps = ident_pressure_steps(default_specimen, experiment=active_experiment_id, thresh=pressure_threshold, RADICAL_data_path=r"V:\\")

# # Save test segments in individual .csv file
# save_test_segments_summary(experiment_steps, default_specimen, active_experiment_id, RADICAL_data_path=r"V:\\")

# # Pastable format for the summary sheet
# bond_angle = 0
# chamfer_side = 'CR'
# DABI_orientation = 'IN'

# # whoops this function doesn't exist yet
# pastable_summary_format(experiment_steps, default_specimen, active_experiment_id, bond_angle, chamfer_side, DABI_orientation, RADICAL_data_path=r"V:\\")