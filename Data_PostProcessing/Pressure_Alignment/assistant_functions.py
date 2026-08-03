import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import io
import re


def gen_aligned(root, dic_file, instrument_file, high_temp, mod):
    #High temp is true, room temp is False
    if high_temp:
        instrument_df = load_data(root, instrument_file, 'ht-dabi')
        if instrument_df.columns.tolist()[0] == 'Dimple':
            raise Exception('This might be a Room Temp DABI dataset. Check flag status (Set to False) to prevent errors')
        dic_df = load_data(root, dic_file, 'dic')
        converted_df = convert_to_unix_ms(instrument_df,  'time', '%m/%d/%Y %H:%M:%S.%f', mod)
        if dic_df['time'].min() > converted_df['time'].max() or dic_df['time'].max() < converted_df['time'].min():
            raise Exception('Wrong times for the DIC and instrument data. No alignment will be found. Check filenames')
        output_file = align_save_data_ht_dabi(dic_df, root, instrument_file, converted_df)
        print(f'Saved the High Temp DABI experiment data to : {output_file}')
    else:
        instrument_df = load_data(root, instrument_file, 'lt-dabi')
        if np.any(instrument_df.columns == 'time'):
            raise Exception('This might be a High Temp DABI dataset. Check flag status (Set to True) to prevent errors')
        dic_df = load_data(root, dic_file, 'dic')
        dic_df['time'] = dic_df['time'] + mod
        if dic_df['time'].min() > instrument_df['TimeStamp'].max() or dic_df['time'].max() < instrument_df['TimeStamp'].min():
            raise Exception('Wrong times for the DIC and instrument data. No alignment will be found. Check filenames')
        dimples_tested = break_up_rt_dabi(root, instrument_file, instrument_df)
        output_files = align_save_data_rt_dabi(dic_df, root, instrument_file, dimples_tested)
        print(f'Saved the Room Temp DABI experiment data to : {output_files}')

def gen_aligned_mod(instrument_root, dic_root, dic_file, instrument_file, high_temp, mod, dimple=[], save_folder=[]):
    #High temp is true, room temp is False
    if high_temp:
        instrument_df = load_data(instrument_root, instrument_file, 'ht-dabi')
        if instrument_df.columns.tolist()[0] == 'Dimple':
            raise Exception('This might be a Room Temp DABI dataset. Check flag status (Set to False) to prevent errors')
        dic_df = load_data(dic_root, dic_file, 'dic')
        converted_df = convert_to_unix_ms(instrument_df,  'time', '%m/%d/%Y %H:%M:%S.%f', mod)
        # converted_df = convert_to_unix_ms(instrument_df,  'time', '%M:%S.%f', mod)
        if dic_df['time'].min() > converted_df['time'].max() or dic_df['time'].max() < converted_df['time'].min():
            raise Exception('Wrong times for the DIC and instrument data. No alignment will be found. Check filenames')
        output_file = align_save_data_ht_dabi(dic_df, instrument_root, instrument_file, converted_df, dimple, save_folder)
        print(f'Saved the High Temp DABI experiment data to : {output_file}')
    else:
        instrument_df = load_data(instrument_root, instrument_file, 'lt-dabi')
        if np.any(instrument_df.columns == 'time'):
            raise Exception('This might be a High Temp DABI dataset. Check flag status (Set to True) to prevent errors')
        dic_df = load_data(dic_root, dic_file, 'dic')
        dic_df['time'] = dic_df['time'] + mod
        if dic_df['time'].min() > instrument_df['TimeStamp'].max() or dic_df['time'].max() < instrument_df['TimeStamp'].min():
            raise Exception('Wrong times for the DIC and instrument data. No alignment will be found. Check filenames')
        dimples_tested = break_up_rt_dabi(instrument_root, instrument_file, instrument_df)
        output_files = align_save_data_rt_dabi(dic_df, instrument_root, instrument_file, dimples_tested)
        print(f'Saved the Room Temp DABI experiment data to : {output_files}')

def load_data(root_path, filename, instrument):
    r'''Loads the csv data into a pandas data structure

    Args:
        root_path (str): folder containing file, without \ at the end of argument 
            Example: 'C:\Users\default_user\data_folder'
        filename (str): file with data and headers associated. Must be csv. Not used for DIC data
        instrument (str): Must be 'dic', 'lt-dabi', or 'ht-dabi' to not throw exception

    Returns:
        DataFrame: DataFrame that contains the csv data with associated headers
    '''
    if instrument.lower() == r'lt-dabi':
        with open(os.path.join(root_path, filename)) as table:
            buffer = io.StringIO('\n'.join(re.sub(r"[\r\t\s]*", "", line) for line in table))
            df = pd.read_table(buffer, low_memory = False, usecols=range(3), header=1, delimiter=',')
        return df
    
    if instrument.lower() == r'dic':
        filepath = os.path.join(root_path, filename)
        output_db = pd.read_csv(filepath, low_memory=False)
        return output_db
        
    elif instrument.lower() == r'ht-dabi':
        return pd.read_csv(os.path.join(root_path, filename), skiprows=1, low_memory = False)
    else:
        raise Exception("\n\nWrong instrument name used.\n" + 
                        f"'{instrument.lower()}' is not an allowed instrument type.\n" +
                        "Please use one of the following: 'dic', 'lt-dabi', or 'ht-dabi'")

def convert_to_unix_ms(df, header, date_string, mod, timezone = 0):
    r'''Converts a string time to unix
    
    Args:
        df (DataFrame): The dataframe with a string as the time
        header (str): The DataFrame header that the time column is labeled
        date_string (str): Format of the data, using datetime notation
        timezone (int): Optional, assumes UTC. Give int for timezone difference from UTC (timezone = 5 for CDT, timezone = 6 for CST)

    Returns:
        DataFrame: Returns the same dataframe structure, but with unix time as a float    
    '''
    unix_time = np.zeros(len(df[header])) #Zero array to start
    for i in range(len(df[header])):
        datetime_obj_local = datetime.strptime(df[header][i], date_string) #Strip the time based on formatting
        datetime_obj_utc = datetime_obj_local + timedelta(hours=timezone) #Set as datetime object with offset based on timezone
        unix_time[i] = datetime_obj_utc.timestamp()*1e3 + mod #.timestep() determines the seconds since Epoch 
    
    df_new = df.copy()
    df_new[header] = unix_time
    return df_new


def interpolate_from_dic(dic_df, inst_df, dic_fh, dic_th, inst_th):
    inst_columns = list(inst_df.columns) #Strip the column names for interpolation
    df_interp = pd.DataFrame(columns=inst_columns.insert(1,'Frame')) #Make new empty df to put data in
    df_interp[inst_th] = dic_df[dic_th] #Set the interp time as the dic time
    df_interp['Frame'] = dic_df[dic_fh] #Set the frames as the dic frames
    inst_columns.remove(inst_th) #Remove the columns that don't need to be interpolated
    inst_columns.remove('Frame')
    #Determine if this is roomtemp data, remove the dimple column from interpolation and set the copied column to the dimple number
    if 'Dimple' in inst_columns:
        inst_columns.remove('Dimple')
        df_interp['Dimple'] = inst_df['Dimple'].reset_index(drop=True).iloc[0]
    #Loop for each column to interpolate the data from dic timestep
    for column in inst_columns:
        new_column_val = np.interp(dic_df[dic_th],inst_df[inst_th],inst_df[column])
        df_interp[column] = new_column_val
    return df_interp
    

def align_save_data_ht_dabi(dic_frame_time, instrument_root, instrument_file, instrument_data, dimple=[], save_folder=[],  dic_time_header='time', dic_frame_header='index', instrument_time_header='time'):
    r'''Aligns and saves data from the high temp insturment and aligns it to DIC

    Args:
        dic_frame_time (DataFrame): DataFrame that has two columns, the unix time and the frame number associated with that time
        instrument_root (str): Root path for the instrument file
        instrument_file (str): Base file name for the recorded data
        instrument_data (DataFrame): HT DABI data to save from the MDI format
        dic_time_header (str): Header name for the time column in the dic file
        dic_frame_header (str): Header name for the frame label column, to be included in the final output dataframe
        instrument_time_header(str): Header for the time column in the instrument data

    Returns:
        filepath_out (str): sting of the filepath to the output saved file
    '''
    
    start_time = np.max([instrument_data[instrument_time_header][0], dic_frame_time[dic_time_header][0]])               # Determines the start time of the region with both DIC and DABI data
    end_time = np.min([instrument_data[instrument_time_header].iloc[-1], dic_frame_time[dic_time_header].iloc[-1]])     # Determines the end time of the region with both the DIC and DABI data
    
    start_ind = (dic_frame_time[dic_time_header] - start_time).abs().idxmin() #Find the start index of the DIC data
    end_ind = (dic_frame_time[dic_time_header] - end_time).abs().idxmin() #Find the end index of the DIC data
    
    #interpolate at each time in the DIC data
    # data_ind_inst = np.array([(instrument_data[instrument_time_header] - dic_frame_time[dic_time_header].iloc[start_ind + i]).abs().idxmin() for i in range(end_ind - start_ind + 1)])
    data_ind_inst = np.array([np.argmin(np.abs(instrument_data[instrument_time_header] - dic_frame_time[dic_time_header].iloc[start_ind + i])) for i in range(end_ind - start_ind + 1)])
    df_aligned = interpolate_from_dic(dic_frame_time.iloc[start_ind:end_ind], instrument_data.iloc[data_ind_inst],dic_frame_header,dic_time_header,instrument_time_header)
    #Save the data
    if len(dimple) == 0:
        ending = '_ALIGNED.csv'
    else:
        ending = '_%s_ALIGNED.csv' % dimple
    if len(save_folder) == 0:
        filepath_out = os.path.join(instrument_root, instrument_file[:-4] + ending)
    else:
        filepath_out = os.path.join(instrument_root, save_folder, instrument_file[:-4] + ending)
    df_aligned.to_csv(filepath_out, index=False)
    return filepath_out

def align_save_data_rt_dabi(dic_frame_time, instrument_root, instrument_file, dimples_tested,  dic_time_header = 'time', 
                            dic_frame_header = 'index', instrument_time_header = 'TimeStamp', save_folder=''):
    r'''Aligns and saves data from the room temp insturment and aligns it to DIC

    Args:
        dic_frame_time (DataFrame): DataFrame that has two columns, the unix time and the frame number associated with that time
        instrument_root (str): Root path for the instrument file
        instrument_file (str): Base file name for the recorded data
        dimples_tested (list of str): Dimple labels which were tested
        dic_time_header (str): Header name for the time column in the dic file
        dic_frame_header (str): Header name for the frame label column, to be included in the final output dataframe
        instrument_time_header(str): Header for the time column in the instrument data
        time_multi (float): Amount (i.e. 1e3, 1e-3, etc) to multiply the DIC time by to match the timestamp of RT DABI data

    Returns:
        DataFrame: The aligned data
    '''
    filepaths = []
    i=0
    for dimple in dimples_tested:
        dataframe_path = os.path.join(instrument_root, instrument_file[:-4] + '_' + dimple + '.csv')
        instrument_data = pd.read_csv(dataframe_path, low_memory = False)

        start_time = np.max([instrument_data[instrument_time_header][0], dic_frame_time[dic_time_header][0]])               # Determines the start time of the region with both DIC and DABI data
        end_time = np.min([instrument_data[instrument_time_header].iloc[-1], dic_frame_time[dic_time_header].iloc[-1]])     # Determines the end time of the region with both the DIC and DABI data
    
        start_ind = (dic_frame_time[dic_time_header] - start_time).abs().idxmin() #Find the start index of the DIC data
        end_ind = (dic_frame_time[dic_time_header] - end_time).abs().idxmin() #Find the end index of the DIC data
    
        #interpolate at each dic time step
        data_ind_inst = np.array([(instrument_data[instrument_time_header] - dic_frame_time[dic_time_header].iloc[start_ind + i]).abs().idxmin() for i in range(end_ind - start_ind + 1)])
        df_aligned = interpolate_from_dic(dic_frame_time.iloc[start_ind:end_ind], instrument_data.iloc[data_ind_inst],dic_frame_header,dic_time_header,instrument_time_header)

        #Save to File
        if save_folder == "":
            filepath_i = os.path.join(instrument_root, instrument_file[:-4] + '_' + dimple + f'_{start_ind}_ALIGNED.csv')
        else:
            filepath_i = os.path.join(instrument_root, save_folder,
                                      instrument_file[:-4] + '_' + dimple + f'_{start_ind}_ALIGNED.csv')
        df_aligned.to_csv(filepath_i, index=False)
        filepaths.append(filepath_i)
        i += 1
        instrument_data = []

    return filepaths

def break_up_rt_dabi(root, filename, df):
    #Drop empty rows and determine the unique dimple IDs
    tested_dimples = df['Dimple'].dropna().unique()
    #For each dimple, split into a new file
    for dimple in tested_dimples:
        selected_rows = df[df['Dimple'] == dimple].reset_index(drop=True)
        filename_dimple = filename[:-4] + '_' + dimple + '.csv'
        path_selected = os.path.join(root, filename_dimple)
        selected_rows.to_csv(path_selected, index=False)

    return tested_dimples