import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpltern
import re
import os
import xml.etree.ElementTree as ET
import json


# create helper function to reassign "Unnamed" columns to previoous column name
def rename_unnamed_columns(df):
    cols = df.columns
    for i in range(1, len(cols)):
        if "Unnamed" in cols[i]:
            df.rename({cols[i]: cols[i - 1]}, axis=1, inplace=True)
    return df

# Function to map string coordinates (e.g. 'I1', 'J3') to 5x5 array indices
# there's probably a way to vectorize this
def coord_to_index(coord):
    col_map = {'I': 0, 'J': 1, 'K': 2, 'L': 3, 'M': 4}
    row_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    
    try:
        col = col_map[coord[0]]
        row = row_map[coord[1]]
    except:
        col = None
        row = None
    
    return row, col

# function to plot on 5x5 array given I-M, 1-5 coordinates and data
def plot_on_dabi_array(data, ax, title, cmap='viridis'):
    # Create a heatmap on the 5x5 array
    cax = ax.matshow(data, cmap=cmap, vmin=2, vmax=24)

    # Set axis labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(['I', 'J', 'K', 'L', 'M'])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, f'{data[i, j]:.1f}', ha="center", va="center", color="w")

    # Set title
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)

    return ax, cbar, cax

# Test case:
# Generate list of coordinates from I1 to M5
test_coords = [f"{chr(i)}{j}" for i in range(ord('I'), ord('M') + 1) for j in range(1, 6)]

# Print test coordinates

# Import ternary data as pandas array
# File location
file = r"V:\FA 1\Simulations\Creep Life of Items\Creep_life_prediction.xlsx"

# Create new directory for figures in same directory as xlsx file
output_dir = os.path.join(os.path.dirname(file), "creep_life_figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

# import columns H:M in "800C_Creep_Life" as creep life
creep_df_raw = rename_unnamed_columns(pd.read_excel(file, sheet_name="800C_Creep_Life", skiprows=1, usecols="A, I:N", index_col=0))

# Import columns A:D as compositions
compositions_df = pd.read_excel(file, sheet_name="800C_Creep_Life", skiprows=2, usecols="A:E", nrows=creep_df_raw.shape[0]-1, index_col=0)

# Convert to multiindex where the second row is the second level of the index 
creep_df_raw.columns = pd.MultiIndex.from_arrays([creep_df_raw.columns, creep_df_raw.iloc[0]], names=["Pressure", "Strain"])
creep_life_df = creep_df_raw.drop(index=0)

# Convert all data to numeric, change '>24' to NaN
creep_life_df = creep_life_df.apply(pd.to_numeric, errors='coerce')

# Set all NaN values to 24
creep_life_df = creep_life_df.fillna(24)

# Create ternary figure for each combination of pressure and strain
pressures = creep_life_df.columns.levels[0].to_numpy()
strains = creep_life_df.columns.levels[1].to_list()

# Take "coordinate" column from compositions_df and create row and column index arrays
# row, col = coord_to_index(compositions_df.loc[:,'Coordinate'].values)
row_indices = []
col_indices = []
for row in compositions_df.loc[:,'Coordinate'].values:
    r, c = coord_to_index(row)
    row_indices = np.append(row_indices, r)
    col_indices = np.append(col_indices, c)

# Loop through pressures and strains to create ternary plots
for pressure in pressures:
    for strain in strains:
        ## Create ternary plot ##
        tern_fig = plt.figure(figsize=(8, 7))
        tern_ax = tern_fig.add_subplot(111, projection='ternary')
        
        # Set axis labels
        tern_ax.set_tlabel('HA25 (wt%)')
        tern_ax.set_llabel('316L (wt%)')
        tern_ax.set_rlabel('IN625 (wt%)')
        
        # Set axis limits
        tern_ax.set_tlim(0, 100)
        tern_ax.set_llim(0, 100)
        tern_ax.set_rlim(0, 100)
        
        # Get proportion of HA25, 316L, and IN625
        wt_HA25 = compositions_df.loc[:,'HA25'].values
        wt_316L = compositions_df.loc[:,'316L'].values
        wt_IN625 = compositions_df.loc[:,'IN625'].values
        
        # Get creep life data for current pressure and strain
        creep_life = creep_life_df[(pressure, strain)].values
        
        # Create contour plot of creep life as a function of composition
        contour = tern_ax.tricontourf(wt_HA25, wt_316L, wt_IN625, creep_life, levels=14, cmap='viridis')
        
        # Alter contour so only values between 2 and 24 are shown
        contour.cmap.set_under('grey')  # Color for values below the minimum
        contour.cmap.set_over('white')    # Color for values above the maximum
        
        # Add colorbar
        creep_cbar = plt.colorbar(contour, ax=tern_ax, orientation='vertical', pad=0.1)
        creep_cbar.set_label('Creep Life (hours)', fontsize=12)
        
        # Remove alpha characters from pressure
        pressure_num = float(re.sub("[^0-9.]", "", str(pressure)))
        
        # Set title
        tern_ax.set_title(f'Creep Life at {np.round(pressure_num, 3)} MPa and {np.round(strain*100, 3)}% Strain', fontsize=14)
        
        # Save figure as svg
        plt.savefig(os.path.join(output_dir, f'creep_life_{np.round(pressure_num, 3)}MPa_{np.round(strain*100, 3)}percent.svg'), format='svg', dpi=300)

        ## Create heatmap in 5 x 5 array ##
        dabi_dmg_fig, dabi_dmg_ax = plt.subplots(figsize=(8, 8))

        # Using coordinates in row_indices and col_indices, create 5x5 array of creep life data
        dabi_creep_array = np.full((5, 5), np.nan)
        for itr in range(len(row_indices)):
            try:
                r = int(row_indices[itr])
                c = int(col_indices[itr])
                
                dabi_creep_array[r, c] = creep_life[itr]
            except TypeError:
                continue
        
        # Plot heatmap on 5x5 array
        dabi_dmg_ax, creep_cbar, creep_cax = plot_on_dabi_array(dabi_creep_array, dabi_dmg_ax, title=f'Creep Life at {np.round(pressure_num, 3)} MPa and {np.round(strain*100, 3)}% Strain', cmap='viridis')
        creep_cbar.set_label('Creep Life (hours)', fontsize=12)

        # Save figure as svg
        plt.savefig(os.path.join(output_dir, f'creep_life_array_{np.round(pressure_num, 3)}MPa_{np.round(strain*100, 3)}percent.svg'), format='svg', dpi=300)

        plt.close('all')

# Creep experiment simulator
# Simulate creep experiments for different compositions and conditions
# Assumptions:
# 1) Uniform temperature
# 2) Failure occurs at 7% strain
# 3) Damage is cumulative and linear w.r.t. time

# Import json module for reading configuration files
# import json

# Creep experiment input dataframe where columns are pressure and time in hours
creep_experiment_df = pd.DataFrame(columns=['Pressure (MPa)', 'Time (hours)'])

# Experiment file location
# Bonded DABI ID
bonded_DABI_ID = r"RAD-BD-006"

# Experiment ID
experiment_ID = r"(250909) Four Stage Creep Exp"

# import experiment conditions from json file
experiment_file = os.path.join(r"V:\Data\Specimen Directory\Bonded_DABI", bonded_DABI_ID, experiment_ID, "nominal_test_conditions.json")

# Import array "steps" from json file "experiment_file"
with open(experiment_file, 'r') as f:
    experiment_data = json.load(f)

# iterate through steps and append pressures, times, and coordinates to lists
pressures = []
times = []
coordinates_test = []
for step in experiment_data['steps']:
    pressures.append(step['pressure'])
    times.append(step['duration'])
    coordinates_test.append(step['cells'])
# pressures = ['8 MPa', '8 MPa', '10 MPa', '10 MPa']
# times = [1., 2.5, 2.5, 6.] # Times in hours

# coordinates_test = [['I1', 'I2', 'I3', 'I4', 'I5',
#             'J1', 'J2', 'J3', 'J4', 'J5',
#             'K1', 'K2', 'K3', 'K4', 'K5',
#             'L1', 'L2', 'L3', 'L4', 'L5',
#             'M1', 'M2', 'M3', 'M4', 'M5'],
#            ['J1', 'J2', 'J3', 'J4', 'J5',
#             'K1', 'K2', 'K3', 'K4', 'K5',
#             'L1', 'L2', 'L3', 'L4', 'L5',
#             'M1', 'M2', 'M3', 'M4', 'M5'],
#            ['K2', 'K3', 'K4', 'K5',
#             'L1', 'L2', 'L3', 'L4', 'L5',
#             'M1', 'M2', 'M3', 'M4', 'M5'],
#            ['K5', 'L5', 'M5', 'M4', 'M3', 'M2']
#            ]

# check that pressures, times, and entries have the same length
assert len(pressures) == len(times) == len(coordinates_test), "Inconsistent lengths"

# Change pressures from float to string with 'MPa' suffix, rounding to 1 decimal place
for itr in range(len(pressures)):
    pressures[itr] = f"{np.round(pressures[itr], 1)} MPa"

creep_dmg_step = np.zeros((5, 5, len(pressures)))
for itr in range(len(pressures)):

    time_elapsed_step = times[itr]
    for coord_jtr in coordinates_test[itr]:

        # Convert string coordinate to row, col indices
        r, c = coord_to_index(coord_jtr)

        # Locate row in compositions_df corresponding to entry ID
        coord_row = compositions_df[compositions_df['Coordinate'] == coord_jtr].index[0]
        
        # Find creep life at pressure itr and 2.5% strain
        creep_life_hrs = creep_life_df[(pressures[itr], 0.025)].loc[coord_row]
        
        # change given time elapsed
        percent_dmg = (time_elapsed_step / creep_life_hrs)
        
        # Assign to row, column in matrix
        creep_dmg_step[r, c, itr] = percent_dmg
        
# sum all three steps
creep_dmg_total = np.sum(creep_dmg_step, axis=2)

# Create plot of total creep damage
dabi_dmg_fig, dabi_dmg_ax = plt.subplots(figsize=(8, 8))
dabi_dmg_ax, damage_cbar, damage_cax = plot_on_dabi_array(creep_dmg_total, dabi_dmg_ax, 'Total Creep Damage (1 = Failure)', cmap='viridis')

# set colorbar label
damage_cbar.set_label('Creep Damage (fraction)', fontsize=12)

# set colorbar range
damage_cax.set_clim(0, 2)

plt.show()