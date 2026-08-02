import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from pathlib import Path
import io
import re
import glob
from assistant_functions import *
from alignment_fns import *
from scipy.interpolate import griddata

root = "250818"
col_header = ['C', 'D']
row_header = [1, 2, 3, 4, 5]
plot_dimples_all = sorted([['%s%i' % (i, j)] for j in row_header for i in col_header])
test_name_all = [i[0] for i in plot_dimples_all]
neg_temp = False
L_offset = -5e2  # Offset to account for time mismatch between pressure and DIC data

# Code aligns using SINGLE dimple data, plot_dimples are all dimples in the same test
# root: test data
# dimple: single dimple to process

max_disp = 0
test_index = 5
dimple_index = 0
mod = 1
save_folder = 'V0'
p_flat_len = 5
t_flat_inds = [0]
p_flat_inds = [0]
t_len = 5
t_range = 1e-4

p_bounds = [1, 0]
d_bounds = [1, -1]

test_name = test_name_all[test_index]
dimple = [plot_dimples_all[test_index][dimple_index]]
print(dimple)
instrument_root = r'/Users/Itzel_Salgado/Downloads/Test_Data_%s' % root
dic_root = r'/Users/Itzel_Salgado/Downloads/Test_Data_%s/Frame_Time_%s' % (root, root)

# Keep following as is, should be False
p_flat_len = 0
mean, custom = [False for i in range(2)]
backwards = False
search_pattern2 = os.path.join(instrument_root, '*%s*.csv' % test_name)
search2 = glob.glob(search_pattern2)[0]
instrument_file = search2.split('/')[-1]
print(instrument_file)

os.makedirs(f"{instrument_root}/{save_folder}", exist_ok=True)

dimple_info = []
for select_dimple in dimple:
    search_pattern2 = os.path.join(dic_root, '*%s.csv' % test_name)
    search2 = glob.glob(search_pattern2)
    print(sorted(search2))
    if len(search2) != 0:
        print(f"#################    {select_dimple}    ###############")
        nfile = open("%s/%s/%s_%s_ALIGNED.log" % (instrument_root, save_folder, test_name, select_dimple), "w")
        dic_file = sorted(search2)[0].split('/')[-1]
        print(test_name)
        print(dic_file)

        instrument_df = load_data(instrument_root, instrument_file, 'ht-dabi')
        if instrument_df.columns.tolist()[0] == 'Dimple':
            raise Exception(
                'This might be a Room Temp DABI dataset. Check flag status (Set to False) to prevent errors')

        # LOAD DIC DATA TIME FRAME AND DISPLACEMENT
        dic_df = load_data(dic_root, dic_file, 'dic')
        converted_df = convert_to_unix_ms(instrument_df, 'time', '%m/%d/%Y %H:%M:%S.%f', 0)
        p_start_inds = 0
        for i in range(len(plot_dimples_all[test_index])):
            if mean:
                actual_peak_displacements = np.load(f"{instrument_root}/{plot_dimples_all[test_index][i]}_mean_displacements.npy")
            else:
                actual_peak_displacements = np.load(f"{instrument_root}/{plot_dimples_all[test_index][i]}_actual_peak_displacements.npy")
            # Find index with MAXIMUM displacement, focus up to that range
            d_max = np.argmax(actual_peak_displacements)
            d = actual_peak_displacements[:d_max]
            ##################
            d_t_init = np.array(dic_df["time"])
            d_t = np.array(dic_df["time"])[:d_max]
            initial_trange = d_t[-1] - d_t[0]
            d_tmod = d_t - d_t[0]
            # LOAD PRESSURE DATA AND CONVERT TIME
            pressure = np.array(converted_df['%s: Press.(PSIG)' % plot_dimples_all[test_index][i]])
            pressure = np.array(converted_df['High Acc. Transducer (PSIG)']) / 145
            t_max = np.argmax(moving_avg(pressure, 9))
            p_t_init = np.array(converted_df['time'])[p_start_inds:]
            p_t = np.array(converted_df['time'])[p_start_inds:t_max]
            p_range = p_t[-1] - initial_trange
            p_inds = np.where(p_t > p_range)[0]

            dimple_info.append([pressure, p_t_init, actual_peak_displacements, d_t_init, d, d_t, initial_trange])

            fig, ax1 = plt.subplots()
            ax1.scatter((p_t_init[p_inds[0]:] - p_t_init[p_inds[0]]) / 1e3, pressure[p_inds[0]:], c='b', s=2,
                        label='Pressure')
            ax1.set_xlabel('Time')  # Change label as appropriate
            ax1.set_ylabel('Pressure [MPa]', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2 = ax1.twinx()
            ax2.scatter((d_t_init - d_t_init[0]) / 1e3, actual_peak_displacements, c='r', s=2,
                        label='Actual Z at Center Point')
            ax2.set_ylabel('Z Displacement [mm]', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
            plt.title("%s: Pressure vs Z-Displacement at Center (not aligned)" % plot_dimples_all[test_index][i])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("%s/%s/%s_%s_dp_unaligned_%i.png" % (instrument_root, save_folder, test_name, plot_dimples_all[test_index][i], i))
            plt.close()

        pressure, p_t_init, actual_peak_displacements, d_t_init, d, d_t, initial_trange = dimple_info[dimple_index]
        if max_disp != 0:
            d_max = np.argmin(np.abs(actual_peak_displacements - max_disp))
            d_t = d_t[:d_max]
            d = d[:d_max]
        d_tmod = d_t - d_t[0]

        temperature = np.array(converted_df["%s: Temp.(C)" % select_dimple])
        t_max = np.argmax(moving_avg(pressure, 9))
        p_t_init = np.array(converted_df['time'])[p_start_inds:]
        p_t = np.array(converted_df['time'])[p_start_inds:t_max]
        p_range = p_t[-1] - initial_trange
        p_inds = p_t > p_range
        p_tmod = p_t[p_inds] - p_t[p_inds][0]

        # Obtain time and use same pressure region as DIC data (here done using initial trange)
        p = pressure[p_start_inds:t_max][p_inds]
        p_avg = moving_avg(p, 9)[5:-5]
        tm = -temperature[p_start_inds:t_max]
        t_avg = moving_avg(tm[p_inds], 9)[5:-5]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if neg_temp:
            axes[0].scatter(p_tmod / 1e3, tm[p_inds], s=5)
            axes[0].set_ylabel("Temperature (-C)")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_title("%s: Negative Temperature" % select_dimple)
        else:
            axes[0].scatter(p_tmod / 1e3, -tm[p_inds], s=5)
            axes[0].set_ylabel("Temperature (C)")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_title("%s: Temperature" % select_dimple)
        axes[1].scatter(p_tmod / 1e3, p, s=5)
        axes[1].set_title("%s: Recorded Pressure" % select_dimple)
        axes[1].set_ylabel("Pressure (MPa)")
        axes[1].set_xlabel("Time (s)")
        axes[2].scatter(d_tmod / 1e3, d, s=5)
        axes[2].set_title("%s: Displacement" % select_dimple)
        axes[2].set_ylabel("Displacement (mm)")
        axes[2].set_xlabel("Time (s)")
        plt.savefig("%s/%s/1_%s_%s_raw_data.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.tight_layout()
        plt.close()
        # plt.show()

        t_hist, t_xedges = np.histogram(d, bins=200)
        p_hist, p_xedges = np.histogram(p, bins=200)
        t_flat, p_flat, t_temp, p_temp = [[] for i in range(4)]
        for i in range(len(t_hist)):
            if t_hist[i] > t_len:
                t_temp.append([t_hist[i], t_xedges[i: i+2]])
            elif len(t_temp) > 0:
                if len(t_temp) == 1:
                    vals = t_temp[0][1]
                else:
                    max_ind = np.argmax([t[0] for t in t_temp])
                    vals = t_temp[max_ind][1]
                t_temp = []
                inds = np.where((d > vals[0]) & (d < vals[1]))[0]
                inds = inds[inds < np.argmax(d)]
                if len(inds) != 0:
                    m = np.mean(d[inds])
                    if len(t_flat) > 0 and np.abs(m - t_flat[-1][0]) < t_range:
                        if len(inds) > len(t_flat[-1][1]):
                            t_flat[-1] = [np.mean(d[inds]), inds]
                    else:
                        t_flat.append([np.mean(d[inds]), inds])
            if p_hist[i] > 10:
                p_temp.append([p_hist[i], p_xedges[i: i+2]])
            elif len(p_temp) > 0:
                if len(p_temp) == 1:
                    vals = p_temp[0][1]
                else:
                    max_ind = np.argmax([t[0] for t in p_temp])
                    vals = p_temp[max_ind][1]
                p_temp = []
                inds = np.where((p > vals[0]) & (p < vals[1]))[0]
                p_flat.append([np.mean(p[inds]), inds])


        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(d_tmod / 1e3, d, s=5)
        p_mean, t_mean = [[] for i in range(2)]
        for t_vals in t_flat:
            t_mean.append(t_vals[0])
            axes[0].scatter(d_tmod[t_vals[1]] / 1e3, d[t_vals[1]], s=5)
        axes[0].set_ylabel("Displacement")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title("View of All Flat Displacement Regions")
        axes[1].scatter(p_tmod / 1e3, p, s=5)
        for t_vals in p_flat:
            p_mean.append(t_vals[0])
            axes[1].scatter(p_tmod[t_vals[1]] / 1e3, p[t_vals[1]], s=5)
        axes[1].set_title("View of All Flat Pressure Regions")
        axes[1].set_ylabel("Pressure")
        axes[1].set_xlabel("Time (s)")
        # plt.show()
        plt.savefig("%s/%s/1_%s_%s_flat_regions.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.close()

        if len(t_flat_inds) != 0:
            t_flat_c = t_flat
            t_flat = []
            for l in t_flat_inds:
                t_flat.append(t_flat_c[l])
        if len(p_flat_inds) != 0:
            p_flat_c = p_flat
            p_flat = []
            for l in p_flat_inds:
                p_flat.append(p_flat_c[l])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(d_tmod/ 1e3, d, s=5)
        p_mean, t_mean = [[] for i in range(2)]
        for t_vals in t_flat:
            t_mean.append(t_vals[0])
            axes[0].scatter(d_tmod[t_vals[1]] / 1e3, d[t_vals[1]], s=5)
        axes[0].set_ylabel("Displacement")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title("View of Flat Displacement Regions")
        axes[1].scatter(p_tmod / 1e3, p, s=5)
        for t_vals in p_flat:
            p_mean.append(t_vals[0])
            axes[1].scatter(p_tmod[t_vals[1]] / 1e3, p[t_vals[1]], s=5)
        axes[1].set_title("View of Flat Pressure Regions")
        axes[1].set_ylabel("Pressure")
        axes[1].set_xlabel("Time (s)")
        plt.show()

        # ESTIMATE CONSTANT LAG # shorten region due to time mismatch
        d_flat_regions, p_flat_regions = [[] for i in range(2)]
        for temp in range(len(p_flat)):
            d_flat_regions.append(t_flat[temp][1][-3:])
            p_flat_regions.append(p_flat[temp][1][-3:])

        L = estimate_constant_lag(p_t[p_inds], p_flat_regions, d_t, d_flat_regions, method="wmean",
                                  include_bounds=True, weights="len")
        nfile.write(f"Constant lag (frames): {L:.1f}\n")
        nfile.write(f"Lag offset: {L_offset}\n")
        nfile.write(f"p_flat_len = {p_flat_len}\nt_flat_inds = {t_flat_inds}\np_flat_inds = {p_flat_inds}\n")
        nfile.write(f"t_len = {t_len}\nt_range = {t_range}\nmax_disp = {max_disp}\n\n")

        x_d2_warped = apply_constant_lag(d_t, L)
        p_int = griddata(p_t[p_inds], p, x_d2_warped, method='linear')
        p_prev = griddata(p_tmod, p, d_tmod, method='linear')
        plt.scatter(p_prev, d, label="Before Alignment", s=5)
        plt.scatter(p_int, d, label="After Alignment", s=5)
        x_lim = [np.min(d) - 0.005, np.max(d) * 1.05]
        y_lim = [np.min(p_prev) - 0.005, np.max(p_prev) * 1.05]
        plt.legend()
        plt.xlabel("Pressure")
        plt.ylabel("Displacement")
        plt.title("%s: Recorded Pressure" % select_dimple)
        # plt.show()
        plt.close()

        x_d2_warped = apply_constant_lag(d_t_init, L)
        print(int(L))
        if mod != 0:
            gen_aligned_mod(instrument_root, dic_root, dic_file, instrument_file, True, -int(L) + L_offset, save_folder=save_folder)
        else:
            gen_aligned_mod(instrument_root, dic_root, dic_file, instrument_file, True, int(d_t[0] - p_t[p_inds][0]), save_folder=save_folder)
        inds = np.where(p_inds)[0]
        p_int = griddata(p_t_init[inds[0]:], pressure[inds[0]:], x_d2_warped, method='linear')
        time_int = griddata(p_t_init[inds[0]:], p_t_init[inds[0]:], x_d2_warped, method='linear')
        np.save("%s/%s/1_%s_%s_prec_aligned.npy" % (instrument_root, save_folder, test_name, select_dimple), p_int.astype(float))
        np.save("%s/%s/1_%s_%s_ptime_aligned.npy" % (instrument_root, save_folder, test_name, select_dimple), time_int.astype(float))
                
        p_prev = griddata(p_t_init[inds[0]:] - p_t_init[inds[0]], pressure[inds[0]:], d_t_init - d_t_init[0], method='linear')
        p_time = griddata(p_t_init[inds[0]:], pressure[inds[0]:], d_t_init, method='linear')
        plt.scatter(p_prev, actual_peak_displacements, label="Before Alignment", s=5)
        plt.scatter(p_time, actual_peak_displacements, label="Time Alignment", s=5)
        df = pd.read_csv("%s/%s/%s_ALIGNED.csv" % (instrument_root, save_folder, instrument_file[:-4]))
        p_aligned = np.array(df['High Acc. Transducer (PSIG)'])  / 145
        t_aligned = np.array(df['time'])
        max_ind = np.min([len(actual_peak_displacements), len(p_aligned), len(t_aligned)])
        # plt.scatter(p_int, actual_peak_displacements, label="After Alignment", s=5)
        plt.scatter(p_aligned[:max_ind], actual_peak_displacements[:max_ind], label="After Alignment", s=5)
        plt.legend()
        plt.xlabel("Pressure")
        plt.ylabel("Displacement")
        plt.title("%s: Recorded Pressure" % select_dimple)
        plt.savefig("%s/%s/1_%s_%s_p_aligned.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.close()

        fig, ax1 = plt.subplots()
        ax1.scatter((p_t_init[inds[0]:] - p_t_init[inds[0]]) / 1e3, pressure[inds[0]:], c='b', s=2, label='Pressure')
        ax1.set_xlabel('Time')  # Change label as appropriate
        ax1.set_ylabel('Pressure [MPa]', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.scatter((d_t_init - d_t_init[0]) / 1e3, actual_peak_displacements, c='r', s=2, label='Actual Z at Center Point')
        ax2.set_ylabel('Z Displacement [mm]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
        plt.title("%s: Pressure vs Z-Displacement at Center (not aligned)" % select_dimple)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("%s/%s/1_%s_%s_dp_unaligned.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.close()

        fig, ax1 = plt.subplots()
        ax1.scatter((x_d2_warped - x_d2_warped[0]) / 1e3, p_int, c='b', s=2, label='Pressure')
        ax1.set_xlabel('Time')  # Change label as appropriate
        ax1.set_ylabel('Pressure [MPa]', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.scatter((x_d2_warped - x_d2_warped[0]) / 1e3, actual_peak_displacements, c='r', s=2, label='Actual Z at Center Point')
        ax2.set_ylabel('Z Displacement [mm]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
        plt.title("%s: Pressure vs Z-Displacement at Center (aligned)" % select_dimple)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("%s/%s/1_%s_%s_dp_aligned.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.close()

        fig, ax1 = plt.subplots()
        t_int = griddata(p_t_init[inds[0]:], -temperature[inds[0]:], x_d2_warped, method='linear')
        np.save("%s/%s/%s_%s_temp_aligned.npy" % (instrument_root, save_folder, test_name, select_dimple), t_int.astype(float))
        if neg_temp:
            ax1.scatter((x_d2_warped - x_d2_warped[0]) / 1e3, t_int, c='b', s=2, label='Temp')
            ax1.set_ylabel('Temp [-C]', color='b')
        else:
            ax1.scatter((x_d2_warped - x_d2_warped[0]) / 1e3, -t_int, c='b', s=2, label='Temp')
            ax1.set_ylabel('Temp [C]', color='b')
        ax1.set_xlabel('Time')  # Change label as appropriate
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.scatter((x_d2_warped - x_d2_warped[0]) / 1e3, actual_peak_displacements, c='r', s=2, label='Actual Z at Center Point')
        ax2.set_ylabel('Z Displacement [mm]', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
        plt.title("%s: Temp vs Z-Displacement at Center (aligned)" % select_dimple)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("%s/%s/1_%s_%s_dt_aligned.png" % (instrument_root, save_folder, test_name, select_dimple))
        plt.close()

for i in range(len(plot_dimples_all[test_index])):
    select_dimple = plot_dimples_all[test_index][i]
    _, p_t_init, actual_peak_displacements, d_t_init, d, d_t, initial_trange = dimple_info[i]
    df = pd.read_csv("%s/%s/%s_ALIGNED.csv" % (instrument_root, save_folder, instrument_file[:-4]))
    pressure = np.array(df['High Acc. Transducer (PSIG)'])  / 145
    time_init = np.array(df['time'])
    max_ind = np.min([len(actual_peak_displacements), len(pressure), len(time_init)])
    p = pressure[:max_ind]
    t = np.array(df["%s: Temp.(C)" % select_dimple])[:max_ind]
    d = actual_peak_displacements[:max_ind]
    time_mod = (time_init[:max_ind] - time_init[0]) / 1e3
    fig, ax1 = plt.subplots()
    ax1.scatter(time_mod, p, c='b', s=2, label='Pressure')
    ax1.set_xlabel('Time')  # Change label as appropriate
    ax1.set_ylabel('Pressure [MPa]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.scatter(time_mod, d, c='r', s=2,
                label='Actual Z at Center Point')
    ax2.set_ylabel('Z Displacement [mm]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    plt.title("%s: Pressure vs Z-Displacement at Center (aligned)" % select_dimple)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("%s/%s/%s_%s_dp_aligned.png" % (instrument_root, save_folder, test_name, select_dimple))
    plt.close()
    fig, ax1 = plt.subplots()
    if neg_temp:
        ax1.scatter(time_mod, -t, c='b', s=2, label='Temperature')
        ax1.set_ylabel('Temperature [-C]', color='b')
    else:
        ax1.scatter(time_mod, t, c='b', s=2, label='Temperature')
        ax1.set_ylabel('Temperature [C]', color='b')
    ax1.set_xlabel('Time')  # Change label as appropriate
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.scatter(time_mod, d, c='r', s=2,
                label='Actual Z at Center Point')
    ax2.set_ylabel('Z Displacement [mm]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    plt.title("%s: Temperature vs Z-Displacement at Center (aligned)" % select_dimple)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("%s/%s/%s_%s_dt_aligned.png" % (instrument_root, save_folder, test_name, select_dimple))
    plt.close()