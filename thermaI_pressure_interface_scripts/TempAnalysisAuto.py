# -*- coding: utf-8 -*-
"""
TempAnalysisAuto.py
-------------------
Batch-load temperature fields from .npy files, align them with elapsed
timestamps from a CSV, compute mean/std inside user-defined circular ROIs,
and provide an interactive viewer with enterable controls (TextBox for
index/vmin/vmax and Buttons to toggle overlays).

Spyder-friendly:
- Edit the CONFIG section and press Run (F5), or:
      import TempAnalysisAuto as taa
      df, df_temporal = taa.run(root="...", rois="circle_rois.csv")

Outputs:
    - stats_per_frame.csv
    - stats_temporal.csv
    - temp_viewer_cache.npz (unless disabled)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================== CONFIG =====================================
# CONFIG = {
#     'AUTO_RUN': True,                # If True and file executed, run with values below
#     'root': os.getcwd(),             # Folder with .npy frames and timestamps CSV
#     'pattern': 'temperature_field_{i}_e=0.99.npy',
#     'timestamps': 'image_timestamps.csv',   # CSV filename or absolute path
#     'rois': None,                    # 'circle_rois.csv' or None to use DEFAULT_GRID
#     'save_npz': True,                # Save NPZ cache
#     'launch_viewer': True            # Launch the interactive viewer
# }

TopRoot = r"V:\Data\Specimen Directory\Bonded_DABI\RAD-BD-005\(250818) Creep + Static Experiment\Test Data\Temperature_Fields"
CONFIG = {
    'AUTO_RUN': True,
    'root': rf'{TopRoot}\numpy',          # <-- set your folder here
    'pattern': 'temperature_field_{i}_e=0.99.npy',
    'timestamps': rf'{TopRoot}\image_timestamps.csv',  # or just 'image_timestamps.csv' if inside root
    'skip_temporal_frames': [1], # [0,1,2,3,4,5,35,36,37,38], # 31,32,33,34,35,36,[1,2,3,4,5,6,7,31,32,33,34,35,36,37,38],
    'rois': None, # r'D:\Lab\TempExp\Run_07\circle_rois.csv',             # or None if using DEFAULT_GRID
    'save_npz': True,
    'launch_viewer': True
}

stats_per_frame = "stats_per_frame"
stats_temporal = "stats_temporal"

# Index / vmin / vmax are now TextBox inputs (type values and hit Apply or press Enter).
# Circles/Text overlays are toggled by buttons (no more checkboxes), and also by keys:
# Press C to toggle circles
# Press T to toggle text

# Example default grid (if you don't want a CSV). Uncomment+edit:
# DEFAULT_GRID = None
DEFAULT_GRID = {
    'rows': ['5','4','3','2','1'],
    'cols': ['A','B','C','D','E'],
    'centers': {
        ('A','1'):(122,430), ('B','1'):(218,430), ('C','1'):(312,430), ('D','1'):(406,430), ('E','1'):(501,430),
        ('A','2'):(122,335), ('B','2'):(217,335), ('C','2'):(311,335), ('D','2'):(405,335), ('E','2'):(499.5,335),
        ('A','3'):(122,240), ('B','3'):(215,240), ('C','3'):(310,240), ('D','3'):(405,240), ('E','3'):(496,240),
        ('A','4'):(122,145), ('B','4'):(215,145), ('C','4'):(309.5,145), ('D','4'):(400,145), ('E','4'):(495,145),
        ('A','5'):(122,50),  ('B','5'):(215,50),  ('C','5'):(309.5,50),  ('D','5'):(400,50),  ('E','5'):(492,50),
    },
    'r': 33, # voxel unit not equal to 40 to avoid boundary
}
# ===========================================================================

# ---------------------------- Utilities ------------------------------------

def find_files(root, pattern):
    """Find all .npy files matching pattern '...{i}...' and return sorted (image_index, path)."""
    gpat = pattern.replace('{i}', '*')
    paths = sorted(glob.glob(os.path.join(root, gpat)))
    out = []
    if '{i}' not in pattern:
        raise ValueError("pattern must include '{i}' placeholder for time step.")
    prefix, suffix = pattern.split('{i}')
    for p in paths:
        fname = os.path.basename(p)
        if not fname.startswith(prefix) or not fname.endswith(suffix):
            continue
        core = fname[len(prefix):-len(suffix)]
        try:
            idx = int(core)
            out.append((idx, p))
        except ValueError:
            continue
    out.sort(key=lambda t: t[0])
    return out

def read_timestamps(csv_path):
    """Read timestamps CSV with columns: image,time,elapsed_time_s. Return dict image->elapsed_time_s."""
    df = pd.read_csv(csv_path)
    if not set(['image', 'elapsed_time_s']).issubset(df.columns):
        raise ValueError("timestamps CSV must contain columns: 'image', 'elapsed_time_s'")
    mapping = {}
    for _, row in df.iterrows():
        try:
            mapping[int(row['image'])] = float(row['elapsed_time_s'])
        except Exception:
            continue
    return mapping

def read_rois(rois_csv=None, default_grid=None):
    """
    Read circular ROIs from CSV (label,x,y,r) or from default_grid structure.
    Returns list of dicts: [{'label','x','y','r'}, ...]
    """
    rois = []
    if rois_csv and os.path.isfile(rois_csv):
        df = pd.read_csv(rois_csv)
        need_cols = {'label', 'x', 'y', 'r'}
        if not need_cols.issubset(df.columns):
            raise ValueError("ROI CSV must contain columns: label,x,y,r")
        for _, row in df.iterrows():
            rois.append({'label': str(row['label']), 'x': float(row['x']), 'y': float(row['y']), 'r': float(row['r'])})
        return rois
    if default_grid is not None:
        rois = []
        for r in default_grid['rows']:
            for c in default_grid['cols']:
                label = f"{c}{r}"
                (x, y) = default_grid['centers'][(c, r)]
                rois.append({'label': label, 'x': float(x), 'y': float(y), 'r': float(default_grid['r'])})
        return rois
    raise ValueError("No ROIs provided. Supply rois CSV or define DEFAULT_GRID.")

def mask_circle(h, w, cx, cy, r):
    """Boolean mask of a filled circle at (cx,cy), radius r for an array (h,w)."""
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx)**2 + (yy - cy)**2 <= r**2

def compute_stats_for_frame(temp2d, rois):
    """Compute mean/std for each circular ROI on a single 2D array."""
    h, w = temp2d.shape
    mean_d, std_d = {}, {}
    for rdef in rois:
        lab = rdef['label']
        cx, cy, rr = rdef['x'], rdef['y'], rdef['r']
        xmin = max(0, int(np.floor(cx - rr)))
        xmax = min(w-1, int(np.ceil (cx + rr)))
        ymin = max(0, int(np.floor(cy - rr)))
        ymax = min(h-1, int(np.ceil (cy + rr)))
        sub = temp2d[ymin:ymax+1, xmin:xmax+1]
        mask = mask_circle(sub.shape[0], sub.shape[1], cx - xmin, cy - ymin, rr)
        vals = sub[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean_d[lab] = np.nan
            std_d[lab]  = np.nan
        else:
            mean_d[lab] = float(np.nanmean(vals))
            std_d[lab]  = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
    return mean_d, std_d

def create_plot_summary(df):
    """Optional: create summary plots of temporal means/stds across ROIs."""
    # create 5 x 5 grid of subplots for each ROI label, showing mean and std over time
    # x-axis: elapsed_time_s, y-axis: mean or std, one line per ROI
    # y-axis: mean, with error bars for std, one line per ROI
    labels = sorted(
        {col[:-5] for col in df.columns if col.endswith('_mean')},
        key=lambda s: (s.rstrip('0123456789'), int(''.join(ch for ch in s if ch.isdigit()) or 0))
    )

    fig, axes = plt.subplots(5, 5, figsize=(18, 14), sharex=True)
    axes = axes.T.ravel()

    for ax, lab in zip(axes, labels):
        mean_col = f'{lab}_mean'
        std_col = f'{lab}_std'
        if mean_col in df.columns and std_col in df.columns:
            ax.errorbar(
                df['elapsed_time_s'],
                df[mean_col],
                yerr=df[std_col],
                marker='o',
                linestyle='-',
                markersize=3,
                linewidth=1
            )
            ax.set_title(lab)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(lab)
            ax.axis('off')

    for ax in axes[len(labels):]:
        ax.axis('off')

    fig.supxlabel('Elapsed Time (s)')
    fig.supylabel('Mean Temperature (°C)')
    plt.tight_layout()
    
    # save stdfig to root directory
    fig.savefig(os.path.join(CONFIG['root'], 'summary_mean_over_time.png'))
    
# ---------------------------- Main Logic -----------------------------------

def analyze(root, pattern, timestamps_csv, rois_csv=None, default_grid=None, save_npz=True):
    files = find_files(root, pattern)
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {root} matching pattern {pattern}")
    ts_map = read_timestamps(timestamps_csv)
    rois = read_rois(rois_csv, default_grid)

    records, temp_cache, image_indices, elapsed_times = [], [], [], []
    for img_idx, path in files:
        temp = np.load(path)
        # temp_flipped = temp[::-1, :] # flip vertically
        temp_flipped = temp
        temp_flipped = np.array(temp_flipped, dtype=float, copy=False)
        temp_cache.append(temp_flipped)
        image_indices.append(img_idx)
        elapsed = ts_map.get(img_idx, np.nan)
        elapsed_times.append(elapsed)

        mean_d, std_d = compute_stats_for_frame(temp_flipped, rois)
        row = {'image': img_idx, 'elapsed_time_s': elapsed}
        for lab in mean_d:
            row[f'{lab}_mean'] = mean_d[lab]
            row[f'{lab}_std']  = std_d[lab]
        records.append(row)

    df = pd.DataFrame.from_records(records).sort_values('image').reset_index(drop=True)

    # Temporal aggregates (skip first N frames for temporal means)
    # Temporal aggregates (skip selected frames by image index for temporal means)
    skip_list = set(int(x) for x in CONFIG.get('skip_temporal_frames', []))
    df_eff = df[~df['image'].isin(skip_list)].reset_index(drop=True)

    # skipN = int(CONFIG.get('skip_initial_frames', 0))
    # df_eff = df.iloc[skipN:].reset_index(drop=True) if skipN < len(df) else df.iloc[0:0]
    
    t_rows = []
    for rdef in rois:
        lab = rdef['label']
    
        # split label "A1" -> col="A", row="1"
        col_label = ''.join(ch for ch in lab if ch.isalpha())
        row_label = ''.join(ch for ch in lab if ch.isdigit())
    
        mean_series_eff = df_eff.get(f'{lab}_mean', pd.Series([], dtype=float))
        std_series_eff  = df_eff.get(f'{lab}_std',  pd.Series([], dtype=float))
    
        t_mean_means = float(np.nanmean(mean_series_eff.values)) if len(mean_series_eff) > 0 else np.nan
        # t_mean_stds  = float(np.nanmean(std_series_eff.values))  if len(std_series_eff)  > 0 else np.nan
        t_mean_stds  = float(np.nanstd(mean_series_eff.values))  if len(mean_series_eff)  > 0 else np.nan
        
        t_rows.append({
            'col': col_label,                      # <— NEW
            'row': row_label,                      # <— NEW
            'label': lab,                          # keep if you still want it
            'temporal_mean_of_means': t_mean_means,
            'temporal_mean_of_stds':  t_mean_stds
        })
    
    df_temporal = pd.DataFrame(t_rows).sort_values(['col','row'])
    
    # Optional: order columns
    df_temporal = df_temporal[['col','row','label','temporal_mean_of_means','temporal_mean_of_stds']]
    
    # Save CSVs
    df.to_csv(os.path.join(root, f'{stats_per_frame}.csv'), index=False)
    df_temporal.to_csv(os.path.join(root, f'{stats_temporal}.csv'), index=False)

    if save_npz:
        np.savez_compressed(os.path.join(root, 'temp_viewer_cache.npz'),
                            stack=np.array(temp_cache, dtype=object),
                            image_indices=np.array(image_indices),
                            elapsed_time_s=np.array(elapsed_times, dtype=float))

    return df, df_temporal, temp_cache, image_indices, elapsed_times, rois

# ---------------------------- Viewer (TextBoxes & Buttons) -----------------

def launch_viewer(stack, image_indices, elapsed_times, rois, cmap='hot'):
    """
    Interactive matplotlib viewer with enterable controls.
    Controls:
        - TextBox: Index (integer)
        - TextBox: vmin, vmax (floats)
        - Buttons: Toggle Circles, Toggle Text
        - Keyboard: 'c' toggles circles, 't' toggles text, 'enter' applies
    """
    if len(stack) == 0:
        raise ValueError("Empty stack.")

    state = {'idx': 0, 'show_circles': True, 'show_text': True}

    fig, ax = plt.subplots(figsize=(10, 7))
    # Leave space at bottom for controls
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.22, top=0.92)

    im = ax.imshow(stack[state['idx']], cmap=cmap, origin='lower')
    cb = fig.colorbar(im, ax=ax)
    ax.set_title(f"Temperature Field | frame={image_indices[state['idx']]} | t={elapsed_times[state['idx']]:.3f}s")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    circle_patches, text_artists = [], []

    def update_overlays(i):
        for art in circle_patches + text_artists:
            try:
                art.remove()
            except Exception:
                pass
        circle_patches.clear()
        text_artists.clear()
        if not (state['show_circles'] or state['show_text']):
            fig.canvas.draw_idle()
            return
        arr = stack[i]
        means, stds = compute_stats_for_frame(arr, rois)
        for rdef in rois:
            lab = rdef['label']; cx, cy, rr = rdef['x'], rdef['y'], rdef['r']
            if state['show_circles']:
                circ = Circle((cx, cy), rr, fill=False, linewidth=1.5)
                ax.add_patch(circ); circle_patches.append(circ)
            if state['show_text']:
                m = means.get(lab, np.nan); s = stds.get(lab, np.nan)
                txt = f"{lab}: {m:.1f}±{s:.1f}°C" if np.isfinite(m) else f"{lab}: nan"
                t = ax.text(cx, cy + 0.05*rr, txt, ha='center', va='bottom', fontsize=8,
                            bbox=dict(fc='white', ec='none', alpha=0.5, pad=0.2))
                text_artists.append(t)
        fig.canvas.draw_idle()

    update_overlays(state['idx'])

    # Controls at the bottom using TextBoxes and Buttons
    from matplotlib.widgets import TextBox, Button

    ax_idx = plt.axes([0.08, 0.12, 0.15, 0.05])
    tb_idx = TextBox(ax_idx, 'Index', initial=str(state['idx']))

    data0 = stack[state['idx']]
    vmin0 = float(np.nanmin(data0)); vmax0 = float(np.nanmax(data0))
    ax_vmin = plt.axes([0.28, 0.12, 0.15, 0.05]); tb_vmin = TextBox(ax_vmin, 'vmin', initial=f"{vmin0:.3f}")
    ax_vmax = plt.axes([0.48, 0.12, 0.15, 0.05]); tb_vmax = TextBox(ax_vmax, 'vmax', initial=f"{vmax0:.3f}")

    ax_apply = plt.axes([0.68, 0.12, 0.08, 0.05]); btn_apply = Button(ax_apply, 'Apply/Enter')
    ax_tc = plt.axes([0.78, 0.12, 0.10, 0.05]); btn_circ = Button(ax_tc, 'Toggle Circles (C)')
    ax_tt = plt.axes([0.89, 0.12, 0.08, 0.05]); btn_text = Button(ax_tt, 'Toggle Text (T)')

    def apply_changes(event=None):
        # index
        try:
            idx = int(tb_idx.text)
            idx = max(0, min(len(stack)-1, idx))
        except Exception:
            idx = state['idx']
        state['idx'] = idx
        im.set_data(stack[idx])
        ax.set_title(f"Temperature Field | frame={image_indices[idx]} | t={elapsed_times[idx]:.3f}s")

        # vmin/vmax
        try:
            vmin = float(tb_vmin.text)
            vmax = float(tb_vmax.text)
            if vmin < vmax:
                im.set_clim(vmin=vmin, vmax=vmax)
        except Exception:
            pass
        update_overlays(idx)

    def toggle_circles(event=None):
        state['show_circles'] = not state['show_circles']
        update_overlays(state['idx'])

    def toggle_text(event=None):
        state['show_text'] = not state['show_text']
        update_overlays(state['idx'])

    btn_apply.on_clicked(apply_changes)
    btn_circ.on_clicked(toggle_circles)
    btn_text.on_clicked(toggle_text)

    def on_key(event):
        if event.key in ('enter', 'return'):
            apply_changes()
        elif event.key and event.key.lower() == 'c':
            toggle_circles()
        elif event.key and event.key.lower() == 't':
            toggle_text()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# ---------------------------- Public API -----------------------------------

def run(root=None, pattern=None, timestamps=None, rois=None, save_npz=True, no_view=False, default_grid=DEFAULT_GRID):
    """Convenience function to run from Spyder Console or scripts."""
    root = root or CONFIG['root']
    pattern = pattern or CONFIG['pattern']
    # Resolve timestamps/rois relative to root if given as filenames
    ts_path = timestamps or CONFIG['timestamps']
    if not os.path.isabs(ts_path):
        ts_path = os.path.join(root, ts_path)
    rois_path = rois or CONFIG['rois']
    if rois_path and not os.path.isabs(rois_path):
        rois_path = os.path.join(root, rois_path)

    df, df_temporal, stack, image_indices, elapsed_times, rois_list = analyze(
        root=root, pattern=pattern, timestamps_csv=ts_path, rois_csv=rois_path,
        default_grid=default_grid, save_npz=save_npz
    )
    print("Per-frame stats ->", os.path.join(root, f'{stats_per_frame}.csv'))
    print("Temporal stats  ->", os.path.join(root, f'{stats_temporal}.csv'))

    if not no_view and CONFIG.get('launch_viewer', True):
        launch_viewer(stack, image_indices, elapsed_times, rois_list)
    return df, df_temporal

# ---------------------------- CLI -----------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Batch temperature analysis with circular ROIs")
    ap.add_argument('--root', type=str, required=False, help='Folder containing .npy frames and timestamps CSV')
    ap.add_argument('--pattern', type=str, default=None, help='Filename pattern with {i} placeholder')
    ap.add_argument('--timestamps', type=str, default=None, help='CSV with columns: image,time,elapsed_time_s')
    ap.add_argument('--rois', type=str, default=None, help='CSV: label,x,y,r (pixels)')
    ap.add_argument('--no_npz', action='store_true', help='Do not save temp_viewer_cache.npz')
    ap.add_argument('--no_view', action='store_true', help='Skip launching viewer')
    args = ap.parse_args()
    run(root=args.root or CONFIG['root'],
        pattern=args.pattern or CONFIG['pattern'],
        timestamps=args.timestamps or CONFIG['timestamps'],
        rois=args.rois or CONFIG['rois'],
        save_npz=not args.no_npz,
        no_view=args.no_view)

if __name__ == '__main__':
    if CONFIG.get('AUTO_RUN', False):
        df, df_temporal = run()
        create_plot_summary(df)
    else:
        _cli()
