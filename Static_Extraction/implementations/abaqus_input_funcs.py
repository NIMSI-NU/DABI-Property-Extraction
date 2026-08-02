import os
import jax
import jax.numpy as np
import jax.flatten_util
import xlsxwriter
from jax._src import source_info_util
from jax._src.interpreters.batching import (BatchTracer as BatchTracer)

# nset definition example (meshio_mesh.point_sets['Fixed_Region'])
# NOT USED BC REFERENCED ONP ARRAYS
# node_inds = onp.argwhere(jax.vmap(fixed_location)(self.mesh.points, np.arange(self.num_total_nodes))).reshape(-1)

def rot_gauss(xy, params):
    xv, yv = xy
    A, x0, y0, sx, sy, theta, p, offset = params
    ct, st = np.cos(theta), np.sin(theta)
    xp = ct*(xv-x0) + st*(yv-y0)
    yp = -st*(xv-x0) + ct*(yv-y0)
    core = (np.abs(xp/sx)**p + np.abs(yp/sy)**p)
    return offset + A * np.exp(-0.5 * core)

def def_fixed_location(nset):
    def fixed_location(point):
        n = np.zeros(point.val.shape[0])
        n = np.array(n.at[nset].set(1), dtype='bool')
        return BatchTracer(trace=point._trace, val=n, batch_dim=0)
    return fixed_location

# '''
def def_load_location(nset):
    def load_location(point):
        n = np.zeros(point.val.shape[0])
        n = np.array(n.at[nset].set(1), dtype='bool')
        return BatchTracer(trace=point._trace, val=n, batch_dim=0)
    return load_location


def format_time(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s"

'''
def def_load_location(nset, num_tot_nodes):
    n = np.zeros(num_tot_nodes, dtype=bool)
    n = n.at[nset].set(True)
    return np.squeeze(np.argwhere(n == True).T)

# MODDED DUE TO ERRORS
def def_fixed_location(nset):
    def fixed_location(point):
        n = np.zeros(point.val.shape[0], dtype=bool)
        n = n.at[nset].set(True)
        return BatchTracer(trace=point._trace, val=n, batch_dim=0)
    return fixed_location


def def_load_location(nset):
    def load_location(point):
        n = np.zeros(point.val.shape[0], dtype=bool)
        n = n.at[nset].set(True)
        return BatchTracer(trace=point._trace, val=n, batch_dim=0)
    return load_location'''

def abaqus_input_disp(data_dir, data_name):
    os.makedirs(data_dir, exist_ok=True)
    x = np.load('x_%s.npy' % data_name)
    np.save(os.path.join(data_dir, 'x_disp_jax'), x)
    y = np.load('y_%s.npy' % data_name)
    np.save(os.path.join(data_dir, 'y_disp_jax'), y)
    z = np.load('z_%s.npy' % data_name)
    np.save(os.path.join(data_dir, 'z_disp_jax'), z)

    x_disp_jax = np.load('%s/x_disp_jax.npy' % data_dir)
    sorted_order = np.argsort(x_disp_jax[0])
    y_disp_jax = np.load('%s/y_disp_jax.npy' % data_dir)
    z_disp_jax = np.load('%s/z_disp_jax.npy' % data_dir)

    def x_disp(scale):
        def val_fn(point):
            x, y, z = point[0], point[1], point[2]
            if scale == 0:
                x_disp = 0
            else:
                x_disp = BatchTracer(trace=point._trace, val= np.take_along_axis(x_disp_jax[scale, :],
                                                                                 sorted_order, axis=0), batch_dim=0)
            return np.nan_to_num(x_disp)
        return val_fn

    def y_disp(scale):
        def val_fn(point):
            x, y, z = point[0], point[1], point[2]
            if scale == 0:
                y_disp = 0
            else:
                y_disp = BatchTracer(trace=point._trace, val=np.take_along_axis(y_disp_jax[scale, :],
                                                                                sorted_order, axis=0), batch_dim=0)
            return np.nan_to_num(y_disp)
        return val_fn

    def z_disp(scale):
        def val_fn(point):
            x, y, z = point[0], point[1], point[2]
            if scale == 0:
                z_disp = 0
            else:
                z_disp = BatchTracer(trace=point._trace, val=np.take_along_axis(z_disp_jax[scale, :],
                                                                                sorted_order, axis=0), batch_dim=0)
            return np.nan_to_num(z_disp)
        return val_fn
    return x_disp, y_disp, z_disp

def dirichlet_val(point):
    return 0.

def get_plots_char(plots, header):
    plots_char = []
    for i in plots:
        char_temp = []
        for j in i:
            char_temp.append(chr(header.index(j) + 65))
        plots_char.append(char_temp)
    return plots_char
def header_expansion(str):
    return ['%s11' % str, '%s22' % str, '%s33' % str, '%s12' % str, '%s13' % str, '%s23' % str]
def single_elem_ordered_list(m):
    return [m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]]

def full_ordered_np_arr(m):
    return np.array([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], m[:, 0, 1], m[:, 0, 2], m[:, 1, 2]]).T

def full_unsym_np_arr(m):
    return np.array([m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 1, 0], m[:, 1, 1], m[:, 1, 2], m[:, 2, 0],
                     m[:, 2, 1], m[:, 2, 2]]).T

def elem_boundary_inds(problem, location_fn):
    load_inds_fn = jax.vmap(location_fn)
    load_nodes = np.nonzero(load_inds_fn(problem.points))[0]
    load_elements = []
    for i in range(len(load_nodes)):
        for j in np.where(load_nodes[i] == problem.cells)[0]:
            load_elements.append(j)
    load_elements_arr = np.unique(np.array(load_elements))
    num_l_elem = load_elements_arr.shape[0]
    elem_b_inds = np.zeros((num_l_elem, 2))
    elem_b_inds = elem_b_inds.at[:, 0].set(load_elements_arr)
    elem_b_inds = elem_b_inds.at[:, 1].set(3 * np.ones(num_l_elem))
    return np.int64(elem_b_inds)

def save_to_xlsx(filename, data, plot_range=None, plot_title=None, axis_title=None):
    workbook = xlsxwriter.Workbook('%s.xlsx' % filename, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()
    worksheet.write_row('A1', data[0])  # Write the header row
    for row_num, row_data in enumerate(data[1:], start=1):
        worksheet.write_row(row_num, 0, row_data)
    if plot_range != None:
        for i in range(len(plot_range)):
            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})  # Scatter plot with markers
            chart.add_series({'name': '=Sheet1!$B$1',
                              'categories': '=Sheet1!$%s$2:$%s$%i' % (plot_range[i][0], plot_range[i][0], row_num),
                              'values': '=Sheet1!$%s$2:$%s$%i' % (plot_range[i][1], plot_range[i][1], row_num),
                              'marker': {'type': 'circle', 'size': 6},  # Optional: Specify marker type and size
                              })
            chart.set_title({'name': '%s' % plot_title[i]})
            chart.set_x_axis({'name': '%s' % axis_title[i][0], 'major_gridlines': {'visible': False}})
            chart.set_y_axis({'name': '%s' % axis_title[i][1], 'major_gridlines': {'visible': False}})

            worksheet.insert_chart('%c2' % chr(65 + i*7), chart)
    workbook.close()


def elem_boundary_inds_file(filename):
    """
    Parses an Abaqus surface definition from an .inp file and converts it
    to JAX-FEM's boundary format. This version correctly handles:
    - The 'generate' keyword in element sets.
    - Mixed parameters on keyword lines.
    - Multiple data lines under a single *Surface definition.

    Args:
        filename (str): The name of the .inp file (without extension) located
                        in the 'abaqus' subdirectory.

    Returns:
        jax.numpy.ndarray: An array of shape (num_boundary_faces, 2) where
                           the first column is the element index (0-based) and the
                           second column is the local face index.
    """
    # Look for abaqus folder in parent directory (../abaqus from implementations/)
    filepath = os.path.join(os.path.dirname(__file__), '..', 'abaqus', f"{filename}.inp")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Abaqus input file not found at: {filepath}")

    with open(filepath, 'r') as f:
        # Read all non-empty lines and remove leading/trailing whitespace
        lines = [line.strip() for line in f if line.strip()]

    # Mapping for Abaqus face identifiers to JAX-FEM local face indices.
    # This can be expanded for other element types, e.g., 3D hexes (S1-S6).
    # abaqus_to_jax_face_map = {'S1': 0, 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5}
    abaqus_to_jax_face_map = {'S1': 0, 'S2': 2, 'S3': 3, 'S4': 1, 'S5': 4, 'S6': 5}
    element_sets = {}
    
    # --- Pass 1: Parse all *Elset definitions and store them ---
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lower().startswith('*elset'):
            # Robustly parse parameters like 'elset=name', 'generate', etc.
            params = {}
            parts = [part.strip() for part in line.split(',')[1:]]
            for p in parts:
                if '=' in p:
                    key, value = p.split('=', 1)
                    params[key.strip().lower()] = value.strip()
                else:
                    params[p.strip().lower()] = True
            
            elset_name = params.get('elset')
            has_generate = params.get('generate', False)
            
            if elset_name:
                elem_numbers = []
                i += 1 # Move to the data lines
                while i < len(lines) and not lines[i].startswith('*'):
                    num_parts = [p.strip() for p in lines[i].split(',') if p.strip()]
                    
                    if len(num_parts) == 3 and has_generate:
                        # Handle 'generate' keyword, e.g., "1, 88, 1"
                        try:
                            start, end, step = map(int, num_parts)
                            elem_numbers.extend(np.arange(start, end + 1, step) - 1)
                        except (ValueError, IndexError):
                            elem_numbers.extend([int(p) - 1 for p in num_parts])
                    else:
                        # Handle a regular list of element numbers
                        elem_numbers.extend([int(p) - 1 for p in num_parts])
                    i += 1
                element_sets[elset_name.lower()] = elem_numbers
                continue # Go to the next line in the outer while loop
        i += 1

    # --- Pass 2: Parse all *Surface definitions using the stored elsets ---
    boundary_faces = []
    i = 0
    while i < len(lines):
        line = lines[i].lower()
        if line.startswith('*surface'):
            i += 1 # Move to the first data line
            # *** NEW: Loop to handle multiple data lines for one surface ***
            while i < len(lines) and not lines[i].startswith('*'):
                parts = lines[i].split(',')
                if len(parts) == 2:
                    elset_name = parts[0].strip().lower()
                    face_identifier = parts[1].strip().upper()

                    local_face_index = abaqus_to_jax_face_map.get(face_identifier)
                    if local_face_index is None:
                        raise ValueError(f"Unknown Abaqus face identifier '{face_identifier}' found.")

                    if elset_name in element_sets:
                        for elem_idx in element_sets[elset_name]:
                            boundary_faces.append([elem_idx, local_face_index])
                    else:
                        raise ValueError(f"Element set '{elset_name}' for surface definition not found.")
                i += 1
            continue # Go to the next line in the outer while loop
        i += 1
            
    if not boundary_faces:
        raise ValueError(f"Could not find a valid *Surface definition in file '{filename}.inp'")

    return np.array(boundary_faces, dtype=np.int64)


def elem_boundary_inds_file1(filename):
    # file = open('abaqus/%s.inp' % filename, 'r')
    file = open(os.path.join(os.path.dirname(__file__), 'abaqus/%s.inp' % filename), 'r')
    lines = file.readlines()
    file.close()
    elem_surfs = []
    elem_len = 0
    l = 0
    while l < len(lines):
        curr = int(lines[l].split(',')[1][-1])
        temp_e = []
        while l < len(lines) - 1 and lines[l + 1][0] != '*':
            t = lines[l + 1].split(',')
            for i in t:
                temp_e.append(int(i) - 1)
            l = l + 1
        e = np.array(temp_e)
        elem_surfs.append([e, curr])
        elem_len = elem_len + len(e)
        l = l + 1

    aba_to_jax = np.array([0,2,3,1], dtype=int)
    # aba_to_jax = np.array([0, 1, 2, 3], dtype=int)
    elem_b_inds = np.zeros((elem_len, 2))
    curr_range = 0
    for i in range(len(elem_surfs)):
        l = len(elem_surfs[i][0])
        elem_b_inds = elem_b_inds.at[curr_range:l + curr_range, 0].set(elem_surfs[i][0])
        elem_b_inds = elem_b_inds.at[curr_range:l + curr_range, 1].set(
            aba_to_jax[elem_surfs[i][1] - 1] * np.ones(l))
        curr_range = curr_range + l
    return np.int64(elem_b_inds)