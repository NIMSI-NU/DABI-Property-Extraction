import os
import sys
import glob
import time
import meshio
import jax
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize, Bounds

# Ensure path includes root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jax_fem import logger
import logging
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh
from implementations.abaqus_input_funcs import def_fixed_location, dirichlet_val, elem_boundary_inds_file
from implementations.axisymmetric_problem_coupled_V11 import *
# Import modular tools
from Solver.solver_mod_2_new import solver, ad_wrapper
from time_stepping_core import TimeStepConfig
from refactored_J_total_DIC_POnly import create_J_total_DIC

# --- Main Simulation Function ---
def simulation(mesh_file, dir_name, dimple_name, test_name, root, version, rho_ini=None, strain=None, Voce=True, rho_fixed = None, rho_fixed_2 = None, mesh_folder='Mesh', skip=None):
    init_rho_ini = np.copy(rho_ini)
    logger = logging.getLogger('jax_fem')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 1. Select Material Model
    if Voce:
        param_bounds = [[100.0e3, 300.0e3], [100.0, 250.0], [80.0, 3000.0], [0.4, 10], [-60.0, 0.1], [3.0, 11.0]]
    else:
        # base_class = Plasticity_tab
        param_bounds = [[100.0e3, 300.0e3], [100.0, 250.0], [-60.0, 0.1], [3.0, 11.0], [5.0, 100.0]]

    # 2. Paths and Folders
    root_path = os.path.join(os.path.dirname(__file__), root)
    sol_dir = f'{root_path}/Gaussian_fit_{dimple_name}_{version}' if test_name == "" else f'{root_path}/{test_name}/{dimple_name}/Mesh_fit_{dimple_name}_{version}'
    data_dir = os.path.join(os.path.dirname(__file__), dir_name)
    os.makedirs(data_dir, exist_ok=True)
    
    for f in glob.glob(os.path.join(data_dir, f'*')):
        if os.path.isfile(f): os.remove(f)

    file_handler = logging.FileHandler('%s/%s_log_file.txt' % (data_dir, dir_name), mode='w')
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Param bounds = {param_bounds}\n")
    
    # 3. Mesh and BCs
    mesh_file_path = f"{root_path}/{mesh_folder}/{mesh_file}.inp"
    meshio_mesh = meshio.read(mesh_file_path)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])
    meshio_mesh_sets = meshio.read(mesh_file_path)
    try:
        top_nodes = meshio_mesh_sets.point_sets['Top']
    except KeyError:
        max_y = np.max(mesh.points[:, 1])
        top_nodes = np.sort(np.where(mesh.points[:, 1] > max_y - 1e-3)[0])

    fixed_location = def_fixed_location(meshio_mesh_sets.point_sets['Fixed'])
    fixed_X0 = def_fixed_location(meshio_mesh_sets.point_sets['X0'])
    load_location = elem_boundary_inds_file(f"../{os.path.dirname(__file__).split('/')[-1]}/" +
                                            f"{root}/{mesh_folder}/{mesh_file}")
    
    dirichlet_bc_info = [[fixed_location] * 2 + [fixed_X0], [0, 1, 0], [dirichlet_val] * 3]

    # Initialize Problem
    problem = PlasticCreepAxisy(mesh=mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info, 
                                location_fns=[load_location], rho_ini=rho_ini,
                                param=param_bounds, p_bounds=True)
    best_file = open('%s/%s_best_rho.log' % (data_dir, dir_name), 'w')
    problem.best_file = best_file
    if not Voce:
        problem.strain_values = strain

    # 4. Extract Loading and Reference Target Data
    pressure_arr = np.load(os.path.join(sol_dir, f'pressure_select.npy'))
    mask = onp.array([True] * len(pressure_arr))
    if skip != None:
        skip_arr = np.array(skip)
        np.save('%s/skip.npy' % data_dir, skip_arr)
        mask[skip_arr] = False
        pressure_arr = pressure_arr[mask]
    max_pressure = np.max(pressure_arr)
    scales_full = pressure_arr / max_pressure
    problem.pressure_mag = -max_pressure

    times = np.load(os.path.join(sol_dir, f'time_select.npy'))
    if skip != None:
        times = times[mask]
    times_array = np.concatenate((np.array([0.0]), times))
    pressure_array = np.concatenate((np.array([0.0]), scales_full))

    # DIC Reference Settings
    radius = mesh.points[top_nodes, 0] < 5.
    dic_inds = top_nodes[radius]
    problem.target_dof = dic_inds
    weight = np.ones(mesh.points[problem.target_dof, 0].shape)
    problem.nodal_weight = weight / np.sum(weight)
    
    dic_disp_full_old = []
    gaussian_param = onp.load(os.path.join(sol_dir, f'Gaussian_param.npy'))
    if skip != None:
        gaussian_param = gaussian_param[mask]
    x_coords = mesh.points[problem.target_dof, 0]
    for i in range(gaussian_param.shape[0]):
        dic_disp_full_old.append(rot_gauss((x_coords, onp.zeros_like(x_coords)), gaussian_param[i]))
    dic_disp_full_old = np.array(dic_disp_full_old)
    dic_disp_full = np.zeros((dic_disp_full_old.shape[0] + 1, dic_disp_full_old.shape[1]))
    dic_disp_full = dic_disp_full.at[1:, :].set(dic_disp_full_old)
    
    problem.sol_reference_target_full = dic_disp_full
    problem.internal_vars_init = problem.internal_vars

    # 5. Build Differentiable Adjoint Solver Wrapper
    solver_options = {'pardiso_solver': {}, 'line_search_flag': True, 'tol': 1e-6, 'rel_tol': 1e-8}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    n_press = np.sum(pressure_arr < 12.) + 1
    time_list = [times_array[:n_press], times_array]
    print(f"First pressure loop: {pressure_array[:n_press] * max_pressure}")
    max_iters = [20, 30]
    logger.info(f"Pressure: {pressure_array * max_pressure}")
    best_file.write(f"Pressure: {pressure_array * max_pressure}\n")
    total_start = time.time()
    for outer_loop in range(0, len(time_list)):
        loop_start = time.time()
        # 6. Configure the Modular Time Stepping
        rho_fixed = None; rho_fixed_2 = None; rho_fixed_3 = None;
        rho_fixed_2 = rho_ini[-2:]
        rho_ini = rho_ini[:-2]
        
        curr_time = time_list[outer_loop]
        problem.sol_reference_target = problem.sol_reference_target_full[:len(curr_time)]
        config = TimeStepConfig(total_time=float(times_array[len(curr_time)]), 
                                initial_dt=float(times_array[1] - times_array[0]), min_dt=1e-5, 
                                max_dt=float(np.max(times_array) * 0.2), target_iters=15, time_tol=1e-3,
                                max_retries=3, increase_factor=1.5, decrease_factor=0.5,
                                time_markers=curr_time[1:].tolist() # Stop exactly at these reference frames
                                )

        J_total_func = create_J_total_DIC(problem=problem, fwd_pred=fwd_pred, fwd_pred_solver_options=solver_options,
                                          times_array=times_array, pressure_array=pressure_array, 
                                          target_sol_array=problem.sol_reference_target, 
                                          nodal_weights=problem.nodal_weight, w_u=1e2, w_dt=0.3, config=config, 
                                          rho_fixed=rho_fixed, rho_fixed_2=rho_fixed_2, fwd=False)

        # 7. Optimizer Execution
        opt_logger = OptimizationLogger(data_dir, len(rho_ini), outer_loop)
        opt_problem = OptimizationProblem(J_total_func, opt_logger, data_dir, dir_name, problem, logger, outer_loop)

        num_params = len(rho_ini)
        bounds = Bounds(lb=tuple(-1.0 * onp.ones(num_params)), ub=tuple(onp.ones(num_params)))
        try:
            result = minimize(fun=opt_problem.objective, x0=onp.array(rho_ini, dtype=onp.float64),
                method='SLSQP', callback=opt_problem.early_stopping_callback, jac=opt_problem.gradient, 
                bounds=bounds, options={'maxiter': max_iters[outer_loop], 'ftol': 1e-10, 'disp': True})
            logger.info(result.message)
            global_params = np.array(result.x)
            if rho_fixed_3 is None:
                parts = [x for x in (rho_fixed, global_params, rho_fixed_2) if x is not None]
            else:
                parts = [x for x in (rho_fixed, global_params[:-1], rho_fixed_3, global_params[-1:]) if x is not None]
            rho_ini = np.concatenate(parts)
        except StopIteration:
            logger.info("\nStop Iteration Triggered!")
            rho_ini = np.array(opt_problem.best_rho)
        print("\nOptimization Finished!")
        logger.info(f"Optimal rho: {rho_ini}")
        best_file.write(opt_problem.best_info + "\n")
        s = f"Loop optimization time: {format_time(time.time() - loop_start)}\n"
        best_file.write(s)
        logger.info(s)
    s = f"\nTotal optimization time: {format_time(time.time() - total_start)}\n"
    logger.info(s)
    best_file.write(s)
    best_file.close()

if __name__ == "__main__":
    # Test Parameters
    E, sig0, Q, b, A, n = 100., 100., 100., 0.4, 1.00E-60, 6.3
    param = [[100.0e3, 300.0e3], [100.0, 250.0], [80.0, 3000.0], [0.4, 10], [-60.0, 0.1], [3.0, 11.0]]
    
    # Map back to normalized [-1, 1] starting values
    rho_ini_full = (2 * np.array([(E - param[0][0]) / param[0][1], (sig0 - param[1][0]) / param[1][1], 
                             (Q - param[2][0]) / param[2][1], (b - param[3][0]) / param[3][1], 
                             (np.log10(A) - param[4][0]) / param[4][1], (n - param[5][0]) / param[5][1]])) - 1
    rho_ini = np.copy(rho_ini_full)
    rho_ini = rho_ini.at[:-2].set(-0.4)
    v = "1"
    root = "HT_250818"
    for i in ['D1', 'D2', 'D3', 'D4', 'D5']:
        mesh_folder = 'Mesh'
        print(f"Running simulation for dimple {i} with mesh: {mesh_folder}/'Dimple_{i}_Axisym_M03S.inp")
        simulation('Dimple_%s_Axisym_M03S.inp' % i, "%s_%sGaussianFit%s_SLSQP_Voce2Step" % ("".join(root.split("_")), i, v), i, "", root, v, rho_ini=rho_ini, mesh_folder=mesh_folder)