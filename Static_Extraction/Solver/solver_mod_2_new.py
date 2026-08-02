from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax.experimental.sparse import BCOO

import scipy
import time
import os
import sys
import atexit
from jax.sharding import SingleDeviceSharding
from jax import checkpoint
from jax.ad_checkpoint import checkpoint_name
from functools import partial
# from petsc4py import PETSc

import scipy.sparse

# --- Initialize PyAMGX once at module level, suppressing deprecation warnings ---
from contextlib import contextmanager

# --- End PyAMGX global init ---
# import dask
# from dask import delayed
from functools import reduce
import operator
# from dask.distributed import Client, LocalCluster
import logging
from jax_fem import logger

# Try importing pardis0, CuPy and related sparse solvers
try:
    import pypardiso
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import spsolve
    from spineax.cudss.solver import CuDSSSolver
except ImportError:
    pass

petsc_row_elim = True  # If True, use PETSc for row elimination, otherwise use JAX
# print(jax.devices())
# print(cp.cuda.Device(0).mem_info)
################################################################################
# JAX solver or scipy solver or PETSc solver

# This can let us use persistent bicgstab but requires transferring data between CPU and GPU
# def jax_solve(A, b, x0, precond):
#     result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
#     return jax.pure_callback(jax_solve_host, result_shape, A, b, x0, precond, vmap_method='sequential')


# Global cache for CuDSS solver to avoid recreating and memory leaks
_cudss_solver_cache = {}

def _get_matrix_signature(csr_offsets, csr_columns):
    """Create a hashable signature for the matrix structure.
    Using buffer hash is much faster and more memory efficient than tuple conversion.
    """
    # Simply use the bytes of the arrays as the key.
    # This assumes these arrays are C-contiguous and have consistent types (int32).
    # We also include shape/size to be safe.
    try:
        # Check if arrays are numpy arrays and contigous
        if not (csr_offsets.flags['C_CONTIGUOUS'] and csr_columns.flags['C_CONTIGUOUS']):
             csr_offsets = onp.ascontiguousarray(csr_offsets)
             csr_columns = onp.ascontiguousarray(csr_columns)
        return (csr_offsets.tobytes(), csr_columns.tobytes())
    except Exception:
        # Fallback for weird array types if needed, though onp.array should handle it
        return (tuple(csr_offsets), tuple(csr_columns))

def clear_cudss_solver_cache():
    """Clear the CuDSS solver cache to free GPU memory."""
    global _cudss_solver_cache
    logger.debug(f"Clearing CuDSS solver cache ({len(_cudss_solver_cache)} solvers)")
    _cudss_solver_cache.clear()

def _cudss_solve_host_wrapper(csr_offsets, csr_columns, csr_values, b):
    # Ensure inputs are proper numpy arrays (pure_callback passes them as such)
    # Cast to correct types expected by CuDSS
    csr_offsets = onp.array(csr_offsets, dtype=onp.int32)
    csr_columns = onp.array(csr_columns, dtype=onp.int32)
    csr_values = onp.array(csr_values, dtype=onp.float64)
    b = onp.array(b, dtype=onp.float64)

    # Use cached solver to avoid memory leaks
    # STRATEGY: Prefer reusing existing solver if dimensions match, to avoid re-analysis overhead.
    # This assumes the sparsity pattern is effectively constant for the same problem size.
    
    cudss_solver = None
    matrix_sig = _get_matrix_signature(csr_offsets, csr_columns)
    
    # check exact match first
    if matrix_sig in _cudss_solver_cache:
        logger.debug("Reusing cached CuDSS solver instance (exact match)")
        cudss_solver = _cudss_solver_cache[matrix_sig]
    else:
        logger.debug("Creating new CuDSS solver instance (structure changed)")
        # Structure changed, need new solver
        # Optional: Clear old cache to save memory since we likely won't need the old structure again for this time step sequence
        # _cudss_solver_cache.clear() 
        
        cudss_solver = CuDSSSolver(
            csr_offsets=csr_offsets,
            csr_columns=csr_columns,
            device_id=0,
            mtype_id=0,
            mview_id=0,
        )
        _cudss_solver_cache[matrix_sig] = cudss_solver
    
    # cudss_solver is now set.
    # Critical fix: JIT-compile the solve function on GPU
    # This ensures that spineax's JAX primitives are executed on the GPU backend
    # even when called from a host callback.
    
    @jax.jit
    def gpu_solve_fn(curr_b, curr_vals):
        return cudss_solver(curr_b, curr_vals)[0] # CuDSSSolver returns (result, info)
        
    # Move data to GPU explicitly
    try:
        gpu_device = jax.devices('cuda')[0]
        b_gpu = jax.device_put(b, gpu_device)
        vals_gpu = jax.device_put(csr_values, gpu_device)
        
        # Run on GPU (blocks until executed)
        result_gpu = gpu_solve_fn(b_gpu, vals_gpu)
        # Move back to host numpy array
        return onp.array(result_gpu)
    except Exception as e:
        logger.error(f"Error in CuDSS solve: {e}")
        raise e

def AMGX_solve_host(A, x, b):
    dtype, shape = b.dtype, b.shape
    b = jax.lax.stop_gradient(b)
    b = np.array(b,dtype=np.float64) # convert to double precision for AmgX
    # indices = A.indices
    # logger.debug(f"Coversion to csr matrix inside solver wrapper...")
    A = A.sum_duplicates(nse=A.nse)
    A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A)
    csr_values  = np.array(A_bcsr.data, dtype=np.float64)
    start_sdv = time.time()
    # logger.info(f" Coversion time csr matrix inside solver wrapper: {solve_time_sdv2} [s]")
    # setup AmgX solver
    # Convert to static numpy arrays to avoid CUDA memory issues
    csr_offsets = np.array(A_bcsr.indptr, dtype=np.int32)
    csr_columns = np.array(A_bcsr.indices, dtype=np.int32)
    cudss_solver = CuDSSSolver(
        csr_offsets=csr_offsets,
        csr_columns=csr_columns,
        device_id=0,
        mtype_id=0,
        mview_id=0,
    )
    result, _ = cudss_solver(b, csr_values)
    end_sdv = time.time()
    solve_time_sdv2 = end_sdv - start_sdv

    # Check convergence
    final_residual = np.linalg.norm(A @ result - b)
    converged = (final_residual <= 1e-9)
    logger.debug(f'CuDSSSolver - Finished solving, linear solve res = {final_residual}')

    return result.astype(dtype).reshape(shape)

# def AMGX_solve(A_sp, b, x0):
#     b_active = b
#     x0_active = x0
#     logger.debug(f"Solving linear system using AMG solver...")
#     start_dom = time.time()
#
#     def matvec(u):
#         Au = A_sp @ u
#         return Au
#
#     result_shape = jax.ShapeDtypeStruct(b_active.shape, b_active.dtype)
#     cust_solver = lambda matvec, b_vec: jax.pure_callback(AMGX_solve_host, result_shape, A_sp, x0_active, b_vec)
#     x_active = jax.lax.custom_linear_solve(matvec, b_active, cust_solver, symmetric=False)
#     x =x_active # x.at[active_dof].set(x_active)
#     end_dom = time.time()
#     solve_time_dom = end_dom - start_dom
#     logger.info(f" AMG solver overall solution time: {solve_time_dom} [s]")
#     return x

def AMGX_solve(A, b, x0):
    A = A.sum_duplicates(nse=A.nse)
    A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A)
    
    # Pass JAX arrays directly to pure_callback. 
    # The callback mechanism converts them to host numpy arrays automatically.
    csr_values = A_bcsr.data
    csr_offsets = A_bcsr.indptr
    csr_columns = A_bcsr.indices
    
    start_time = time.time()
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    result = jax.pure_callback(
        _cudss_solve_host_wrapper, result_shape, csr_offsets, 
        csr_columns, csr_values, b)
    end_time = time.time()
    solve_time = end_time - start_time
    logger.debug(f" CuDSS solver overall solution time: {solve_time} [s]")

    final_residual = np.linalg.norm(A @ result - b)
    # Use logger.debug instead of jax.debug.print: the latter forces a GPU sync
    # to fetch `final_residual` on every linear solve (i.e. every Newton iter),
    # which adds a host-device round-trip to every Newton step.
    logger.debug(f"CuDSSSolver - Finished solving, linear solve res: {final_residual}")
    return result

def pardiso_solve(A, b, x0, solver_options):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    # Pass underlying arrays instead of the BCOO object to ensure JAX compatibility
    return jax.pure_callback(pardiso_solve_host, result_shape, A.data, A.indices, b)

def umfpack_solve1(A, b):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    timer_start = time.time()
    result = jax.pure_callback(umfpack_solve_host, result_shape, A, b) #, vmap_method='sequential'
    timer_end = time.time()
    timer_duration = timer_end - timer_start
    logger.debug(f" UMFPACK solver overall solution time: {timer_duration} [s]")
    return result

def umfpack_solve(A, b):
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    # indptr, indices, data = A.getValuesCSR()
    # Asp = A #scipy.sparse.csr_matrix((data, indices, indptr))
    # x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))
    pyamgx = None
    b = onp.array(b)
    # print(f"Matrix b: {b}")
    x_guess = onp.zeros_like(b)
    # logger.info(f" Coversion time csr matrix inside solver wrapper: {solve_time_sdv2} [s]")
    # setup AmgX solver
    # Initialize PyAMGX
    pyamgx.initialize()
    # Create resources
    cfg = pyamgx.Config().create_from_dict({
         "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",  # "CG", BICGSTAB
            "use_scalar_norm": 1,
            "norm": "L2",
            "tolerance": 1e-10,
            "monitor_residual": 1,
            "max_iters": 10000,
            "convergence": "ABSOLUTE",  # RELATIVE_INI_CORE
            "monitor_residual": 1,
            # "print_solve_stats": 1,
            "preconditioner": {
                "scope": "amg",
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "smoother": "JACOBI",
                "cycle": "V",
                "max_levels": 10,
                "max_iters": 2
            }
        }
    })

    resources = pyamgx.Resources().create_simple(cfg)

    solver = pyamgx.Solver().create(resources, cfg)
    # Create matrix and vector objects
    A_amg = pyamgx.Matrix().create(resources)
    b_amg = pyamgx.Vector().create(resources)
    x_amg = pyamgx.Vector().create(resources)
    # ======
    # Upload data to PyAMGX objects
    # A_amg.upload(onp.array(A.indices[:, 0]), onp.array(A.indices[:, 1]), onp.array(A.data))
    A_amg.upload_CSR(A)
    b_amg.upload(b)
    x_amg.upload(x_guess)

    # Solve Ax = b
    # logger.debug(f"Setting up the AMGx solver...")
    solver.setup(A_amg)
    solver.solve(b_amg, x_amg)

    # logger.info(f" AmgX solver Setting up time: {solve_time_sdv} [s]")
    # logger.info(f" AmgX solver solve time: {solve_time_sdv2} [s]")
    # logger.info(f" AmgX solver set+solve time: {solve_time_sdv + solve_time_sdv2} [s]")

    # final_residual = solver.get_residual()
    # logger.info(f" Final residual: : {final_residual} [s]")
    # Download the result
    result = x_amg.download()
    # print(f"Result: {result.astype(dtype).reshape(shape)}")

    # Cleanup
    x_amg.destroy()
    b_amg.destroy()
    A_amg.destroy()
    solver.destroy()
    cfg.destroy()
    resources.destroy()
    # Finalize PyAMGX
    pyamgx.finalize()
    x = result
    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(f'Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(A @ x - b)}')
    return x

def cupy_solve(A, b):
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(cupy_solve_host, result_shape, A, b) #, vmap_method='sequential'

def petsc_solve(A, b, ksp_type, pc_type):
    # Note that for jacrev compatitibility, we hard code ksptype and pc_type inside the host function
    # Only functions with JAX-type arguments can be transformed by pure_callback for batch-tracing
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    return jax.pure_callback(petsc_solve_host, result_shape, A, b)

# def jax_solve_host(A, b, x0, precond):
def jax_solve(A, b, x0, precond):
    """Solves the equilibrium equation using a JAX solver.
    Is fully traceable and runs on GPU.

    Parameters
    ----------
    precond
        Whether to calculate the preconditioner or not
    """
    logger.debug(f"JAX Solver - Solving linear system")
    # indptr, indices, data = A.getValuesCSR()
    # A_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=A.getSize())
    # A = BCOO.from_scipy_sparse(A_sp_scipy).sort_indices()
    # jacobi = np.array(A_sp_scipy.diagonal())

    # move things to gpu
    # A = jax.device_put(A, jax.devices('gpu')[0])
    # b = jax.device_put(b, jax.devices('gpu')[0])
    # x0 = jax.device_put(x0, jax.devices('gpu')[0])

    # diagonal_mask = A.indices[:,0] == A.indices[:,1]
    # jacobi = np.zeros(A.shape[0])
    # jacobi = jacobi.at[A.indices[diagonal_mask,0]].add(A.data[diagonal_mask])
    # diag = np.arange(A.shape[0])
    # jacobi = A[diag,diag]
    # jacobi = jacobi.todense()
    # pc = lambda x: x * (1. / jacobi) if precond else None
    pc =None
    # while True: # workaround for if bicgstab does not converge at first
    rel_tol = 1e-10
    abs_tol = 1e-10
    start_time = time.time()
    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                            b,
                                            x0=x0,
                                            M=pc,
                                            tol=rel_tol,
                                            atol=abs_tol,
                                            maxiter=10000)
    x.block_until_ready() # Ensure computation finishes before timing
    end_time = time.time()
    solve_time = end_time - start_time
    logger.debug(f" JAX bicgstab solver time: {solve_time}")

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    norm_b = np.linalg.norm(b)
    logger.debug(f"JAX Solver - Finished solving, |res| = {err}, |b| = {norm_b}") # info

    # if err <= max(rel_tol*norm_b, abs_tol):
    #     break
    # logger.warning(f"JAX Solver - Bicgstab did not converge, |res| = {err}, |b| = {norm_b}")

    # assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    # x = np.where(err < 0.1, x, np.nan) # For assert purpose, some how this also affects bicgstab.

    return x.astype(b.dtype), err

def pardiso_solve_host(data, indices, b):
    logger.debug(f"Pardiso Solver - Solving linear system")
    # A = jax.lax.stop_gradient(A)
    # b = jax.lax.stop_gradient(b)
    # If you need to convert PETSc to scipy
    start = time.time()
    # Reconstruct Scipy CSR matrix from components. 
    # BCOO indices are (row, col) pairs.
    rows = indices[:, 0]
    cols = indices[:, 1]
    # Note: We don't know the shape here, so we infer it from b
    n = len(b)
    Asp = scipy.sparse.csr_array((data, (rows, cols)), shape=(n, n))
    x = pypardiso.spsolve(Asp, onp.array(b))
    end = time.time()
    logger.debug(f'Pardiso Solver - Finished solving, linear solve res = {onp.linalg.norm(Asp @ x - onp.array(b))}')
    logger.debug(f" Pardiso solver time: {end - start} [s]")
    return x.astype(b.dtype).reshape(b.shape)

def umfpack_solve_host(A, b):
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    # indptr, indices, data = A.getValuesCSR()
    # Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    A = jax.lax.stop_gradient(A)
    b = jax.lax.stop_gradient(b)
    dtype, shape = b.dtype, b.shape
    Asp = scipy.sparse.csr_array((A.data, (A.indices[:,0], A.indices[:,1])),
                                 shape=A.shape)
    x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(f'umfpack Solver- Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}')
    return x.astype(b.dtype)

def cupy_solve_host(A, b):
    logger.debug(f"Cupy Solver - Solving linear system with Cupy")
    A = jax.lax.stop_gradient(A)
    b = jax.lax.stop_gradient(b)
    dtype, shape = b.dtype, b.shape
    # indptr, indices, data = A.getValuesCSR()
    # Asp = scipy.sparse.csr_matrix((data, indices, indptr))
    Asp = csr_matrix((cp.array(A.data), (cp.array(A.indices[:,0]), cp.array(A.indices[:,1]))),
                                 shape=A.shape,dtype=cp.float32) #,dtype=cp.float32
    b_cp =  cp.array(b, dtype=cp.float32)
    x = spsolve(Asp, b_cp) #,dtype=cp.float32
    x = x.get()
    # logger.debug(f'Cupy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b_cp)}')
    return x.astype(dtype).reshape(shape)

def petsc_solve_host(A, b):
    from petsc4py import PETSc
    ksp_type = 'bcgsl'
    pc_type = 'ilu'
    A_sp_scipy = scipy.sparse.csr_array((A.data, (A.indices[:,0], A.indices[:,1])),
                                 shape=A.shape)
    A = PETSc.Mat().createAIJ(size=A_sp_scipy.shape,
                              csr=(A_sp_scipy.indptr.astype(PETSc.IntType, copy=False),
                                   A_sp_scipy.indices.astype(PETSc.IntType, copy=False),
                                   A_sp_scipy.data))

    rhs = PETSc.Vec().createSeq(len(b))
    rhs.setValues(range(len(b)), onp.array(b))
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType(ksp_type)
    ksp.pc.setType(pc_type)

    # TODO: This works better. Do we need to generalize the code a little bit?
    if ksp_type == 'tfqmr':
        ksp.pc.setFactorSolverType('mumps')

    logger.debug(f'PETSc Solver - Solving linear system with ksp_type = {ksp.getType()}, pc = {ksp.pc.getType()}')
    x = PETSc.Vec().createSeq(len(b))
    ksp.solve(rhs, x)

    # Verify convergence
    y = PETSc.Vec().createSeq(len(b))
    A.mult(x, y)

    err = np.linalg.norm(y.getArray() - rhs.getArray())
    logger.debug(f"PETSc Solver - Finished solving, linear solve res = {err}")
    assert err < 0.1, f"PETSc linear solver failed to converge, err = {err}"

    return x.getArray().astype(b.dtype)


def custom_solver(A_sp, b_active, x0_active,solver_options):
    def matvec(u):
        Au = A_sp @ u
        return Au
    result_shape = jax.ShapeDtypeStruct(b_active.shape, b_active.dtype)
    cust_solver = lambda matvec, b_vec: jax.pure_callback(AMGX_solve_host, result_shape, A_sp, x0_active, b_vec)
    x_active = jax.lax.custom_linear_solve(matvec, b_active, cust_solver, symmetric=True)
    return x_active

def linear_solver(A, b, x0, solver_options):

    # If user does not specify any solver, set jax_solver as the default one.
    if  len(solver_options.keys() & {'jax_solver', 'umfpack_solver', 'petsc_solver',
                                     'AMGX_solver', 'custom_solver','pardiso_solver','cupy_solver'}) == 0:
        solver_options['jax_solver'] = {}

    if 'jax_solver' in solver_options:
        precond = solver_options['jax_solver']['precond'] if 'precond' in solver_options['jax_solver'] else True
        x, res_info = jax_solve(A, b, x0, precond)
    elif 'umfpack_solver' in solver_options:
        x = umfpack_solve1(A, b)
    elif 'petsc_solver' in solver_options:
        ksp_type = solver_options['petsc_solver']['ksp_type'] if 'ksp_type' in solver_options['petsc_solver'] else  'bcgsl'
        pc_type = solver_options['petsc_solver']['pc_type'] if 'pc_type' in solver_options['petsc_solver'] else 'ilu'
        x = petsc_solve(A, b, ksp_type, pc_type)
    elif 'AMGX_solver' in solver_options:
        x = AMGX_solve(A, b, x0)
        # x = AMGX_solve_host_gpu(A, b, x0)
    elif 'pardiso_solver' in solver_options:
        x = pardiso_solve(A, b, x0, solver_options)
    elif 'cupy_solver' in solver_options:
        x = cupy_solve(A, b)
    elif 'custom_solver' in solver_options:
        # Users can define their own solver
        # custom_solver = solver_options['custom_solver']
        x = custom_solver(A, b, x0, solver_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    if 'res_info' not in locals():
        res_info = 0.0
    return x, res_info


################################################################################
# "row elimination" solver

def apply_bc_vec(res_vec, dofs, problem, scale=1.):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i]*scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.):
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)
    return res_fn_bc


def assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_ones_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(1.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(0.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = (new_sol.at[fe.node_inds_list[i],
                                  fe.vec_inds_list[i]].set(sol[fe.node_inds_list[i],
                                          fe.vec_inds_list[i]]))
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list, problem):

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


def linear_incremental_solver(problem, res_vec, A, dofs, solver_options):
    """
    Linear solver at each Newton's iteration
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if hasattr(problem, 'P_mat'):
        x0_2 = copy_bc(problem.P_mat @ dofs, problem)
        x0 = problem.P_mat.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2
    # TO DO: add term for column elimination
    # x0_1 is all zero and true BC
    # x0_2 is all zero exept at BC, where is current solution
    # x0 is all zero except at BC, where is the difference between current and true BC
    # because this would be the desired increment at BC
    # So, something like linear_solver(A, b +-? A @ x0, x0, solver_options) would be correct when column eliminating with nonzero Dirichlet BC

    inc, res_info = linear_solver(A, b, x0, solver_options)
    # print(f"inc = {inc}, dofs = {dofs}")
    if res_info > 1.0: # linear residual too large => reject step
        logger.warning(f"Linear solver did not converge well, linear residual = {res_info}")
        status = ["rejected", "linear solver did not converge well"]
        return dofs, status

    line_search_flag = solver_options['line_search_flag'] if 'line_search_flag' in solver_options else False
    if line_search_flag:
        line_search_options = solver_options.get('line_search_options', {})
        dofs, status = line_search_scherzinger2017(problem, dofs, inc, r0=res_vec, **line_search_options)
    else:
        dofs = dofs + inc
        status = ["accepted", "no line search"]

    return dofs, status

def line_search_scherzinger2017(problem, dofs, inc,
                               max_ls=3,
                               beta=1.e-4,          # paper: β in Goldstein condition (Eq. 51)
                               eta=0.1,            # paper: η minimum reduction factor (Eq. 52)
                               damping=1.0,        # initial "try" step (often 1.0 in the paper)
                               min_alpha=0.4,     # numerical floor (not in paper; practical)
                               maxstep=None,       # optional cap on ||inc||
                               accept_check=None,  # optional filter: accept_check(x_trial)->(ok, reason)
                               fail_shrink=0.5,    # used only if accept_check rejects
                               merit_fn=None,      # optional: merit_fn(x, r)->float
                               merit_mode="auto",  # "auto" | "force_scaled" | "preconditioned"
                               precond_apply=None, # optional: z = M^{-1} r
                               K0=None,            # optional tangent used to estimate ||K0||
                               stiffness_scale=None, # optional precomputed ||K0||
                               U_ref=None,         # optional displacement reference scale
                               force_scale=None,   # optional frozen force scale F_ref
                               verbose=True,
                               r0=None):
    """
    Scherzinger (2017) / Perez-Foguet & Armero style line search.

    Merit ψ(x):
      - custom merit_fn(x, r), or
      - preconditioned ψ=0.5 r^T M^{-1} r, or
      - force-scaled ψ=0.5 ||r/F_ref||^2 with frozen F_ref (displacement-control friendly).

    For force-scaled merit, F_ref is frozen at line-search start:
      F_ref ~ ||K0|| * U_ref, with fallback F_ref=max(||r0||, 1).

    Evaluate ψ(0) and ψ(1). If ψ(1) satisfies sufficient decrease, take full step.
    Otherwise:
      α0 = ψ(0) / (ψ(0) + ψ(1))                          (Eq. 49)
      α_{j+1} = ψ(0) / (ψ(0) + ψ(α_j))                   (Eq. 50)
      accept when ψ(α_{j+1}) < (1 - 2β α_j) ψ(α_j)       (Eq. 51)
      enforce α_{j+1} = max(η α_j, α_{j+1})              (Eq. 52)

    Returns (x_new, status) where status = ["accepted"/"rejected", reason]
    """

    # Residual wrapper (same pattern as your code)
    res_fn_core = problem.compute_residual
    res_fn_core = get_flatten_fn(res_fn_core, problem)
    res_fn_core = apply_bc(res_fn_core, problem)

    def residual(x):
        if hasattr(problem, 'P_mat'):
            x_full = problem.P_mat @ x
            r_full = res_fn_core(x_full)
            r = problem.P_mat.T @ r_full
        else:
            r = res_fn_core(x)
        return np.asarray(r, dtype=float)

    def _estimate_stiffness_scale(K):
        if K is None:
            return None
        try:
            if scipy.sparse.issparse(K):
                diag = onp.abs(K.diagonal())
                diag_max = float(diag.max()) if diag.size else 0.0
                row_sum = onp.asarray(onp.abs(K).sum(axis=1)).ravel()
                row_max = float(row_sum.max()) if row_sum.size else 0.0
                return max(diag_max, row_max)

            if isinstance(K, BCOO):
                inds = onp.asarray(K.indices)
                vals = onp.abs(onp.asarray(K.data))
                if inds.size == 0 or vals.size == 0:
                    return 0.0
                row_sums = onp.bincount(inds[:, 0], weights=vals, minlength=K.shape[0])
                row_max = float(row_sums.max()) if row_sums.size else 0.0
                diag_vals = vals[inds[:, 0] == inds[:, 1]]
                diag_max = float(diag_vals.max()) if diag_vals.size else 0.0
                return max(diag_max, row_max)

            K_arr = onp.asarray(K)
            if K_arr.ndim == 2:
                diag_max = float(onp.abs(onp.diag(K_arr)).max()) if K_arr.size else 0.0
                row_max = float(onp.abs(K_arr).sum(axis=1).max()) if K_arr.size else 0.0
                return max(diag_max, row_max)
            if K_arr.ndim == 1:
                return float(onp.abs(K_arr).max()) if K_arr.size else 0.0
        except Exception:
            return None
        return None

    x0 = np.asarray(dofs, dtype=float)
    p = np.asarray(inc, dtype=float)

    # Optional maxstep scaling
    if maxstep is not None:
        pn = np.linalg.norm(p)
        if pn > maxstep and pn > 0.0:
            p = p * (maxstep / pn)

    # Compute r0, ψ0
    if r0 is None:
        r0 = residual(x0)
    else:
        r0 = np.asarray(r0, dtype=float)

    # if merit_fn is None:
        mode = merit_mode.lower()
        if mode not in ("auto", "force_scaled", "preconditioned"):
            raise ValueError("merit_mode must be 'auto', 'force_scaled', or 'preconditioned'")

        if precond_apply is None:
            for attr in ("apply_preconditioner", "line_search_preconditioner", "preconditioner_apply"):
                maybe_pc = getattr(problem, attr, None)
                if callable(maybe_pc):
                    precond_apply = maybe_pc
                    break

        use_preconditioned = (mode == "preconditioned") or (mode == "auto" and callable(precond_apply))
        if use_preconditioned:
            if not callable(precond_apply):
                raise ValueError("preconditioned merit requested but no precond_apply callable was provided")

            def merit_fn(x, r):  # noqa: F811
                z = np.asarray(precond_apply(r), dtype=float)
                return 0.5 * float(r @ z)
        else:
            if force_scale is None:
                k_scale = stiffness_scale if stiffness_scale is not None else _estimate_stiffness_scale(K0)
                if U_ref is None:
                    U_ref_eff = max(float(np.linalg.norm(x0)),
                                    float(np.linalg.norm(p)),
                                    1.0)
                else:
                    U_ref_eff = max(float(U_ref), 1.0)

                if k_scale is not None and np.isfinite(k_scale) and k_scale > 0.0:
                    force_scale_eff = max(float(k_scale) * U_ref_eff, 1.0)
                else:
                    force_scale_eff = max(float(np.linalg.norm(r0)), 1.0)
            else:
                force_scale_eff = max(float(force_scale), 1.0)

            inv_scale_sq = 1.0 / (force_scale_eff * force_scale_eff)

            def merit_fn(x, r):  # noqa: F811
                return 0.5 * float(r @ r) * inv_scale_sq

    def merit_fn(x, r):  # noqa: F811
        return 0.5 * float(r @ r)

    psi0 = merit_fn(x0, r0)
    if not np.isfinite(psi0) or psi0 == 0.0:
        return x0, ["accepted", "psi0 non-finite or zero"]

    def trial(alpha, r_provided=None):
        x = x0 + alpha * p

        if accept_check is not None:
            ok, reason = accept_check(x)
            if not ok:
                return x, np.inf, ["rejected", reason], None

        r = r_provided if r_provided is not None else residual(x)

        # Treat NaN/Inf residual as infeasible => infinite merit
        if not np.all(np.isfinite(r)):
            return x, np.inf, ["rejected", "non-finite residual"], r

        psi = merit_fn(x, r)
        if not np.isfinite(psi):
            psi = np.inf
            return x, psi, ["rejected", "non-finite merit"], r

        return x, psi, ["ok", ""], r

    # --- 1) Try the "full" step first (paper: α=1 is the Newton step) ---
    alpha_full = float(np.clip(damping, 0.0, 1.0))
    x1, psi1, st1, r1 = trial(alpha_full)

    # If accept_check rejected, shrink a couple of times (this is your practical add-on)
    if st1[0] == "rejected" and accept_check is not None:
        alpha_tmp = alpha_full
        for _ in range(3):
            alpha_tmp *= fail_shrink
            if alpha_tmp < min_alpha:
                break
            x1, psi1, st1, r1 = trial(alpha_tmp)
            if st1[0] != "rejected":
                alpha_full = alpha_tmp
                break

    # Paper sufficient decrease for taking full step (α=1 case):
    # ψ(1) < (1 - 2β) ψ(0)
    if np.isfinite(psi1) and (psi1 < (1.0 - 2.0 * beta) * psi0):
        if verbose:
            logger.debug(f"[LS] accept full: alpha={alpha_full:.6g}, psi0={psi0:.3e}, psi1={psi1:.3e}")
        return x1, ["accepted", "full step"]

    # --- 2) Otherwise, compute the quadratic minimizer α0 (Eq. 49) ---
    denom = psi0 + psi1
    if not np.isfinite(denom) or denom <= 0.0:
        alpha = max(min_alpha, eta * alpha_full)
    else:
        alpha = psi0 / denom
        alpha = float(np.clip(alpha, min_alpha, 1.0))

    # Evaluate ψ(α0)
    xa, psia, sta, ra = trial(alpha)
    if verbose:
        logger.debug(f"[LS] start backtrack: first itr alpha0={alpha:.6g}, psi(alpha0)={psia:.3e}, psi1 at alpha_full={psi1:.3e}\
               , psi0={psi0:.3e}, suff. decrease thresh at alpha0={(1.0 - 2.0 * beta * alpha) * psi0:.3e}")

    # --- 3) Iterate α updates (Eqs. 50–52) until Goldstein is satisfied or max_ls hit ---
    alpha_j = alpha
    psi_j = psia

    for j in range(max_ls):
        # Goldstein sufficient decrease (Eq. 51):
        # ψ(α_{j+1}) < (1 - 2β α_j) ψ(α_j)
        # But Eq. 51 is written comparing successive ψ(·) values; implement as:
        # accept current α_j if it already gives enough decrease from ψ0:
        # (common practical interpretation for monotone decrease)
        if np.isfinite(psi_j) and (psi_j < (1.0 - 2.0 * beta * alpha_j) * psi0):
            if verbose:
                logger.debug(f"[LS] accept: alpha={alpha_j:.6g}, psi={psi_j:.3e}, suff. decrease={(1.0 - 2.0 * beta * alpha_j) * psi0:.3e}")
            return xa, ["accepted", f"alpha={alpha_j:.6g}"]

        # Eq. (50): α_{j+1} = ψ(0) / (ψ(0) + ψ(α_j))
        denom = psi0 + psi_j
        if not np.isfinite(denom) or denom <= 0.0:
            alpha_next = eta * alpha_j
        else:
            alpha_next = psi0 / denom

        # Eq. (52): α_{j+1} = max(η α_j, α_{j+1})
        alpha_next = max(eta * alpha_j, float(alpha_next))
        alpha_next = float(np.clip(alpha_next, min_alpha, 1.0))
            
        if alpha_next <= min_alpha:
            if verbose:
                if psi0  > 50. * psi_j:
                    alpha_next = 1.0 # if we got huge decrease, try full step next time instead of stagnating at floor
                else:
                    alpha_next=0.5 # accept half step if hit floor, to avoid stagnation
                logger.debug(f"[LS] accept at alpha floor: alpha={alpha_next:.6g}")
            return x0 + alpha_next * p, ["accepted", f"alpha={alpha_next:.6g} (alpha floor)"]

        if j == 0 and alpha_next <0.5:
            alpha_next = 0.5 # heuristic for first backtrack step to improve robustness

        # Evaluate at α_{j+1}
        xa_next, psi_next, st_next, ra_next = trial(alpha_next)

        # If trial is infeasible (NaN residual / accept_check), force shrink and retry
        if st_next[0] == "rejected":
            alpha_next = max(min_alpha, eta * alpha_next)
            xa_next, psi_next, st_next, ra_next = trial(alpha_next)

        if verbose:
            logger.debug(f"[LS] j={j+1}, alpha={alpha_next:.6g},\
                   psi={psi_next:.3e},psi0={psi0:.3e},suff. decrease thresh={(1.0 - 2.0 * beta * alpha_next) * psi0:.3e}")

        alpha_j, psi_j = alpha_next, psi_next
        xa, psi_j, sta, ra = xa_next, psi_next, st_next, ra_next

    logger.warning("Line search: max_ls reached or alpha hit floor; returning last trial.")
    return xa, ["rejected", "max_ls / alpha floor"]


def line_search_scherzinger2017_mod(problem, dofs, inc,
                               max_ls=10,
                               beta=1.e-4,          # paper: β in Goldstein condition (Eq. 51)
                               eta=0.1,            # paper: η minimum reduction factor (Eq. 52)
                               damping=1.0,        # initial "try" step (often 1.0 in the paper)
                               min_alpha=0.4,     # numerical floor (not in paper; practical)
                               maxstep=None,       # optional cap on ||inc||
                               accept_check=None,  # optional filter: accept_check(x_trial)->(ok, reason)
                               fail_shrink=0.5,    # used only if accept_check rejects
                               merit_fn=None,      # optional: merit_fn(x, r)->float
                               merit_mode="auto",  # "auto" | "force_scaled" | "preconditioned"
                               precond_apply=None, # optional: z = M^{-1} r
                               K0=None,            # optional tangent used to estimate ||K0||
                               stiffness_scale=None, # optional precomputed ||K0||
                               U_ref=None,         # optional displacement reference scale
                               force_scale=None,   # optional frozen force scale F_ref
                               residual_ratio_cutoff=1e8,  # early reject if ||r|| / ||r0|| exceeds this
                               residual_abs_cutoff=np.inf, # early reject if ||r|| exceeds this absolute threshold
                               merit_ratio_cutoff=1e16,    # early reject if ψ / ψ0 exceeds this
                               alpha_floor_reject_ratio=200.0,  # reject when floor is hit with strong residual growth
                               verbose=True,
                               r0=None):
    """
    Scherzinger (2017) / Perez-Foguet & Armero style line search.

    Merit ψ(x):
      - custom merit_fn(x, r), or
      - preconditioned ψ=0.5 r^T M^{-1} r, or
      - force-scaled ψ=0.5 ||r/F_ref||^2 with frozen F_ref (displacement-control friendly).

    For force-scaled merit, F_ref is frozen at line-search start:
      F_ref ~ ||K0|| * U_ref, with fallback F_ref=max(||r0||, 1).

    Evaluate ψ(0) and ψ(1). If ψ(1) satisfies sufficient decrease, take full step.
    Otherwise:
      α0 = ψ(0) / (ψ(0) + ψ(1))                          (Eq. 49)
      α_{j+1} = ψ(0) / (ψ(0) + ψ(α_j))                   (Eq. 50)
      accept when ψ(α_{j+1}) < (1 - 2β α_j) ψ(α_j)       (Eq. 51)
      enforce α_{j+1} = max(η α_j, α_{j+1})              (Eq. 52)

    Returns (x_new, status) where status = ["accepted"/"rejected", reason]
    """

    # Residual wrapper (same pattern as your code)
    res_fn_core = problem.compute_residual
    res_fn_core = get_flatten_fn(res_fn_core, problem)
    res_fn_core = apply_bc(res_fn_core, problem)

    def residual(x):
        if hasattr(problem, 'P_mat'):
            x_full = problem.P_mat @ x
            r_full = res_fn_core(x_full)
            r = problem.P_mat.T @ r_full
        else:
            r = res_fn_core(x)
        return np.asarray(r, dtype=float)

    def _estimate_stiffness_scale(K):
        if K is None:
            return None
        try:
            if scipy.sparse.issparse(K):
                diag = onp.abs(K.diagonal())
                diag_max = float(diag.max()) if diag.size else 0.0
                row_sum = onp.asarray(onp.abs(K).sum(axis=1)).ravel()
                row_max = float(row_sum.max()) if row_sum.size else 0.0
                return max(diag_max, row_max)

            if isinstance(K, BCOO):
                inds = onp.asarray(K.indices)
                vals = onp.abs(onp.asarray(K.data))
                if inds.size == 0 or vals.size == 0:
                    return 0.0
                row_sums = onp.bincount(inds[:, 0], weights=vals, minlength=K.shape[0])
                row_max = float(row_sums.max()) if row_sums.size else 0.0
                diag_vals = vals[inds[:, 0] == inds[:, 1]]
                diag_max = float(diag_vals.max()) if diag_vals.size else 0.0
                return max(diag_max, row_max)

            K_arr = onp.asarray(K)
            if K_arr.ndim == 2:
                diag_max = float(onp.abs(onp.diag(K_arr)).max()) if K_arr.size else 0.0
                row_max = float(onp.abs(K_arr).sum(axis=1).max()) if K_arr.size else 0.0
                return max(diag_max, row_max)
            if K_arr.ndim == 1:
                return float(onp.abs(K_arr).max()) if K_arr.size else 0.0
        except Exception:
            return None
        return None

    x0 = np.asarray(dofs, dtype=float)
    p = np.asarray(inc, dtype=float)

    # Optional maxstep scaling
    if maxstep is not None:
        pn = np.linalg.norm(p)
        if pn > maxstep and pn > 0.0:
            p = p * (maxstep / pn)

    # Compute r0, ψ0
    if r0 is None:
        r0 = residual(x0)
    else:
        r0 = np.asarray(r0, dtype=float)
    r0_norm = float(np.linalg.norm(r0))
    r0_norm_ref = max(r0_norm, 1.0)

    if merit_fn is None:
        mode = merit_mode.lower()
        if mode not in ("auto", "force_scaled", "preconditioned"):
            raise ValueError("merit_mode must be 'auto', 'force_scaled', or 'preconditioned'")

        if precond_apply is None:
            for attr in ("apply_preconditioner", "line_search_preconditioner", "preconditioner_apply"):
                maybe_pc = getattr(problem, attr, None)
                if callable(maybe_pc):
                    precond_apply = maybe_pc
                    break

        use_preconditioned = (mode == "preconditioned") or (mode == "auto" and callable(precond_apply))
        if use_preconditioned:
            if not callable(precond_apply):
                raise ValueError("preconditioned merit requested but no precond_apply callable was provided")

            def merit_fn(x, r):  # noqa: F811
                z = np.asarray(precond_apply(r), dtype=float)
                return 0.5 * float(r @ z)
        else:
            if force_scale is None:
                k_scale = stiffness_scale if stiffness_scale is not None else _estimate_stiffness_scale(K0)
                if U_ref is None:
                    U_ref_eff = max(float(np.linalg.norm(x0)),
                                    float(np.linalg.norm(p)),
                                    1.0)
                else:
                    U_ref_eff = max(float(U_ref), 1.0)

                if k_scale is not None and np.isfinite(k_scale) and k_scale > 0.0:
                    force_scale_eff = max(float(k_scale) * U_ref_eff, 1.0)
                else:
                    force_scale_eff = max(float(np.linalg.norm(r0)), 1.0)
            else:
                force_scale_eff = max(float(force_scale), 1.0)

            inv_scale_sq = 1.0 / (force_scale_eff * force_scale_eff)

            def merit_fn(x, r):  # noqa: F811
                return 0.5 * float(r @ r) * inv_scale_sq
    # Keep the merit definition consistent with the current solver behavior.
    def merit_fn(x, r):  # noqa: F811
        return 0.5 * float(r @ r)

    psi0 = merit_fn(x0, r0)
    if not np.isfinite(psi0) or psi0 == 0.0:
        return x0, ["accepted", "psi0 non-finite or zero"]
    psi0_ref = max(float(psi0), 1.0)

    def _is_cutoff_status(status):
        return status[0] == "rejected" and "cutoff" in status[1]

    def trial(alpha, r_provided=None):
        x = x0 + alpha * p

        if accept_check is not None:
            ok, reason = accept_check(x)
            if not ok:
                return x, np.inf, ["rejected", reason], None

        r = r_provided if r_provided is not None else residual(x)

        # Treat NaN/Inf residual as infeasible => infinite merit
        if not np.all(np.isfinite(r)):
            return x, np.inf, ["rejected", "non-finite residual"], r
        r_norm = float(np.linalg.norm(r))
        if (residual_abs_cutoff is not None and np.isfinite(residual_abs_cutoff)
                and residual_abs_cutoff > 0.0 and r_norm > float(residual_abs_cutoff)):
            return x, np.inf, ["rejected", "residual abs cutoff"], r
        if (residual_ratio_cutoff is not None and np.isfinite(residual_ratio_cutoff)
                and residual_ratio_cutoff > 0.0 and r_norm > float(residual_ratio_cutoff) * r0_norm_ref):
            return x, np.inf, ["rejected", "residual ratio cutoff"], r

        psi = merit_fn(x, r)
        if not np.isfinite(psi):
            psi = np.inf
            return x, psi, ["rejected", "non-finite merit"], r
        if (merit_ratio_cutoff is not None and np.isfinite(merit_ratio_cutoff)
                and merit_ratio_cutoff > 0.0 and psi > float(merit_ratio_cutoff) * psi0_ref):
            return x, psi, ["rejected", "merit ratio cutoff"], r

        return x, psi, ["ok", ""], r

    # --- 1) Try the "full" step first (paper: α=1 is the Newton step) ---
    alpha_full = float(np.clip(damping, 0.0, 1.0))
    x1, psi1, st1, r1 = trial(alpha_full)
    if _is_cutoff_status(st1):
        if verbose:
            print(f"[LS] reject early: {st1[1]} at alpha={alpha_full:.6g}")
        logger.warning(f"Line search rejected early due to {st1[1]} at alpha={alpha_full:.6g}")
        return x0, st1

    # If accept_check rejected, shrink a couple of times (this is your practical add-on)
    if st1[0] == "rejected" and accept_check is not None:
        alpha_tmp = alpha_full
        for _ in range(3):
            alpha_tmp *= fail_shrink
            if alpha_tmp < min_alpha:
                break
            x1, psi1, st1, r1 = trial(alpha_tmp)
            if st1[0] != "rejected":
                alpha_full = alpha_tmp
                break

    # Paper sufficient decrease for taking full step (α=1 case):
    # ψ(1) < (1 - 2β) ψ(0)
    if np.isfinite(psi1) and (psi1 < (1.0 - 2.0 * beta) * psi0):
        if verbose:
            print(f"[LS] accept full: alpha={alpha_full:.6g}, psi0={psi0:.3e}, psi1={psi1:.3e}")
        return x1, ["accepted", "full step"]

    # --- 2) Otherwise, compute the quadratic minimizer α0 (Eq. 49) ---
    denom = psi0 + psi1
    if not np.isfinite(denom) or denom <= 0.0:
        alpha = max(min_alpha, eta * alpha_full)
    else:
        alpha = psi0 / denom
        alpha = float(np.clip(alpha, min_alpha, 1.0))

    # Evaluate ψ(α0)
    xa, psia, sta, ra = trial(alpha)
    if _is_cutoff_status(sta):
        if verbose:
            print(f"[LS] reject early: {sta[1]} at alpha0={alpha:.6g}")
        logger.warning(f"Line search rejected early due to {sta[1]} at alpha0={alpha:.6g}")
        return x0, sta
    if verbose:
        print(f"[LS] start backtrack: first itr alpha0={alpha:.6g}, psi(alpha0)={psia:.3e}, psi1 at alpha_full={psi1:.3e}\
               , psi0={psi0:.3e}, suff. decrease thresh at alpha0={(1.0 - 2.0 * beta * alpha) * psi0:.3e}")

    # --- 3) Iterate α updates (Eqs. 50–52) until Goldstein is satisfied or max_ls hit ---
    alpha_j = alpha
    psi_j = psia

    for j in range(max_ls):
        # Goldstein sufficient decrease (Eq. 51):
        # ψ(α_{j+1}) < (1 - 2β α_j) ψ(α_j)
        # But Eq. 51 is written comparing successive ψ(·) values; implement as:
        # accept current α_j if it already gives enough decrease from ψ0:
        # (common practical interpretation for monotone decrease)
        if np.isfinite(psi_j) and (psi_j < (1.0 - 2.0 * beta * alpha_j) * psi0):
            if verbose:
                print(f"[LS] accept: alpha={alpha_j:.6g}, psi={psi_j:.3e}, suff. decrease={(1.0 - 2.0 * beta * alpha_j) * psi0:.3e}")
            return xa, ["accepted", f"alpha={alpha_j:.6g}"]

        # Eq. (50): α_{j+1} = ψ(0) / (ψ(0) + ψ(α_j))
        denom = psi0 + psi_j
        if not np.isfinite(denom) or denom <= 0.0:
            alpha_next = eta * alpha_j
        else:
            alpha_next = psi0 / denom

        # Eq. (52): α_{j+1} = max(η α_j, α_{j+1})
        alpha_next = max(eta * alpha_j, float(alpha_next))
        alpha_next = float(np.clip(alpha_next, min_alpha, 1.0))
            
        if alpha_next <= min_alpha:
            if (j > 0 and alpha_floor_reject_ratio is not None and np.isfinite(alpha_floor_reject_ratio)
                    and alpha_floor_reject_ratio > 0.0 and np.isfinite(psi_j)
                    and psi_j > float(alpha_floor_reject_ratio) * psi0):
                msg = "alpha floor residual blow-up cutoff"
                if verbose:
                    ratio = psi_j / psi0
                    print(f"[LS] reject early: {msg}, psi/psi0={ratio:.3e}")
                logger.warning(f"Line search rejected early: {msg}")
                return x0, ["rejected", msg]
            if verbose:
                if psi0  > 10. * psi_j:
                    alpha_next = 1.0 # if we got huge decrease, try full step next time instead of stagnating at floor
                else:
                    alpha_next=0.5 # accept half step if hit floor, to avoid stagnation
                print(f"[LS] accept at alpha floor: alpha={alpha_next:.6g}")
            return x0 + alpha_next * p, ["accepted", f"alpha={alpha_next:.6g} (alpha floor)"]

        if j == 0 and alpha_next <0.5:
            alpha_next = 0.5 # heuristic for first backtrack step to improve robustness

        # Evaluate at α_{j+1}
        xa_next, psi_next, st_next, ra_next = trial(alpha_next)
        if _is_cutoff_status(st_next):
            if verbose:
                print(f"[LS] reject early: {st_next[1]} at alpha={alpha_next:.6g}")
            logger.warning(f"Line search rejected early due to {st_next[1]} at alpha={alpha_next:.6g}")
            return x0, st_next

        # If trial is infeasible (NaN residual / accept_check), force shrink and retry
        if st_next[0] == "rejected":
            alpha_next = max(min_alpha, eta * alpha_next)
            xa_next, psi_next, st_next, ra_next = trial(alpha_next)
            if _is_cutoff_status(st_next):
                if verbose:
                    print(f"[LS] reject early: {st_next[1]} at alpha={alpha_next:.6g}")
                logger.warning(f"Line search rejected early due to {st_next[1]} at alpha={alpha_next:.6g}")
                return x0, st_next

        if verbose:
            print(f"[LS] j={j+1}, alpha={alpha_next:.6g},\
                   psi={psi_next:.3e},psi0={psi0:.3e},suff. decrease thresh={(1.0 - 2.0 * beta * alpha_next) * psi0:.3e}")

        alpha_j, psi_j = alpha_next, psi_next
        xa, psi_j, sta, ra = xa_next, psi_next, st_next, ra_next

    logger.warning("Line search: max_ls reached or alpha hit floor; returning last trial.")
    return xa, ["rejected", "max_ls / alpha floor"]

def line_search3(problem, dofs, inc,
                max_ls=3,
                minlambda=0.5,        # PETSc-style (replace alpha_min)
                c1=1e-4,               # PETSc: -snes_linesearch_alpha
                damping=1.0,           # PETSc: -snes_linesearch_damping
                shrink_lo=0.1,         # safeguard: lambda_new >= shrink_lo*lambda
                shrink_hi=0.5,         # safeguard: lambda_new <= shrink_hi*lambda
                slope_mode="newton",       # "fd" robust, "newton" faster if exact Newton
                fd_eps=None,
                maxstep=None,          # optional cap on ||inc||
                accept_check=None,     # optional filter: accept_check(x_trial)->(ok, reason)
                fail_shrink=0.5,       # shrink factor when accept_check fails
                verbose=False,
                r0=None):
    """
    PETSc SNESLINESEARCHBT-style backtracking line search (clean drop-in).

    Uses merit: phi(x)=0.5||R(x)||^2
    Armijo: phi(x+λp) <= phi(x) + c1*λ*phi'(0)

    Step updates: safeguarded quadratic model (PETSc-style).
    Optional accept_check lets you reject invalid trial states cheaply (MFEM/Abaqus-like).
    """

    logger.debug("Starting PETSc-style backtracking line search")

    # Residual wrapper (same pattern as your code)
    res_fn_core = problem.compute_residual
    res_fn_core = get_flatten_fn(res_fn_core, problem)
    res_fn_core = apply_bc(res_fn_core, problem)

    def residual(x):
        if hasattr(problem, 'P_mat'):
            x_full = problem.P_mat @ x
            r_full = res_fn_core(x_full)
            r = problem.P_mat.T @ r_full
        else:
            r = res_fn_core(x)
        return np.asarray(r, dtype=float)

    def merit_from_r(r):
        return 0.5 * float(np.dot(r, r))

    x0 = np.asarray(dofs, dtype=float)
    p = np.asarray(inc, dtype=float)

    # Optional maxstep scaling (PETSc-like)
    if maxstep is not None:
        pn = np.linalg.norm(p)
        if pn > maxstep and pn > 0.0:
            p = p * (maxstep / pn)

    # phi(0)
    if r0 is not None:
        r0 = np.asarray(r0, dtype=float)
    else:
        r0 = residual(x0)

    phi0 = merit_from_r(r0)
    if not np.isfinite(phi0) or phi0 == 0.0:
        return x0

    # Estimate phi'(0) = R^T (J p)
    if slope_mode == "newton":
        # Only reliable if p is a true Newton step: Jp ≈ -R
        dphi0 = -2.0 * phi0
    elif slope_mode == "fd":
        # Finite-difference JVP: Jp ≈ (R(x+eps p) - R(x)) / eps
        pn = np.linalg.norm(p)
        if pn == 0.0 or not np.isfinite(pn):
            return x0
        if fd_eps is None:
            fd_eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x0)) / (pn + 1e-30)
        r_eps = residual(x0 + fd_eps * p)
        Jp = (r_eps - r0) / fd_eps
        dphi0 = float(np.dot(r0, Jp))
    else:
        raise ValueError("slope_mode must be 'newton' or 'fd'")

    # Need descent slope for Armijo
    if (not np.isfinite(dphi0)) or dphi0 >= 0.0:
        logger.warning("Line search: non-descent direction (phi'(0) >= 0). Forcing conservative slope.")
        dphi0 = -abs(dphi0) if np.isfinite(dphi0) else -1.0

    # Initial lambda
    lam = float(max(damping, minlambda))
    
    # Check for immediate NaNs with full step
    x_check = x0 + lam * p
    r_check = residual(x_check)
    if not np.all(np.isfinite(r_check)):
        logger.warning(f"Line search: NaN detected with damping={lam}. Reducing initial step.")
        # print(f'full residual NaN at initial step: {r_check}')
        # Try reducing iteratively until not NaN. 
        # In NaN crisis, we allow going lower than standard minlambda (e.g. to 0.05 or 1e-3)
        nan_crisis_min = 0.05 
        for _ in range(1): # try reducing more times
             lam *= 0.5
             if lam < nan_crisis_min: 
                 break 
             x_check = x0 + lam * p 
             r_check = residual(x_check)
             if np.all(np.isfinite(r_check)):
                 logger.debug(f"Line search: Found stable initial step lam={lam}")
                 break
        if not np.all(np.isfinite(r_check)):
             logger.warning("Line search: Could not recover from NaN even after reducing damping.")


    def trial_eval(lam_try, r_provided=None):
        x_try = x0 + lam_try * p

        # Optional cheap acceptability filter (detF>0, finite stress, etc.)
        if accept_check is not None:
            ok, reason = accept_check(x_try)
            if not ok:
                return x_try, np.inf, ["rejected", reason]

        if r_provided is not None:
            r_try = r_provided
        else:
            r_try = residual(x_try)
        
        phi_try = merit_from_r(r_try)
        return x_try, phi_try, ["ok", ""]
    
    x_try, phi_try, status = trial_eval(lam, r_provided=r_check)

    for it in range(1, max_ls + 1):
        # If accept_check rejected, shrink and retry
        if status[0] == "rejected":
            lam_new = max(minlambda, fail_shrink * lam)
            verbose = True
            if verbose:
                print(f"[LS] reject({status[1]}) -> lambda {lam:.3e}->{lam_new:.3e}")
            lam = lam_new
            if lam <= minlambda:
                lam =minlambda
                logger.warning(f"Line search: accept_check keeps failing; reached minlambda={minlambda}.")
                return x0 + lam * p, status
            logger.warning(f"[LS] retrying with lambda={lam}")
            x_try, phi_try, status = trial_eval(lam)
            continue

        # Armijo sufficient decrease
        if np.isfinite(phi_try) and (phi_try <= phi0 + c1 * lam * dphi0):
            logger.debug(f"[LS] Armijo: phi0={phi0}, phi={phi_try}, lambda {lam}")
            logger.debug(f"Line search accepted: lambda={lam:.6g}, iters={it}")
            if phi_try >= phi0:
                logger.warning("Line search: accepted step did not reduce merit function.")
                status[0] = "rejected"
                status[1] = "no merit decrease"
            else:
                status[0] = "accepted"
                status[1] = ""
            return x_try, status

        # Terminate if lambda too small
        if lam <= minlambda:
            if phi_try > phi0:
                x_try = x0 + minlambda * p
            logger.warning(f"[LS] Armijo: phi0={phi0}, phi={phi_try}, lambda {lam}")
            logger.warning(f"Line search: reached minlambda={minlambda:.3e}; returning last trial.")
            return x_try, status

        # PETSc-style safeguarded quadratic step:
        # phi(lam) ≈ phi0 + dphi0*lam + c*lam^2
        denom = (phi_try - phi0 - dphi0 * lam)
        if np.isfinite(denom) and abs(denom) > 1e-30:
            c = denom / (lam * lam)
            if c > 0.0:
                lam_q = -dphi0 / (2.0 * c)
            else:
                lam_q = 0.5 * lam
        else:
            lam_q = 0.5 * lam

        # Safeguards like PETSc: do not shrink too little or too much
        lam_new = float(np.clip(lam_q, shrink_lo * lam, shrink_hi * lam))
        lam_new = max(lam_new, minlambda)
       

        lam = lam_new
        x_try, phi_try, status = trial_eval(lam)
    
    logger.warning("Line search: max_ls reached; returning last trial.")
    return x_try, status


def line_search1(problem, dofs, inc, r0=None):
    """
    TODO: This is useful for finite deformation plasticity.
    """
    logger.debug("Starting line search with basic halving")
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    # grad_res_norm_fn = jax.grad(res_norm_fn)
    # hess_res_norm_fn = jax.hessian(res_norm_fn)

    # tol = 1e-3
    # alpha = 1.
    # lr = 1.
    # grad_alpha = 1.
    # while np.abs(grad_alpha) > tol:
    #     grad_alpha = grad_res_norm_fn(alpha)
    #     hess_alpha = hess_res_norm_fn(alpha)
    #     alpha = alpha - 1./hess_alpha*grad_alpha
    #     print(f"alpha = {alpha}, grad_alpha = {grad_alpha}, hess_alpha = {hess_alpha}")

    alpha = 1.
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half

    status = ["accepted", "Basic halving line search"]
    return dofs + alpha*inc, status

# # @jax.jit
# def temp(A, row_inds):
#     zero_mask = np.isin(np.arange(A.shape[0]), row_inds)
#     nonzero_mask = np.invert(zero_mask)
#     elim_rows = BCOO((nonzero_mask.astype(int), np.vstack((np.arange(A.shape[0]), np.arange(A.shape[0]))).T), shape=A.shape)
#     recover_diag = BCOO((zero_mask.astype(int), np.vstack((np.arange(A.shape[0]), np.arange(A.shape[0]))).T), shape=A.shape)
#     A = elim_rows @ A + recover_diag
#     return A

# # @jax.jit
# def temp2(A, row_inds):
#     mask = np.isin(A.indices[:,0], row_inds, invert=True)
#     # mask_columns = np.isin(A.indices[:,1], row_inds, invert=True)
#     # mask = np.logical_and(mask, mask_columns)

#     # Note concrete masks are required for jit
#     ij = np.vstack((A.indices[mask,:],
#                     np.repeat(row_inds[:,None],2, axis=1)))
#     v = np.concatenate((A.data[mask], np.ones(len(row_inds))))
#     # Assemble A with zero rows and ones on the diagonal of the zero rows
#     A = BCOO((v, ij), shape=A.shape)
#     return A

@jax.jit
def row_elimination_jax(A, row_inds):
    mask = np.isin(A.indices[:,0], row_inds, invert=True)
    # mask_columns = np.isin(A.indices[:,1], row_inds, invert=True)
    # mask = np.logical_and(mask, mask_columns)

    # Note concrete masks are required for jit
    ij = np.vstack((A.indices,
                    np.repeat(row_inds[:,None],2, axis=1)))
    v = np.concatenate((np.where(mask,A.data,0.0), np.ones(len(row_inds))))
    # Assemble A with zero rows and ones on the diagonal of the zero rows
    A = BCOO((v, ij), shape=A.shape)
    return A

def row_elimination_petsc(A, row_inds):
    pass



def get_A(problem, solver_options):
    """
    Optimized sparse matrix assembly.
    For small systems (~6000 DOFs), the overhead of CSR construction and 
    PETSc conversion is significant. This version uses a cached sparsity 
    pattern and direct row elimination.
    """
    # 1. Transfer Jacobian values from GPU to CPU
    V_cpu = onp.asarray(problem.V)

    # 2. Build or Update Scipy CSR matrix
    # Check if we have a cached structure on the problem object
    if not hasattr(problem, '_cached_csr_structure'):
        logger.debug("Creating initial CSR sparsity pattern cache...")
        # Create a temporary coo matrix to get the sorted CSR structure
        A_initial = scipy.sparse.csr_array((V_cpu, (problem.I, problem.J)),
                                            shape=(problem.num_total_dofs_all_vars, 
                                                   problem.num_total_dofs_all_vars))
        # Cache the structure
        problem._cached_csr_structure = A_initial
    else:
        # Update existing matrix data in-place (much faster)
        # Note: This assumes problem.I and problem.J (the sparsity pattern) are constant.
        # If problem.V is already sorted by (I, J), we can use this. 
        # Since we use scipy.sparse.csr_array above, we should rebuild if the mapping is complex,
        # but for small 6000 DOF systems, even a fresh csr_array with pre-sorted indices is fast.
        problem._cached_csr_structure = scipy.sparse.csr_array((V_cpu, (problem.I, problem.J)),
                                            shape=(problem.num_total_dofs_all_vars, 
                                                   problem.num_total_dofs_all_vars))

    A_sp = problem._cached_csr_structure

    # 3. Direct Row/Column Elimination (Bypass PETSc)
    # This is much faster for small systems than converting to/from PETSc.
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind],
                                 dtype=onp.int32)
            
            # Row elimination: zero the row and set diagonal to 1.0
            for row_ind in row_inds:
                start, end = A_sp.indptr[row_ind], A_sp.indptr[row_ind+1]
                A_sp.data[start:end] = 0.0
                diag_idx = onp.where(A_sp.indices[start:end] == row_ind)[0]
                if len(diag_idx) > 0:
                    A_sp.data[start + diag_idx[0]] = 1.0

    # Convert back to JAX BCOO for the solver interface.
    # JAX callbacks require valid JAX types; Scipy objects are not allowed in the trace.
    return BCOO.from_scipy_sparse(A_sp).sort_indices()



################################################################################
# The "row elimination" solver

def solver(problem, solver_options={}):
    """
    Specify exactly either 'jax_solver' or 'umfpack_solver' or 'petsc_solver'

    Examples:
    (1) solver_options = {'jax_solver': {}}
    (2) solver_options = {'umfpack_solver': {}}
    (3) solver_options = {'petsc_solver': {'ksp_type': 'bcgsl', 'pc_type': 'jacobi'}, 'initial_guess': some_guess}

    Default parameters will be used if no instruction is found:

    solver_options =
    {
        # If multiple solvers are specified or no solver is specified, 'jax_solver' will be used.
        'jax_solver':
        {
            # The JAX built-in linear solver
            # Reference: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.bicgstab.html
            'precond': True,
        }

        'umfpack_solver':
        {
            # The scipy solver that calls UMFPACK
            # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
        }

        'petsc_solver':
        {
            # PETSc solver
            # For more ksp_type and pc_type: https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/index.html
            'ksp_type': 'bcgsl', # e.g., 'minres', 'gmres', 'tfqmr'
            'pc_type': 'ilu', # e.g., 'jacobi'
        }

        'line_search_flag': False, # Line search method
        'initial_guess': initial_guess, # Same shape as sol_list
        'tol': 1e-5, # Absolute tolerance for residual vector (l2 norm), used in Newton's method
        'rel_tol': 1e-8, # Relative tolerance for residual vector (l2 norm), used in Newton's method
    }

    The solver imposes Dirichlet B.C. with "row elimination" method.

    Some memo:

    res(u) = D*r(u) + (I - D)u - u_b
    D = [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 0]
         [0 0 0 1]]
    I = [[1 0 0 0]
         [0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]
    A = d(res)/d(u) = D*dr/du + (I - D)

    TODO: linear multipoint constraint

    The function newton_update computes r(u) and dr/du
    """
    logger.debug(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if 'initial_guess' in solver_options:
        # We dont't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if hasattr(problem, 'P_mat'):
            dofs = np.zeros(problem.P_mat.shape[1]) # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = solver_options['rel_tol'] if 'rel_tol' in solver_options else 1e-8
    tol = solver_options['tol'] if 'tol' in solver_options else 1e-9

    def newton_update_helper(dofs):
        if hasattr(problem, 'P_mat'):
            dofs = problem.P_mat @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        if hasattr(problem, 'P_mat'):
            res_vec = problem.P_mat.T @ res_vec

        A = get_A(problem, solver_options)
        return res_vec, A

    res_vec, A = newton_update_helper(dofs)
    # print('A_matrix',A.todense())
    # print('res_vec',res_vec)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = np.maximum(res_val, 1.e-8) # Avoid division by zero
    rel_res_val = res_val / res_val_initial
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = {rel_res_val}")
    
    # check for nan/inf in initial residual
    if not (np.all(np.isfinite(res_vec)) and np.all(np.isfinite(dofs))):
        logger.warning(f"NaN or Inf detected in initial solution or residual. Aborting solve attempt.")
        sol_list = problem.unflatten_fn_sol_list(dofs)
        if solver_options.get('return_full_info', False):
            return sol_list, 0, False
        elif solver_options.get('iteration_count', False):
            return [sol_list, 0]
        else:
            return sol_list
    # Get solver parameters
    iteration_counter = 0
    max_iters = solver_options.get('max_iters', 50) # Default to 50 iterations if not specified
    update_jacobian_freq = solver_options.get('jacobian_freq', 1)
    max_newton_time = solver_options['max_newton_time'] if 'max_newton_time' in solver_options else None
    
    has_converged = False
    nan_detected = False
    no_merit_decrease = False
    res_val_prev = res_val

    # Main Newton-Raphson iteration loop
    newton_loop_start = time.time()
    while (rel_res_val > rel_tol and res_val > tol) and (iteration_counter < max_iters):
        if max_newton_time is not None:
            elapsed_newton = time.time() - newton_loop_start
            if elapsed_newton > max_newton_time:
                raise AssertionError(
                    f"Newton loop exceeded max_newton_time={max_newton_time:.1f}s "
                    f"(elapsed={elapsed_newton:.1f}s, iterations={iteration_counter})"
                )
        dofs,status = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val / res_val_initial

        # print('A_matrix',A.todense())
        # print('res_vec',res_vec)
        # Update Jacobian matrix at a specified frequency
        if rel_res_val > rel_tol and res_val > tol:
            if (iteration_counter + 1) % update_jacobian_freq == 0:
                A = get_A(problem, solver_options)

        res_val_prev = res_val
        logger.debug(f"Iteration {iteration_counter + 1}: l_2 res = {res_val:.4e}, relative l_2 res = {rel_res_val:.4e}")
        iteration_counter += 1
        
        if status[0] == "rejected":
            logger.debug(f"Step rejected at iteration {iteration_counter}, due to {status[1]}. Retrying with smaller step.")
            no_merit_decrease = True
            break
        # --- NaN and Inf Check ---
        # Instead of asserting, check for non-finite values and break the loop if found.
        if not (np.all(np.isfinite(res_val)) and np.all(np.isfinite(dofs))):
            logger.warning(f"NaN or Inf detected in solution or residual at iteration {iteration_counter + 1}. Aborting step attempt.")
            nan_detected = True
            break # Exit the Newton loop immediately

    # Check for convergence status after the loop
    if nan_detected or no_merit_decrease:
        # If the loop was broken by a NaN, it has not converged.
        has_converged = False
    elif rel_res_val <= rel_tol or res_val <= tol:
        # If tolerances are met, it has converged.
        has_converged = True
        logger.info(f"Converged in {iteration_counter} iterations.")
    else:
        # If loop finished due to max_iters, it has not converged.
        has_converged = False
        logger.warning(f"Solver did not converge after {iteration_counter} iterations (max_iters reached).")

    if hasattr(problem, 'P_mat'):
        dofs = problem.P_mat @ dofs

    sol_list = problem.unflatten_fn_sol_list(dofs)

    # Store convergence info on problem for callers that need it (e.g. ad_wrapper path)
    problem._solver_converged = has_converged
    problem._solver_num_iters = iteration_counter

    # Cache the converged tangent for the adjoint pass to reuse, avoiding a
    # redundant assembly + factorization in implicit_vjp.  Only cache on success;
    # on failure leave the cache cleared so the adjoint can rebuild safely.
    #
    # IMPORTANT: the Newton loop above intentionally *skips* the final A update
    # when the convergence test passes, so the local `A` variable holds the
    # tangent at iterate (n-1), not at the converged solution.  For the adjoint
    # pass we need the tangent at the converged state -- otherwise solving
    # A^T lambda = v against a near-singular stale tangent produces gradients
    # of arbitrary magnitude (e.g. 1e+55 blow-ups under highly nonlinear
    # constitutive responses such as exp-map plasticity through eigh).
    #
    # Assemble the tangent one more time at the converged dofs before caching.
    if has_converged:
        # newton_update() refreshes problem internal state (residual, weak form
        # cell jacobian, etc.) at the current dofs; calling it here ensures the
        # subsequent get_A() builds the tangent at the converged solution.
        _ = problem.newton_update(sol_list)
        A = get_A(problem, solver_options)
        problem._cached_converged_A = A
    else:
        problem._cached_converged_A = None

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve attempt took {solve_time:.2f} [s]")
    logger.debug(f"max of dofs = {np.max(dofs)}")
    logger.debug(f"min of dofs = {np.min(dofs)}")
    # jax.debug.print('dofs,{}',dofs)
    # Return convergence information if requested
    if solver_options.get('return_full_info', False):
        return sol_list, iteration_counter, has_converged
    elif solver_options.get('iteration_count', False):
        return [sol_list, iteration_counter]
    else:
        return sol_list

    # return sol_list


################################################################################
# The "arc length" solver
# Reference: Vasios, Nikolaos. "Nonlinear analysis of structures." The Arc-Length method. Harvard (2015).
# Our implementation follows the Crisfeld's formulation

# TODO: Do we want to merge displacement-control and force-control codes?

def arc_length_solver_disp_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem, lamda)
        A = get_A(problem, solver_options={'umfpack_solver':{}})
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(u_b**2.)

    u_vec = prev_u_vec
    lamda = prev_lamda

    u_b = assign_bc(np.zeros_like(prev_u_vec), problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-9
    res_val = 1.
    while res_val > tol:

        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        delta_u_bar = umfpack_solve(A, -res_vec)
        delta_u_t = umfpack_solve(A, u_b)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(u_b**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(u_b**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(u_b**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")

    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def arc_length_solver_force_driven(problem, prev_u_vec, prev_lamda, prev_Delta_u_vec, prev_Delta_lamda, q_vec, Delta_l=0.1, psi=1.):
    """
    TODO: Does not support periodic B.C., need some work here.
    """
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem, solver_options={'umfpack_solver':{}})
        return res_vec, A

    def u_lamda_dot_product(Delta_u_vec1, Delta_lamda1, Delta_u_vec2, Delta_lamda2):
        return np.sum(Delta_u_vec1*Delta_u_vec2) + psi**2.*Delta_lamda1*Delta_lamda2*np.sum(q_vec_mapped**2.)

    u_vec = prev_u_vec
    lamda = prev_lamda
    q_vec_mapped = assign_zeros_bc(q_vec, problem)

    Delta_u_vec_dir = prev_Delta_u_vec
    Delta_lamda_dir = prev_Delta_lamda

    tol = 1e-9
    res_val = 1.
    while res_val > tol:
        res_vec, A = newton_update_helper(u_vec)
        res_val = np.linalg.norm(res_vec + lamda*q_vec_mapped)
        logger.debug(f"Arc length solver: res_val = {res_val}")

        # TODO: the scipy umfpack solver seems to be far better than the jax linear solver, so we use umfpack solver here.
        # x0_1 = assign_bc(np.zeros_like(u_vec), problem)
        # x0_2 = copy_bc(u_vec, problem)
        # delta_u_bar = jax_solve(problem, A, -(res_vec + lamda*q_vec_mapped), x0=x0_1 - x0_2, precond=True)
        # delta_u_t = jax_solve(problem, A, -q_vec_mapped, x0=np.zeros_like(u_vec), precond=True)

        delta_u_bar = umfpack_solve(A, -(res_vec + lamda*q_vec_mapped))
        delta_u_t = umfpack_solve(A, -q_vec_mapped)

        Delta_u_vec = u_vec - prev_u_vec
        Delta_lamda = lamda - prev_lamda
        a1 = np.sum(delta_u_t**2.) + psi**2.*np.sum(q_vec_mapped**2.)
        a2 = 2.* np.sum((Delta_u_vec + delta_u_bar)*delta_u_t) + 2.*psi**2.*Delta_lamda*np.sum(q_vec_mapped**2.)
        a3 = np.sum((Delta_u_vec + delta_u_bar)**2.) + psi**2.*Delta_lamda**2.*np.sum(q_vec_mapped**2.) - Delta_l**2.

        delta_lamda1 = (-a2 + np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)
        delta_lamda2 = (-a2 - np.sqrt(a2**2. - 4.*a1*a3))/(2.*a1)

        logger.debug(f"Arc length solver: delta_lamda1 = {delta_lamda1}, delta_lamda2 = {delta_lamda2}")
        assert np.isfinite(delta_lamda1) and np.isfinite(delta_lamda2), f"No valid solutions for delta lambda, a1 = {a1}, a2 = {a2}, a3 = {a3}"

        delta_u_vec1 = delta_u_bar + delta_lamda1 * delta_u_t
        delta_u_vec2 = delta_u_bar + delta_lamda2 * delta_u_t

        Delta_u_vec_dir1 = u_vec + delta_u_vec1 - prev_u_vec
        Delta_lamda_dir1 = lamda + delta_lamda1 - prev_lamda
        dot_prod1 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir1, Delta_lamda_dir1)

        Delta_u_vec_dir2 = u_vec + delta_u_vec2 - prev_u_vec
        Delta_lamda_dir2 = lamda + delta_lamda2 - prev_lamda
        dot_prod2 = u_lamda_dot_product(Delta_u_vec_dir, Delta_lamda_dir, Delta_u_vec_dir2, Delta_lamda_dir2)

        if np.abs(dot_prod1) < 1e-10 and np.abs(dot_prod2) < 1e-10:
            # At initial step, (Delta_u_vec_dir, Delta_lamda_dir) is zero, so both dot_prod1 and dot_prod2 are zero.
            # We simply select the larger value for delta_lamda.
            delta_lamda = np.maximum(delta_lamda1, delta_lamda2)
        elif dot_prod1 > dot_prod2:
            delta_lamda = delta_lamda1
        else:
            delta_lamda = delta_lamda2

        lamda = lamda + delta_lamda
        delta_u = delta_u_bar + delta_lamda * delta_u_t
        u_vec = u_vec + delta_u

        Delta_u_vec_dir = u_vec - prev_u_vec
        Delta_lamda_dir = lamda - prev_lamda

    logger.debug(f"Arc length solver: finished for one step, with Delta lambda = {lamda - prev_lamda}")

    return u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir


def get_q_vec(problem):
    """
    Used in the arc length method only, to get the external force vector q_vec
    """
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list = problem.newton_update(sol_list)
    q_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    return q_vec


################################################################################
# Dynamic relaxation solver

def assembleCSR(problem, dofs):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    problem.newton_update(sol_list)
    A_sp_scipy = scipy.sparse.csr_array((problem.V, (problem.I, problem.J)),
        shape=(problem.fes[0].num_total_dofs, problem.fes[0].num_total_dofs))

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(fe.node_inds_list[i] * fe.vec + fe.vec_inds_list[i] + problem.offset[ind], dtype=onp.int32)
            for row_ind in row_inds:
                A_sp_scipy.data[A_sp_scipy.indptr[row_ind]: A_sp_scipy.indptr[row_ind + 1]] = 0.
                A_sp_scipy[row_ind, row_ind] = 1.

    return A_sp_scipy


def calC(t, cmin, cmax):

    if t < 0.: t = 0.

    c = 2. * onp.sqrt(t)
    if (c < cmin): c = cmin
    if (c > cmax): c = cmax

    return c


def printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force):

    ## printing control
    if nIters % nPrint == 1:
        #logger.info('\t------------------------------------')
        if info_force == True:
            print(('\nDR Iteration %d: Max force (residual error) = %g (tol = %g)' +
                   'Max velocity = %g') % (nIters, error, tol,
                                            np.max(np.absolute(qdot))))
        if info == True:
            print('\nDamping t: ',t, );
            print('Damping coefficient: ', c)
            print('Max epsilon: ',np.max(eps))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))


def dynamic_relax_solve(problem, tol=1e-9, nKMat=50, nPrint=500, info=True, info_force=True, initial_guess=None):
    """
    Implementation of

    Luet, David Joseph. Bounding volume hierarchy and non-uniform rational B-splines for contact enforcement
    in large deformation finite element analysis of sheet metal forming. Diss. Princeton University, 2016.
    Chapter 4.3 Nonlinear System Solution

    Particularly good for handling buckling behavior.
    There is a FEniCS version of this dynamic relaxation algorithm.
    The code below is a direct translation from the FEniCS version.


    TODO: Does not support periodic B.C., need some work here.
    """
    solver_options = {'umfpack_solver': {}}

    # TODO: combine these into initial guess
    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)
        A = get_A(problem, solver_options)
        return res_vec, A

    dofs = np.zeros(problem.num_total_dofs_all_vars)
    res_vec, A = newton_update_helper(dofs)
    dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)

    if initial_guess is not None:
        dofs = initial_guess
        dofs = assign_bc(dofs, problem)

    # parameters not to change
    cmin = 1e-3
    cmax = 3.9
    h_tilde = 1.1
    h = 1.

    # initialize all arrays
    N = len(dofs)  #print("--------num of DOF's: %d-----------" % N)
    #initialize displacements, velocities and accelerations
    q, qdot, qdotdot = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize displacements, velocities and accelerations from a previous time step
    q_old, qdot_old, qdotdot_old = onp.zeros(N), onp.zeros(N), onp.zeros(N)
    #initialize the M, eps, R_old arrays
    eps, M, R, R_old = onp.zeros(N), onp.zeros(N), onp.zeros(N), onp.zeros(N)

    @jax.jit
    def assembleVec(dofs):
        res_fn = get_flatten_fn(problem.compute_residual, problem)
        res_vec = res_fn(dofs)
        res_vec = assign_zeros_bc(res_vec, problem)
        return res_vec

    R = onp.array(assembleVec(dofs))
    KCSR = assembleCSR(problem, dofs)

    M[:] = h_tilde * h_tilde / 4. * onp.array(
        onp.absolute(KCSR).sum(axis=1)).squeeze()
    q[:] = dofs
    qdot[:] = -h / 2. * R / M
    # set the counters for iterations and
    nIters, iKMat = 0, 0
    error = 1.0
    timeZ = time.time() #Measurement of loop time.

    assert onp.all(onp.isfinite(M)), f"M not finite"
    assert onp.all(onp.isfinite(q)), f"q not finite"
    assert onp.all(onp.isfinite(qdot)), f"qdot not finite"

    error = onp.max(onp.absolute(R))

    while error > tol:

        print(f"error = {error}")
        # marching forward
        q_old[:] = q[:]; R_old[:] = R[:]
        q[:] += h*qdot; dofs = np.array(q)

        R = onp.array(assembleVec(dofs))

        nIters += 1
        iKMat += 1
        error = onp.max(onp.absolute(R))

        # damping calculation
        S0 = onp.dot((R - R_old) / h, qdot)
        t = S0 / onp.einsum('i,i,i', qdot, M, qdot)
        c = calC(t, cmin, cmax)

        # determine whether to recal KMat
        eps = h_tilde * h_tilde / 4. * onp.absolute(
            onp.divide((qdotdot - qdotdot_old), (q - q_old),
                       out=onp.zeros_like((qdotdot - qdotdot_old)),
                       where=(q - q_old) != 0))

        # calculating the jacobian matrix
        if ((onp.max(eps) > 1) and (iKMat > nKMat)): #SPR JAN max --> min
            if info == True:
                print('\nRecalculating the tangent matrix: ', nIters)

            iKMat = 0
            KCSR = assembleCSR(problem, dofs)
            M[:] = h_tilde * h_tilde / 4. * onp.array(
                onp.absolute(KCSR).sum(axis=1)).squeeze()

        # compute new velocities and accelerations
        qdot_old[:] = qdot[:]; qdotdot_old[:] = qdotdot[:];
        qdot = (2.- c*h)/(2 + c*h) * qdot_old - 2.*h/(2.+c*h)* R / M
        qdot_old[:] = qdot[:]
        qdotdot = qdot - qdot_old

        # output on screen
        printInfo(error, t, c, tol, eps, qdot, qdotdot, nIters, nPrint, info, info_force)

    # check if converged
    convergence = True
    if onp.isnan(onp.max(onp.absolute(R))):
        convergence = False

    # print final info
    if convergence:
        print("DRSolve finished in %d iterations and %fs" % \
              (nIters, time.time() - timeZ))
    else:
        print("FAILED to converged")

    sol_list = problem.unflatten_fn_sol_list(dofs)

    return sol_list[0]


################################################################################
# Implicit differentiation with the adjoint method
def get_gpu_memory_info(gpu_id=0):
    """Get GPU memory statistics in bytes"""
    device = jax.devices()[gpu_id]
    mem_stats = device.memory_stats()
    
    bytes_in_use = mem_stats['bytes_in_use']
    bytes_limit = mem_stats['bytes_limit']
    free_mem = bytes_limit - bytes_in_use
    
    return free_mem, bytes_limit

def alternative_vjp_with_forward(problem, sol_list, params, v_list, adjoint_solver_options):

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)
        """
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(sol_list)
        def vjp_linear_fn(v_list):
            try:
                _, f_vjp = jax.vjp(partial_c_fn, params)
                val, = f_vjp(v_list)
                return val
            except NotImplementedError as err:
                # Some custom-JVP paths are not reverse-transposable (e.g., stop_gradient transpose).
                # Fallback: compute v^T (dc/dp) using forward-mode JVP over parameter basis vectors.
                if "stop_gradient" not in str(err):
                    raise

                p_flat, unflatten_p = jax.flatten_util.ravel_pytree(params)
                v_flat, _ = jax.flatten_util.ravel_pytree(v_list)

                def c_flat(pf):
                    p_tree = unflatten_p(pf)
                    c_tree = partial_c_fn(p_tree)
                    c_vec, _ = jax.flatten_util.ravel_pytree(c_tree)
                    return c_vec

                eye = np.eye(p_flat.shape[0], dtype=p_flat.dtype)

                def one_basis_col(ei):
                    _, jvp_col = jax.jvp(c_flat, (p_flat,), (ei,))
                    return np.vdot(v_flat, jvp_col)

                grad_flat = jax.vmap(one_basis_col)(eye)
                return unflatten_p(grad_flat)
        return vjp_linear_fn
    # Checkpoint the VJP computation to save memory
    # def get_vjp_contraint_fn_params(params, sol_list):
    #     partial_c_fn = lambda p: constraint_fn_sol_to_sol(sol_list, p)
    #     @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    #     def vjp_linear_fn(v_list):
    #         primals, f_vjp = jax.vjp(partial_c_fn, params)
    #         val, = f_vjp(v_list)
    #         return val
    #     return vjp_linear_fn

    problem.set_params(params)

    # Adjoint Jacobian reuse: the forward Newton solver caches the converged
    # tangent on `problem._cached_converged_A`.  If present, skip redundant
    # assembly + factorization here -- the tangent at the converged point is
    # identical to what `get_A` would rebuild, since set_params was called with
    # the parameters that produced the converged state.
    cached_A = getattr(problem, '_cached_converged_A', None)
    if cached_A is not None:
        A = cached_A
    else:
        problem.newton_update(sol_list)
        A = get_A(problem, adjoint_solver_options)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if hasattr(problem, 'P_mat'):
        v_vec = problem.P_mat.T @ v_vec

    if not np.all(np.isfinite(v_vec)):
        raise FloatingPointError(
            "Adjoint RHS vector contains non-finite values before linear solve."
        )
    if hasattr(A, "data") and (not np.all(np.isfinite(A.data))):
        raise FloatingPointError(
            "Adjoint Jacobian matrix contains non-finite entries before linear solve."
        )

    adjoint_vec, _ = linear_solver(
        A.transpose(), v_vec, np.zeros_like(v_vec), adjoint_solver_options
    )
    if not np.all(np.isfinite(adjoint_vec)):
        raise FloatingPointError(
            "Adjoint linear solve returned non-finite values."
        )

    if hasattr(problem, 'P_mat'):
        adjoint_vec = problem.P_mat @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)

    # Checkpoint when actually calling the function
    # @partial(jax.checkpoint,
    #          policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def checkpointed_call(v):
        return vjp_linear_fn(v)
    
    # free, total = get_gpu_memory_info(0)
    # print(f"Before vjp_result:Free: {free / 1e9:.2f} GB, Total: {total / 1e9:.2f} GB")
    vjp_result = checkpointed_call(problem.unflatten_fn_sol_list(adjoint_vec))
    # vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree.map(lambda x: -x, vjp_result)

    return vjp_result

def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options):

    def constraint_fn(dofs, params):
        """c(u, p)
        """
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)
        """
        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)
        """
        partial_c_fn = get_partial_params_c_fn(sol_list)
        def vjp_linear_fn(v_list):
            primals, f_vjp = jax.vjp(partial_c_fn, params)
            val, = f_vjp(v_list)
            return val
        return vjp_linear_fn
    # Checkpoint the VJP computation to save memory
    # def get_vjp_contraint_fn_params(params, sol_list):
    #     partial_c_fn = lambda p: constraint_fn_sol_to_sol(sol_list, p)
    #     @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    #     def vjp_linear_fn(v_list):
    #         primals, f_vjp = jax.vjp(partial_c_fn, params)
    #         val, = f_vjp(v_list)
    #         return val
    #     return vjp_linear_fn

    problem.set_params(params)

    problem.newton_update(sol_list)
    A = get_A(problem, adjoint_solver_options)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if hasattr(problem, 'P_mat'):
        v_vec = problem.P_mat.T @ v_vec

    if not np.all(np.isfinite(v_vec)):
        raise FloatingPointError(
            "Adjoint RHS vector contains non-finite values before linear solve."
        )
    
    if hasattr(A, "data") and (not np.all(np.isfinite(A.data))):
        raise FloatingPointError(
            "Adjoint Jacobian matrix contains non-finite entries before linear solve."
        )

    adjoint_vec, _ = linear_solver(
        A.transpose(), v_vec, np.zeros_like(v_vec), adjoint_solver_options
    )
    if not np.all(np.isfinite(adjoint_vec)):
        raise FloatingPointError(
            "Adjoint linear solve returned non-finite values."
        )
    
    v_norm = float(onp.linalg.norm(onp.array(v_vec)))
    adj_norm = float(onp.linalg.norm(onp.array(adjoint_vec)))
    _MAX_ADJOINT_AMPLIFICATION = 1e6  # tune if needed
    if v_norm > 0 and adj_norm > _MAX_ADJOINT_AMPLIFICATION * v_norm:
        logger.warning(
            f"implicit_vjp: adjoint blowup detected. "
            f"|adjoint|={adj_norm:.3e}, |rhs|={v_norm:.3e}, ratio={adj_norm/v_norm:.3e}. "
            f"Clamping adjoint to prevent cascading gradient explosion."
        )
        scale = (_MAX_ADJOINT_AMPLIFICATION * v_norm) / adj_norm
        adjoint_vec = adjoint_vec * scale

    if hasattr(problem, 'P_mat'):
        adjoint_vec = problem.P_mat @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)

    # Checkpoint when actually calling the function
    # @partial(jax.checkpoint,
    #          policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def checkpointed_call(v):
        return vjp_linear_fn(v)
    
    # free, total = get_gpu_memory_info(0)
    # print(f"Before vjp_result:Free: {free / 1e9:.2f} GB, Total: {total / 1e9:.2f} GB")
    vjp_result = checkpointed_call(problem.unflatten_fn_sol_list(adjoint_vec))
    # vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree.map(lambda x: -x, vjp_result)

    return vjp_result



def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        safe_solver_opts = dict(solver_options)
        if 'initial_guess' in safe_solver_opts:
            if isinstance(params, (list, tuple)) and len(params) >= 3:
                safe_solver_opts['initial_guess'] = [params[2]]
            else:
                safe_solver_opts.pop('initial_guess')
        sol_list = solver(problem, safe_solver_opts)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.debug("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred



def ad_wrapper_discrete3(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    # Set up shardings for device and host memory
    s_dev = SingleDeviceSharding(jax.devices()[0]) #, memory_kind="device"
    s_host = SingleDeviceSharding(jax.devices()[0]) #, memory_kind="pinned_host"
    
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = checkpoint(time_stepping_forward(params, problem, solver, solver_options))
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        # Offload trajectories to host memory to free GPU memory
        sol_traj_host = jax.device_put(sol_traj, s_host)
        params_traj_host = jax.device_put(params_traj, s_host)
        del params_traj
        # Return solution for loss (keep on device), save offloaded data for backward
        return sol_traj, (params_traj_host, sol_traj_host)

    def f_bwd(res, v_traj):
        """
        Executes the full discrete adjoint backward pass, accounting for both the
        implicit solve and the explicit internal variable update.
        """
        params_traj, sol_traj = res
        global_params = params_traj[0][3]
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)

        # --- Initialize sensitivities ---
        sol_final = sol_traj[-1]
        iv_prev_final = params_traj[-1][0]  # This is iv_{N-1}
        iv_final = problem.update_int_vars_gp(sol_final, iv_prev_final)

        # Sensitivity of the loss wrt the final internal variables (iv_N) is zero.
        propagated_sensitivity_iv = jax.tree_util.tree_map(np.zeros_like, iv_final)
        # Sensitivity from the future (step N) starts at zero for the solution variables.
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_traj[-1])

        # Iterate backwards from step n = N-1 down to 0.
        for i in range(len(params_traj) - 1, -1, -1):
            sol_n = sol_traj[i]
            params_for_step_n = params_traj[i]
            # Get iv_{n-1} from the parameters used for step n.
            iv_prev = params_for_step_n[0]

            # --- 1. Backward pass through `update_int_vars_gp(sol_n, iv_{n-1})` ---
            # Propagate sensitivity from iv_n back to sol_n and iv_{n-1}.
            update_fn = lambda s, iv: problem.update_int_vars_gp(s, iv)
            _, vjp_update_fn = jax.vjp(update_fn, sol_n, iv_prev)

            # Get sensitivities wrt the inputs of the update function.
            sens_from_iv_wrt_sol_n, sens_from_iv_wrt_iv_prev = vjp_update_fn(propagated_sensitivity_iv)

            # --- 2. Construct Total RHS for the Main Adjoint Solve at step n ---
            adjoint_rhs_list = jax.tree_util.tree_map(
                lambda loss, prop_u, prop_iv: loss + prop_u + prop_iv,
                v_traj[i],
                propagated_sensitivity_u,
                sens_from_iv_wrt_sol_n
            )

            # --- 3. Solve Main Adjoint System for λ_n and get all VJPs ---
            # FIX 1: Always force a fresh Jacobian assembly inside implicit_vjp.
            # The cached _cached_converged_A holds the Jacobian from the most
            # recent FORWARD step, not the current adjoint step.  Reusing it
            # across steps with accumulated plasticity/creep causes the adjoint
            # solve to blow up.  Clear the cache before every adjoint call so
            # implicit_vjp is forced to call newton_update + get_A for the
            # correct step state.
            if hasattr(problem, '_cached_converged_A'):
                del problem._cached_converged_A

            vjp_result = implicit_vjp(
                problem, [sol_n], params_for_step_n, adjoint_rhs_list, adjoint_solver_options
            )

            # --- 4. Accumulate Gradient wrt Global Parameters 'p' ---
            grad_wrt_p = vjp_result[3]
            sens_from_C_wrt_p = sens_from_iv_wrt_iv_prev[-1]
            sens_from_C_wrt_p_total = 0. * np.sum(sens_from_C_wrt_p, axis=(0, 1))

            total_grad_p = jax.tree_util.tree_map(
                lambda total, gR, gC: total + gR + gC,
                total_grad_p,
                grad_wrt_p,
                sens_from_C_wrt_p_total,
            )

            # --- 5. Compute and Propagate Total Sensitivities to the PREVIOUS step (n-1) ---
            if i > 0:
                # Sensitivity wrt u_{n-1} comes from this step's implicit solve.
                _prop = vjp_result[2]

                # FIX 2: Guard against cascading blowup in propagated_sensitivity_u.
                # If the adjoint solve at this step was poorly conditioned (e.g. due
                # to a near-singular Jacobian at high plastic/creep strain), the
                # propagated sensitivity can be orders of magnitude larger than the
                # adjoint RHS.  Detect this and scale it down before it poisons all
                # prior steps.
                _prop_flat = jax.flatten_util.ravel_pytree(_prop)[0]
                _rhs_flat  = jax.flatten_util.ravel_pytree(adjoint_rhs_list)[0]
                _prop_norm = float(onp.linalg.norm(onp.array(_prop_flat)))
                _rhs_norm  = float(onp.linalg.norm(onp.array(_rhs_flat)))
                _MAX_PROP_RATIO = 1e5
                if _rhs_norm > 0 and _prop_norm > _MAX_PROP_RATIO * _rhs_norm:
                    logger.warning(
                        f"f_bwd step {i}: propagated_sensitivity_u blowup detected "
                        f"(|prop|={_prop_norm:.3e}, |rhs|={_rhs_norm:.3e}, "
                        f"ratio={_prop_norm / _rhs_norm:.3e}). Scaling down."
                    )
                    _scale = (_MAX_PROP_RATIO * _rhs_norm) / _prop_norm
                    _prop = jax.tree_util.tree_map(lambda x: x * _scale, _prop)
                propagated_sensitivity_u = _prop

                # Sensitivity wrt iv_{n-1} comes from TWO places.
                propagated_sensitivity_iv = jax.tree_util.tree_map(
                    lambda from_solve, from_update: from_solve + from_update,
                    vjp_result[0],          # Contribution from the implicit solve
                    sens_from_iv_wrt_iv_prev  # Contribution from the explicit update
                )

        return (total_grad_p,)
    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred

def ad_wrapper_discrete2(time_stepping_forward, problem, solver_options={}, adjoint_solver_options={}):
    s_dev = SingleDeviceSharding(jax.devices()[0])
    s_host = SingleDeviceSharding(jax.devices()[0])

    @jax.custom_vjp
    def fwd_pred(params):
        # Forward run; backward will need full history
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Offload history to host to save device memory
        sol_traj_host = jax.device_put(sol_traj, s_host)
        params_traj_host = jax.device_put(params_traj, s_host)
        del params_traj
        return sol_traj, (params_traj_host, sol_traj_host)

    def f_bwd(res, v_traj):
        params_traj_host, sol_traj_host = res

        # Initialize gradient accumulator
        global_params = jax.device_put(params_traj_host[0][3], s_dev)
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)

        # Future sensitivities start at zero
        sol_final = jax.device_put(sol_traj_host[-1], s_dev)
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_final)

        # Backward sweep
        for i in range(len(params_traj_host) - 1, -1, -1):
            sol_n = jax.device_put(sol_traj_host[i], s_dev)
            params_n = jax.device_put(params_traj_host[i], s_dev)

            # RHS: loss at step n + propagated sensitivity from future
            adjoint_rhs = jax.tree_util.tree_map(
                lambda loss, prop: loss + prop,
                v_traj[i],
                propagated_sensitivity_u,
            )

            # Solve adjoint and get VJPs wrt (iv_{n-1}, …, u_{n-1}, p)
            vjp_result = implicit_vjp(problem, [sol_n], params_n, adjoint_rhs, adjoint_solver_options)

            # Accumulate param gradient
            total_grad_p = jax.tree_util.tree_map(
                lambda total, contrib: total + contrib,
                total_grad_p,
                vjp_result[3],
            )

            # Propagate u-sensitivity to previous step
            if i > 0:
                propagated_sensitivity_u = vjp_result[2]

        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred

def ad_wrapper_discrete(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        return sol_traj, (params_traj, sol_traj)

    def f_bwd(res, v_traj):
        """
        Executes the full discrete adjoint backward pass, accounting for both the
        implicit solve and the explicit internal variable update.
        """
        params_traj, sol_traj = res
        global_params = params_traj[0][3]
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, global_params)

        # --- Initialize sensitivities ---
        # We need the *final* state of the internal variables (iv_N) to initialize.
        # We compute it from the inputs to the very last step.
        sol_final = sol_traj[-1]
        iv_prev_final = params_traj[-1][0] # This is iv_{N-1}
        iv_final = problem.update_int_vars_gp(sol_final, iv_prev_final)
        
        # Sensitivity of the loss wrt the final internal variables (iv_N) is zero.
        propagated_sensitivity_iv = jax.tree_util.tree_map(np.zeros_like, iv_final)
        # Sensitivity from the future (step N) starts at zero for the solution variables.
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_traj[-1])

        # Iterate backwards from step n = N-1 down to 0.
        for i in range(len(params_traj) - 1, -1, -1):
            sol_n = sol_traj[i]
            params_for_step_n = params_traj[i]
            # Get iv_{n-1} from the parameters used for step n, as you specified.
            iv_prev = params_for_step_n[0]
            
            iv_core = iv_prev[0:-1]
            param = iv_prev[-1]
            def update_iv_core(sol_loc,iv_core_local,param_loc):
                iv_prev_local = [iv_core_local, param_loc]
                return problem.update_int_vars_gp(sol_loc, iv_prev_local)
            # --- 1. Backward pass through `update_int_vars_gp(sol_n, iv_{n-1})` ---
            # Propagate sensitivity from iv_n back to sol_n and iv_{n-1}.
            update_fn = lambda s, iv: problem.update_int_vars_gp(s, iv)
            _, vjp_update_fn = jax.vjp(update_fn, sol_n, iv_prev)
            
            # Get sensitivities wrt the inputs of the update function.
            sens_from_iv_wrt_sol_n, sens_from_iv_wrt_iv_prev = vjp_update_fn(propagated_sensitivity_iv)
            
            # --- 2. Construct Total RHS for the Main Adjoint Solve at step n ---
            # The total sensitivity for sol_n is the sum of three contributions.
            adjoint_rhs_list = jax.tree_util.tree_map(
                lambda loss, prop_u, prop_iv: loss + prop_u + prop_iv,
                v_traj[i],
                propagated_sensitivity_u,
                sens_from_iv_wrt_sol_n
            )
                
            # --- 3. Solve Main Adjoint System for λ_n and get all VJPs ---
            vjp_result = implicit_vjp(
                problem, [sol_n], params_for_step_n, adjoint_rhs_list, adjoint_solver_options
            )
            
            # --- 4. Accumulate Gradient wrt Global Parameters 'p' ---
            grad_wrt_p = vjp_result[3]
            # total_grad_p = jax.tree_util.tree_map(lambda total, contrib: total + contrib, total_grad_p, grad_wrt_p)
            sens_from_C_wrt_p = sens_from_iv_wrt_iv_prev[-1]
            sens_from_C_wrt_p_total =  0. * np.sum(sens_from_C_wrt_p, axis=(0, 1))
            # breakpoint()
            
            total_grad_p = jax.tree_util.tree_map(
                lambda total, gR, gC: total + gR + gC,
                total_grad_p,
                grad_wrt_p,
                sens_from_C_wrt_p_total,
            )

            # --- 5. Compute and Propagate Total Sensitivities to the PREVIOUS step (n-1) ---
            if i > 0:
                # Sensitivity wrt u_{n-1} comes from this step's implicit solve.
                propagated_sensitivity_u = vjp_result[2]
                
                # Sensitivity wrt iv_{n-1} comes from TWO places.
                propagated_sensitivity_iv = jax.tree_util.tree_map(
                    lambda from_solve, from_update: from_solve + from_update,
                    vjp_result[0], # Contribution from the implicit solve
                    sens_from_iv_wrt_iv_prev # Contribution from the explicit update
                )
                
        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred

def ad_wrapper_discrete_(time_stepping_forward,problem,solver_options={},adjoint_solver_options={}):
    @jax.custom_vjp
    def fwd_pred(params):
        # The VJP only needs the final trajectory for the loss function,
        # but the backward pass needs the full history.
        sol_traj, _ = time_stepping_forward(params, problem, solver, solver_options)
        return sol_traj

    def f_fwd(params):
        sol_traj, params_traj = time_stepping_forward(params, problem, solver, solver_options)
        # Return solution trajectory for the loss function, but save
        # params and full history for the backward pass.
        return sol_traj, (params_traj, sol_traj)

    def f_bwd(res, v_traj):
        params_traj, sol_traj = res
        total_grad_p = jax.tree_util.tree_map(np.zeros_like, params_traj[0][3])

        sol_final = sol_traj[-1]
        iv_prev_final = params_traj[-1][0]
        param_final = params_traj[-1][3]
        iv_final = problem.update_int_vars_gp(sol_final, iv_prev_final, param_final)
        propagated_sensitivity_iv = jax.tree_util.tree_map(np.zeros_like, iv_final)
        propagated_sensitivity_u = jax.tree_util.tree_map(np.zeros_like, sol_final)

        for i in range(len(params_traj) - 1, -1, -1):
            sol_n = sol_traj[i]
            iv_prev = params_traj[i][0]
            param_n = params_traj[i][3]

            update_fn = lambda s, iv, p: problem.update_int_vars_gp(s, iv, p)
            _, vjp_update_fn = jax.vjp(update_fn, sol_n, iv_prev, param_n)
            sens_sol_C, sens_iv_prev_C, sens_param_C = vjp_update_fn(propagated_sensitivity_iv)
            sens_param_C = jax.tree_util.tree_map(lambda x: 0. * x, sens_param_C)

            adjoint_rhs_list = jax.tree_util.tree_map(
                lambda loss, prop_u, sen_C: loss + prop_u + sen_C,
                v_traj[i],
                propagated_sensitivity_u,
                sens_sol_C,
            )

            vjp_result = implicit_vjp(problem, [sol_n], params_traj[i], adjoint_rhs_list, adjoint_solver_options)

            # Accumulate parameter sensitivity from both the implicit residual
            # and the explicit internal-variable update.
            total_grad_p = jax.tree_util.tree_map(
                lambda total, gR, gC: total + gR - gC,
                total_grad_p,
                vjp_result[3],
                sens_param_C,
            )

            if i > 0:
                propagated_sensitivity_u = vjp_result[2]
                propagated_sensitivity_iv = jax.tree_util.tree_map(
                    lambda from_solve, from_update: from_solve + from_update,
                    vjp_result[0],
                    sens_iv_prev_C,
                )

        return (total_grad_p,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred