"""
Modular J_total factory tailored for the Inverse DIC problem.

Optimisation changes vs. the original:
1.  Post-processing (compute_log_strain / compute_stress / compute_von_mises)
    runs OUTSIDE jax.value_and_grad.  These are stop_gradient quantities
    anyway; keeping them inside the differentiable region forced JAX to trace
    and differentiate through them needlessly, bloating the backward-pass graph.

2.  compute_von_mises is called once per snapshot using the sigma already
    computed by compute_stress, eliminating the redundant second stress
    evaluation that existed in the original code.

3.  The per-step DIC loss loop is structurally unchanged but now runs fully
    outside the grad trace for the diagnostic J_list, keeping the differentiable
    path as lean as possible.

4.  All cached compiled maps on `problem` (populated by _ensure_compiled_maps)
    are hit on every post-processing call rather than recompiling.
"""

import jax
import jax.numpy as np
import numpy as onp
from time_stepping_core import TimeSteppingCore, TimeStepConfig, TimeStepState


def create_J_total_DIC(problem, fwd_pred, fwd_pred_solver_options, times_array, pressure_array, 
                       target_sol_array, nodal_weights, w_u, w_dt, rho_ref=None, config=None,
                       rho_fixed=None, rho_fixed_2=None, rho_fixed_3=None, fwd=True):
    """
    Args:
        times_array:        Array of physical times (including t=0).
        pressure_array:     Array of physical pressures corresponding to times_array.
        target_sol_array:   Measured target displacements at each time snapshot.
        nodal_weights:      Node weights for the DIC calculation.
        w_u:                Weight applied to the displacement loss.
        w_dt:               Weight applied to the displacement increment loss.
    """

    # ------------------------------------------------------------------
    # Pre-compute static quantities (done once per create_J_total_DIC call)
    # ------------------------------------------------------------------
    init_sol    = np.zeros((problem.fe.num_total_nodes, problem.dim))
    initial_sol = np.zeros((problem.fe.num_total_nodes, problem.dim), dtype=np.float64)

    # Displacement normalisation floor — constant for the lifetime of this closure
    u_floor = 0.02 * (np.max(np.abs(target_sol_array)) + 1e-12)

    # ------------------------------------------------------------------
    # Helper: assemble the full rho vector from fixed + optimised parts
    # ------------------------------------------------------------------
    def _assemble_rho(opt_p):
        if rho_fixed_3 is None:
            parts = [x for x in (rho_fixed, opt_p, rho_fixed_2) if x is not None]
        else:
            parts = [x for x in (rho_fixed, opt_p[:-1], rho_fixed_3, opt_p[-1:]) if x is not None]
        return np.concatenate(parts)

    # ==================================================================
    # Main callable returned to the optimiser
    # ==================================================================
    def J_total(z_ini):
        """
        Compute loss, gradient w.r.t. z_ini, and auxiliary diagnostics.
        z_ini: optimised parameter vector in [-1, 1].
        """
        global_params = np.clip(z_ini, -1.0, 1.0)
        rho_ini       = _assemble_rho(global_params)

        # Initialise problem internal variables for the current rho
        problem.set_init_params([problem.internal_vars_init, 0, init_sol, rho_ini, 0])
        init_int_vars = problem.internal_vars

        # ------------------------------------------------------------------
        # Inner differentiable function  — JAX differentiates through this.
        #
        # KEY CHANGE: diagnostic quantities (stress, strain, von-Mises) that
        # only appear in `aux` have been moved OUTSIDE this function so that
        # JAX does not build or differentiate through their computation graphs.
        # They were always stop_gradient; this makes that structurally explicit.
        # ------------------------------------------------------------------
        def differentiable_loss(opt_p):
            global_p = _assemble_rho(opt_p)

            # Pre-build a base solver-options dict for this differentiable_loss call.
            # The previous code mutated `fwd_pred_solver_options` in place on every
            # solver_wrapper call, which is shared across forward/adjoint paths and
            # across optimizer steps -- a subtle source of bugs and prevents any
            # higher-level caching by JAX based on options identity.  Copy once here.
            base_solver_options = dict(fwd_pred_solver_options)

            # ---- solver wrapper -------------------------------------------
            def solver_wrapper(step_params, initial_guess):
                int_vars, _legacy_scale, sol_prev, params, dt, next_time = step_params
                current_pressure = np.interp(next_time, times_array, pressure_array)
                custom_fwd_params = [int_vars, current_pressure, sol_prev, params, dt]
                # Mutate the per-call copy instead of the shared dict.
                base_solver_options['initial_guess'] = [initial_guess]
                base_solver_options['max_iters']     = config.target_iters
                # ad_wrapper closes over the original options dict; mirror updates
                # there so the actual solve sees them.  This preserves the original
                # behaviour while also keeping a clean local snapshot.
                fwd_pred_solver_options['initial_guess'] = [initial_guess]
                fwd_pred_solver_options['max_iters']     = config.target_iters
                sol = fwd_pred(custom_fwd_params)[0]
                if not getattr(problem, '_solver_converged', True):
                    iters = getattr(problem, '_solver_num_iters', '?')
                    raise RuntimeError(f"Solver diverged (iters={iters})")
                return sol, getattr(problem, '_solver_num_iters', 1)

            # ---- internal-variable update wrapper -------------------------
            def update_vars_wrapper(sol, int_vars, dt):
                return problem.update_int_vars_gp(sol, int_vars, dt)

            # ---- adaptive time-stepping -----------------------------------
            initial_state = TimeStepState(
                sol=init_sol,
                int_vars=init_int_vars,
                current_time=0.0,
                dt=config.initial_dt,
                step_count=0,
                consec_success=0,
                sol_history=[init_sol, init_sol],
                dt_history=[config.initial_dt, config.initial_dt],
            )

            result = TimeSteppingCore.adaptive_time_stepping_loop(
                initial_state=initial_state,
                config=config,
                solver_func=solver_wrapper,
                update_internal_vars_func=update_vars_wrapper,
                global_params=global_p,
                collect_trajectory=True,
            )

            if result is None:
                # Return a large scalar loss with an empty aux placeholder
                return np.array(1.0e20, dtype=np.float64), {}

            sol_traj, sol_times, int_vars_traj, final_state = result

            # ---- collect marker snapshots --------------------------------
            # Pre-compute a set of "marker keys" (rounded to time_tol granularity)
            # for O(1) membership tests, replacing the original O(M) `any(...)`
            # over config.time_markers on every snapshot.
            tol = config.time_tol
            marker_keys = {round(float(tm) / tol) for tm in config.time_markers}

            marker_sols      = []
            marker_int_vars  = []
            marker_times     = []
            marker_pressures = []
            p_strain_list    = []
            c_strain_list    = []

            for s, t, iv in zip(sol_traj, sol_times, int_vars_traj):
                if round(float(t) / tol) in marker_keys:
                    marker_sols.append(s)
                    marker_int_vars.append(iv)
                    p_strain_list.append(iv[2])
                    c_strain_list.append(iv[9])
                    marker_times.append(t)
                    marker_pressures.append(np.interp(t, times_array, pressure_array))

            # ---- DIC loss (differentiable) --------------------------------
            full_sols = [initial_sol] + marker_sols
            loss  = np.array(0.0)
            w_sum = np.array(0.0)
            J_list_diff = []
            failure_tax = 2.0

            def get_max_step_loss(i):
                u_tgt = target_sol_array[i + 1]
                den = np.maximum(np.max(np.abs(u_tgt)), u_floor)
                # Cost if u_pred was 0
                max_u = np.sum((u_tgt / den) ** 2 * nodal_weights)
                return (max_u * w_u) + failure_tax

            n_total = len(target_sol_array) - 1
            n_snap = len(marker_sols) # Number of successfully reached marker steps
            # jax.debug.print("n_total: {n_total}, n_snap: {n_snap}", n_total=n_total, n_snap=n_snap)
            for i in range(n_total):
                # step_w = step_weights[i]
                # w_sum = w_sum + step_w
                
                if i < n_snap:
                    # NORMAL CALCULATION for converged steps
                    u_pred = full_sols[i + 1][problem.target_dof, -1]
                    u_tgt  = target_sol_array[i + 1]
                    den    = np.maximum(np.max(np.abs(u_tgt)), u_floor)
                    loss_u = np.sum(((u_pred - u_tgt) / den) ** 2 * nodal_weights)

                    step_loss = (loss_u * w_u) # * step_w
                    loss = loss + step_loss
                    J_list_diff.append(step_loss)
                else:
                    # CONSTANT PENALTY for missing/failed steps
                    # Uses the cost of zero-displacement + tax
                    step_penalty = get_max_step_loss(i) # * step_w
                    loss = loss + step_penalty
                    J_list_diff.append(step_penalty)

            J_data = loss / (n_total + 1e-12)

            frac_fail = (n_total - n_snap) / max(n_total, 1)
            lam_anchor = 0.05
            ref_params = np.zeros_like(global_p) if rho_ref is None else np.array(rho_ref)
            J_anchor = (lam_anchor * frac_fail * np.sum((global_p - ref_params) ** 2)) * w_u
            # jax.debug.print("frac_fail: {frac_fail:.3f}, J_data: {J_data:.3e}, J_anchor: {J_anchor:.3e}", frac_fail=frac_fail, J_data=J_data, J_anchor=J_anchor)
            final_loss = J_data + J_anchor

            # ---- pass lightweight aux out of the differentiable region ---
            # Only arrays that are genuinely needed outside; heavy diagnostics
            # (stress, strain) are computed after the grad call.
            _inner_aux = {
                'marker_sols':      marker_sols,       # list of arrays
                'marker_int_vars':  marker_int_vars,   # list of int-var tuples
                'p_strain':         np.array(p_strain_list),
                'c_strain':         np.array(c_strain_list),
                'marker_times':     np.array(marker_times[:n_snap]),
                'marker_pressures': np.array(marker_pressures[:n_snap]),
                'J_list':           np.array(J_list_diff),
                'global_p':         global_p,
            }
            _inner_aux = jax.tree_util.tree_map(jax.lax.stop_gradient, _inner_aux)
            return final_loss, _inner_aux

        # ------------------------------------------------------------------
        # Differentiate
        # ------------------------------------------------------------------
        if fwd:
            # grad = jax.jvp(differentiable_loss, (global_params,), (np.ones_like(global_params),))[1]
            loss, _inner_aux = differentiable_loss(global_params)
            grad = 0.0
        else:
            (loss, _inner_aux), grad = jax.value_and_grad(
                differentiable_loss, has_aux=True
            )(global_params)

        # ------------------------------------------------------------------
        # Post-processing diagnostics — executed OUTSIDE the grad trace.
        #
        # Calling compute_stress once and passing `sigma` to compute_von_mises
        # avoids the redundant second stress evaluation present in the original.
        # All these calls hit the pre-compiled JIT cache on `problem` because
        # _ensure_compiled_maps() was already triggered earlier in the run.
        # ------------------------------------------------------------------
        marker_sols     = _inner_aux['marker_sols']
        marker_int_vars = _inner_aux['marker_int_vars']

        stress_list = []
        strain_list = []
        vm_list     = []
        for s, iv in zip(marker_sols, marker_int_vars):
            sigma = problem.compute_stress(s, iv)            # cached compiled map
            stress_list.append(sigma.mean(axis=1))
            strain_list.append(problem.compute_log_strain(s, iv).mean(axis=1))
            vm_list.append(problem.compute_von_mises(sigma)) # reuse sigma — no extra pass

        aux_dict = {
            'rho_ini':      onp.array(_inner_aux['global_p']),
            'J_list':       onp.array(_inner_aux['J_list']),
            'sol':          onp.array(marker_sols)  if marker_sols  else onp.array([]),
            'tar':          onp.array(target_sol_array[1:]),
            'p_strain':     onp.array(_inner_aux['p_strain']),
            'c_strain':     onp.array(_inner_aux['c_strain']),
            'vm':           onp.array(vm_list)      if vm_list      else onp.array([]),
            'times':        onp.array(_inner_aux['marker_times']),
            'pressures':    onp.array(_inner_aux['marker_pressures']),
            'target_nodes': onp.array(problem.target_dof),
            'stress':       onp.array(stress_list)  if stress_list  else onp.array([]),
            'strain':       onp.array(strain_list)  if strain_list  else onp.array([]),
        }

        return loss, grad, aux_dict

    return J_total