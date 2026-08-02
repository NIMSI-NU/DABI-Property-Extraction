"""
Modular Time Stepping Core - Pure functional implementation
that can be used both for regular forward simulations and within
differentiable loss functions.
"""

import jax
import jax.numpy as np
from matplotlib.pyplot import step
import numpy as onp
from typing import NamedTuple, Callable, List, Any


class TimeStepConfig(NamedTuple):
    """Configuration for time stepping algorithm."""
    total_time: float
    initial_dt: float
    min_dt: float
    max_dt: float
    time_tol: float = 1.e-3
    max_retries: int = 3
    increase_factor: float = 1.5
    decrease_factor: float = 0.25
    consec_success_for_aggressive_increase: int = 2
    time_markers: List[float] = None
    target_iters: int = 15
    marker_buffer: float = 1.0e-2
    warm_start_merge_small_steps: bool = False
    warm_start_small_dt_threshold: float = 0.0
    warm_start_merge_max_steps: int = 1


class TimeStepState(NamedTuple):
    """State during time stepping - all JAX traceable."""
    sol: Any  # Current solution
    int_vars: Any  # Internal variables
    current_time: float
    dt: float
    step_count: int
    consec_success: int
    sol_history: List[Any]  # Last 2 solutions for extrapolation
    dt_history: List[float]  # Last 2 time steps
    disable_warm_start: bool = False  # Set to True after any step failure


class TimeSteppingCore:
    """
    Pure functional time stepping core that can be used in both
    differentiable and non-differentiable contexts.
    """
    
    @staticmethod
    def prepare_time_markers(time_markers, total_time, time_tol):
        """Validate/sort marker times and always include total_time."""
        if time_markers is None:
            markers = []
        else:
            markers = sorted(float(t) for t in time_markers)

        valid = []
        for t in markers:
            if t <= time_tol:
                continue
            if t > total_time + time_tol:
                continue
            t = min(t, float(total_time))
            if not valid or abs(t - valid[-1]) > time_tol:
                valid.append(t)

        if not valid or abs(valid[-1] - total_time) > time_tol:
            valid.append(float(total_time))
        return valid

    @staticmethod
    def next_marker_gap(markers, current_time, time_tol):
        """Gap from current_time to the next required marker, if any."""
        for marker in markers:
            if marker > current_time + time_tol:
                return marker - current_time
        return None

    @staticmethod
    def snap_time(t, markers, total_time, time_tol):
        """Snap to marker/total_time when very close to avoid floating drift."""
        for marker in markers:
            if abs(t - marker) <= time_tol:
                return marker
        if abs(t - total_time) <= time_tol:
            return total_time
        return t

    @staticmethod
    def enforce_time_markers(dt_candidate, current_time, config, markers, step_counter):
        """Limit dt so the next step cannot jump over required marker times."""
        dt = float(dt_candidate)
        remaining_total = config.total_time - current_time
        if remaining_total <= config.time_tol:
            return 0.0, step_counter

        dt = min(dt, config.max_dt)
        dt = min(dt, remaining_total)
        
        next_gap = TimeSteppingCore.next_marker_gap(markers, current_time, config.time_tol)
        if next_gap is not None:
            if dt > next_gap:
                dt = next_gap
                step_counter += 1
                # The previous jax.debug.print here forced a GPU sync per step.
                # `step_counter` and `config.time_markers` are pure Python; use plain print.
                print(f"Approx step {step_counter} of {len(config.time_markers)} completed")
            elif next_gap - dt <= config.marker_buffer and next_gap - dt > config.time_tol:
                dt = next_gap
        return dt, step_counter

    @staticmethod
    def is_forced_boundary_step(dt_value, current_time, config, markers):
        """True when dt is exactly the remaining distance to marker/final time."""
        remaining_total = config.total_time - current_time
        if abs(dt_value - remaining_total) <= config.time_tol:
            return True
        next_gap = TimeSteppingCore.next_marker_gap(markers, current_time, config.time_tol)
        return next_gap is not None and abs(dt_value - next_gap) <= config.time_tol

    @staticmethod
    def tree_all_finite(tree):
        """Check if all leaves in a PyTree are finite."""
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return True
        checks = [np.all(np.isfinite(x)) for x in leaves]
        return bool(np.all(np.stack(checks)))

    @staticmethod
    def get_extrapolated_guess(state: TimeStepState, dt_trial: float):
        """Computes an initial guess for the next step using linear extrapolation."""
        if state.step_count <= 1:
            return state.sol
        
        u_n_minus_1 = state.sol_history[-1]
        u_n_minus_2 = state.sol_history[-2]
        dt_n_minus_1 = state.dt_history[-1]
        
        velocity = (u_n_minus_1 - u_n_minus_2) / (dt_n_minus_1 + 1e-16)
        return state.sol + velocity * dt_trial

    @staticmethod
    def is_time_marker(t, markers, time_tol):
        return any(abs(t - marker) <= time_tol for marker in markers)

    @staticmethod
    def get_warm_start_step_data(warm_start_data, current_time, config, markers):
        """Look up the previous epoch's accepted step that started at current_time."""
        if warm_start_data is None:
            return None, None

        start_times = warm_start_data.get('start_times', [])
        dt_schedule = warm_start_data.get('dt_schedule', [])
        end_times = warm_start_data.get('times', [])
        warm_solutions = warm_start_data.get('solutions', [])
        num_entries = min(len(start_times), len(dt_schedule), len(end_times), len(warm_solutions))
        if num_entries == 0:
            return None, None

        for idx in range(num_entries):
            if abs(start_times[idx] - current_time) <= config.time_tol:
                base_dt = float(dt_schedule[idx])
                if (
                    not config.warm_start_merge_small_steps
                    or config.warm_start_merge_max_steps <= 1
                    or config.warm_start_small_dt_threshold <= 0.0
                    or base_dt > config.warm_start_small_dt_threshold
                ):
                    return base_dt, warm_solutions[idx]

                merged_dt = base_dt
                final_idx = idx
                max_steps = min(num_entries - idx, config.warm_start_merge_max_steps)
                for offset in range(1, max_steps):
                    candidate_idx = idx + offset
                    prev_end_time = float(end_times[candidate_idx - 1])
                    candidate_dt = float(dt_schedule[candidate_idx])

                    if TimeSteppingCore.is_time_marker(prev_end_time, markers, config.time_tol):
                        break
                    if candidate_dt > config.warm_start_small_dt_threshold:
                        break
                    if merged_dt + candidate_dt > config.max_dt:
                        break

                    merged_dt += candidate_dt
                    final_idx = candidate_idx

                    if TimeSteppingCore.is_time_marker(float(end_times[final_idx]), markers, config.time_tol):
                        break

                return merged_dt, warm_solutions[final_idx]

        return None, None

    @staticmethod
    def get_nearest_warm_solution(warm_start_data, next_time):
        """Fallback warm solution lookup by nearest accepted time."""
        if warm_start_data is None:
            return None

        warm_times = warm_start_data.get('times', [])
        warm_solutions = warm_start_data.get('solutions', [])
        if len(warm_times) != len(warm_solutions) or len(warm_times) == 0:
            return None

        nearest_idx = min(
            range(len(warm_times)),
            key=lambda idx: abs(warm_times[idx] - next_time),
        )
        return warm_solutions[nearest_idx]

    @staticmethod
    def update_dt_after_success(dt_used, dt_trial, state, config, time_tol, num_iters=None):
        """Update time step after successful convergence."""
        dt_for_update = float(dt_used)
        
        marker_clipped = dt_used + time_tol < dt_trial
        if marker_clipped:
            return float(onp.clip(dt_trial, config.min_dt, config.max_dt)), 0
        
        if num_iters is not None and num_iters > 0:
            if num_iters <= 6:
                new_consec_success = state.consec_success + 1
            else:
                new_consec_success = 0
        else:
            new_consec_success = state.consec_success + 1
        
        new_dt = dt_for_update
        if new_consec_success >= config.consec_success_for_aggressive_increase:
            new_dt = min(config.max_dt, new_dt * config.increase_factor)
        
        return float(onp.clip(new_dt, config.min_dt, config.max_dt)), new_consec_success

    @staticmethod
    def adaptive_time_stepping_loop(
        initial_state: TimeStepState,
        config: TimeStepConfig,
        solver_func: Callable,
        update_internal_vars_func: Callable,
        global_params: Any,
        collect_trajectory: bool = False,
        warm_start_data: Any = None,
    ):
        """Pure functional adaptive time stepping loop."""
        step_counter = 0
        markers = TimeSteppingCore.prepare_time_markers(
            config.time_markers, config.total_time, config.time_tol
        )
        
        state = initial_state
        sol_traj, sol_times, int_vars_traj = [[] for i in range(3)]
        
        while state.current_time < config.total_time and not np.isclose(
            state.current_time, config.total_time
        ):
            state = state._replace(step_count=state.step_count + 1)
            retries = 0
            step_accepted = False
            dt_trial = state.dt
            scheduled_warm_guess = None
            
            if warm_start_data is not None and not state.disable_warm_start:
                preferred_dt, scheduled_warm_guess = TimeSteppingCore.get_warm_start_step_data(
                    warm_start_data, state.current_time, config, markers
                )
                if preferred_dt is not None:
                    dt_trial = float(preferred_dt)
            dt_this_step, step_counter = TimeSteppingCore.enforce_time_markers(
                dt_trial, state.current_time, config, markers, step_counter
            )
            
            while (not step_accepted) and (retries < config.max_retries):
                if retries > 0:
                    dt_trial *= config.decrease_factor
                    dt_this_step, step_counter = TimeSteppingCore.enforce_time_markers(
                        dt_trial, state.current_time, config, markers, step_counter
                    )
                    state = state._replace(disable_warm_start=True)
                    initial_guess = TimeSteppingCore.get_extrapolated_guess(state, dt_this_step)
                
                if dt_this_step < config.min_dt and not TimeSteppingCore.is_forced_boundary_step(
                    dt_this_step, state.current_time, config, markers
                ):
                    break
                
                next_time = TimeSteppingCore.snap_time(
                    state.current_time + dt_this_step, markers, config.total_time, config.time_tol
                )
                scale = next_time / config.total_time

                warm_guess = None
                if retries == 0 and warm_start_data is not None and not state.disable_warm_start:
                    warm_guess = scheduled_warm_guess
                    if warm_guess is None:
                        warm_guess = TimeSteppingCore.get_nearest_warm_solution(
                            warm_start_data, next_time
                        )

                initial_guess = warm_guess
                if initial_guess is None:
                    initial_guess = TimeSteppingCore.get_extrapolated_guess(state, dt_this_step)
                
                # --- CRITICAL MODIFICATION --- 
                # Passing physical time context (dt_this_step and next_time) at the end 
                # so specific wrappers can calculate non-linear loading conditions.
                step_params = [state.int_vars, scale, state.sol, global_params, dt_this_step, next_time]
                
                try:
                    solver_result = solver_func(step_params, initial_guess)
                    
                    if isinstance(solver_result, tuple):
                        sol_step, num_iters = solver_result
                    else:
                        sol_step = solver_result
                        num_iters = None
                    
                    if not TimeSteppingCore.tree_all_finite(sol_step):
                        raise FloatingPointError("Non-finite solution.")
                    
                    next_int_vars = update_internal_vars_func(sol_step, state.int_vars, dt_this_step)
                    
                    if not TimeSteppingCore.tree_all_finite(next_int_vars):
                        raise FloatingPointError("Non-finite internal vars.")
                    
                    step_accepted = True
                    
                except (RuntimeError, FloatingPointError):
                    retries += 1
            
            if not step_accepted:
                break
            
            accepted_time = TimeSteppingCore.snap_time(
                state.current_time + dt_this_step, markers, config.total_time, config.time_tol
            )
            
            new_sol_history = state.sol_history[1:] + [state.sol]
            new_dt_history = state.dt_history[1:] + [dt_this_step]
            
            new_dt, new_consec_success = TimeSteppingCore.update_dt_after_success(
                dt_this_step, dt_trial, state, config, config.time_tol, num_iters
            )
            
            state = TimeStepState(
                sol=sol_step,
                int_vars=next_int_vars,
                current_time=accepted_time,
                dt=new_dt,
                step_count=state.step_count,
                consec_success=new_consec_success,
                sol_history=new_sol_history,
                dt_history=new_dt_history,
                disable_warm_start=state.disable_warm_start,
            )
            
            if collect_trajectory:
                sol_traj.append(sol_step)
                sol_times.append(accepted_time)
                int_vars_traj.append(next_int_vars)
        
        if collect_trajectory:
            return sol_traj, sol_times, int_vars_traj, state
        else:
            return state.sol, state.int_vars, state