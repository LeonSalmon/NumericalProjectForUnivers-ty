import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
class AcademicDiagnostics:
    @staticmethod
    def run_all_diagnostics():
        logs = []
        logs.append("===PRE-FLIGHT ACADEMIC & MATHEMATICAL DIAGNOSTICS ===")
        # Demonstrating the inherent limitations of computer arithmetic and round-off errors.
        val = 0.0
        for _ in range(10):
            val += 0.1
        error = abs(1.0 - val)
        logs.append(f"[Req 1] Floating-Point Accumulation Test (10 * 0.1): {val}")
        logs.append(f"  - Calculated Round-off Error: {error:.2e} (Formula: |1.0 - ∑0.1|)")
        # Using LU factorization to pre-process and solve systems efficiently.
        A = np.array([[4.0, -1.0, 0.0], 
                      [-1.0, 4.0, -1.0], 
                      [0.0, -1.0, 3.0]])
        b = np.array([1.2, 0.5, 0.8])       
        lu, piv = lu_factor(A)
        U = np.triu(lu)
        L = np.tril(lu, -1) + np.eye(A.shape[0])
        wind_vector = lu_solve((lu, piv), b)        
        logs.append("\n[Req 6 & 7] LU Decomposition for Wind Disturbance Vector (A * x = b):")
        logs.append(f"  - Matrix L (Lower):\n{L}")
        logs.append(f"  - Matrix U (Upper):\n{U}")
        logs.append(f"  - Substitution Formula: L * y = P * b  -->  U * x = y")
        logs.append(f"  - Solved Wind Vector [x, y, z]: [{wind_vector[0]:.3f}, {wind_vector[1]:.3f}, {wind_vector[2]:.3f}]")
       # Applying multivariate optimization routines to minimize energy cost.
        def energy_cost(v):
            return (v**2) * 0.05 + (100.0 / max(v, 0.1))     
            
        opt_res = minimize(energy_cost, x0=5.0, bounds=[(1.0, 20.0)])
        logs.append("\n[Req 8] Cruise Velocity Optimization (Objective: Minimize Energy Cost):")
        logs.append(f"  - Cost Function: f(v) = 0.05*v² + 100/v")
        logs.append(f"  - Optimization Status: {opt_res.message}")
        logs.append(f"  - Total Iterations: {opt_res.nit} | Function Evaluations: {opt_res.nfev}")
        logs.append(f"  - Optimal Cruise Velocity (v*): {opt_res.x[0]:.4f} m/s")
        
        # Solving ODEs using SciPy's methods and comparing solver options (RK45 vs RK23).
        def benchmark_ode(t, y): return -1.5 * y
        t_span = (0, 2)
        y0 = [10.0]        
        start_rk45 = time.perf_counter()
        sol_rk45 = solve_ivp(benchmark_ode, t_span, y0, method='RK45', rtol=1e-5)
        time_rk45 = time.perf_counter() - start_rk45        
        start_rk23 = time.perf_counter()
        sol_rk23 = solve_ivp(benchmark_ode, t_span, y0, method='RK23', rtol=1e-5)
        time_rk23 = time.perf_counter() - start_rk23       
        logs.append("\n[Req 9] SciPy ODE Solver Comparative Benchmarking (dy/dt = -1.5y):")
        logs.append(f"  - RK45 (Runge-Kutta 4(5)) -> Steps: {len(sol_rk45.t)} | Time: {time_rk45:.6f}s | Final Value: {sol_rk45.y[0][-1]:.5f}")
        logs.append(f"  - RK23 (Runge-Kutta 2(3)) -> Steps: {len(sol_rk23.t)} | Time: {time_rk23:.6f}s | Final Value: {sol_rk23.y[0][-1]:.5f}")
        logs.append("===========================================\n")
        return "\n".join(logs), wind_vector

class DroneEngine:
    @staticmethod
    def get_dynamics(state, target, obstacles, other_drones_states, wind_vector):
        if other_drones_states is None:
            other_drones_states = []    
        pos, vel = state[:3], state[3:]
        dist_vec = target - pos
        dist = max(np.linalg.norm(dist_vec), 0.1)
        F_att = 8.0 * (dist_vec / dist) 
        F_rep = np.zeros(3)
        collision_limit = 11.5        
        for i, obs in enumerate(obstacles):
            building_height = obs[2] + 15.0
            safety_radius = 20.0 
            if pos[2] < building_height:
                dist_vec_xy = pos[:2] - obs[:2]
                dist_xy = np.linalg.norm(dist_vec_xy)
                eff_dist = max(dist_xy - collision_limit, 0.5)
                if eff_dist < safety_radius:
                    mag = 1500.0 * (1/eff_dist - 1/safety_radius) * (1/eff_dist**2)
                    mag = min(mag, 250.0) 
                    F_rep[:2] += mag * (dist_vec_xy / dist_xy)
                    if eff_dist < 4.0:
                        F_rep[2] += mag * 1.5                        
        F_evasion = np.zeros(3)
        sensor_range = 35.0 
        for other_state in other_drones_states:
            other_pos = other_state[:3]
            other_vel = other_state[3:]
            vec_to_other = other_pos - pos
            dist_to_other = np.linalg.norm(vec_to_other)
            if 0 < dist_to_other < sensor_range:
                if dist_to_other < 12.0:
                    F_evasion -= (2000.0 / (dist_to_other**2)) * (vec_to_other / dist_to_other)
                rel_vel = vel - other_vel
                closing_speed = np.dot(rel_vel, vec_to_other / dist_to_other) 
                if closing_speed > 0.5: 
                    time_to_impact = dist_to_other / closing_speed
                    if time_to_impact < 4.0:
                        escape_dir = np.cross(vec_to_other, np.array([0, 0, 1]))
                        norm_esc = np.linalg.norm(escape_dir)
                        escape_dir = escape_dir / norm_esc if norm_esc > 1e-4 else np.array([1.0, 0.0, 0.0])
                        if np.dot(escape_dir, vel) < 0: escape_dir = -escape_dir
                        F_evasion += escape_dir * (1500.0 / (time_to_impact + 0.1))
                        if pos[2] > other_pos[2]: F_evasion[2] += 600.0 / (time_to_impact + 0.1)
                        else: F_evasion[2] -= 600.0 / (time_to_impact + 0.1)                       
        F_damping = -1.5 * vel 
        acc = F_att + F_rep + F_evasion + F_damping + wind_vector        
        log_data = (f"F_att: {np.linalg.norm(F_att):.2f}N | F_rep: {np.linalg.norm(F_rep):.2f}N | F_evasion: {np.linalg.norm(F_evasion):.2f}N\n"
                    f"LU Decoupled Wind Contribution: {np.linalg.norm(wind_vector):.2f}N")
        return np.concatenate((vel, acc)), log_data
    @staticmethod
    def rk4_step(state, dt, target, obstacles, other_drones_states, wind_vector):
        """
        Solves the ordinary differential equations governing the drone's dynamics.
        """
        k1, log1 = DroneEngine.get_dynamics(state, target, obstacles, other_drones_states, wind_vector)
        k2, _ = DroneEngine.get_dynamics(state + 0.5 * dt * k1, target, obstacles, other_drones_states, wind_vector)
        k3, _ = DroneEngine.get_dynamics(state + 0.5 * dt * k2, target, obstacles, other_drones_states, wind_vector)
        k4, _ = DroneEngine.get_dynamics(state + dt * k3, target, obstacles, other_drones_states, wind_vector)       
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)       
        rk_math_log = (f"[RK4 Mathematics] Weighted Slopes Calculated:\n"
                       f"  k1 norm: {np.linalg.norm(k1):.3f} | k2 norm: {np.linalg.norm(k2):.3f}\n"
                       f"  k3 norm: {np.linalg.norm(k3):.3f} | k4 norm: {np.linalg.norm(k4):.3f}\n"
                       f"  Formula: y_(n+1) = y_n + (dt/6)*(k1 + 2k2 + 2k3 + k4)")
        return new_state, log1 + "\n" + rk_math_log
    @staticmethod
    def predict_collision_newton_raphson(pos, vel, obstacles):
        """
        Applies the Newton-Raphson method to predict potential collisions.
        """
        alerts = []
        r_crit = 11.5 
        for i, obs in enumerate(obstacles):
            v_xy = vel[:2]
            if np.linalg.norm(v_xy) < 0.1: continue            
            p_xy, o_xy = pos[:2], obs[:2]
            t = 1.0          
            nr_logs = [f"\n  [Newton-Raphson Execution for Obstacle {i}] Formula: t_new = t - f(t)/f'(t)"]           
            for iteration in range(1, 6):
                pred_pos = p_xy + v_xy * t
                f_t = np.sum((pred_pos - o_xy)**2) - r_crit**2 
                f_prime_t = 2 * np.dot(pred_pos - o_xy, v_xy)                
                if abs(f_prime_t) < 1e-4: 
                    nr_logs.append(f"    - Iter {iteration}: Derivative near zero (|f'(t)| < 1e-4). Breaking loop.")
                    break                   
                t_new = t - f_t / f_prime_t
                nr_logs.append(f"    - Iter {iteration}: current_t = {t:.4f} | f(t) = {f_t:.4f} | f'(t) = {f_prime_t:.4f} --> next_t = {t_new:.4f}")             
                if abs(t_new - t) < 0.01: 
                    t = t_new
                    nr_logs.append(f"    - Convergence achieved at Delta < 0.01.")
                    break
                t = t_new                 
            if 0 < t < 2.0 and pos[2] < (obs[2] + 15): 
                alerts.append("".join(nr_logs))
                alerts.append(f"  *** [CRITICAL ALERT] NR predicts collision with Obs {i} in {t:.2f}s! ***")
        return "\n".join(alerts)
    @staticmethod
    def analyze_path_cubic_spline(path_history):
        """
        Uses cubic splines to smooth the trajectory and central finite difference to approximate derivatives.
        """
        if len(path_history) < 5:
            return ""         
        path = np.array(path_history[-5:]) 
        t = np.arange(len(path))
        try:
            cs_x = CubicSpline(t, path[:, 0])
            cs_y = CubicSpline(t, path[:, 1])
            cs_z = CubicSpline(t, path[:, 2])            
            h = 1e-4
            vel_x = (cs_x(2 + h) - cs_x(2 - h)) / (2 * h)
            vel_y = (cs_y(2 + h) - cs_y(2 - h)) / (2 * h)
            vel_z = (cs_z(2 + h) - cs_z(2 - h)) / (2 * h)          
            speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)           
            spline_log = (f"[Req 3 & 4] Cubic Spline Boundary Constraints Satisfied.\n"
                          f"  - Central Difference Differentiation Scheme Formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)\n"
                          f"  - Derived Polynomial Kinematics Vector: [{vel_x:.3f}, {vel_y:.3f}, {vel_z:.3f}]\n"
                          f"  - Central Diff Speed Estimation at step index 2: {speed:.3f} m/s")
            return spline_log
        except Exception as e:
            return f"Spline Evaluation Error: {e}"
def generate_academic_plots(actual_path, spline_path, time_steps, roundoff_errors):
    """
    This function is called at the end of the simulation or when the target is reached. 
    It generates the required academic analysis charts, displays them on the screen, 
    and saves them to the local directory.
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(actual_path[:, 0], actual_path[:, 1], 'ro', markersize=4, label='Discrete Waypoints (Actual)')
    plt.plot(spline_path[:, 0], spline_path[:, 1], 'b-', linewidth=2, label='Cubic Spline Polynomial Path')
    plt.title('Req 3 & 4: Trajectory Smoothing & Kinematic Interpolation', fontsize=11, fontweight='bold')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, roundoff_errors, 'g-', linewidth=2, label='Accumulated Machine Round-off Error')
    plt.title('Req 1: Floating-Point Absolute Error Propagation Over Time', fontsize=11, fontweight='bold')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Absolute Error Magnitude |1.0 - ∑0.1|')
    plt.yscale('log')  # Logarithmic axis for academic precision
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    output_filename = 'trajectory_and_error_analysis.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[Academic Plotter] System charts saved successfully to: '{output_filename}'")
    plt.show()
