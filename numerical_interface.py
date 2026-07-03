import matplotlib
matplotlib.use('TkAgg') 
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numerical import DroneEngine, AcademicDiagnostics, generate_academic_plots
from scipy.interpolate import CubicSpline

class DroneApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1600x900")
        self.is_running = False
        self.camera_mode = False 
        self.show_trajectory = True       
        self.drones = [] 
        self.colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink']         
        self.grid_columnconfigure(1, weight=1)       
        
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)     
        
        ctk.CTkLabel(self.sidebar, text="SIMULATION CONTROL", font=("Arial", 16, "bold")).pack(pady=10)       
        ctk.CTkButton(self.sidebar, text="Scenario 1: Standard (5 Buildings)", command=lambda: self.set_scenario(5)).pack(pady=5, padx=10)
        ctk.CTkButton(self.sidebar, text="Scenario 2: Hard (15 Buildings)", command=lambda: self.set_scenario(15)).pack(pady=5, padx=10)      
        
        ctk.CTkLabel(self.sidebar, text="SWARM CONTROL", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text="Add New Drone", command=self.add_random_drone, fg_color="#3498db").pack(pady=5, padx=10)        
        ctk.CTkButton(self.sidebar, text="START / STOP", command=self.toggle, fg_color="#2ecc71").pack(pady=20, padx=10)
        ctk.CTkButton(self.sidebar, text="Toggle Chasing Camera", command=self.toggle_camera).pack(pady=5, padx=10)
        ctk.CTkButton(self.sidebar, text="Toggle Trajectory", command=self.toggle_trajectory).pack(pady=5, padx=10)       
        
        # Console for logging mathematical operations and numerical alerts
        ctk.CTkLabel(self.sidebar, text="MATHEMATICAL OPERATION LOG", font=("Arial", 12, "bold")).pack()
        self.log_console = ctk.CTkTextbox(self.sidebar, height=400, fg_color="#000000", text_color="#00FF41", font=("Consolas", 11))
        self.log_console.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.fig = plt.figure(figsize=(12, 9), facecolor='#111111')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")        
        
        # Execute Pre-Flight Academic Diagnostics (Req 1, 6, 7, 8, 9)
        diag_logs, self.wind_vector = AcademicDiagnostics.run_all_diagnostics()
        self.log_console.insert("0.0", diag_logs + "\n")        
        
        self.set_scenario(5)

    def create_drone_plots(self, color):
        """Initializes the 3D plot objects for a new drone instance."""
        path_line, = self.ax.plot([], [], [], color=color, lw=3, zorder=10)
        drone_body, = self.ax.plot([], [], [], marker='o', markersize=15, color='#aaaaaa', zorder=15)
        drone_arms = [self.ax.plot([], [], [], color='#777777', lw=4, zorder=15)[0] for _ in range(6)]
        drone_props = [self.ax.plot([], [], [], color='red', lw=3, zorder=15)[0] for _ in range(6)]
        drone_legs = [self.ax.plot([], [], [], color='silver', lw=5, zorder=15)[0] for _ in range(4)]
        
        path_line.set_visible(self.show_trajectory)        
        return {
            'path_line': path_line, 'drone_body': drone_body, 
            'drone_arms': drone_arms, 'drone_props': drone_props, 'drone_legs': drone_legs
        }

    def add_random_drone(self):
        """Instantiates a new drone with randomized start and target coordinates."""
        start_x, start_y, start_z = np.random.uniform(-10, 20), np.random.uniform(-10, 20), np.random.uniform(5, 20)
        target_x, target_y, target_z = np.random.uniform(70, 110), np.random.uniform(70, 110), np.random.uniform(60, 90)
        color = self.colors[len(self.drones) % len(self.colors)]      
        
        # Plot target marker
        self.ax.scatter(target_x, target_y, target_z, color=color, s=150, marker='*')
        plots = self.create_drone_plots(color)        
        
        # Drone state array structure: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        new_drone = {
            'id': len(self.drones) + 1,
            'state': np.array([start_x, start_y, start_z, 0.0, 0.0, 0.0]),
            'target': np.array([target_x, target_y, target_z]),
            'path_history': [], 'color': color, 'plots': plots
        }      
        
        self.drones.append(new_drone)
        self.canvas.draw_idle()

    def set_scenario(self, n_obs):
        """Generates random obstacle matrices representing buildings."""
        self.is_running = False
        self.obstacles = np.random.rand(n_obs, 3)
        self.obstacles[:, 0] = self.obstacles[:, 0] * 60 + 20 
        self.obstacles[:, 1] = self.obstacles[:, 1] * 60 + 20 
        self.obstacles[:, 2] = self.obstacles[:, 2] * 40 + 30        
        self.drones = [] 
        
        self.ax.clear()
        self.ax.set_facecolor('#111111') 
        self.ax.set_xlim([-20, 120])
        self.ax.set_ylim([-20, 120])
        self.ax.set_zlim([0, 120])        
        
        # Draw ground plane
        self.ax.plot_surface(np.array([[-20, 120], [-20, 120]]), np.array([[-20, -20], [120, 120]]), np.array([[0, 0], [0, 0]]), color='#0a0a0a', alpha=0.5)
        
        # Render obstacles
        for obs in self.obstacles: 
            building_height = obs[2] + 15 
            self.ax.bar3d(obs[0]-4, obs[1]-4, 0, 8, 8, building_height, color='#555555', edgecolor='white', alpha=0.3, shade=True)     
            
        self.ax.view_init(elev=30, azim=45) 
        self.add_random_drone()
        self.canvas.draw()

    def toggle_trajectory(self):
        """Toggles the visibility of the drone path interpolation line."""
        self.show_trajectory = not self.show_trajectory
        for drone in self.drones: 
            drone['plots']['path_line'].set_visible(self.show_trajectory) 
        self.canvas.draw_idle()

    def toggle_camera(self):
        """Toggles dynamic chasing camera mode."""
        self.camera_mode = not self.camera_mode

    def toggle(self):
        """Starts or pauses the numerical simulation engine."""
        self.is_running = not self.is_running
        if self.is_running: self.update_sim()

    def draw_hexacopter(self, pos, frame_count, plots):
        """Calculates and updates the geometric coordinates for drone rendering."""
        x, y, z = pos[0], pos[1], pos[2]
        body_coords = [(x+2, y, z), (x, y+2, z), (x-2, y, z), (x, y-2, z), (x+2, y, z)]
        plots['drone_body'].set_data_3d([c[0] for c in body_coords], [c[1] for c in body_coords], [c[2] for c in body_coords])     
        rotation = frame_count * 0.8
        
        for i in range(6):
            angle = i * (np.pi / 3)
            ex, ey = x + 8.0 * np.cos(angle), y + 8.0 * np.sin(angle)
            plots['drone_arms'][i].set_data_3d([x, ex], [y, ey], [z, z])        
            px = np.array([ex - 1.2*np.sin(angle + rotation), ex + 1.2*np.sin(angle + rotation)])
            py = np.array([ey + 1.2*np.cos(angle + rotation), ey - 1.2*np.cos(angle + rotation)])
            plots['drone_props'][i].set_data_3d(px, py, [z+0.5, z+0.5])
            
        leg_positions = [(x+1.5, y+1.5), (x+1.5, y-1.5), (x-1.5, y+1.5), (x-1.5, y-1.5)]
        for i, (lx, ly) in enumerate(leg_positions):
            plots['drone_legs'][i].set_data_3d([lx, lx], [ly, ly], [z, z-3.5])

    def update_sim(self):
        """Main simulation loop: Computes numerical step, handles plotting and state transitions."""
        if not self.is_running: return
        try:
            all_states = [d['state'] for d in self.drones]            
            for drone in self.drones:
                other_states = [state for i, state in enumerate(all_states) if i != (drone['id']-1)]                                              
                
                # Execute RK4 step for ODE integration
                new_state, log_msg = DroneEngine.rk4_step(drone['state'], 0.1, drone['target'], self.obstacles, other_states, self.wind_vector)              
                
                # Check for numerical instability (NaN or Inf values)
                if np.isnan(new_state).any() or np.isinf(new_state).any():
                    raise ValueError(f"Numerical Instability detected in Drone {drone['id']}!")              
                
                # Predict collisions using Newton-Raphson
                nr_alert = DroneEngine.predict_collision_newton_raphson(new_state[:3], new_state[3:], self.obstacles)              
                drone['state'] = new_state
                current_pos = drone['state'][:3].copy()
                drone['path_history'].append(current_pos)                                            
                
                spline_log = ""                
                if drone['id'] == 1 and len(drone['path_history']) % 5 == 0:
                    spline_log = DroneEngine.analyze_path_cubic_spline(drone['path_history'])              
                
                if nr_alert or drone['id'] == 1:
                    full_log = f"D{drone['id']} T={len(drone['path_history'])}:\n{nr_alert}\n{log_msg}\n{spline_log}\n" + "-"*35 + "\n"
                    self.log_console.insert("0.0", full_log)                
                
                if len(drone['path_history']) > 1:
                    path = np.array(drone['path_history'])
                    drone['plots']['path_line'].set_data_3d(path[:, 0], path[:, 1], path[:, 2])              
                self.draw_hexacopter(current_pos, len(drone['path_history']), drone['plots'])          
            
            if self.camera_mode and len(self.drones) > 0:
                lead = self.drones[0]
                c_pos, vx, vy = lead['state'][:3], lead['state'][3], lead['state'][4]
                azim = np.degrees(np.arctan2(vy, vx)) if (abs(vx)>0.1 or abs(vy)>0.1) else self.ax.azim              
                self.ax.set_xlim([c_pos[0]-40, c_pos[0]+40])
                self.ax.set_ylim([c_pos[1]-40, c_pos[1]+40])
                self.ax.set_zlim([max(0, c_pos[2]-15), c_pos[2]+40])
                self.ax.view_init(elev=20, azim=azim)
            else:
                self.ax.set_xlim([-20, 120])
                self.ax.set_ylim([-20, 120])
                self.ax.set_zlim([0, 120])            
            
            self.canvas.draw_idle() 
            
            if len(self.drones) > 0:
                lead_drone = self.drones[0]
                dist_to_target = np.linalg.norm(lead_drone['state'][:3] - lead_drone['target'])
                
                # If the drone approaches within 5 meters of the target, halt and plot
                if dist_to_target < 5.0 and self.is_running:
                    self.is_running = False
                    self.log_console.insert("0.0", "\nTARGET REACHED\n")
                    
                    # Prepare discrete waypoint arrays
                    actual_path = np.array(lead_drone['path_history'])
                    
                    if len(actual_path) > 5:
                        t = np.arange(len(actual_path))
                        
                        spline_path = np.array([CubicSpline(t, actual_path[:,0])(t), 
                                                CubicSpline(t, actual_path[:,1])(t), 
                                                CubicSpline(t, actual_path[:,2])(t)]).T
                        time_steps = np.linspace(0, len(actual_path)*0.1, len(actual_path))
                        # Simulating machine precision loss as steps accumulate
                        errors = [abs(1.0 - sum([0.1]*int(i))) for i in np.linspace(1, 100, len(actual_path))]
                        
                        # Trigger the secondary academic visualization engine
                        generate_academic_plots(actual_path, spline_path, time_steps, errors)                        
        except Exception as e:
            self.is_running = False
            self.log_console.insert("0.0", f"\n hated\nReason: {str(e)}\n")         
            
        if self.is_running:
            self.after(15, self.update_sim)

if __name__ == "__main__":
    app = DroneApp()
    app.mainloop()
