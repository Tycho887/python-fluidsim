import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utility import initialize_fields, build_up_b, pressure_poisson, update_velocity
from boundary_functions import apply_boundary_conditions

class PipeFlowSimulation:
    def __init__(self, Lx=1.0, Ly=1.0, nx=100, ny=100, rho=10.00, nu=1.0, g=0.0, dt=1e-5, nt=1000):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.dx, self.dy = Lx / (nx - 1), Ly / (ny - 1)
        self.rho, self.nu = rho, nu
        self.g = -g
        self.dt, self.nt = dt, nt

        self.u, self.v, self.p, self.b = initialize_fields(self.ny, self.nx)
        self.u, self.v, self.p = apply_boundary_conditions(self.u, self.v, self.p)

    def single_time_step(self):
        """
        Run a single time step of the simulation, updating u, v, and p fields.
        """
        print(f"Running")
        self.b = build_up_b(self.u, self.v, self.dx, self.dy, self.dt, self.rho)
        self.p = pressure_poisson(self.p, self.b, self.dx, self.dy, max_iterations=100, rho=self.rho, g=self.g)
        self.u, self.v = update_velocity(self.u, self.v, self.p, self.dx, self.dy, self.dt, self.rho, self.nu)
        self.u, self.v, self.p = apply_boundary_conditions(self.u, self.v, self.p)

    def animate_simulation(self):
        """
        Animate the simulation results with both velocity and pressure fields.
        """
        fig, (ax_vel, ax_pres) = plt.subplots(1, 2, figsize=(12, 6))
        x, y = np.linspace(0, self.Lx, self.nx), np.linspace(0, self.Ly, self.ny)
        
        # Initial plot of the velocity magnitude
        vel_magnitude = np.sqrt(self.u**2 + self.v**2)
        CS_vel = ax_vel.contourf(x, y, vel_magnitude, alpha=0.5)
        cbar_vel = fig.colorbar(CS_vel, ax=ax_vel)
        cbar_vel.ax.set_ylabel('Velocity Magnitude')
        
        # Initial plot of the pressure field
        CS_pres = ax_pres.contourf(x, y, self.p, alpha=0.5)
        cbar_pres = fig.colorbar(CS_pres, ax=ax_pres)
        cbar_pres.ax.set_ylabel('Pressure')
        
        def animate(i):
            # Advance the simulation by a single time step
            self.single_time_step()

            ax_vel.clear()
            ax_pres.clear()
            
            # Update velocity plot
            vel_magnitude = np.sqrt(self.u**2 + self.v**2)
            ax_vel.quiver(x, y, self.u, self.v, scale=10)
            ax_vel.contourf(x, y, vel_magnitude, alpha=0.5)
            ax_vel.set_title(f"Velocity Field (Time step: {i})")
            
            # Update pressure plot
            ax_pres.contourf(x, y, self.p, alpha=0.5)
            ax_pres.set_title("Pressure Field")
        
        ani = animation.FuncAnimation(fig, animate, frames=range(0, self.nt, self.nt // 100), repeat=False)
        plt.tight_layout()
        ani.save('pipe_flow_simulation.mp4', writer='ffmpeg', fps=5)
        plt.close()

# Initialize and run the animation
simulation = PipeFlowSimulation()
simulation.animate_simulation()
