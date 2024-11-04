import numpy as np

def circular_velocity(u, v, r, n):
    """
    Set the initial velocity field to be a circular vortex.
    """
    right_arc = np.linspace(-np.pi/4, np.pi/4, n)
    left_arc = np.linspace(3*np.pi/4, 5*np.pi/4, n)
    top_arc = np.linspace(3*np.pi/4, 5*np.pi/4, n)
    bottom_arc = np.linspace(-np.pi/4, np.pi/4, n)

    v[-1,:] = np.cos(right_arc)
    v[0, :] = np.cos(left_arc)
    u[:,-1] = np.cos(top_arc)
    u[:, 0] = np.cos(bottom_arc)

    return u, v

def _circular_velocity(u, v, velocity_magnitude, grid_size):
    """
    Apply a circular velocity around the boundary edges.
    """
    center_x, center_y = grid_size / 2, grid_size / 2  # Center of the grid

    # Top edge (y = 0)
    for x in range(u.shape[1]):
        angle = np.arctan2(0 - center_y, x - center_x)
        u[0, x] = velocity_magnitude * np.cos(angle)
        v[0, x] = velocity_magnitude * np.sin(angle)

    # Bottom edge (y = -1)
    for x in range(u.shape[1]):
        angle = np.arctan2(u.shape[0] - 1 - center_y, x - center_x)
        u[-1, x] = velocity_magnitude * np.cos(angle)
        v[-1, x] = velocity_magnitude * np.sin(angle)

    # Left edge (x = 0)
    for y in range(v.shape[0]):
        angle = np.arctan2(y - center_y, 0 - center_x)
        u[y, 0] = velocity_magnitude * np.cos(angle)
        v[y, 0] = velocity_magnitude * np.sin(angle)

    # Right edge (x = -1)
    for y in range(v.shape[0]):
        angle = np.arctan2(y - center_y, u.shape[1] - 1 - center_x)
        u[y, -1] = velocity_magnitude * np.cos(angle)
        v[y, -1] = velocity_magnitude * np.sin(angle)

    return u, v

def simple(u,v):
    u[0, :] = 0
    v[0, :] = 0
    u[-1, :] = 0
    v[-1, :] = 0
    u[:, 0] = 0
    v[:, 0] = 0
    u[:, -1] = 0
    v[:, -1] = 0
    return u, v

def chaos(u,v):
    u[-1, :] = -0.1  # Bottom wall
    v[-1, :] = 0
    u[0, :] = 0.1    # Top wall
    v[0, :] = 0

    u[:, -1] = 0     # Right wall
    v[:, -1] = 0.1
    u[:, 0] = 0      # Left wall
    v[:, 0] = -0.1

def right_flow(u,v,flowrate=0.10):
    u[:, -1] = flowrate*np.sin(np.linspace(0, np.pi, u.shape[0]))
    v[:, -1] = 0

    u[:, 0] = flowrate*np.sin(np.linspace(0, np.pi, u.shape[0]))
    v[:, 0] = 0

    u[0, :] = 0
    v[0, :] = 0
    # no slip on other walls

    u[-1, :] = 0
    v[-1, :] = 0

    return u, v

def obstacle(u, v):
    """
    Apply obstacle in the middle of the domain.
    """
    u[10:20, 10:-10] = 0
    v[10:20, 10:-10] = 0
    return u, v


def apply_boundary_conditions(u, v, p):
    """
    Apply boundary conditions:
    - No-slip on all walls for u and v (static fluid)
    - Constant pressure on top and bottom boundaries
    - Zero-gradient on left and right boundaries
    """
    
    # Set constant pressure at top and bottom boundaries
    p[0, :] = 0      # Top boundary pressure
    p[-1, :] = 0     # Bottom boundary pressure
    
    # Apply zero-gradient (Neumann) condition on left and right boundaries
    p[:, 0] = p[:, 1]  # Left boundary
    p[:, -1] = p[:, -2]  # Right boundary

    u,v = right_flow(u,v)
    
    return u, v, p
