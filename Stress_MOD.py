import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import streamlit as st
import lasio
import io
import tempfile
import os

# Page configuration
st.set_page_config(layout="wide")
st.title("Wellbore Stress Analysis with LAS Data")
st.markdown("""
This app calculates hoop stress distribution around a wellbore using stress field data from LAS files.
All results are displayed in psi (pressure units).
""")

def main():
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Parameters")
        
        # LAS file upload
        las_file = st.file_uploader("Upload LAS File", type=['las'])
        
        if las_file:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp:
                    tmp.write(las_file.getvalue())
                    tmp_path = tmp.name
                
                # Read LAS file
                las = lasio.read(tmp_path)
                os.unlink(tmp_path)  # Delete temporary file
                
                st.success("LAS file successfully loaded!")
                
                # Get available curves
                available_curves = list(las.curves.keys())
                st.write("Available curves:", ", ".join(available_curves))
                
                # Wellbore geometry
                wellbore_radius = st.number_input("Wellbore Radius (ft)", 0.1, 2.0, 0.328, 0.01)
                
                # Depth range selection
                depth_curve_options = [c for c in available_curves if c.upper() in ['DEPT', 'DEPTH']]
                if not depth_curve_options:
                    st.error("No DEPT or DEPTH curve found in LAS file")
                    return
                
                depth_curve = depth_curve_options[0]
                depth_data = las[depth_curve]
                valid_depth_indices = ~np.isnan(depth_data)
                depth_data = depth_data[valid_depth_indices]
                
                if len(depth_data) == 0:
                    st.error("No valid depth data found")
                    return
                
                default_min = float(np.nanmin(depth_data))
                default_max = float(np.nanmax(depth_data))
                
                min_depth = st.number_input("Minimum Depth (ft)", 
                                          min_value=default_min, 
                                          max_value=default_max, 
                                          value=default_min)
                max_depth = st.number_input("Maximum Depth (ft)", 
                                          min_value=default_min, 
                                          max_value=default_max, 
                                          value=default_max)
                
                # Stress field selection
                st.subheader("Select Stress Curves")
                sigma_H_curve = st.selectbox("Maximum Horizontal Stress (σH) curve", 
                                           options=available_curves,
                                           index=0)
                sigma_h_curve = st.selectbox("Minimum Horizontal Stress (σh) curve", 
                                           options=available_curves,
                                           index=min(1, len(available_curves)-1))
                Pp_curve = st.selectbox("Pore Pressure (Pp) curve", 
                                      options=available_curves,
                                      index=min(2, len(available_curves)-1))
                
                # Visualization settings
                st.subheader("Visualization Settings")
                threshold_percent = st.slider("Stress Concentration Threshold (%)", 30, 90, 50, 5)
                resolution = st.selectbox("Model Resolution", ["Low", "Medium", "High"], index=1)
                
                # Get stress data
                sigma_H_data = las[sigma_H_curve][valid_depth_indices]
                sigma_h_data = las[sigma_h_curve][valid_depth_indices]
                Pp_data = las[Pp_curve][valid_depth_indices]
                
                # Create stress vs depth plot
                st.subheader("Stress Field Profile")
                stress_plot = plot_stress_vs_depth(depth_data, sigma_H_data, sigma_h_data, Pp_data, min_depth, max_depth)
                st.pyplot(stress_plot)
                
                # Run calculations
                results = calculate_stresses(
                    depth_data, sigma_H_data, sigma_h_data, Pp_data,
                    min_depth, max_depth, wellbore_radius, resolution
                )
                
                if results:
                    X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d = results
                    mid_depth_idx = len(np.linspace(min_depth, max_depth, 5)) // 2
                    current_depth = np.linspace(min_depth, max_depth, 5)[mid_depth_idx]
                    
                    # Create analysis plots
                    st.subheader("Wellbore Stress Analysis")
                    fig = create_analysis_plots(
                        X, Y, Z, R, Theta, Depth, hoop_stress_fd, 
                        sigma_H_3d, sigma_h_3d, Pp_3d,
                        wellbore_radius, min_depth, max_depth,
                        mid_depth_idx, current_depth, threshold_percent
                    )
                    st.pyplot(fig)
                    
                    # Show raw stress data
                    st.subheader("Stress Data at Selected Depth")
                    st.write(f"Maximum Horizontal Stress (σH) at {current_depth:.0f} ft: {sigma_H_3d[0,0,mid_depth_idx]:.0f} psi")
                    st.write(f"Minimum Horizontal Stress (σh) at {current_depth:.0f} ft: {sigma_h_3d[0,0,mid_depth_idx]:.0f} psi")
                    st.write(f"Pore Pressure (Pp) at {current_depth:.0f} ft: {Pp_3d[0,0,mid_depth_idx]:.0f} psi")
                
            except Exception as e:
                st.error(f"Error processing LAS file: {str(e)}")
        else:
            st.warning("Please upload a LAS file to proceed")

def plot_stress_vs_depth(depth, sigma_H, sigma_h, Pp, min_depth, max_depth):
    fig, ax = plt.subplots(figsize=(8, 10))
    
    ax.plot(sigma_H, depth, 'r-', label='σH (Max Horizontal Stress)')
    ax.plot(sigma_h, depth, 'b-', label='σh (Min Horizontal Stress)')
    ax.plot(Pp, depth, 'g-', label='Pp (Pore Pressure)')
    ax.axhspan(min_depth, max_depth, color='yellow', alpha=0.3, label='Selected Depth Range')
    
    ax.set_xlabel('Stress (psi)')
    ax.set_ylabel('Depth (ft)')
    ax.set_title('Stress Field vs Depth')
    ax.grid(True)
    ax.legend()
    ax.invert_yaxis()
    
    return fig

def calculate_stresses(depth_data, sigma_H_data, sigma_h_data, Pp_data,
                     min_depth, max_depth, wellbore_radius, resolution):
    # Set resolution parameters
    if resolution == "Low":
        depth_points = 3
        theta_points = 18
        radial_points = 10
    elif resolution == "Medium":
        depth_points = 5
        theta_points = 36
        radial_points = 20
    else:  # High
        depth_points = 7
        theta_points = 72
        radial_points = 30
    
    try:
        # Create depth range
        depth_range = np.linspace(min_depth, max_depth, depth_points)
        
        # Interpolate to our depth points
        sigma_H = np.interp(depth_range, depth_data, sigma_H_data)
        sigma_h = np.interp(depth_range, depth_data, sigma_h_data)
        Pp = np.interp(depth_range, depth_data, Pp_data)
        
        # Create grid
        theta = np.linspace(0, 2*np.pi, theta_points)
        r = np.linspace(wellbore_radius, 5*wellbore_radius, radial_points)
        
        # 3D grid for finite difference
        R, Theta, Depth = np.meshgrid(r, theta, depth_range, indexing='ij')
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = Depth
        
        # Expand stress arrays to 3D
        sigma_H_3d = np.zeros_like(R)
        sigma_h_3d = np.zeros_like(R)
        Pp_3d = np.zeros_like(R)
        
        for i in range(R.shape[2]):
            sigma_H_3d[:,:,i] = sigma_H[i]
            sigma_h_3d[:,:,i] = sigma_h[i]
            Pp_3d[:,:,i] = Pp[i]
        
        # Finite difference setup
        dr = r[1] - r[0]
        dtheta = theta[1] - theta[0]
        dz = depth_range[1] - depth_range[0]
        
        # Sparse matrix construction
        num_points = np.prod(R.shape)
        A = csr_matrix((num_points, num_points))
        b = np.zeros(num_points)
        
        def get_index(i, j, k):
            return i * (Theta.shape[1] * Theta.shape[2]) + j * Theta.shape[2] + k
        
        # Build FD system
        for i in range(1, R.shape[0]-1):
            for j in range(R.shape[1]):
                for k in range(R.shape[2]):
                    idx = get_index(i, j, k)
                    
                    # Radial terms
                    A[idx, get_index(i+1, j, k)] = 1/dr**2 + 1/(2*R[i,j,k]*dr)
                    A[idx, get_index(i-1, j, k)] = 1/dr**2 - 1/(2*R[i,j,k]*dr)
                    A[idx, idx] = -2/dr**2 - 2/(R[i,j,k]**2 * dtheta**2) - 2/dz**2
                    
                    # Theta terms
                    A[idx, get_index(i, (j+1)%R.shape[1], k)] = 1/(R[i,j,k]**2 * dtheta**2)
                    A[idx, get_index(i, (j-1)%R.shape[1], k)] = 1/(R[i,j,k]**2 * dtheta**2)
                    
                    # Depth terms
                    if k > 0:
                        A[idx, get_index(i, j, k-1)] = 1/dz**2
                    if k < R.shape[2]-1:
                        A[idx, get_index(i, j, k+1)] = 1/dz**2
                    
                    # Source term
                    b[idx] = -sigma_H_3d[i,j,k] * (1 + wellbore_radius**2/R[i,j,k]**2)
        
        # Boundary conditions
        for j in range(R.shape[1]):
            for k in range(R.shape[2]):
                idx = get_index(0, j, k)
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = (sigma_H_3d[0,j,k] + sigma_h_3d[0,j,k])/2 * (1 - 2*np.cos(2*Theta[0,j,k])) - Pp_3d[0,j,k]
        
        # Solve system
        hoop_stress_fd = spsolve(A, b).reshape(R.shape)
        
        return X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d
    
    except Exception as e:
        st.error(f"Error in stress calculations: {str(e)}")
        return None

def create_analysis_plots(X, Y, Z, R, Theta, Depth, hoop_stress_fd, 
                         sigma_H_3d, sigma_h_3d, Pp_3d,
                         wellbore_radius, min_depth, max_depth,
                         mid_depth_idx, current_depth, threshold_percent):
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 3D Wellbore Stress Distribution
    ax1 = fig.add_subplot(231, projection='3d')
    wellbore_surface = ax1.plot_surface(
        X[:,:,mid_depth_idx], Y[:,:,mid_depth_idx], Z[:,:,mid_depth_idx], 
        facecolors=cm.jet(hoop_stress_fd[:,:,mid_depth_idx]/hoop_stress_fd.max()),
        rstride=1, cstride=1, alpha=0.8
    )
    ax1.set_title('3D Wellbore Hoop Stress (psi)')
    ax1.set_xlabel('X (ft)')
    ax1.set_ylabel('Y (ft)')
    ax1.set_zlabel('Depth (ft)')
    
    # 2. 3D Stress Concentration
    ax2 = fig.add_subplot(232, projection='3d')
    threshold = hoop_stress_fd.max() * (threshold_percent/100)
    mask = hoop_stress_fd > threshold
    scatter = ax2.scatter(
        X[mask], Y[mask], Z[mask], 
        c=hoop_stress_fd[mask], 
        cmap='hot', 
        s=30,
        alpha=0.8
    )
    ax2.set_title(f'3D Stress Concentration (>{threshold_percent}% of max)')
    ax2.set_xlabel('X (ft)')
    ax2.set_ylabel('Y (ft)')
    ax2.set_zlabel('Depth (ft)')
    
    # Add wellbore outline
    theta_wall = np.linspace(0, 2*np.pi, 50)
    x_wall = wellbore_radius * np.cos(theta_wall)
    y_wall = wellbore_radius * np.sin(theta_wall)
    z_wall = np.linspace(min_depth, max_depth, 50)
    X_wall, Z_wall = np.meshgrid(x_wall, z_wall)
    Y_wall, _ = np.meshgrid(y_wall, z_wall)
    ax2.plot_wireframe(X_wall, Y_wall, Z_wall, color='black', linewidth=0.5, alpha=0.3)
    
    # 3. Circumferential Stress at Wellbore Wall
    theta_fine = np.linspace(0, 2*np.pi, 360)
    stress_at_wall = kirsch_hoop_stress(
        wellbore_radius, theta_fine, 
        sigma_H_3d[0,0,mid_depth_idx], sigma_h_3d[0,0,mid_depth_idx], 
        wellbore_radius, Pp_3d[0,0,mid_depth_idx]
    )
    
    ax3 = fig.add_subplot(233)
    ax3.plot(np.degrees(theta_fine), stress_at_wall, 'b-')
    ax3.axhline(
        (sigma_H_3d[0,0,mid_depth_idx] + sigma_h_3d[0,0,mid_depth_idx])/2 - Pp_3d[0,0,mid_depth_idx], 
        color='r', linestyle='--', label='Avg Stress - Pp'
    )
    ax3.set_title(f'Hoop Stress at Wellbore Wall\n(Depth = {current_depth:.0f} ft)')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Hoop Stress (psi)')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Radial Stress Distribution
    r_fine = np.linspace(wellbore_radius, 5*wellbore_radius, 100)
    stress_0deg = kirsch_hoop_stress(
        r_fine, 0, sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx], wellbore_radius, Pp_3d[0,0,mid_depth_idx]
    )
    stress_90deg = kirsch_hoop_stress(
        r_fine, np.pi/2, sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx], wellbore_radius, Pp_3d[0,0,mid_depth_idx]
    )
    
    ax4 = fig.add_subplot(234)
    ax4.plot(r_fine/wellbore_radius, stress_0deg, 'r-', label='θ=0° (σH direction)')
    ax4.plot(r_fine/wellbore_radius, stress_90deg, 'b-', label='θ=90° (σh direction)')
    ax4.axhline(sigma_H_3d[0,0,mid_depth_idx], color='r', linestyle=':', label='Far-field σH')
    ax4.axhline(sigma_h_3d[0,0,mid_depth_idx], color='b', linestyle=':', label='Far-field σh')
    ax4.set_title('Radial Stress Decay')
    ax4.set_xlabel('Normalized Radius (r/r_w)')
    ax4.set_ylabel('Hoop Stress (psi)')
    ax4.grid(True)
    ax4.legend()
    
    # 5. 2D Polar Contour
    R_2D, Theta_2D = np.meshgrid(r_fine, theta_fine)
    X_2D = R_2D * np.cos(Theta_2D)
    Y_2D = R_2D * np.sin(Theta_2D)
    hoop_stress_2D = kirsch_hoop_stress(
        R_2D, Theta_2D, sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx], wellbore_radius, Pp_3d[0,0,mid_depth_idx]
    )
    
    ax5 = fig.add_subplot(235, polar=True)
    contour = ax5.contourf(Theta_2D, R_2D, hoop_stress_2D, 20, cmap='jet')
    ax5.set_title('2D Polar Stress Distribution (psi)')
    
    # 6. Cartesian Cross-Section
    ax6 = fig.add_subplot(236)
    contour = ax6.contourf(X_2D, Y_2D, hoop_stress_2D, 20, cmap='jet')
    ax6.set_title('Cartesian Stress Distribution (psi)')
    ax6.set_aspect('equal')
    
    # Add colorbars
    fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax1, label='Hoop Stress (psi)')
    fig.colorbar(scatter, ax=ax2, label='Hoop Stress (psi)')
    fig.colorbar(contour, ax=ax5, label='Hoop Stress (psi)')
    fig.colorbar(contour, ax=ax6, label='Hoop Stress (psi)')
    
    plt.tight_layout()
    return fig

def kirsch_hoop_stress(r, theta, sigma_H, sigma_h, wellbore_radius, Pp):
    term1 = (sigma_H + sigma_h)/2 * (1 + wellbore_radius**2/r**2)
    term2 = (sigma_H - sigma_h)/2 * (1 + 3*wellbore_radius**4/r**4) * np.cos(2*theta)
    term3 = -Pp * wellbore_radius**2/r**2
    return term1 + term2 + term3

if __name__ == "__main__":
    main()
