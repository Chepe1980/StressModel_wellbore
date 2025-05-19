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
                
                # Well orientation data
                st.subheader("Well Orientation")
                azimuth_curve_options = [c for c in available_curves if c.upper() in ['AZIMUTH', 'AZI', 'AZ']]
                if azimuth_curve_options:
                    azimuth_curve = st.selectbox("Azimuth Curve", options=azimuth_curve_options)
                    azimuth_data = las[azimuth_curve][valid_depth_indices]
                    default_azimuth = np.nanmean(azimuth_data)
                    if np.isnan(default_azimuth):
                        default_azimuth = 0.0
                    manual_azimuth = st.number_input("Manual Azimuth Override (°)", 
                                                   min_value=0.0, 
                                                   max_value=360.0, 
                                                   value=float(default_azimuth),
                                                   help="Use this if azimuth curve is not available or needs adjustment")
                else:
                    st.warning("No azimuth curve found in LAS file")
                    manual_azimuth = st.number_input("Enter Wellbore Azimuth (°)", 
                                                    min_value=0.0, 
                                                    max_value=360.0, 
                                                    value=0.0,
                                                    help="0° = North, 90° = East")
                    azimuth_data = np.full_like(depth_data, manual_azimuth)
                
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
                    depth_data, sigma_H_data, sigma_h_data, Pp_data, azimuth_data,
                    min_depth, max_depth, wellbore_radius, resolution
                )
                
                if results:
                    X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d, azimuth_3d = results
                    mid_depth_idx = len(np.linspace(min_depth, max_depth, 5)) // 2
                    current_depth = np.linspace(min_depth, max_depth, 5)[mid_depth_idx]
                    current_azimuth = azimuth_3d[0,0,mid_depth_idx]
                    
                    # Create analysis plots in main area
                    st.subheader("Wellbore Stress Analysis")
                    st.write(f"Current depth: {current_depth:.0f} ft | Azimuth: {current_azimuth:.1f}°")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["3D Visualization", "2D Plots", "Stress Profiles"])
                    
                    with tab1:
                        fig_3d = create_3d_visualization(
                            X, Y, Z, hoop_stress_fd, 
                            wellbore_radius, current_depth, current_azimuth,
                            threshold_percent
                        )
                        st.pyplot(fig_3d)
                    
                    with tab2:
                        fig_2d = create_2d_plots(
                            R, Theta, Depth, hoop_stress_fd,
                            sigma_H_3d, sigma_h_3d, Pp_3d,
                            wellbore_radius, current_depth, current_azimuth
                        )
                        st.pyplot(fig_2d)
                    
                    with tab3:
                        fig_profiles = create_stress_profiles(
                            R, Theta, Depth, hoop_stress_fd,
                            sigma_H_3d, sigma_h_3d, Pp_3d,
                            wellbore_radius, current_depth, current_azimuth
                        )
                        st.pyplot(fig_profiles)
                    
                    # Show raw stress data
                    st.subheader("Stress Data at Selected Depth")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"σH at {current_depth:.0f} ft", f"{sigma_H_3d[0,0,mid_depth_idx]:.0f} psi")
                    with col2:
                        st.metric(f"σh at {current_depth:.0f} ft", f"{sigma_h_3d[0,0,mid_depth_idx]:.0f} psi")
                    with col3:
                        st.metric(f"Pp at {current_depth:.0f} ft", f"{Pp_3d[0,0,mid_depth_idx]:.0f} psi")
                    st.metric(f"Wellbore Azimuth", f"{current_azimuth:.1f}°")
                
            except Exception as e:
                st.error(f"Error processing LAS file: {str(e)}")
        else:
            st.warning("Please upload a LAS file to proceed")

def plot_stress_vs_depth(depth, sigma_H, sigma_h, Pp, min_depth, max_depth):
    fig, ax = plt.subplots(figsize=(8, 6))
    
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

def calculate_stresses(depth_data, sigma_H_data, sigma_h_data, Pp_data, azimuth_data,
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
        azimuth = np.interp(depth_range, depth_data, azimuth_data)
        
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
        azimuth_3d = np.zeros_like(R)
        
        for i in range(R.shape[2]):
            sigma_H_3d[:,:,i] = sigma_H[i]
            sigma_h_3d[:,:,i] = sigma_h[i]
            Pp_3d[:,:,i] = Pp[i]
            azimuth_3d[:,:,i] = azimuth[i]
        
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
        
        return X, Y, Z, R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d, azimuth_3d
    
    except Exception as e:
        st.error(f"Error in stress calculations: {str(e)}")
        return None

def create_3d_visualization(X, Y, Z, hoop_stress_fd, wellbore_radius, current_depth, current_azimuth, threshold_percent):
    fig = plt.figure(figsize=(12, 8))
    
    # 3D Wellbore Stress Distribution
    ax1 = fig.add_subplot(121, projection='3d')
    mid_depth_idx = Z.shape[2] // 2
    surf = ax1.plot_surface(
        X[:,:,mid_depth_idx], Y[:,:,mid_depth_idx], Z[:,:,mid_depth_idx], 
        facecolors=cm.jet(hoop_stress_fd[:,:,mid_depth_idx]/hoop_stress_fd.max()),
        rstride=1, cstride=1, alpha=0.8
    )
    
    # Add North arrow
    ax1.quiver(0, 0, current_depth, 
               0, wellbore_radius*2, 0,
               color='k', arrow_length_ratio=0.1, linewidth=2)
    ax1.text(0, wellbore_radius*2.2, current_depth, 'N', color='k')
    
    # Add stress direction indicator
    theta_offset = np.radians(current_azimuth)
    ax1.quiver(0, 0, current_depth,
               wellbore_radius*2*np.cos(theta_offset), 
               wellbore_radius*2*np.sin(theta_offset), 
               0,
               color='r', arrow_length_ratio=0.1, linewidth=2)
    ax1.text(wellbore_radius*2.2*np.cos(theta_offset), 
             wellbore_radius*2.2*np.sin(theta_offset), 
             current_depth, 'σH', color='r')
    
    ax1.set_title(f'3D Wellbore Hoop Stress\nAzimuth: {current_azimuth:.1f}°')
    ax1.set_xlabel('X (ft)')
    ax1.set_ylabel('Y (ft)')
    ax1.set_zlabel('Depth (ft)')
    
    # 3D Stress Concentration
    ax2 = fig.add_subplot(122, projection='3d')
    threshold = hoop_stress_fd.max() * (threshold_percent/100)
    mask = hoop_stress_fd > threshold
    scatter = ax2.scatter(
        X[mask], Y[mask], Z[mask], 
        c=hoop_stress_fd[mask], 
        cmap='hot', 
        s=30,
        alpha=0.8
    )
    
    # Add wellbore outline
    theta_wall = np.linspace(0, 2*np.pi, 20)
    x_wall = wellbore_radius * np.cos(theta_wall)
    y_wall = wellbore_radius * np.sin(theta_wall)
    z_wall = np.linspace(Z.min(), Z.max(), 20)
    X_wall, Z_wall = np.meshgrid(x_wall, z_wall)
    Y_wall, _ = np.meshgrid(y_wall, z_wall)
    ax2.plot_wireframe(X_wall, Y_wall, Z_wall, color='black', linewidth=0.5, alpha=0.3)
    
    ax2.set_title(f'3D Stress Concentration (>{threshold_percent}% of max)')
    ax2.set_xlabel('X (ft)')
    ax2.set_ylabel('Y (ft)')
    ax2.set_zlabel('Depth (ft)')
    
    # Add colorbars
    fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax1, label='Hoop Stress (psi)')
    fig.colorbar(scatter, ax=ax2, label='Hoop Stress (psi)')
    
    plt.tight_layout()
    return fig

def create_2d_plots(R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d, 
                   wellbore_radius, current_depth, current_azimuth):
    fig = plt.figure(figsize=(12, 6))
    
    # Get current depth index and azimuth
    mid_depth_idx = Depth.shape[2] // 2
    theta_offset = np.radians(current_azimuth)
    
    # 1. Polar plot
    r_fine = np.linspace(wellbore_radius, 5*wellbore_radius, 100)
    theta_fine = np.linspace(0, 2*np.pi, 360)
    R_2D, Theta_2D = np.meshgrid(r_fine, theta_fine)
    
    hoop_stress_2D = kirsch_hoop_stress(
        R_2D, Theta_2D - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    
    ax1 = fig.add_subplot(121, polar=True)
    contour = ax1.contourf(Theta_2D, R_2D, hoop_stress_2D, 20, cmap='jet')
    
    # Set North to top and clockwise azimuth
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    
    # Add compass directions
    ax1.text(0, np.max(r_fine)*1.1, 'N', ha='center', va='center')
    ax1.text(np.pi/2, np.max(r_fine)*1.1, 'E', ha='center', va='center')
    ax1.text(np.pi, np.max(r_fine)*1.1, 'S', ha='center', va='center')
    ax1.text(3*np.pi/2, np.max(r_fine)*1.1, 'W', ha='center', va='center')
    
    # Add stress direction markers
    ax1.plot([theta_offset, theta_offset], [0, wellbore_radius*1.5], 'r-', linewidth=2, label='σH Direction')
    
    ax1.set_title('2D Polar Stress Distribution (psi)')
    ax1.legend()
    
    # 2. Cartesian plot
    X_2D = R_2D * np.cos(Theta_2D)
    Y_2D = R_2D * np.sin(Theta_2D)
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X_2D, Y_2D, hoop_stress_2D, 20, cmap='jet')
    
    # Add North arrow
    ax2.arrow(0, 0, 0, wellbore_radius*1.2, head_width=0.2, head_length=0.3, fc='k', ec='k')
    ax2.text(0, wellbore_radius*1.3, 'N', ha='center', va='center')
    
    # Add stress direction indicator
    ax2.plot([0, wellbore_radius*1.5*np.cos(theta_offset)], 
             [0, wellbore_radius*1.5*np.sin(theta_offset)], 
             'r-', linewidth=2, label='σH Direction')
    
    ax2.set_title('Cartesian Stress Distribution (psi)')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Add colorbars
    fig.colorbar(contour, ax=ax1, label='Hoop Stress (psi)')
    fig.colorbar(contour, ax=ax2, label='Hoop Stress (psi)')
    
    plt.tight_layout()
    return fig

def create_stress_profiles(R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d,
                         wellbore_radius, current_depth, current_azimuth):
    fig = plt.figure(figsize=(12, 8))
    
    # Get current depth index and azimuth
    mid_depth_idx = Depth.shape[2] // 2
    theta_offset = np.radians(current_azimuth)
    
    # 1. Circumferential stress at wellbore wall
    theta_fine = np.linspace(0, 2*np.pi, 360)
    stress_at_wall = kirsch_hoop_stress(
        wellbore_radius, theta_fine - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    
    ax1 = fig.add_subplot(221)
    ax1.plot(np.degrees(theta_fine), stress_at_wall, 'b-')
    ax1.axvline(current_azimuth, color='r', linestyle='--', label='σH Direction')
    ax1.axvline((current_azimuth + 90) % 360, color='g', linestyle='--', label='σh Direction')
    ax1.set_title(f'Hoop Stress at Wellbore Wall\n(Depth = {current_depth:.0f} ft)')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Hoop Stress (psi)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Radial stress distribution
    r_fine = np.linspace(wellbore_radius, 5*wellbore_radius, 100)
    stress_0deg = kirsch_hoop_stress(
        r_fine, 0 - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    stress_90deg = kirsch_hoop_stress(
        r_fine, np.pi/2 - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    
    ax2 = fig.add_subplot(222)
    ax2.plot(r_fine/wellbore_radius, stress_0deg, 'r-', label='σH direction')
    ax2.plot(r_fine/wellbore_radius, stress_90deg, 'b-', label='σh direction')
    ax2.set_title('Radial Stress Decay')
    ax2.set_xlabel('Normalized Radius (r/r_w)')
    ax2.set_ylabel('Hoop Stress (psi)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Depth profile of maximum stress
    max_stress = np.max(hoop_stress_fd, axis=(0,1))
    min_stress = np.min(hoop_stress_fd, axis=(0,1))
    
    ax3 = fig.add_subplot(212)
    ax3.plot(max_stress, Depth[0,0,:], 'r-', label='Max Hoop Stress')
    ax3.plot(min_stress, Depth[0,0,:], 'b-', label='Min Hoop Stress')
    ax3.set_title('Hoop Stress Extremes vs Depth')
    ax3.set_xlabel('Hoop Stress (psi)')
    ax3.set_ylabel('Depth (ft)')
    ax3.grid(True)
    ax3.legend()
    ax3.invert_yaxis()
    
    plt.tight_layout()
    return fig

def kirsch_hoop_stress(r, theta, sigma_H, sigma_h, wellbore_radius, Pp):
    term1 = (sigma_H + sigma_h)/2 * (1 + wellbore_radius**2/r**2)
    term2 = (sigma_H - sigma_h)/2 * (1 + 3*wellbore_radius**4/r**4) * np.cos(2*theta)
    term3 = -Pp * wellbore_radius**2/r**2
    return term1 + term2 + term3

if __name__ == "__main__":
    main()
