import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import streamlit as st
import lasio
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
                os.unlink(tmp_path)
                
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
                                                   value=float(default_azimuth))
                else:
                    st.warning("No azimuth curve found in LAS file")
                    manual_azimuth = st.number_input("Enter Wellbore Azimuth (°)", 
                                                    min_value=0.0, 
                                                    max_value=360.0, 
                                                    value=0.0)
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
                st.plotly_chart(stress_plot, use_container_width=True)
                
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
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                    with tab2:
                        fig_2d = create_2d_plots(
                            R, Theta, Depth, hoop_stress_fd,
                            sigma_H_3d, sigma_h_3d, Pp_3d,
                            wellbore_radius, current_depth, current_azimuth
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                    
                    with tab3:
                        fig_profiles = create_stress_profiles(
                            R, Theta, Depth, hoop_stress_fd,
                            sigma_H_3d, sigma_h_3d, Pp_3d,
                            wellbore_radius, current_depth, current_azimuth
                        )
                        st.plotly_chart(fig_profiles, use_container_width=True)
                    
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
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sigma_H, y=depth,
        mode='lines',
        name='σH (Max Horizontal Stress)',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=sigma_h, y=depth,
        mode='lines',
        name='σh (Min Horizontal Stress)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=Pp, y=depth,
        mode='lines',
        name='Pp (Pore Pressure)',
        line=dict(color='green')
    ))
    
    fig.add_shape(type="rect",
        x0=min(sigma_H.min(), sigma_h.min(), Pp.min()),
        x1=max(sigma_H.max(), sigma_h.max(), Pp.max()),
        y0=min_depth, y1=max_depth,
        fillcolor="yellow", opacity=0.2,
        line=dict(width=0)
    )
    
    fig.update_layout(
        title='Stress Field vs Depth',
        xaxis_title='Stress (psi)',
        yaxis_title='Depth (ft)',
        yaxis=dict(autorange="reversed"),
        height=600,
        hovermode="closest"
    )
    
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
    mid_depth_idx = Z.shape[2] // 2
    
    # Create 3D surface plot
    fig1 = go.Figure(data=[
        go.Surface(
            x=X[:,:,mid_depth_idx],
            y=Y[:,:,mid_depth_idx],
            z=Z[:,:,mid_depth_idx],
            surfacecolor=hoop_stress_fd[:,:,mid_depth_idx],
            colorscale='jet',
            colorbar=dict(title='Hoop Stress (psi)'),
            opacity=0.8
        )
    ])
    
    # Add North arrow
    fig1.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, wellbore_radius*2],
        z=[current_depth, current_depth],
        mode='lines+text',
        line=dict(color='black', width=4),
        text=['', 'N'],
        textposition='top center'
    ))
    
    # Add stress direction indicator
    theta_offset = np.radians(current_azimuth)
    fig1.add_trace(go.Scatter3d(
        x=[0, wellbore_radius*2*np.cos(theta_offset)],
        y=[0, wellbore_radius*2*np.sin(theta_offset)],
        z=[current_depth, current_depth],
        mode='lines+text',
        line=dict(color='red', width=4),
        text=['', 'σH'],
        textposition='top center'
    ))
    
    fig1.update_layout(
        title=f'3D Wellbore Hoop Stress<br>Azimuth: {current_azimuth:.1f}°',
        scene=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth (ft)',
            zaxis=dict(autorange="reversed")
        ),
        height=700
    )
    
    # Create 3D stress concentration plot
    threshold = hoop_stress_fd.max() * (threshold_percent/100)
    mask = hoop_stress_fd > threshold
    
    fig2 = go.Figure(data=[
        go.Scatter3d(
            x=X[mask],
            y=Y[mask],
            z=Z[mask],
            mode='markers',
            marker=dict(
                size=4,
                color=hoop_stress_fd[mask],
                colorscale='hot',
                colorbar=dict(title='Hoop Stress (psi)'),
                opacity=0.8
            )
        )
    ])
    
    # Add wellbore outline
    theta_wall = np.linspace(0, 2*np.pi, 20)
    x_wall = wellbore_radius * np.cos(theta_wall)
    y_wall = wellbore_radius * np.sin(theta_wall)
    z_wall = np.linspace(Z.min(), Z.max(), 20)
    
    for i in range(len(theta_wall)):
        fig2.add_trace(go.Scatter3d(
            x=[x_wall[i], x_wall[(i+1)%len(theta_wall)]],
            y=[y_wall[i], y_wall[(i+1)%len(theta_wall)]],
            z=[z_wall[0], z_wall[0]],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))
        fig2.add_trace(go.Scatter3d(
            x=[x_wall[i], x_wall[(i+1)%len(theta_wall)]],
            y=[y_wall[i], y_wall[(i+1)%len(theta_wall)]],
            z=[z_wall[-1], z_wall[-1]],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))
    
    fig2.update_layout(
        title=f'3D Stress Concentration (>{threshold_percent}% of max)',
        scene=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth (ft)',
            zaxis=dict(autorange="reversed")
        ),
        height=700
    )
    
    return fig1, fig2

def create_2d_plots(R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d, 
                   wellbore_radius, current_depth, current_azimuth):
    mid_depth_idx = Depth.shape[2] // 2
    theta_offset = np.radians(current_azimuth)
    
    # Create polar plot data
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
    
    # Polar plot
    polar_fig = go.Figure(data=[
        go.Contour(
            r=R_2D.flatten(),
            theta=np.degrees(Theta_2D).flatten(),
            z=hoop_stress_2D.flatten(),
            colorscale='jet',
            contours=dict(
                coloring='heatmap',
                showlines=False
            ),
            colorbar=dict(title='Hoop Stress (psi)')
        )
    ])
    
    # Add compass directions
    compass_radius = 1.1 * R_2D.max()
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        polar_fig.add_trace(go.Scatterpolar(
            r=[0, compass_radius],
            theta=[angle, angle],
            mode='lines+text',
            line=dict(color='black', width=1),
            text=['', label],
            textposition='top center',
            showlegend=False
        ))
    
    # Add stress direction marker
    polar_fig.add_trace(go.Scatterpolar(
        r=[0, wellbore_radius*1.5],
        theta=[current_azimuth, current_azimuth],
        mode='lines',
        line=dict(color='red', width=3),
        name='σH Direction'
    ))
    
    polar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(
                rotation=90,
                direction="clockwise"
            )
        ),
        title='2D Polar Stress Distribution (psi)',
        height=600
    )
    
    # Cartesian plot
    X_2D = R_2D * np.cos(Theta_2D)
    Y_2D = R_2D * np.sin(Theta_2D)
    
    cartesian_fig = go.Figure(data=[
        go.Contour(
            x=X_2D[0,:],
            y=Y_2D[:,0],
            z=hoop_stress_2D,
            colorscale='jet',
            contours=dict(
                coloring='heatmap',
                showlines=False
            ),
            colorbar=dict(title='Hoop Stress (psi)')
        )
    ])
    
    # Add North arrow
    cartesian_fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, wellbore_radius*1.2],
        mode='lines+text',
        line=dict(color='black', width=2),
        text=['', 'N'],
        textposition='top center',
        showlegend=False
    ))
    
    # Add stress direction indicator
    cartesian_fig.add_trace(go.Scatter(
        x=[0, wellbore_radius*1.5*np.cos(theta_offset)],
        y=[0, wellbore_radius*1.5*np.sin(theta_offset)],
        mode='lines+text',
        line=dict(color='red', width=2),
        text=['', 'σH'],
        textposition='top center',
        name='σH Direction'
    ))
    
    cartesian_fig.update_layout(
        title='Cartesian Stress Distribution (psi)',
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        height=600
    )
    
    return polar_fig, cartesian_fig

def create_stress_profiles(R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d,
                         wellbore_radius, current_depth, current_azimuth):
    mid_depth_idx = Depth.shape[2] // 2
    theta_offset = np.radians(current_azimuth)
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Hoop Stress at Wellbore Wall',
                                      'Radial Stress Decay',
                                      'Hoop Stress Extremes vs Depth'))
    
    # 1. Circumferential stress at wellbore wall
    theta_fine = np.linspace(0, 2*np.pi, 360)
    stress_at_wall = kirsch_hoop_stress(
        wellbore_radius, theta_fine - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    
    fig.add_trace(go.Scatter(
        x=np.degrees(theta_fine),
        y=stress_at_wall,
        mode='lines',
        name='Hoop Stress',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_vline(x=current_azimuth, line=dict(color='red', dash='dash'),
                 annotation_text="σH Direction", row=1, col=1)
    fig.add_vline(x=(current_azimuth + 90) % 360, line=dict(color='green', dash='dash'),
                 annotation_text="σh Direction", row=1, col=1)
    
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
    
    fig.add_trace(go.Scatter(
        x=r_fine/wellbore_radius,
        y=stress_0deg,
        mode='lines',
        name='σH direction',
        line=dict(color='red')
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=r_fine/wellbore_radius,
        y=stress_90deg,
        mode='lines',
        name='σh direction',
        line=dict(color='blue')
    ), row=1, col=2)
    
    # 3. Depth profile of maximum stress
    max_stress = np.max(hoop_stress_fd, axis=(0,1))
    min_stress = np.min(hoop_stress_fd, axis=(0,1))
    
    fig.add_trace(go.Scatter(
        x=max_stress,
        y=Depth[0,0,:],
        mode='lines',
        name='Max Hoop Stress',
        line=dict(color='red')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=min_stress,
        y=Depth[0,0,:],
        mode='lines',
        name='Min Hoop Stress',
        line=dict(color='blue')
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Angle (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="Hoop Stress (psi)", row=1, col=1)
    fig.update_xaxes(title_text="Normalized Radius (r/r_w)", row=1, col=2)
    fig.update_yaxes(title_text="Hoop Stress (psi)", row=1, col=2)
    fig.update_xaxes(title_text="Hoop Stress (psi)", row=2, col=1)
    fig.update_yaxes(title_text="Depth (ft)", row=2, col=1)
    
    # Reverse depth axis
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    
    return fig

def kirsch_hoop_stress(r, theta, sigma_H, sigma_h, wellbore_radius, Pp):
    term1 = (sigma_H + sigma_h)/2 * (1 + wellbore_radius**2/r**2)
    term2 = (sigma_H - sigma_h)/2 * (1 + 3*wellbore_radius**4/r**4) * np.cos(2*theta)
    term3 = -Pp * wellbore_radius**2/r**2
    return term1 + term2 + term3

if __name__ == "__main__":
    main()
