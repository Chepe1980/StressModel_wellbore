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

# add logo
logo = Image.open('logo.png')
stsidebar.image(logo, width=200)
st.markdown("""
This app calculates hoop stress distribution around a wellbore using stress field data from LAS files.
All results are displayed in psi (pressure units).
""")

def kirsch_hoop_stress(r, theta, sigma_H, sigma_h, wellbore_radius, Pp):
    term1 = (sigma_H + sigma_h)/2 * (1 + wellbore_radius**2/r**2)
    term2 = (sigma_H - sigma_h)/2 * (1 + 3*wellbore_radius**4/r**4) * np.cos(2*theta)
    term3 = -Pp * wellbore_radius**2/r**2
    return term1 + term2 + term3

def calculate_zoback_polygon(Sv, Pp):
    """
    Calculate Zoback stress polygon coordinates
    Returns: (x, y) coordinates for plotting the polygon
    """
    pp_norm = Pp/Sv
    
    # Polygon vertices (normalized coordinates)
    vertices = np.array([
        [pp_norm, pp_norm],               # Lower left
        [3 - pp_norm, pp_norm],            # Lower right
        [3 - pp_norm, 1 + 2*pp_norm],      # Upper right
        [1 + 2*pp_norm, 3 - pp_norm],      # Top right
        [pp_norm, 3 - pp_norm],            # Top left
        [pp_norm, pp_norm]                 # Close polygon
    ])
    
    # Convert back to real stress values
    vertices *= Sv
    
    return vertices[:,0], vertices[:,1]  # Sh_min, Sh_max coordinates

def plot_zoback_polygon(Sv, Sh_min, Sh_max, Pp, depth):
    """
    Create Zoback stress polygon plot with current stress state
    """
    # Calculate polygon coordinates
    sh_poly, sH_poly = calculate_zoback_polygon(Sv, Pp)
    
    fig = go.Figure()
    
    # Add stress polygon
    fig.add_trace(go.Scatter(
        x=sh_poly,
        y=sH_poly,
        mode='lines',
        fill='toself',
        fillcolor='rgba(100, 100, 255, 0.2)',
        line=dict(color='blue', width=2),
        name='Possible Stress States'
    ))
    
    # Add current stress point
    fig.add_trace(go.Scatter(
        x=[Sh_min],
        y=[Sh_max],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Current Stress State'
    ))
    
    # Add reference lines
    fig.add_shape(type='line',
        x0=0, y0=0, x1=1.5*Sv, y1=1.5*Sv,
        line=dict(color='black', dash='dash'),
        name='Sh_min = Sh_max'
    )
    
    fig.add_shape(type='line',
        x0=0, y0=Sv, x1=Sv, y1=Sv,
        line=dict(color='green', dash='dot'),
        name='Sh_max = Sv'
    )
    
    fig.add_shape(type='line',
        x0=Sv, y0=0, x1=Sv, y1=Sv,
        line=dict(color='green', dash='dot'),
        name='Sh_min = Sv'
    )
    
    # Add stress regime annotations
    fig.add_annotation(x=0.3*Sv, y=0.3*Sv, text="Normal Faulting",
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.7*Sv, y=1.3*Sv, text="Strike-Slip",
                      showarrow=False, font=dict(size=10))
    fig.add_annotation(x=1.3*Sv, y=1.3*Sv, text="Reverse Faulting",
                      showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title=f'Zoback Stress Polygon at {depth:.0f} ft',
        xaxis_title='Minimum Horizontal Stress (σh, psi)',
        yaxis_title='Maximum Horizontal Stress (σH, psi)',
        showlegend=True,
        height=600
    )
    
    return fig

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
        height=500,
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
        
        return {
            'X': X,
            'Y': Y,
            'Z': Z,
            'R': R,
            'Theta': Theta,
            'Depth': Depth,
            'hoop_stress_fd': hoop_stress_fd,
            'sigma_H_3d': sigma_H_3d,
            'sigma_h_3d': sigma_h_3d,
            'Pp_3d': Pp_3d,
            'azimuth_3d': azimuth_3d,
            'sigma_H': sigma_H,
            'sigma_h': sigma_h,
            'Pp': Pp
        }
    
    except Exception as e:
        st.error(f"Error in stress calculations: {str(e)}")
        return None

def create_3d_visualization(X, Y, Z, hoop_stress_fd, wellbore_radius, current_depth, 
                           current_azimuth, threshold_percent, sigma_H, sigma_h, Pp):
    mid_depth_idx = Z.shape[2] // 2
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=(
            f'3D Wellbore Hoop Stress at {current_depth:.0f} ft',
            f'3D Stress Concentration (>{threshold_percent}% of max)'
        )
    )
    
    # Enhanced polar stress visualization (left plot)
    fig.add_trace(
        go.Surface(
            x=X[:,:,mid_depth_idx],
            y=Y[:,:,mid_depth_idx],
            z=Z[:,:,mid_depth_idx],
            surfacecolor=hoop_stress_fd[:,:,mid_depth_idx],
            colorscale='jet',
            colorbar=dict(x=0.45, title='Hoop Stress (psi)'),
            showscale=True,
            opacity=0.9,
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            )
        ),
        row=1, col=1
    )
    
    # Add wellbore wall as a separate surface
    theta = np.linspace(0, 2*np.pi, 100)
    z_values = np.linspace(Z[:,:,mid_depth_idx].min(), Z[:,:,mid_depth_idx].max(), 10)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    
    x_wall = wellbore_radius * np.cos(theta_grid)
    y_wall = wellbore_radius * np.sin(theta_grid)
    
    # Calculate hoop stress at the wellbore wall
    wall_stress = kirsch_hoop_stress(
        wellbore_radius, 
        theta_grid - np.radians(current_azimuth),
        sigma_H[mid_depth_idx], 
        sigma_h[mid_depth_idx],
        wellbore_radius, 
        Pp[mid_depth_idx]
    )
    
    fig.add_trace(
        go.Surface(
            x=x_wall,
            y=y_wall,
            z=z_grid,
            surfacecolor=wall_stress,
            colorscale='jet',
            showscale=False,
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Enhanced stress concentration plot (right plot)
    threshold = hoop_stress_fd.max() * (threshold_percent/100)
    mask = hoop_stress_fd > threshold
    
    # Create a shale-like wellbore model
    shale_colors = np.zeros_like(hoop_stress_fd)
    shale_colors[mask] = hoop_stress_fd[mask]
    
    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=shale_colors.flatten(),
            isomin=threshold,
            isomax=hoop_stress_fd.max(),
            opacity=0.2,
            surface_count=25,
            colorscale='hot',
            colorbar=dict(title='Hoop Stress (psi)'),
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        row=1, col=2
    )
    
    # Add wellbore outline to both subplots
    theta_wall = np.linspace(0, 2*np.pi, 20)
    x_outline = wellbore_radius * np.cos(theta_wall)
    y_outline = wellbore_radius * np.sin(theta_wall)
    
    for col in [1, 2]:
        # Bottom outline
        fig.add_trace(
            go.Scatter3d(
                x=x_outline,
                y=y_outline,
                z=np.full_like(x_outline, Z[:,:,mid_depth_idx].min()),
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Top outline
        fig.add_trace(
            go.Scatter3d(
                x=x_outline,
                y=y_outline,
                z=np.full_like(x_outline, Z[:,:,mid_depth_idx].max()),
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Vertical lines
        for i in [0, 5, 10, 15]:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_outline[i], x_outline[i]],
                    y=[y_outline[i], y_outline[i]],
                    z=[Z[:,:,mid_depth_idx].min(), Z[:,:,mid_depth_idx].max()],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ),
                row=1, col=col
            )
    
    # Add North and stress direction indicators with improved visibility
    arrow_length = wellbore_radius * 2.5
    
    for col in [1, 2]:
        # North arrow
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, arrow_length],
                z=[current_depth, current_depth],
                mode='lines+text',
                line=dict(color='black', width=6),
                text=['', 'N'],
                textfont=dict(size=14),
                textposition='top center',
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Stress direction
        fig.add_trace(
            go.Scatter3d(
                x=[0, arrow_length*np.cos(np.radians(current_azimuth))],
                y=[0, arrow_length*np.sin(np.radians(current_azimuth))],
                z=[current_depth, current_depth],
                mode='lines+text',
                line=dict(color='red', width=6),
                text=['', 'σH'],
                textfont=dict(size=14),
                textposition='top center',
                showlegend=False
            ),
            row=1, col=col
        )
    
    # Update layout for better viewing
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth (ft)',
            zaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        scene2=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth (ft)',
            zaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    return fig

def create_2d_plots(R, Theta, Depth, hoop_stress_fd, sigma_H_3d, sigma_h_3d, Pp_3d, 
                   wellbore_radius, current_depth, current_azimuth):
    mid_depth_idx = Depth.shape[2] // 2
    theta_offset = np.radians(current_azimuth)
    
    # Create polar plot data with higher resolution
    r_fine = np.linspace(wellbore_radius, 5*wellbore_radius, 200)
    theta_fine = np.linspace(0, 2*np.pi, 360)
    R_2D, Theta_2D = np.meshgrid(r_fine, theta_fine)
    
    hoop_stress_2D = kirsch_hoop_stress(
        R_2D, Theta_2D - theta_offset,
        sigma_H_3d[0,0,mid_depth_idx], 
        sigma_h_3d[0,0,mid_depth_idx],
        wellbore_radius, 
        Pp_3d[0,0,mid_depth_idx]
    )
    
    # Create polar figure with improved styling
    polar_fig = go.Figure()
    
    polar_fig.add_trace(go.Scatterpolar(
        r=R_2D.flatten(),
        theta=np.degrees(Theta_2D).flatten(),
        mode='markers',
        marker=dict(
            color=hoop_stress_2D.flatten(),
            colorscale='jet',
            showscale=True,
            size=6,
            opacity=0.8,
            colorbar=dict(
                title='Hoop Stress (psi)',
                thickness=20,
                len=0.75
            ),
            cmin=hoop_stress_2D.min(),
            cmax=hoop_stress_2D.max()
        ),
        hoverinfo='r+theta+text',
        text=[f'Stress: {stress:.0f} psi' for stress in hoop_stress_2D.flatten()],
        showlegend=False
    ))
    
    # Add wellbore wall
    polar_fig.add_trace(go.Scatterpolar(
        r=[wellbore_radius]*360,
        theta=np.linspace(0, 360, 360),
        mode='lines',
        line=dict(color='black', width=3),
        name='Wellbore',
        hoverinfo='none'
    ))
    
    # Add compass directions with better styling
    compass_radius = 1.15 * R_2D.max()
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        polar_fig.add_trace(go.Scatterpolar(
            r=[wellbore_radius, compass_radius],
            theta=[angle, angle],
            mode='lines+text',
            line=dict(color='black', width=2),
            text=['', label],
            textfont=dict(size=14),
            textposition='top center',
            showlegend=False
        ))
    
    # Add stress direction marker with arrow
    polar_fig.add_trace(go.Scatterpolar(
        r=[wellbore_radius, wellbore_radius*1.8],
        theta=[current_azimuth, current_azimuth],
        mode='lines+text',
        line=dict(color='red', width=4),
        text=['', 'σH'],
        textfont=dict(size=14),
        textposition='top center',
        name='Max Stress Direction'
    ))
    
    polar_fig.update_layout(
        title=f'Polar Stress Distribution at {current_depth:.0f} ft<br><sup>Azimuth: {current_azimuth:.1f}°</sup>',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, R_2D.max()],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickfont=dict(size=12)
            ),
            bgcolor='#f0f0f0'
        ),
        showlegend=True,
        height=550,
        margin=dict(t=100)
    )
    
    # Create Cartesian figure with fixed contour plot
    cartesian_fig = go.Figure()
    
    # Convert to Cartesian coordinates
    X_2D = R_2D * np.cos(Theta_2D)
    Y_2D = R_2D * np.sin(Theta_2D)
    
    # Add contour plot with improved settings
    cartesian_fig.add_trace(go.Contour(
        x=X_2D[0,:],
        y=Y_2D[:,0],
        z=hoop_stress_2D,
        colorscale='jet',
        contours=dict(
            coloring='heatmap',
            showlines=True,
            start=hoop_stress_2D.min(),
            end=hoop_stress_2D.max(),
            size=(hoop_stress_2D.max()-hoop_stress_2D.min())/20
        ),
        colorbar=dict(
            title='Hoop Stress (psi)',
            thickness=20,
            len=0.75
        ),
        line=dict(width=0.5, color='white'),
        hoverinfo='x+y+z'
    ))
    
    # Add wellbore circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = wellbore_radius * np.cos(theta_circle)
    y_circle = wellbore_radius * np.sin(theta_circle)
    
    cartesian_fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        line=dict(color='black', width=3),
        name='Wellbore',
        hoverinfo='none'
    ))
    
    # Add North arrow
    arrow_length = wellbore_radius * 1.5
    cartesian_fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, arrow_length],
        mode='lines+text',
        line=dict(color='black', width=3),
        text=['', 'N'],
        textfont=dict(size=14),
        textposition='top center',
        showlegend=False
    ))
    
    # Add stress direction indicator with arrow
    cartesian_fig.add_trace(go.Scatter(
        x=[0, arrow_length*np.cos(theta_offset)],
        y=[0, arrow_length*np.sin(theta_offset)],
        mode='lines+text',
        line=dict(color='red', width=3),
        text=['', 'σH'],
        textfont=dict(size=14),
        textposition='top center',
        name='Max Stress Direction'
    ))
    
    # Add quadrant lines for reference
    for angle in [0, 90, 180, 270]:
        x_line = arrow_length * 1.2 * np.cos(np.radians(angle))
        y_line = arrow_length * 1.2 * np.sin(np.radians(angle))
        cartesian_fig.add_trace(go.Scatter(
            x=[0, x_line],
            y=[0, y_line],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='none'
        ))
    
    cartesian_fig.update_layout(
        title=f'Cartesian Stress Distribution at {current_depth:.0f} ft<br><sup>Azimuth: {current_azimuth:.1f}°</sup>',
        xaxis=dict(
            title='X (ft)',
            scaleanchor="y",
            scaleratio=1,
            constrain='domain'
        ),
        yaxis=dict(
            title='Y (ft)',
            scaleanchor="x",
            scaleratio=1
        ),
        height=550,
        showlegend=True,
        margin=dict(t=100),
        plot_bgcolor='#f0f0f0'
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
        height=700,
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

def create_3d_stress_distribution(X, Y, Z, hoop_stress, wellbore_radius, current_depth, current_azimuth):
    """Create interactive 3D stress distribution plot with zoom capabilities"""
    fig = go.Figure()
    
    # Create a 3D surface plot of the stress distribution
    fig.add_trace(go.Surface(
        x=X[:,:,0],
        y=Y[:,:,0],
        z=Z[:,:,0],
        surfacecolor=hoop_stress[:,:,0],
        colorscale='jet',
        colorbar=dict(title='Hoop Stress (psi)'),
        opacity=0.9,
        name='Stress Distribution'
    ))
    
    # Add wellbore wall
    theta = np.linspace(0, 2*np.pi, 100)
    z_values = np.linspace(Z[:,:,0].min(), Z[:,:,0].max(), 10)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    
    x_wall = wellbore_radius * np.cos(theta_grid)
    y_wall = wellbore_radius * np.sin(theta_grid)
    
    fig.add_trace(go.Surface(
        x=x_wall,
        y=y_wall,
        z=z_grid,
        surfacecolor=np.zeros_like(z_grid),  # Uniform color for wellbore
        colorscale=[[0, 'gray'], [1, 'gray']],
        showscale=False,
        opacity=0.7,
        name='Wellbore'
    ))
    
    # Add direction indicators
    arrow_length = wellbore_radius * 2
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, arrow_length],
        z=[current_depth, current_depth],
        mode='lines+text',
        line=dict(color='black', width=6),
        text=['', 'N'],
        textposition='top center',
        name='North'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, arrow_length*np.cos(np.radians(current_azimuth))],
        y=[0, arrow_length*np.sin(np.radians(current_azimuth))],
        z=[current_depth, current_depth],
        mode='lines+text',
        line=dict(color='red', width=6),
        text=['', 'σH'],
        textposition='top center',
        name='Max Stress Direction'
    ))
    
    # Add depth reference plane
    fig.add_trace(go.Surface(
        x=X[:,:,0],
        y=Y[:,:,0],
        z=np.full_like(X[:,:,0], current_depth),
        colorscale=[[0, 'rgba(0,0,0,0.1)'], [1, 'rgba(0,0,0,0.1)']],
        showscale=False,
        opacity=0.3,
        name=f'Depth: {current_depth:.0f} ft'
    ))
    
    # Update layout for better interactivity
    fig.update_layout(
        title=f'3D Interactive Stress Distribution at {current_depth:.0f} ft',
        scene=dict(
            xaxis_title='X (ft)',
            yaxis_title='Y (ft)',
            zaxis_title='Depth (ft)',
            zaxis=dict(autorange="reversed"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Add interactive controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"camera.eye.x": 1.5, "camera.eye.y": 1.5, "camera.eye.z": 0.8}],
                        label="Default View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"camera.eye.x": 0, "camera.eye.y": 0, "camera.eye.z": 2}],
                        label="Top View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"camera.eye.x": 0, "camera.eye.y": 2, "camera.eye.z": 0}],
                        label="Side View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"camera.eye.x": 2, "camera.eye.y": 0, "camera.eye.z": 0}],
                        label="Front View",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    return fig

def main():
    # Main window layout with columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
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
                Sv_curve = st.selectbox("Vertical Stress (Sv) curve", 
                                      options=available_curves,
                                      index=min(3, len(available_curves)-1))
                
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
                Sv_data = las[Sv_curve][valid_depth_indices]
                
                # Run calculations button
                if st.button("Run Analysis"):
                    with st.spinner("Calculating stresses..."):
                        # Create stress vs depth plot
                        with col2:
                            st.subheader("Stress Field Profile")
                            stress_plot = plot_stress_vs_depth(depth_data, sigma_H_data, sigma_h_data, Pp_data, min_depth, max_depth)
                            st.plotly_chart(stress_plot, use_container_width=True)
                        
                        # Run calculations
                        results = calculate_stresses(
                            depth_data, sigma_H_data, sigma_h_data, Pp_data, azimuth_data,
                            min_depth, max_depth, wellbore_radius, resolution
                        )
                        
                        if results:
                            mid_depth_idx = len(np.linspace(min_depth, max_depth, 5)) // 2
                            current_depth = np.linspace(min_depth, max_depth, 5)[mid_depth_idx]
                            current_azimuth = results['azimuth_3d'][0,0,mid_depth_idx]
                            
                            # Get current stress values for polygon
                            Sv = np.interp(current_depth, depth_data, Sv_data)
                            Sh_min = results['sigma_h'][mid_depth_idx]
                            Sh_max = results['sigma_H'][mid_depth_idx]
                            Pp = results['Pp'][mid_depth_idx]
                            
                            # Show raw stress data
                            with col2:
                                st.subheader("Analysis Results")
                                st.write(f"Current depth: {current_depth:.0f} ft | Azimuth: {current_azimuth:.1f}°")
                                
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric(f"Sv at {current_depth:.0f} ft", f"{Sv:.0f} psi")
                                with col_b:
                                    st.metric(f"σH at {current_depth:.0f} ft", f"{Sh_max:.0f} psi")
                                with col_c:
                                    st.metric(f"σh at {current_depth:.0f} ft", f"{Sh_min:.0f} psi")
                                with col_d:
                                    st.metric(f"Pp at {current_depth:.0f} ft", f"{Pp:.0f} psi")
                            
                            # Create analysis plots
                            with col2:
                                st.subheader("Zoback Stress Polygon")
                                polygon_fig = plot_zoback_polygon(Sv, Sh_min, Sh_max, Pp, current_depth)
                                st.plotly_chart(polygon_fig, use_container_width=True)
                                
                                st.subheader("3D Visualization")
                                fig_3d = create_3d_visualization(
                                    results['X'], results['Y'], results['Z'], results['hoop_stress_fd'],
                                    wellbore_radius, current_depth, current_azimuth,
                                    threshold_percent,
                                    results['sigma_H'], results['sigma_h'], results['Pp']
                                )
                                st.plotly_chart(fig_3d, use_container_width=True)
                                
                                st.subheader("Interactive 3D Stress Distribution")
                                interactive_3d_fig = create_3d_stress_distribution(
                                    results['X'], results['Y'], results['Z'], 
                                    results['hoop_stress_fd'],
                                    wellbore_radius, current_depth, current_azimuth
                                )
                                st.plotly_chart(interactive_3d_fig, use_container_width=True)
                                
                                st.subheader("2D Plots")
                                polar_fig, cartesian_fig = create_2d_plots(
                                    results['R'], results['Theta'], results['Depth'], results['hoop_stress_fd'],
                                    results['sigma_H_3d'], results['sigma_h_3d'], results['Pp_3d'],
                                    wellbore_radius, current_depth, current_azimuth
                                )
                                st.plotly_chart(polar_fig, use_container_width=True)
                                st.plotly_chart(cartesian_fig, use_container_width=True)
                                
                                st.subheader("Stress Profiles")
                                fig_profiles = create_stress_profiles(
                                    results['R'], results['Theta'], results['Depth'], results['hoop_stress_fd'],
                                    results['sigma_H_3d'], results['sigma_h_3d'], results['Pp_3d'],
                                    wellbore_radius, current_depth, current_azimuth
                                )
                                st.plotly_chart(fig_profiles, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing LAS file: {str(e)}")
        else:
            st.warning("Please upload a LAS file to proceed")

if __name__ == "__main__":
    main()
