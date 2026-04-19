import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.express as px
import webbrowser
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# For static 3D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# Creating the data

def create_data(crop_name, df_final, resampled_df):
    # Filter for tomato_healthy class
    real_data = df_final[df_final['label'] == crop_name].copy()
    synthetic_data = resampled_df[resampled_df['label'] == crop_name].copy()
    
    print(f"{crop_name} real data shape:", real_data.shape)
    print(f"{crop_name} synthetic data shape:", synthetic_data.shape)

    # Select numerical columns (values only)
    numerical_cols = real_data.select_dtypes(include=[np.number]).columns
    X_real = real_data[numerical_cols].copy()
    X_synthetic = synthetic_data[numerical_cols].copy()

    # Standardize the features
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_synthetic_scaled = scaler.transform(X_synthetic)  # Use same scaler for consistency
    
    return X_real_scaled, X_synthetic_scaled



def plot_3D(crop_name, plot_df, ptype=""):
    # Split plot_df into Real and Synthetic for explicit trace control
    real_df = plot_df[plot_df['Dataset'] == 'Real']
    synthetic_df = plot_df[plot_df['Dataset'] == 'Synthetic']

    # Create figure with separate traces for Real and Synthetic
    fig = go.Figure()

    # Add Real data trace
    fig.add_trace(
        go.Scatter3d(
            x=real_df[ptype+'_1'],
            y=real_df[ptype+'_2'],
            z=real_df[ptype+'_3'],
            mode='markers',
            name='Real',
            marker=dict(size=5, color='blue', opacity=0.6)
        )
    )

    # Add Synthetic data trace
    fig.add_trace(
        go.Scatter3d(
            x=synthetic_df[ptype+'_1'],
            y=synthetic_df[ptype+'_2'],
            z=synthetic_df[ptype+'_3'],
            mode='markers',
            name='Synthetic',
            marker=dict(size=2, color='red', opacity=0.6)
        )
    )
    

    tickfontSize = 12
    titlefontSize = 15
    # Update layout with Times New Roman, font size 30 for everything
    fig.update_layout(
        title=dict(
            text=f'{ptype} 3D Projection of {crop_name} Class (Real vs Synthetic)',
            font=dict(family="Times New Roman", size=30)
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text=f'{ptype} Component 1', font=dict(family="Times New Roman", size=titlefontSize)),
                tickfont=dict(family="Times New Roman", size=tickfontSize)
            ),
            yaxis=dict(
                title=dict(text=f'{ptype} Component 2', font=dict(family="Times New Roman", size=titlefontSize)),
                tickfont=dict(family="Times New Roman", size=tickfontSize)
            ),
            zaxis=dict(
                title=dict(text=f'{ptype} Component 3', font=dict(family="Times New Roman", size=titlefontSize)),
                tickfont=dict(family="Times New Roman", size=tickfontSize)
            )
        ),
        legend=dict(font=dict(family="Times New Roman", size=30)),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )

    # Show plot in new browser window
    print(f"Showing {ptype} plot in new window")
    fig.show(renderer="browser")
    
    return None




def plot_3D_static(crop_name, plot_df, ptype=""):
    # Split plot_df into Real and Synthetic
    real_df = plot_df[plot_df['Dataset'] == 'Real']
    synthetic_df = plot_df[plot_df['Dataset'] == 'Synthetic']

    # Create a figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot Real data
    ax.scatter(
        real_df[ptype+'_1'],
        real_df[ptype+'_2'],
        real_df[ptype+'_3'],
        c ='blue', s=50, alpha=0.6, label="Real"
    )

    # Plot Synthetic data
    ax.scatter(
        synthetic_df[ptype+'_1'],
        synthetic_df[ptype+'_2'],
        synthetic_df[ptype+'_3'],
        c='red', s=20, alpha=0.6, label="Synthetic"
    )

    # Set titles and labels with Times New Roman font
    # ax.set_title(f'{ptype} 3D Projection of {crop_name} Class (Real vs Synthetic)', fontfamily="Times New Roman", fontsize=30)
    
    Font_size_ticks = 10
    Font_size_xyz = 20
    Font_size_legend = 20
    
    ax.set_xlabel(f'Component 1', fontfamily="Times New Roman", fontsize = Font_size_xyz)
    ax.set_ylabel(f'Component 2', fontfamily="Times New Roman", fontsize = Font_size_xyz)
    ax.set_zlabel(f'Component 3', fontfamily="Times New Roman", fontsize = Font_size_xyz)

    # Set tick label font
    ax.tick_params(axis='both', labelsize = Font_size_ticks)  
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontname("Times New Roman")

    # Legend
    # ax.legend(prop={'family': "Times New Roman", 'size': Font_size_legend}, loc="lower left", bbox_to_anchor=(0.20, 0.20))

    plt.show()




# PCA-------------------------------------------------------------------
def apply_pca(X_real_scaled, X_synthetic_scaled):
    # Apply PCA
    pca = PCA(n_components=3, random_state=42)
    real_pca = pca.fit_transform(X_real_scaled)
    synthetic_pca = pca.transform(X_synthetic_scaled)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # Combine results for plotting
    plot_df = pd.DataFrame({
        'PCA_1': np.concatenate([real_pca[:, 0], synthetic_pca[:, 0]]),
        'PCA_2': np.concatenate([real_pca[:, 1], synthetic_pca[:, 1]]),
        'PCA_3': np.concatenate([real_pca[:, 2], synthetic_pca[:, 2]]),
        'Dataset': ['Real'] * len(real_pca) + ['Synthetic'] * len(synthetic_pca)
    })
    
    return plot_df


# TSNE------------------------------------------------------------------------
def apply_TSNE(X_real_scaled, X_synthetic_scaled):
    # Apply t-SNE
    tsne = TSNE(n_components=3, random_state=42, n_iter=1000, perplexity=min(30, len(X_real_scaled)-1), init="pca", learning_rate="auto")
    real_tsne = tsne.fit_transform(X_real_scaled)
    synthetic_tsne = tsne.fit_transform(X_synthetic_scaled)  # Separate fit for synthetic data

    # Combine results for plotting
    plot_df = pd.DataFrame({
        'TSNE_1': np.concatenate([real_tsne[:, 0], synthetic_tsne[:, 0]]),
        'TSNE_2': np.concatenate([real_tsne[:, 1], synthetic_tsne[:, 1]]),
        'TSNE_3': np.concatenate([real_tsne[:, 2], synthetic_tsne[:, 2]]),
        'Dataset': ['Real'] * len(real_tsne) + ['Synthetic'] * len(synthetic_tsne)
    })
    
    return plot_df



# UMAP-----------------------------------------------------------------
def apply_UMAP(X_real_scaled, X_synthetic_scaled):
    umap_model = umap.UMAP(
        n_components=3,
        random_state=42,
        n_neighbors=min(15, len(X_real_scaled)-1),
        min_dist=0.1
    )
    real_umap = umap_model.fit_transform(X_real_scaled)
    synthetic_umap = umap_model.transform(X_synthetic_scaled)
    
    # Combine results for plotting
    plot_df = pd.DataFrame({
        'UMAP_1': np.concatenate([real_umap[:, 0], synthetic_umap[:, 0]]),
        'UMAP_2': np.concatenate([real_umap[:, 1], synthetic_umap[:, 1]]),
        'UMAP_3': np.concatenate([real_umap[:, 2], synthetic_umap[:, 2]]),
        'Dataset': ['Real'] * len(real_umap) + ['Synthetic'] * len(synthetic_umap)
    })
    
    return plot_df


