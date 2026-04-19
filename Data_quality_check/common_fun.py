# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:52:45 2025

@author: IISTDST
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import textwrap
from sklearn.ensemble import RandomForestClassifier


# ****************************** data normalization *****************************************
def min_max_scaling(data):
    """
    Apply Min-Max Scaling to normalize data to a range [0, 1].
    
    Parameters:
        data (numpy.ndarray or pandas.DataFrame): Input data matrix (samples x wavelengths)
        
    Returns:
        numpy.ndarray: Min-Max scaled data
    """
    # Convert to numpy array if input is a DataFrame
    data = data.to_numpy() if isinstance(data, pd.DataFrame) else np.array(data)
    
    # Compute min and max for each wavelength (column)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Avoid division by zero
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1
    
    # Apply min-max scaling
    scaled_data = (data - min_vals) / denominator
    return scaled_data


def standard_normal_variate(data):
    """
    Apply Standard Normal Variate (SNV) normalization to remove sensor-specific or environmental biases.
    
    Parameters:
        data (numpy.ndarray or pandas.DataFrame): Input data matrix (samples x wavelengths)
        
    Returns:
        numpy.ndarray: SNV normalized data
    """
    # Convert to numpy array if input is a DataFrame
    data = data.to_numpy() if isinstance(data, pd.DataFrame) else np.array(data)
    
    # Initialize output array
    snv_data = np.zeros_like(data, dtype=float)
    
    # Apply SNV for each sample (row)
    for i in range(data.shape[0]):
        sample = data[i, :]
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        # Avoid division by zero
        if std == 0:
            std = 1
        snv_data[i, :] = (sample - mean) / std
    return snv_data


def normalize_datasets(datasets, method='min_max'):
    """
    Normalize X1 and X2 data for each crop in the datasets dictionary.
    
    Parameters:
        datasets (dict): Dictionary with crop names as keys and tuples (X1, X2, columns) as values
        method (str): Normalization method ('min_max' or 'snv')
        
    Returns:
        dict: Dictionary with normalized data in the same format
    """
    normalized_datasets = {}
    
    # Select normalization function
    normalize_fn = min_max_scaling if method == 'min_max' else standard_normal_variate
    
    # Process each crop
    for crop, (X1, X2, columns) in datasets.items():
        # Normalize X1 and X2
        X1_normalized = normalize_fn(X1)
        X2_normalized = normalize_fn(X2)
        
        # Convert back to DataFrame to preserve column names
        X1_normalized_df = pd.DataFrame(X1_normalized, columns=columns)
        X2_normalized_df = pd.DataFrame(X2_normalized, columns=columns)
        
        # Store in the same format
        normalized_datasets[crop] = (X1_normalized_df, X2_normalized_df, columns)
    
    return normalized_datasets



# *******************************Univariate Measures***************************

# Function to calculate Hellinger distance for two discrete distributions
def hellinger_distance(p, q):
    return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))



# Function to calculate hellinger_distance between real and synthetic data
def calculate_hellinger_distances(real_data, synthetic_data, columns, n_bins):
    hellinger_distances = []
    
    for column in columns:
        # Discretize the data
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        
        # Fit the discretizer on the real data and transform both real and synthetic data
        real_discretized = discretizer.fit_transform(real_data[[column]]).flatten()
        synthetic_discretized = discretizer.transform(synthetic_data[[column]]).flatten()
        
        # Calculate the proportion of counts in each bin
        real_counts, _ = np.histogram(real_discretized, bins=n_bins, range=(0, n_bins), density=False)
        synthetic_counts, _ = np.histogram(synthetic_discretized, bins=n_bins, range=(0, n_bins), density=False)
        
        real_probs = real_counts / real_counts.sum()
        synthetic_probs = synthetic_counts / synthetic_counts.sum()
        
        # Compute Hellinger distance
        distance = hellinger_distance(real_probs, synthetic_probs)
        hellinger_distances.append(distance)
    
    return hellinger_distances
    
    # Ensure that the number of Hellinger distances matches the number of columns
    assert len(hellinger_distances) == len(columns)
    
    
    



# *******************************Bivariate Measures************************************

# Function to calculate pairwise_correlation_difference between real and synthetic data
def pairwise_correlation_difference(df1, df2):
   
    # Compute the correlation matrices
    corr_X = df1.corr().values
    corr_Y = df2.corr().values
    
    # Calculate the difference between the correlation matrices
    diff = corr_X - corr_Y
    
    # Compute the Frobenius norm of the difference matrix
    frobenius_norm = np.linalg.norm(diff, 'fro')
    
    return frobenius_norm




# ******************************Multivariate Measures***********************************
def calculate_propensity_score(real_data, synthetic_data):
   # Combine datasets
   combined_data = pd.concat([real_data, synthetic_data])
   labels = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])

   # Train a classifier
   classifier = RandomForestClassifier(n_estimators=100, random_state=42)
   classifier.fit(combined_data, labels)

   # Predict probabilities
   predicted_probs = classifier.predict_proba(combined_data)[:, 1]

   # Calculate pMSE propensity score
   pMSE = np.mean((predicted_probs - 0.5) ** 2)

   return pMSE



# ******************************All Measures simeltaneously***********************************
# function to wrap labels in the boxplot
def wrap_labels(labels, width=10):
    return [textwrap.fill(label, width) for label in labels]


# Funciton to plot hellinger distances, calulate PWC_disfference, and Propensity Score (pMSE)
def cal_bi_multi_variate_plot_hellinger_distances(base_path, datasets, n_bins=10, slug="", save_img=False):
    all_distances = []
    labels = []
    pwc_diff = []
    PMSE =[]

    for dataset_name, (real_data, synthetic_data, columns) in datasets.items():
        distances = calculate_hellinger_distances(real_data, synthetic_data, columns, n_bins)
        all_distances.append(distances)
        labels.append(dataset_name)
        pwc_diff.append(pairwise_correlation_difference(real_data, synthetic_data))
        PMSE.append(calculate_propensity_score(real_data, synthetic_data))
        
    print("labels:", labels)
    print("pwc_diff:", pwc_diff)
    wrapped_labels = [f"{label}\n{pwc:0.2f}" for label, pwc in zip(labels, pwc_diff)]
    print("wrapped_labels:\n", wrapped_labels)
    
    # sys.exit(0)

    # --- Box Plotting ---
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig, ax = plt.subplots(figsize=(15, 15))
    fontsize = 50
    
    # --- Create and Customize the Box Plot ---
    bplot = ax.boxplot(all_distances, vert=True, patch_artist=True, labels=wrapped_labels)
    
    # Define a list of 6 colors for your 6 groups
    # colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845', '#2a9d8f']
    colors = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
    
    # 1. Customize the boxes
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        patch.set_alpha(0.8)
    
    # 2. Customize the whiskers and caps
    for whisker, cap in zip(bplot['whiskers'], bplot['caps']):
        whisker.set(color='#404040', linewidth=2, linestyle=':')
        cap.set(color='#404040', linewidth=2)
    
    # 3. Customize the medians
    for median in bplot['medians']:
        median.set(color='black', linewidth=3)
    
    # 4. Customize the fliers (outliers)
    for flier in bplot['fliers']:
        flier.set(marker='D', markerfacecolor='#e7298a', markeredgecolor='none', markersize=10, alpha=0.7)
    
    # --- Apply Titles and Labels ---
    ax.set_title(f'{slug}', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('\n Hellinger Distance', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel('\n Crops with pairwise correlation difference', fontsize=fontsize, fontweight='bold')
    
    # Set basic tick parameters
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Set the font weight for tick labels separately
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # --- Add Grid and Customize Spines ---
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    ax.set_ylim(0, 0.8) # Adjust these values as needed

    plt.tight_layout()
    plt.show()

    
    # Saving the box plot image 
    if save_img:
        print("\nSaving image at location", base_path, "\n")
        plt.savefig(base_path + fr"{slug}.png", dpi=500, bbox_inches="tight")  # Increase DPI for higher quality
    plt.show()
    
    for label, pwc in zip(labels, pwc_diff):
       print(f"Bivariate Pairwise Correlation of {label} original vs synthetic data: {pwc:.2f}")
    
    print("\n")
    for label, PMSE in zip(labels, PMSE):
       print(f"pMSE of {label} original vs synthetic data: {PMSE:.2f}")





# ****************************** SAM calculations using average spectras ***********************************

def spectral_angle(spectrum1, spectrum2):
    """
    Compute the spectral angle between two spectra.
    
    Parameters:
        spectrum1 (numpy.ndarray): First spectrum (1 x wavelengths)
        spectrum2 (numpy.ndarray): Second spectrum (1 x wavelengths)
        
    Returns:
        float: Spectral angle in radians
    """
    # Ensure inputs are 1D arrays
    spectrum1 = np.ravel(spectrum1)
    spectrum2 = np.ravel(spectrum2)
    
    # Compute dot product and norms
    dot_product = np.dot(spectrum1, spectrum2)
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return np.pi / 2  # Maximum angle (90 degrees)
    
    # Compute cosine of the angle
    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
    
    # Return angle in radians
    return np.arccos(cos_angle)


def spectral_angle_mapper_mean(datasets):
    """
    Compute SAM between the mean spectra of X1 (original) and X2 (synthetic) for each crop.
    Assumes input data is already normalized.
    
    Parameters:
        datasets (dict): Dictionary with crop names as keys and tuples (X1, X2, columns) as values
        
    Returns:
        dict: Dictionary with crop names as keys and tuples (X1_mean, X2_mean, columns, angle)
              where X1_mean and X2_mean are DataFrames with one row (mean spectrum)
    """
    sam_results = {}
    
    # Process each crop
    for crop, (X1, X2, columns) in datasets.items():
        # Convert to numpy arrays
        X1_array = X1.to_numpy() if isinstance(X1, pd.DataFrame) else np.array(X1)
        X2_array = X2.to_numpy() if isinstance(X2, pd.DataFrame) else np.array(X2)
        
        # Compute mean spectra
        X1_mean = np.mean(X1_array, axis=0)
        X2_mean = np.mean(X2_array, axis=0)
        
        # Compute spectral angle between mean spectra
        angle = spectral_angle(X1_mean, X2_mean)
        
        # Convert mean spectra to DataFrames for consistency
        X1_mean_df = pd.DataFrame([X1_mean], columns=columns)
        X2_mean_df = pd.DataFrame([X2_mean], columns=columns)
        
        # Store results
        sam_results[crop] = (X1_mean_df, X2_mean_df, columns, angle)
    
    return sam_results








