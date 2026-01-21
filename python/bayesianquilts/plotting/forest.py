import matplotlib.pyplot as plt
import numpy as np


def plot_top_k_forest(data, k, parameter_names=None):
    """
    Calculates statistics, ranks parameters by mean magnitude, and generates a forest plot.

    Args:
        data (np.ndarray): An S x P numpy array with S samples of P coefficients.
        k (int): The number of top parameters to plot.
        parameter_names (list of str, optional): A list of names for the P parameters.
                                                 If None, parameters will be labeled by index.
    """
    # --- 1. Input Validation ---
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array.")
    
    num_samples, num_parameters = data.shape
    
    if k > num_parameters:
        print(f"Warning: k ({k}) is larger than the number of parameters ({num_parameters})." \
              f" Plotting all {num_parameters} parameters.")
        k = num_parameters

    if parameter_names and len(parameter_names) != num_parameters:
        raise ValueError("The length of 'parameter_names' must match the number of columns in 'data'.")

    # --- 2. Calculate Mean and Standard Deviation ---
    # axis=0 calculates stats along the columns (for each parameter)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # --- 3. Rank Parameters by Absolute Mean ---
    # Get the indices that would sort the array based on the absolute mean values
    sorted_indices = np.argsort(np.abs(means))[::-1] # [::-1] for descending order

    # --- 4. Select Top K Parameters ---
    top_k_indices = sorted_indices[:k]
    top_k_means = means[top_k_indices]
    top_k_stds = stds[top_k_indices]
    
    if parameter_names:
        top_k_names = [parameter_names[i] for i in top_k_indices]
    else:
        top_k_names = [f"Parameter {i}" for i in top_k_indices]

    # --- 5. Generate the Forest Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, k * 0.5)) # Adjust figure size based on k

    # y-positions for the plots
    y_pos = np.arange(len(top_k_names))

    # Plot the error bars (standard deviation)
    # The 'xerr' parameter defines the horizontal error bar width.
    ax.errorbar(x=top_k_means, y=y_pos, xerr=top_k_stds, fmt='o',
                capsize=5, markersize=8, elinewidth=2,
                color='darkcyan', ecolor='lightseagreen',
                label='Mean with Std. Dev.')

    # Add a vertical line at x=0 for reference
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    # --- 6. Formatting and Labels ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_k_names, fontsize=12)
    ax.invert_yaxis()  # Invert y-axis to have the top-ranked parameter at the top

    ax.set_xlabel("Coefficient Value", fontsize=14)
    ax.set_ylabel("Parameters", fontsize=14)
    ax.set_title(f"Forest Plot of Top {k} Ranked Parameters", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax, top_k_names
    # plt.show()




