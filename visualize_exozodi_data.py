#!/usr/bin/env python3
"""
Visualization script for exozodi data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_exozodi_data():
    """Load exozodi data from the .npy file."""
    data_path = 'PPop/ExozodiModels/ExozodiNominal.npy'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    
    # Handle different data shapes
    if len(data.shape) == 2:
        # Data is (1, N) - extract the first row
        exozodi_levels = data[0]
    else:
        # Data is (N,) - use directly
        exozodi_levels = data
    
    return exozodi_levels

def analyze_exozodi_data(exozodi_levels):
    """Analyze the exozodi data and return statistics."""
    stats_dict = {
        'count': len(exozodi_levels),
        'min': np.min(exozodi_levels),
        'max': np.max(exozodi_levels),
        'mean': np.mean(exozodi_levels),
        'median': np.median(exozodi_levels),
        'std': np.std(exozodi_levels),
        'q25': np.percentile(exozodi_levels, 25),
        'q75': np.percentile(exozodi_levels, 75),
        'iqr': np.percentile(exozodi_levels, 75) - np.percentile(exozodi_levels, 25),
    }
    
    # Calculate additional statistics
    stats_dict['geometric_mean'] = np.exp(np.mean(np.log(exozodi_levels[exozodi_levels > 0])))
    stats_dict['log_mean'] = np.mean(np.log(exozodi_levels[exozodi_levels > 0]))
    stats_dict['log_std'] = np.std(np.log(exozodi_levels[exozodi_levels > 0]))
    
    return stats_dict

def create_exozodi_visualization():
    """Create comprehensive visualization of exozodi data."""
    
    # Load data
    print("Loading exozodi data...")
    exozodi_levels = load_exozodi_data()
    
    # Analyze data
    print("Analyzing data...")
    stats_dict = analyze_exozodi_data(exozodi_levels)
    
    # Print summary statistics
    print("\n=== Exozodi Data Summary ===")
    print(f"Number of systems: {stats_dict['count']:,}")
    print(f"Range: {stats_dict['min']:.2e} to {stats_dict['max']:.2e}")
    print(f"Mean: {stats_dict['mean']:.2f}")
    print(f"Median: {stats_dict['median']:.2f}")
    print(f"Standard deviation: {stats_dict['std']:.2f}")
    print(f"Geometric mean: {stats_dict['geometric_mean']:.2f}")
    print(f"Log-space mean: {stats_dict['log_mean']:.2f}")
    print(f"Log-space std: {stats_dict['log_std']:.2f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Histogram (linear scale)
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(exozodi_levels, bins=100, alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('Exozodi Level')
    plt.ylabel('Density')
    plt.title('Exozodi Level Distribution (Linear Scale)')
    plt.axvline(stats_dict['mean'], color='red', linestyle='--', label=f'Mean: {stats_dict["mean"]:.2f}')
    plt.axvline(stats_dict['median'], color='orange', linestyle='--', label=f'Median: {stats_dict["median"]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Histogram (log scale)
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(exozodi_levels, bins=np.logspace(np.log10(stats_dict['min']), np.log10(stats_dict['max']), 100), 
             alpha=0.7, edgecolor='black', density=True)
    plt.xscale('log')
    plt.xlabel('Exozodi Level')
    plt.ylabel('Density')
    plt.title('Exozodi Level Distribution (Log Scale)')
    plt.axvline(stats_dict['geometric_mean'], color='red', linestyle='--', 
                label=f'Geometric Mean: {stats_dict["geometric_mean"]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Log-normal fit
    ax3 = plt.subplot(3, 3, 3)
    log_data = np.log(exozodi_levels[exozodi_levels > 0])
    plt.hist(log_data, bins=50, alpha=0.7, edgecolor='black', density=True, label='Data')
    
    # Fit log-normal distribution
    mu, sigma = stats.norm.fit(log_data)
    x = np.linspace(log_data.min(), log_data.max(), 100)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', linewidth=2, label=f'Log-normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
    
    plt.xlabel('Log(Exozodi Level)')
    plt.ylabel('Density')
    plt.title('Log-Space Distribution with Log-Normal Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box plot
    ax4 = plt.subplot(3, 3, 4)
    plt.boxplot(exozodi_levels, vert=True)
    plt.ylabel('Exozodi Level')
    plt.title('Box Plot of Exozodi Levels')
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(3, 3, 5)
    sorted_data = np.sort(exozodi_levels)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cumulative, 'b-', linewidth=2)
    plt.xlabel('Exozodi Level')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # Add percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = float(np.percentile(exozodi_levels, p))
        plt.axvline(value, color='red', alpha=0.5, linestyle=':')
        plt.text(value, 0.5, f'{p}%', rotation=90, verticalalignment='center')
    
    # 6. Log-space cumulative distribution
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(sorted_data, cumulative, 'b-', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Exozodi Level')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 7. Q-Q plot (log-normal)
    ax7 = plt.subplot(3, 3, 7)
    stats.probplot(log_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Log-Normal)')
    plt.grid(True, alpha=0.3)
    
    # 8. Exozodi level ranges
    ax8 = plt.subplot(3, 3, 8)
    
    # Define ranges
    ranges = [
        (0, 1, 'Very Low (0-1)'),
        (1, 10, 'Low (1-10)'),
        (10, 100, 'Moderate (10-100)'),
        (100, 1000, 'High (100-1000)'),
        (1000, float('inf'), 'Very High (>1000)')
    ]
    
    counts = []
    labels = []
    for min_val, max_val, label in ranges:
        if max_val == float('inf'):
            count = np.sum((exozodi_levels >= min_val))
        else:
            count = np.sum((exozodi_levels >= min_val) & (exozodi_levels < max_val))
        counts.append(count)
        labels.append(label)
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution by Exozodi Level Ranges')
    
    # 9. Comparison with solar system
    ax9 = plt.subplot(3, 3, 9)
    
    # Solar system exozodi level is typically around 1
    solar_system_level = 1.0
    
    plt.hist(exozodi_levels, bins=np.logspace(np.log10(stats_dict['min']), np.log10(stats_dict['max']), 100), 
             alpha=0.7, edgecolor='black', density=True, label='Model Distribution')
    plt.axvline(solar_system_level, color='red', linewidth=3, linestyle='--', 
                label=f'Solar System Level ({solar_system_level})')
    
    # Calculate percentage of systems with higher/lower exozodi than solar system
    higher_than_solar = np.sum(exozodi_levels > solar_system_level) / len(exozodi_levels) * 100
    lower_than_solar = np.sum(exozodi_levels < solar_system_level) / len(exozodi_levels) * 100
    
    plt.text(0.05, 0.95, f'Higher than solar: {higher_than_solar:.1f}%\nLower than solar: {lower_than_solar:.1f}%', 
             transform=ax9.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xscale('log')
    plt.xlabel('Exozodi Level')
    plt.ylabel('Density')
    plt.title('Comparison with Solar System Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = 'exozodi_data_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Additional analysis
    print("\n=== Additional Analysis ===")
    print(f"Systems with exozodi level > 10: {np.sum(exozodi_levels > 10):,} ({np.sum(exozodi_levels > 10)/len(exozodi_levels)*100:.1f}%)")
    print(f"Systems with exozodi level > 100: {np.sum(exozodi_levels > 100):,} ({np.sum(exozodi_levels > 100)/len(exozodi_levels)*100:.1f}%)")
    print(f"Systems with exozodi level > 1000: {np.sum(exozodi_levels > 1000):,} ({np.sum(exozodi_levels > 1000)/len(exozodi_levels)*100:.1f}%)")
    
    # Calculate detection impact
    print("\n=== Detection Impact Analysis ===")
    print("Assuming exozodi level affects detection probability:")
    
    # Simple model: detection probability decreases with exozodi level
    # P_detect ∝ 1 / (1 + exozodi_level)
    detection_probabilities = 1 / (1 + exozodi_levels)
    
    print(f"Average detection probability: {np.mean(detection_probabilities):.3f}")
    print(f"Median detection probability: {np.median(detection_probabilities):.3f}")
    print(f"Systems with detection probability < 0.1: {np.sum(detection_probabilities < 0.1):,} ({np.sum(detection_probabilities < 0.1)/len(detection_probabilities)*100:.1f}%)")
    
    return stats_dict, exozodi_levels

if __name__ == "__main__":
    try:
        stats_dict, exozodi_levels = create_exozodi_visualization()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 