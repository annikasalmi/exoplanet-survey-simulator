import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple

from tools.paths import PLOTS_DIR

class BasePlotter:
    """
    Base class for all plotting classes with common functionality.
    Provides shared initialization, data handling, and plotting utilities.
    """
    
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize base plotter with common parameters."""
        self.df = df.copy()
        self.nruns = nruns
        self.star_catalog = star_catalog
        self.name = name
        self.data_dir = self._make_output_dir()
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Cache for computed values to avoid recalculation
        self._cache = {}
        
        # Pre-compute commonly used values
        self._precompute_values()

    def _make_output_dir(self):
        """Create and return output directory for plots."""
        out_dir = os.path.join(PLOTS_DIR, str(self.name)+'_'+str(self.nruns)+'_'+str(self.star_catalog))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    
    def _output_filename(self, plot_type: str, suffix: Optional[str] = None) -> str:
        """Centralize output filename formatting."""
        base = f"{plot_type}_{self.name}_nruns{self.nruns}_{self.star_catalog}"
        if suffix:
            base += f"_{suffix}"
        return base + ".png"

    def _precompute_values(self):
        """Pre-compute values that are used multiple times."""
        if 'detection_masks' not in self._cache:
            self._cache['detection_masks'] = self._get_detection_masks()

    def _save_plot(self, fig, filename: str, suffix: Optional[str] = None) -> None:
        """Save plot with consistent settings."""
        plt.tight_layout()
        outfile = self._output_filename(filename, suffix)
        plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _get_detection_masks(self) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Get detection masks from cache."""
        return self._cache['detection_masks']

    def _validate_data(self) -> bool:
        """Validate that the dataframe has required data."""
        if self.df.empty:
            print("Warning: DataFrame is empty")
            return False
        return True

    def _add_percentage_labels(self, ax, x, y_values, total_values, y_errors=None, 
                              offset=1.0, fontsize=8, color='black'):
        """Add percentage labels to bars."""
        for i, (y_val, total_val) in enumerate(zip(y_values, total_values)):
            if y_val > 0 and total_val > 0:
                pct = int(round(100 * y_val / total_val))
                y_err = y_errors[i] if y_errors is not None else 0
                ax.text(x[i], y_val + y_err + offset, f"{pct}%", 
                       ha='center', fontsize=fontsize, color=color)

    def _setup_plot_style(self, ax, xlabel: str, ylabel: str, title: str, 
                         legend_title: Optional[str] = None):
        """Setup common plot styling."""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if legend_title:
            ax.legend(title=legend_title)
        else:
            ax.legend()

    def _calculate_efficiency_data(self, df, x_col: str, bins: np.ndarray):
        """Calculate detection efficiency data for given dataframe and bins."""
        # Ensure df is a DataFrame
        if hasattr(df, 'to_frame'):
            df = df.to_frame()
        
        total_counts, _ = np.histogram(df[x_col], bins=bins)
        total_counts = total_counts / self.nruns
        
        mask_best, _ = self._get_detection_masks()
        detected_counts, _ = np.histogram(df[mask_best][x_col], bins=bins)
        detected_counts = detected_counts / self.nruns
        
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = np.true_divide(detected_counts, total_counts)
            efficiency[np.isnan(efficiency)] = 0.0
        
        return total_counts, detected_counts, efficiency, mask_best

    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, 
                           total_errors=None, detected_errors=None,
                           bar_width=0.8, total_color='lightgray', 
                           detected_color='green', detected_hatch=None):
        """Create overlay bar plot with total and detected bars."""
        # Plot total bars (background)
        ax.bar(x, total_heights, width=bar_width, color=total_color, 
               alpha=0.5, edgecolor='black', yerr=total_errors, capsize=3,
               label='Total')
        
        # Plot detected bars (overlay)
        ax.bar(x, detected_heights, width=bar_width, color=detected_color,
               alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3,
               bottom=np.zeros_like(detected_heights), hatch=detected_hatch,
               label='Detected')
        
        return detected_heights, detected_errors or np.zeros_like(detected_heights)

    def plot_all(self) -> None:
        """Base plot_all method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement plot_all()") 