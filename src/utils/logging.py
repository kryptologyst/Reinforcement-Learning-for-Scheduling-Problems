"""Utility modules for logging and metrics tracking."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns


class Logger:
    """Simple logger for training."""
    
    def __init__(self, log_file: Path):
        """Initialize logger.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics tracker.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.current_episode = 0
    
    def update(self, episode_metrics: Dict[str, Any]) -> None:
        """Update metrics with new episode data.
        
        Args:
            episode_metrics: Dictionary of metrics for current episode
        """
        self.current_episode += 1
        
        # Store episode-level metrics
        for key, value in episode_metrics.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append(value)
            elif isinstance(value, dict):
                # Flatten nested metrics
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.metrics[f"{key}_{sub_key}"].append(sub_value)
    
    def get_metric(self, metric_name: str) -> List[float]:
        """Get history of a specific metric.
        
        Args:
            metric_name: Name of metric to retrieve
            
        Returns:
            List of metric values
        """
        return list(self.metrics[metric_name])
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """Get all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their histories
        """
        return {key: list(values) for key, values in self.metrics.items()}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest values of all metrics.
        
        Returns:
            Dictionary mapping metric names to their latest values
        """
        return {key: values[-1] if values else 0.0 for key, values in self.metrics.items()}
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a specific metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dictionary with mean, std, min, max
        """
        values = self.get_metric(metric_name)
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def save_metrics(self, filepath: Path) -> None:
        """Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        metrics_dict = self.get_all_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def load_metrics(self, filepath: Path) -> None:
        """Load metrics from file.
        
        Args:
            filepath: Path to load metrics from
        """
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        
        for key, values in metrics_dict.items():
            self.metrics[key] = deque(values, maxlen=self.max_history)
