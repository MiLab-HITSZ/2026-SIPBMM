import os
import sys
import json
import hashlib
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from evalscope import run_task
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Import required modules


def generate_model_cache_key(model_path: str) -> str:
    """
    Generate a unique cache key for a model based on its path
    
    Args:
        model_path: Path to the model
    
    Returns:
        str: Cache key
    """
    # Generate hash key using model path
    return hashlib.md5(model_path.encode()).hexdigest()

def get_model_cache_path(checkpoint_dir: str, model_key: str, cache_type: str = 'original') -> str:
    """
    Get the cache path for a model evaluation result
    
    Args:
        checkpoint_dir: Checkpoint directory
        model_key: Model cache key
        cache_type: Type of cache ('original' or 'solution')
    
    Returns:
        str: Cache file path
    """
    cache_dir = os.path.join(checkpoint_dir, "evaluation_cache", cache_type)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model_key}.json")

def load_cached_results(cache_path: str) -> Optional[Dict]:
    """
    Load cached evaluation results if they exist
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Optional[Dict]: Cached results or None if not found
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                print(f"Loading cached results from: {cache_path}")
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache from {cache_path}: {e}")
    return None

def save_results_to_cache(cache_path: str, results: Dict):
    """
    Save evaluation results to cache
    
    Args:
        cache_path: Path to cache file
        results: Results to cache
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Add timestamp to cached results
        cached_data = results.copy()
        cached_data['cached_at'] = datetime.now().isoformat()
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cached_data, f, indent=2, ensure_ascii=False)
        print(f"Results cached to: {cache_path}")
    except Exception as e:
        print(f"Error saving cache to {cache_path}: {e}")



def load_latest_checkpoint(checkpoint_dir: str, task_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the latest solution set for a specified task ID from the checkpoint directory
    
    Args:
        checkpoint_dir: Checkpoint root directory
        task_id: Task ID
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (decision variables, objective function values)
    """
    task_dir = os.path.join(checkpoint_dir, task_id)
    latest_checkpoint_path = os.path.join(task_dir, 'checkpoint_latest_selected.pt')
    if os.path.exists(latest_checkpoint_path):
        if not os.path.exists(latest_checkpoint_path):
            print(f"Failed to find latest checkpoint (after selection) for task {task_id}, trying before selection: {latest_checkpoint_path}")
            latest_checkpoint_path = os.path.join(task_dir, 'checkpoint_latest.pt')
            if not os.path.exists(latest_checkpoint_path):
                raise FileNotFoundError(f"Failed to find latest checkpoint for task {task_id}: {latest_checkpoint_path}") 
    print(f"Loading checkpoint from {latest_checkpoint_path}...")
    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    
    # Get decision variables and objective function values
    train_x = checkpoint['train_x']
    train_obj_true = checkpoint['train_obj_true']
    
    print(f"Successfully loaded {train_x.shape[0]} candidate solutions")
    return train_x, train_obj_true


def get_pareto_optimal_points(train_x: torch.Tensor, train_obj: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Pareto optimal solutions from candidate solutions
    
    Args:
        train_x: Decision variables for all candidate solutions
        train_obj: Objective function values for all candidate solutions
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Pareto decision variables, Pareto objective function values)
    """
    # Convert to numpy arrays
    if isinstance(train_x, torch.Tensor):
        train_x = train_x.numpy()
    if isinstance(train_obj, torch.Tensor):
        train_obj = train_obj.numpy()
    
    # Mark whether each point is Pareto optimal
    is_pareto = np.ones(train_x.shape[0], dtype=bool)
    for i in range(train_x.shape[0]):
        # Check if each point is dominated by any other point
        for j in range(train_x.shape[0]):
            if i != j and is_pareto[j]:
                # If point j dominates point i
                if np.all(train_obj[j] >= train_obj[i]) and np.any(train_obj[j] > train_obj[i]):
                    is_pareto[i] = False
                    break
    
    pareto_x = train_x[is_pareto]
    pareto_y = train_obj[is_pareto]
    print(f"Extracted {pareto_x.shape[0]} Pareto optimal solutions from {train_x.shape[0]} candidate solutions")
    
    return pareto_x, pareto_y


def weighted_fusion(pareto_x: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Fusion of Pareto optimal solutions based on weights
    
    Args:
        pareto_x: Decision variables of Pareto optimal solutions
        weights: Weight array, default is equal weights
    
    Returns:
        np.ndarray: Fused decision variables
    """
    if weights is None:
        # Use equal weights by default
        weights = np.ones(pareto_x.shape[0]) / pareto_x.shape[0]
    else:
        # Ensure weights are normalized
        weights = np.array(weights) / np.sum(weights)
    
    print(f"Fusing {pareto_x.shape[0]} Pareto optimal solutions with weights {weights}")
    
    # Fusion by weights
    fused_x = np.zeros(pareto_x.shape[1])
    for i in range(pareto_x.shape[0]):
        fused_x += weights[i] * pareto_x[i]
    
    print(f"Fused decision variables: {fused_x}")
    return fused_x


def create_eval_task_config(model_path: str, max_tokens: int = 26000) -> Any:
    """
    Create evaluation task configuration
    
    Args:
        model_path: Model path
    
    Returns:
        TaskConfig object
    """
    # Import configuration from config_manager
    from src.config_manager import config_manager
    
    # Use create_triojetivc_vaild_task_config method from config_manager
    task_cfg = config_manager.create_triojetivc_vaild_task_config(model_path, max_tokens)
    
    # Set work_dir
    task_cfg.work_dir = './output/evalscope_logs'
    
    return task_cfg


def run_task_with_server(port: int, task_cfg: Any) -> Dict:
    """
    Execute task on server at specified port
    
    Args:
        port: Server port
        task_cfg: Task configuration object
    
    Returns:
        Task execution result
    """
    # Update API URL to use correct port
    task_cfg.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
    
    print(f"Executing task on port {port}: model={task_cfg.model}, datasets={task_cfg.datasets}, repeats={task_cfg.repeats}")
    
    try:
        # Execute task
        result = run_task(task_cfg=task_cfg)
        print(f"Task execution completed on port {port}")
        return result
    except Exception as e:
        print(f"Task execution error on port {port}: {e}")
        return {"error": str(e)}

import json
import os
import sys

# Ensure correct import of project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ResultProcessor:
    """
    Result processor for parsing and saving task execution results
    Supports converting complex report objects into serializable formats and saving as JSON and TXT files
    """
    
    def __init__(self, base_output_dir=None):
        """
        Initialize result processor
        
        Args:
            base_output_dir: Base output directory, default is outputs subdirectory under evoMI project directory
        """
        if base_output_dir is None:
            # Save to outputs directory under evoMI project directory
            current_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Get evoMI directory from src/evoMI/result_processor.py
            evoMI_dir = os.path.dirname(current_script_dir)
            self.base_output_dir = os.path.join(evoMI_dir, 'output/results_logs')
        else:
            self.base_output_dir = base_output_dir
        
    def process_and_save(self, results):
        """
        Process task execution results and save as JSON and TXT files
        
        Args:
            results: Task execution results dictionary
            
        Returns:
            list: Structured results array, each element is a task results dictionary
                  Format: [{problem1: {metric1: value1, metric2: value2, ...}}, {problem2: {metric1: value1, ...}}, ...]
        """
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(self.base_output_dir, timestamp)
        os.makedirs(result_dir, exist_ok=True)
        
        # Parse results into serializable format
        parsed_results = self._parse_results(results)
        
        # Save as JSON file
        json_path = os.path.join(result_dir, 'results.json')
        self._save_as_json(parsed_results, json_path)
        
        # Generate and save TXT file
        txt_path = os.path.join(result_dir, 'results_summary.txt')
        self._save_as_txt(parsed_results, txt_path, timestamp)
        
        print(f"\nResults saved:")
        print(f"  JSON file: {json_path}")
        print(f"  Text summary: {txt_path}")
        
        # Generate structured results array
        structured_results = self._generate_structured_results(results)
        
        return structured_results
    
    def _parse_results(self, results):
        """
        Parse results into serializable format
        
        Args:
            results: Original task execution results
            
        Returns:
            dict: Parsed results dictionary
        """
        parsed_results = {}
        
        for task_id, task_result in results.items():
            if isinstance(task_result, dict) and 'error' in task_result:
                parsed_results[task_id] = {'error': task_result['error']}
                continue
            
            task_parsed = {}
            
            # Iterate through dataset reports
            if hasattr(task_result, '__dict__'):
                # Handle single report object case
                dataset_key = 'single_report'
                report = task_result
                task_parsed[dataset_key] = self._parse_report(report)
            else:
                # Handle multiple dataset reports case
                try:
                    for dataset_key, report in task_result.items():
                        if hasattr(report, '__dict__'):
                            task_parsed[dataset_key] = self._parse_report(report)
                        else:
                            task_parsed[dataset_key] = str(report)
                except (AttributeError, TypeError):
                    # If task_result is not an iterable dictionary type, convert to string directly
                    task_parsed['unknown_result'] = str(task_result)
            
            parsed_results[task_id] = task_parsed
        
        return parsed_results
    
    def _parse_report(self, report):
        """
        Parse Report object into dictionary format
        
        Args:
            report: Report object
            
        Returns:
            dict: Parsed report dictionary
        """
        parsed = {
            'name': getattr(report, 'name', 'N/A'),
            'dataset_name': getattr(report, 'dataset_name', 'N/A'),
            'dataset_pretty_name': getattr(report, 'dataset_pretty_name', 'N/A'),
            'dataset_description': getattr(report, 'dataset_description', 'N/A'),
            'model_name': getattr(report, 'model_name', 'N/A'),
            'score': getattr(report, 'score', 'N/A'),
            'analysis': getattr(report, 'analysis', 'N/A'),
            'metrics': []
        }
        
        # Parse metrics
        metrics = getattr(report, 'metrics', [])
        for metric in metrics:
            if hasattr(metric, '__dict__'):
                metric_dict = {
                    'name': getattr(metric, 'name', 'Unknown'),
                    'num': getattr(metric, 'num', 'N/A'),
                    'score': getattr(metric, 'score', 'N/A'),
                    'macro_score': getattr(metric, 'macro_score', 'N/A')
                }
                parsed['metrics'].append(metric_dict)
        
        return parsed
    
    def _generate_structured_results(self, results):
        """
        Generate structured results array
        
        Args:
            results: Original task execution results
            
        Returns:
            list: Structured results array
                  Format: [{problem1: {metric1: value1, metric2: value2, ...}}, {problem2: {metric1: value1, ...}}, ...]
        """
        structured_results = []
        
        # Sort by task ID to ensure consistent order in result array
        sorted_task_ids = sorted(results.keys())
        
        for task_id in sorted_task_ids:
            task_result = results[task_id]
            task_structured = {}
            
            # Skip error tasks
            if isinstance(task_result, dict) and 'error' in task_result:
                structured_results.append({task_id: {'status': 'error', 'error': task_result['error']}})
                continue
            
            # Process report objects
            if hasattr(task_result, '__dict__'):
                # Single report object
                report = task_result
                # Use task ID as question identifier
                metrics_dict = {}
                
                # Extract all metrics
                metrics = getattr(report, 'metrics', [])
                for metric in metrics:
                    if hasattr(metric, '__dict__'):
                        metric_name = getattr(metric, 'name', 'Unknown')
                        metric_score = getattr(metric, 'score', 'N/A')
                        metrics_dict[metric_name] = metric_score
                
                # If no metrics, add basic information
                if not metrics_dict:
                    metrics_dict['score'] = getattr(report, 'score', 'N/A')
                
                task_structured[task_id] = metrics_dict
            else:
                # Multiple dataset reports case
                try:
                    # Assume first dataset as main question identifier
                    for dataset_key, report in task_result.items():
                        if hasattr(report, '__dict__'):
                            metrics_dict = {}
                            
                            # Extract all metrics
                            metrics = getattr(report, 'metrics', [])
                            for metric in metrics:
                                if hasattr(metric, '__dict__'):
                                    metric_name = getattr(metric, 'name', 'Unknown')
                                    metric_score = getattr(metric, 'score', 'N/A')
                                    metrics_dict[metric_name] = metric_score
                            
                            # If no metrics, add basic information
                            if not metrics_dict:
                                metrics_dict['score'] = getattr(report, 'score', 'N/A')
                            
                            task_structured[dataset_key] = metrics_dict
                        else:
                            # If not a report object, save directly
                            task_structured[dataset_key] = {'value': str(report)}
                except (AttributeError, TypeError):
                    # If cannot iterate, treat entire result as single question
                    task_structured[task_id] = {'value': str(task_result)}
            
            structured_results.append(task_structured)
        
        return structured_results
    
    def _save_as_json(self, parsed_results, json_path):
        """
        Save parsed results as JSON file
        
        Args:
            parsed_results: Parsed results dictionary
            json_path: JSON file path
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_as_txt(self, parsed_results, txt_path, timestamp):
        """
        Save parsed results as TXT summary file
        
        Args:
            parsed_results: Parsed results dictionary
            txt_path: TXT file path
            timestamp: Timestamp string
        """
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Task Execution Results Summary - {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            for task_id, task_data in parsed_results.items():
                f.write(f"Task ID: {task_id}\n")
                f.write("-" * 60 + "\n")
                
                if 'error' in task_data:
                    f.write(f"Status: Failed\n")
                    f.write(f"Error Message: {task_data['error']}\n")
                else:
                    f.write(f"Status: Success\n")
                    
                    # Iterate through each dataset's report
                    for dataset_key, report_data in task_data.items():
                        if report_data and isinstance(report_data, dict):
                            f.write(f"\nDataset: {dataset_key}\n")
                            f.write(f"Model: {report_data.get('model_name', 'N/A')}\n")
                            f.write(f"Total Score: {report_data.get('score', 'N/A')}\n")
                            
                            # Write metric information
                            if 'metrics' in report_data:
                                f.write("Metric Details:\n")
                                for metric in report_data['metrics']:
                                    f.write(f"  - {metric.get('name', 'Unknown')}: {metric.get('score', 'N/A')} (Sample Count: {metric.get('num', 'N/A')})\n")
                        elif report_data:
                            f.write(f"\nDataset: {dataset_key}\n")
                            f.write(f"Result: {str(report_data)}\n")
                
                f.write("\n" + "="*80 + "\n\n")

"""
Plotting Tool Class - Central management of all plotting-related functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Optional, Dict, Any


class VisualizationTool:
    """Plotting tool class, providing various optimization result visualization functions"""
    
    @staticmethod
    def visualize_optimization_results(result_dict: dict, output_dir: str) -> None:
        """
        Visualize optimization results
        
        Args:
            result_dict: Optimization results dictionary
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get Pareto front and corresponding decision variables
        pareto_x = result_dict.get('pareto_x', np.array([]))
        pareto_y = result_dict.get('pareto_y', np.array([]))
        
        if len(pareto_y) == 0:
            print("No Pareto front found")
            return
        
        # Plot 3D Pareto front
        VisualizationTool._plot_pareto_3d(pareto_y, result_dict.get('all_y', np.array([])), output_dir)
        
        # Plot 2D Pareto front for each pair of objectives
        VisualizationTool._plot_pareto_2d_pairs(pareto_y, result_dict.get('all_y', np.array([])), output_dir)
        
        # Plot hypervolume history over iterations
        hypervolumes = result_dict.get('hypervolume_history', [])
        if hypervolumes:
            VisualizationTool.plot_hypervolume_history(hypervolumes, "Multi-objective Optimization", os.path.join(output_dir, 'hypervolume_history.png'))
        
        print(f"Visualization results saved to {output_dir}")
    
    @staticmethod
    def _plot_pareto_3d(pareto_y: np.ndarray, all_y: np.ndarray, output_dir: str) -> None:
        """
        Plot 3D Pareto front
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Pareto front points
        scatter = ax.scatter(
            pareto_y[:, 0], 
            pareto_y[:, 1], 
            pareto_y[:, 2], 
            c='r', 
            marker='o', 
            s=50, 
            label='Pareto Front'
        )
        
        # Add numbering labels for each Pareto point
        for i in range(len(pareto_y)):
            ax.text(
                pareto_y[i, 0], 
                pareto_y[i, 1], 
                pareto_y[i, 2], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
            )
        
        # Plot all evaluation points
        if len(all_y) > 0 and all_y.ndim == 2:
            # Ensure all_y is a 2D array with enough columns
            if all_y.shape[1] >= 3:
                ax.scatter(
                    all_y[:, 0], 
                    all_y[:, 1], 
                    all_y[:, 2], 
                    c='b', 
                    marker='x', 
                    s=20, 
                    alpha=0.5, 
                    label='Evaluation Points'
                )
        
        # Set axis labels
        ax.set_xlabel('F1: aime25 acc')
        ax.set_ylabel('F2: apqa acc')
        ax.set_zlabel('F3: tokens + ifeval')
        
        ax.set_title('Pareto Front of Multi-objective Optimization')
        ax.legend()
        
        # Save 3D plot
        plt.savefig(os.path.join(output_dir, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_pareto_2d_pairs(pareto_y: np.ndarray, all_y: np.ndarray, output_dir: str) -> None:
        """
        Plot 2D Pareto fronts between each pair of objectives
        """
        pairs = [(0, 1), (0, 2), (1, 2)]
        labels = [
            ['F1: aime25 acc', 'F2: apqa acc'],
            ['F1: aime25 acc', 'F3: tokens + ifeval'],
            ['F2: apqa acc', 'F3: tokens + ifeval']
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (idx1, idx2) in enumerate(pairs):
            ax = axes[i]
            
            # Plot all evaluation points
            if len(all_y) > 0 and all_y.ndim == 2:
                if all_y.shape[1] >= max(idx1, idx2) + 1:
                    ax.scatter(all_y[:, idx1], all_y[:, idx2], c='b', marker='x', s=20, alpha=0.3, label='Evaluation Points')
            
            # Plot Pareto front points
            ax.scatter(pareto_y[:, idx1], pareto_y[:, idx2], c='r', marker='o', s=50, label='Pareto Front')
            
            # Add numbering labels for each Pareto point
            for i in range(len(pareto_y)):
                ax.text(
                    pareto_y[i, idx1], 
                    pareto_y[i, idx2], 
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
            
            # Connect Pareto points (sorted by first metric)
            sorted_indices = np.argsort(pareto_y[:, idx1])
            ax.plot(
                pareto_y[sorted_indices, idx1], 
                pareto_y[sorted_indices, idx2], 
                'r-', 
                alpha=0.7
            )
            
            ax.set_xlabel(labels[i][0])
            ax.set_ylabel(labels[i][1])
            ax.set_title(f'Pareto Front: {labels[i][0]} vs {labels[i][1]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pareto_front_2d_pairs.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_pareto_results(train_obj_true: torch.Tensor, 
                          problem_name: str = "Multi-objective Problem", 
                          true_pareto_front: Optional[Tuple[np.ndarray, ...]] = None,
                          ref_point: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot Pareto front results
        
        Args:
        ----------
        train_obj_true : torch.Tensor
            True objective function values for all evaluation points
        problem_name : str, optional
            Problem name, default is "Multi-objective Problem"
        true_pareto_front : tuple of np.ndarray, optional
            True Pareto front objective values, format (f1, f2, ..., fn)
        ref_point : torch.Tensor, optional
            Reference point
        save_path : str, optional
            Path to save the image
        """
        from .qehvi_optimizer import get_pareto_optimal_points
        
        # Ensure tensors are on CPU for matplotlib plotting
        if isinstance(train_obj_true, torch.Tensor):
            train_obj_true = train_obj_true.cpu()
        
        # Get Pareto optimal solutions
        pareto_points = get_pareto_optimal_points(train_obj_true)
        
        # Ensure pareto_points is also on CPU
        if isinstance(pareto_points, torch.Tensor):
            pareto_points = pareto_points.cpu()
        
        # Handle three-objective problems
        if true_pareto_front is not None and len(true_pareto_front) == 3:
            VisualizationTool._plot_pareto_3d_comparison(pareto_points, true_pareto_front, problem_name, save_path)
            return
        
        # Handle two-objective problems
        plt.figure(figsize=(8, 6))
        
        # Plot true Pareto front if provided
        if true_pareto_front is not None and len(true_pareto_front) == 2:
            f1, f2 = true_pareto_front
            plt.scatter(f1, f2, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # Plot obtained Pareto optimal solutions
        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', s=40, 
                   label='Obtained Pareto Solutions', edgecolors='black', linewidths=0.5)
        
        # Add numbered labels for each Pareto solution
        for i in range(len(pareto_points)):
            plt.text(
                pareto_points[i, 0], 
                pareto_points[i, 1], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.3)
            )
        
        # Plot reference point if provided
        if ref_point is not None and len(ref_point) == 2:
            # Ensure reference point is on CPU
            if isinstance(ref_point, torch.Tensor):
                ref_point = ref_point.cpu()
            plt.scatter(ref_point[0], ref_point[1], c='green', s=100, 
                       label='Reference Point', marker='*', edgecolors='black')
        
        # Set plot properties
        plt.xlabel('Objective Function 1', fontsize=12)
        plt.ylabel('Objective Function 2', fontsize=12)
        plt.title(f'Pareto Solutions for {problem_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results figure saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def _plot_pareto_3d_comparison(pareto_points: np.ndarray, 
                                 true_pareto_front: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 problem_name: str, 
                                 save_path: Optional[str]) -> None:
        """
        Plot 3D Pareto front comparison
        """
        ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
        f1, f2, f3 = true_pareto_front
        
        # Plot true Pareto front
        ax.scatter(f1, f2, f3, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # Plot obtained Pareto solutions
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], 
                  c='blue', s=40, label='Obtained Pareto Solutions', 
                  edgecolors='black', linewidths=0.5)
        
        # Add numbered labels for each Pareto solution
        for i in range(len(pareto_points)):
            ax.text(
                pareto_points[i, 0], 
                pareto_points[i, 1], 
                pareto_points[i, 2], 
                str(i), 
                fontsize=10, 
                ha='center', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.3)
            )
        
        # Set axis labels
        ax.set_xlabel('Objective Function 1', fontsize=12)
        ax.set_ylabel('Objective Function 2', fontsize=12)
        ax.set_zlabel('Objective Function 3', fontsize=12)
        ax.set_title(f'Pareto Solutions for {problem_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results figure saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def plot_hypervolume_history(hvs: List[float], 
                               problem_name: str = "Multi-objective Problem",
                               save_path: Optional[str] = None) -> None:
        """
        Plot hypervolume change over iterations
        
        Parameters:
        ----------
        hvs : list
            Hypervolume values for each iteration
        problem_name : str, optional
            Problem name
        save_path : str, optional
            Path to save the image
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(hvs)), hvs, 'b-o', linewidth=2, markersize=5)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Hypervolume', fontsize=12)
        plt.title(f'Hypervolume History for {problem_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hypervolume history saved as '{save_path}'")
        
        plt.close()
    
    @staticmethod
    def plot_zdt3_results(train_obj_true: torch.Tensor, 
                         hvs: List[float],
                         compute_true_pareto_func,
                         get_pareto_optimal_func) -> None:
        """
        Plot ZDT3 problem results
        
        Parameters:
        ----------
        train_obj_true : torch.Tensor
            True objective function values for all evaluation points
        hvs : list
            Hypervolume values for each iteration
        compute_true_pareto_func : callable
            Function to compute true Pareto front
        get_pareto_optimal_func : callable
            Function to get Pareto optimal solutions
        """
        # Compute true Pareto front
        true_f1, true_f2 = compute_true_pareto_func()
        
        # Get Pareto optimal solutions
        pareto_points = get_pareto_optimal_func(train_obj_true)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot true Pareto front
        plt.scatter(true_f1, true_f2, c='red', s=20, label='True Pareto Front', alpha=0.6)
        
        # Plot obtained Pareto optimal solutions
        plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', s=40, 
                   label='Obtained Pareto Solutions', edgecolors='black', linewidths=0.5)
        
        # Set plot properties
        plt.xlabel('Objective Function 1', fontsize=12)
        plt.ylabel('Objective Function 2', fontsize=12)
        plt.title('Ideal vs Obtained Pareto Solutions for ZDT3 Problem', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Adjust layout to ensure all points are visible
        plt.tight_layout()
        
        # Save figure
        plt.savefig('qehvi_zdt3_results.png', dpi=300, bbox_inches='tight')
        print("Results figure saved as 'qehvi_zdt3_results.png'")
        plt.close()
        
        # If hypervolume history is provided, plot hypervolume curve
        if hvs:
            VisualizationTool.plot_hypervolume_history(hvs, "ZDT3 Problem", "qehvi_zdt3_hypervolume.png")


    @staticmethod
    def plot_3d_objectives(objectives_list: List[List], output_dir: str) -> None:
        """
        Visualize all solutions in 3D objective space and add numbered labels to solution points
        
        Parameters:
            objectives_list: List of objective function values, format [[f1, f2, f3, type], ...]
            output_dir: Output directory
        """
        if not objectives_list:
            print("No objective value data, skipping 3D visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Separate different types of objective points
        solution_obj = []
        original_obj = []
        solution_indices = []
        
        for idx, obj in enumerate(objectives_list):
            if len(obj) > 3 and obj[3] == 'solution':
                solution_obj.append(obj[:3])
                solution_indices.append(idx)
            else:
                original_obj.append(obj[:3])
        
        solution_obj = np.array(solution_obj) if solution_obj else np.array([])
        original_obj = np.array(original_obj) if original_obj else np.array([])
        
        # Plot solution points
        if len(solution_obj) > 0:
            f1_sol = solution_obj[:, 0]
            f2_sol = solution_obj[:, 1]
            f3_sol = solution_obj[:, 2]
            scatter_sol = ax.scatter(f1_sol, f2_sol, f3_sol, c=f1_sol, cmap='viridis', 
                                    s=50, alpha=0.7, label='Merged Models')
            
            # Add numbered labels for each solution point
            for i, idx in enumerate(solution_indices):
                ax.text(
                    solution_obj[i, 0]+0.01, 
                    solution_obj[i, 1]+0.01, 
                    solution_obj[i, 2]+0.01, 
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.1)
                )
            
            # Add colorbar
            cbar = plt.colorbar(scatter_sol, ax=ax)
            cbar.set_label('F1 Score')
        
        # Plot original model points
        if len(original_obj) > 0:
            f1_orig = original_obj[:, 0]
            f2_orig = original_obj[:, 1]
            f3_orig = original_obj[:, 2]
            scatter_orig = ax.scatter(f1_orig, f2_orig, f3_orig, c='red', 
                                     s=80, alpha=0.9, marker='^', label='Original Models')
        
        # Set labels
        ax.set_xlabel('F1 (AIME25 & GPQA Avg Accuracy)', fontsize=12, labelpad=10)
        ax.set_ylabel('F2 (-tokens_num/20000)', fontsize=12, labelpad=10)
        ax.set_zlabel('F3 (IFEval Avg Accuracy)', fontsize=12, labelpad=10)
        
        # Set title
        ax.set_title('Distribution of Solutions in 3D Objective Space', fontsize=14, pad=20)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set viewing angle
        ax.view_init(30, 45)
        
        # Save image
        plt.tight_layout()
        output_path = os.path.join(output_dir, "3d_objectives_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D objective space plot saved to: {output_path}")
    
    @staticmethod
    def plot_dataset_metrics(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        Visualize the relationship between token count and accuracy for each dataset
        
        Parameters:
            output_dir: Output directory
            metrics_list: List of metrics for all solutions and original models
        """
        if not metrics_list:
            print("No metrics data, skipping dataset visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = ['aime25', 'gpqa_diamond', 'ifeval']
        dataset_names = {
            'aime25': 'AIME25',
            'gpqa_diamond': 'GPQA Diamond',
            'ifeval': 'IFEval'
        }
        
        for dataset in datasets:
            # Extract token count and accuracy for current dataset, separating solutions and original models
            solution_tokens = []
            solution_acc = []
            original_tokens = []
            original_acc = []
            original_names = []
            
            for metrics in metrics_list:
                if dataset in metrics:
                    tokens = metrics[dataset].get('mean_tokens_num', 5000)
                    # Get accuracy based on dataset
                    if 'mean_acc' in metrics[dataset]:
                        acc = metrics[dataset]['mean_acc']
                    elif dataset == 'ifeval' and 'mean_prompt_level_strict' in metrics[dataset]:
                        acc = (metrics[dataset]['mean_prompt_level_strict'] + 
                              metrics[dataset]['mean_inst_level_strict'] + 
                              metrics[dataset]['mean_prompt_level_loose'] + 
                              metrics[dataset]['mean_inst_level_loose']) / 4
                    else:
                        continue
                    # Separate by type
                    if metrics.get('type') == 'solution':
                        solution_tokens.append(tokens)
                        solution_acc.append(acc)
                    else:
                        original_tokens.append(tokens)
                        original_acc.append(acc)
                        original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
            
            # Create chart
            plt.figure(figsize=(10, 6))
            
            # Plot solution points
            if solution_tokens and solution_acc:
                scatter_sol = plt.scatter(solution_tokens, solution_acc, c='black', 
                                         cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
                
                # Add numbered labels for each solution point
                for i, (tokens, acc) in enumerate(zip(solution_tokens, solution_acc)):
                    # Calculate upward offset for y-axis (dynamically adjusted based on chart range)
                    if solution_acc:
                        y_range = max(solution_acc) - min(solution_acc)
                        y_offset = y_range * 0.02  # 2% upward offset of y-axis range
                    else:
                        y_offset = 0.02
                        
                    plt.text(
                        tokens, 
                        acc + y_offset,  # Move up slightly
                        str(i), 
                        fontsize=10, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                    )
            
            # Plot original model points
            if original_tokens and original_acc:
                scatter_orig = plt.scatter(original_tokens, original_acc, c='red', 
                                          s=70, alpha=0.8, marker='^', label='Original Models')
                
                # Add name labels for original model points
                for i, (tokens, acc, name) in enumerate(zip(original_tokens, original_acc, original_names)):
                    plt.text(
                        tokens, 
                        acc, 
                        name, 
                        fontsize=9, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                    )
            
            # Set chart properties
            plt.xlabel('Mean Tokens Used', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Token Usage vs Accuracy for {dataset_names[dataset]} Dataset', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # Save chart
            output_path = os.path.join(output_dir, f"{dataset}_tokens_vs_accuracy.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{dataset_names[dataset]} dataset visualization saved to: {output_path}")
    
    @staticmethod
    def plot_token_avg_vs_model_ability(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        Visualize the relationship between average token count across all metrics and model ability (f1+f3)
        
        Parameters:
            output_dir: Output directory
            metrics_list: List of metrics for all solutions and original models
        """
        if not metrics_list:
            print("No metrics data, skipping token average vs model ability visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        solution_avg_tokens = []
        solution_model_ability = []
        original_avg_tokens = []
        original_model_ability = []
        original_names = []
        
        for metrics in metrics_list:
            # Calculate average token count across all datasets
            tokens_list = []
            if 'aime25' in metrics:
                tokens_list.append(metrics['aime25'].get('mean_tokens_num', 0))
            if 'gpqa_diamond' in metrics:
                tokens_list.append(metrics['gpqa_diamond'].get('mean_tokens_num', 0))
            if 'ifeval' in metrics:
                tokens_list.append(metrics['ifeval'].get('mean_tokens_num', 0))
            
            if not tokens_list:
                continue
            
            avg_tokens = np.mean(tokens_list)
            
            # Calculate model ability: f1 + f3
            # Get f1, f2, f3 from metrics (these are added in save_solution_results)
            f1 = metrics.get('f1', 0)
            f3 = metrics.get('f3', 0)
            model_ability = f1 + f3
            
            # Separate solutions and original models
            if metrics.get('type') == 'solution':
                solution_avg_tokens.append(avg_tokens)
                solution_model_ability.append(model_ability)
            else:
                original_avg_tokens.append(avg_tokens)
                original_model_ability.append(model_ability)
                original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Plot solution points
        if solution_avg_tokens and solution_model_ability:
            scatter_sol = plt.scatter(solution_avg_tokens, solution_model_ability, c='black', 
                                     cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
            
            # Add numbered labels for each solution point
            for i, (tokens, ability) in enumerate(zip(solution_avg_tokens, solution_model_ability)):
                y_range = max(solution_model_ability) - min(solution_model_ability)
                y_offset = y_range * 0.02  # 2% upward offset of y-axis range
                
                plt.text(
                    tokens, 
                    ability + y_offset,  # Move up slightly
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
        
        # Plot original model points
        if original_avg_tokens and original_model_ability:
            scatter_orig = plt.scatter(original_avg_tokens, original_model_ability, c='red', 
                                      s=70, alpha=0.8, marker='^', label='Original Models')
            
            # Add name labels for original model points
            for tokens, ability, name in zip(original_avg_tokens, original_model_ability, original_names):
                plt.text(
                    tokens, 
                    ability, 
                    name, 
                    fontsize=9, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                )
        
        # Set chart properties
        plt.xlabel('Average Tokens Used Across All Datasets', fontsize=12)
        plt.ylabel('Model Ability (F1 + F3)', fontsize=12)
        plt.title('Average Tokens vs Model Ability (F1 + F3)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(output_dir, "token_avg_vs_model_ability.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Token average vs model ability visualization saved to: {output_path}")
    
    @staticmethod
    def plot_f1_vs_f3(output_dir: str, metrics_list: List[Dict]) -> None:
        """
        Visualize the relationship between model abilities f1 and f3
        
        Parameters:
            output_dir: Output directory
            metrics_list: List of metrics for all solutions and original models
        """
        if not metrics_list:
            print("No metrics data, skipping f1 vs f3 visualization")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        solution_f1 = []
        solution_f3 = []
        original_f1 = []
        original_f3 = []
        original_names = []
        
        for metrics in metrics_list:
            # Get f1 and f3 from metrics
            f1 = metrics.get('f1', 0)
            f3 = metrics.get('f3', 0)
            
            # Separate solutions and original models
            if metrics.get('type') == 'solution':
                solution_f1.append(f1)
                solution_f3.append(f3)
            else:
                original_f1.append(f1)
                original_f3.append(f3)
                original_names.append(metrics.get('name', f'Original_{len(original_names)}'))
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        # Plot solution points
        if solution_f1 and solution_f3:
            scatter_sol = plt.scatter(solution_f1, solution_f3, c='black', 
                                     cmap='plasma', s=50, alpha=0.7, label='evoMI(ours)')
            
            # Add numbered labels for each solution point
            for i, (f1_val, f3_val) in enumerate(zip(solution_f1, solution_f3)):
                y_range = max(solution_f3) - min(solution_f3)
                y_offset = y_range * 0.02  # 2% upward offset of y-axis range
                
                plt.text(
                    f1_val, 
                    f3_val + y_offset,  # Move up slightly
                    str(i), 
                    fontsize=10, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
                )
        
        # Plot original model points
        if original_f1 and original_f3:
            scatter_orig = plt.scatter(original_f1, original_f3, c='red', 
                                      s=70, alpha=0.8, marker='^', label='Original Models')
            
            # Add name labels for original model points
            for f1_val, f3_val, name in zip(original_f1, original_f3, original_names):
                plt.text(
                    f1_val, 
                    f3_val, 
                    name, 
                    fontsize=9, 
                    ha='center', 
                    va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.5)
                )
        
        # Set chart properties
        plt.xlabel('Model Ability F1', fontsize=12)
        plt.ylabel('Model Ability F3', fontsize=12)
        plt.title('Model Ability F1 vs F3', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(output_dir, "f1_vs_f3.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"f1 vs f3 visualization saved to: {output_path}")

# Create visualization tool instance for external use
visualizer = VisualizationTool()