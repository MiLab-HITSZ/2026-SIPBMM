#!/usr/bin/env python3
"""
Model Optimization and Merging Module - Using SAASBO+qNEHVI Algorithm - Version 2
Integrated module providing model merging and multi-objective optimization functionality, supporting dual parameter settings for weights and density
"""

import os
import sys
import time
import uuid
import numpy as np
import torch
from evalscope import run_task

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from src.vllm_server_manager import VllmServerManager
from src.config_manager import config_manager
from src.sipbmm.saasbo_qnehvi_optimizer import saasbo_qnehvi_optimizer
from src.sipbmm.block_fusion import mi_block_fusion
from src.sipbmm.tool import visualizer, ResultProcessor, generate_model_cache_key, get_model_cache_path, load_cached_results, save_results_to_cache, run_task_with_server


available_gpus_global = [0,1,2,3]
# Global variables for storing base and expert model evaluation results
base_model_results = None
expert_model_results = None

# Use configuration management class to get model configuration
# Model paths will be specified through command line arguments
# Cache directory
checkpoint_dir = config_manager.checkpoint_dir


# Initialization function - Get evaluation results for base and expert models
def initialize_model_evaluations(base_model, expert_model, max_tokens: int = 35000, max_model_len: int = None):
    """
    Initialize evaluation results for base and expert models
    
    Args:
        base_model: List of base model paths
        expert_model: List of expert model paths
        max_tokens: Maximum number of generated tokens
        max_model_len: Maximum model length
    
    Returns:
        tuple: (base_model_results, expert_model_results)
    """
    print("\n===== Initializing Model Evaluation Results =====")
    
   # Evaluate base models
    print(f"\nEvaluating base models: {base_model}")
    base_model_results = []
    
    for model_path in base_model:
        model_key = generate_model_cache_key(model_path)
        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
        cached_result = load_cached_results(cache_path)
        
        if cached_result:
            print(f"Using cached results for base model: {model_path}")
            base_model_results.append(cached_result)
        else:
            print(f"No cached results found for base model, starting evaluation: {model_path}")
            
            # Implement evaluation logic directly in mi_opt_saasbo2.py using bi-objective configuration
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            # Create evaluation task configuration - Using bi-objective configuration, without ifeval
            task_cfg = config_manager.create_biojective_eval_task_config(model_path, max_tokens)
            
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            # Use VllmServerManager to run tasks
            print(f"Starting to evaluate model: {model_path}")
            start_time = time.time()
            try:
                # If max_model_len is not specified, set to max_tokens + 3000
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"max_model_len not specified, using default value: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=max_model_len) as server_manager:
                    results = server_manager.run_series_tasks([task])
                    
                print(f"Evaluation completed, time taken: {time.time() - start_time:.2f} seconds")
                
                # Use ResultProcessor to process results
                print("Processing evaluation results...")
                result_processor = ResultProcessor()
                res = result_processor.process_and_save(results)
                
                # Extract metrics
                metrics = {}
                if isinstance(res, dict) and 'processed_results' in res:
                    results_list = res['processed_results']
                elif isinstance(res, list):
                    results_list = res
                else:
                    results_list = [res]
                
                for result in results_list:
                    try:
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                if dataset_name not in metrics:
                                    metrics[dataset_name] = {}
                                # Directly copy all key-value pairs to metrics
                                for key, value in result[dataset_name].items():
                                    metrics[dataset_name][key] = value
                    except Exception as e:
                        print(f"Error extracting metrics: {e}")
                
            except Exception as e:
                print(f"Error occurred during evaluation: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            base_model_results.append(model_result)
            # Save to cache
            save_results_to_cache(cache_path, model_result)
    
    # Evaluate expert models
    print(f"\nEvaluating expert models: {expert_model}")
    expert_model_results = []
    
    for model_path in expert_model:
        model_key = generate_model_cache_key(model_path)
        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
        cached_result = load_cached_results(cache_path)
        
        if cached_result:
            print(f"Using cached results for expert model: {model_path}")
            expert_model_results.append(cached_result)
        else:
            print(f"No cached results found for expert model, starting evaluation: {model_path}")
            
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            task_cfg = config_manager.create_biojective_eval_task_config(model_path, max_tokens)
            
            # Create task dictionary
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            print(f"Starting model evaluation: {model_path}")
            start_time = time.time()
            try:
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"max_model_len not specified, using default: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=list[int](range(torch.cuda.device_count())), 
                                     max_model_len=max_model_len) as server_manager:
                    results = server_manager.run_series_tasks([task])
                    
                print(f"Evaluation completed, time taken: {time.time() - start_time:.2f} seconds")
                
                # Use ResultProcessor to process results
                print("Processing evaluation results...")
                result_processor = ResultProcessor()
                res = result_processor.process_and_save(results)
                
                metrics = {}
                if isinstance(res, dict) and 'processed_results' in res:
                    results_list = res['processed_results']
                elif isinstance(res, list):
                    results_list = res
                else:
                    results_list = [res]
                
                for result in results_list:
                    try:
                        # Simply extract all metrics from all datasets, using keys as metric names
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                if dataset_name not in metrics:
                                    metrics[dataset_name] = {}
                                # Directly copy all key-value pairs to metrics
                                for key, value in result[dataset_name].items():
                                    metrics[dataset_name][key] = value
                    except Exception as e:
                        print(f"Error extracting metrics: {e}")
                
            except Exception as e:
                print(f"Error occurred during evaluation: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            expert_model_results.append(model_result)
            # Save to cache
            save_results_to_cache(cache_path, model_result)
    
    print("\n===== Model Evaluation Initialization Completed =====")
    return base_model_results, expert_model_results


def process_decision_variables(decision_matrix, base_model_path="models/Qwen3-4B", 
                              task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"], 
                              base_output_dir="output/mi_optimizer",
                              max_tokens: int = 20000, max_model_len: int = None,
                              merged_blocks: list = None,
                              num_blocks: int = 8,
                              fusion_method="breadcrumbs",
                              base_model_results=None,
                              expert_model_results=None,
                              optimize_density=1):
    """
    Process decision variable matrix, create merged model for each decision variable and evaluate performance
    
    Args:
        decision_matrix: numpy array with shape (n, num_blocks+1), containing only weight parameters
        base_model_path: Base model path
        task_model_paths: List of task model paths
        base_output_dir: Base output directory
        max_tokens: Maximum number of generated tokens
        max_model_len: Maximum model length
        merged_blocks: Precomputed list of merged blocks to avoid redundant calculations
        num_blocks: Number of blocks
        fusion_method: Model fusion method
        base_model_results: List of evaluation results for base models
        expert_model_results: List of evaluation results for expert models
        optimize_density: int, optional
            Density optimization mode: Only supports 1 - No density optimization, only optimize existing weights
    
    Returns:
        numpy array: Shape (n, 2), each row represents 2 objective function values for the corresponding candidate solution
    """
    # Ensure decision matrix is a numpy array
    if isinstance(decision_matrix, list):
        decision_matrix = np.array(decision_matrix)
    
    # Validate decision matrix dimensions
    # Only optimize weights, parameter count = (number of blocks + 1)
    expected_dim = num_blocks + 1
    
    if decision_matrix.ndim != 2 or decision_matrix.shape[1] != expected_dim:
        raise ValueError(f"Decision variable matrix must be a 2D array, each row containing {expected_dim} decision variables")
    
    num_candidates = decision_matrix.shape[0]
    print(f"Starting to process {num_candidates} candidate solutions")
    
    # Ensure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create task list
    tasks = []
    model_paths = []
    
    # Create merged model for each candidate solution
    for i in range(num_candidates):
        # Extract decision variables from current row
        # All parameters are weight parameters
        block_weights = decision_matrix[i, :].tolist()
        block_densities = None  # Do not specify density, mi_block_fusion uses default value
        gamma_params = None
        print(f"\nProcessing candidate solution {i+1}/{num_candidates}:")
        print(f"  Weight parameters: {block_weights}")
        print(f"  Density parameters: Using default values")
        
        # Generate unique model ID
        model_id = f"merged_model_{i}_{uuid.uuid4().hex[:8]}"
        model_output_dir = os.path.join(base_output_dir, model_id)
        
        # Call mi_block_fusion method for model fusion
        success = mi_block_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            block_weights=block_weights,
            block_densities=block_densities,
            block_gammas=gamma_params,
            output_dir=model_output_dir,
            fusion_method=fusion_method,
            copy_from_base=True,
            merged_blocks=merged_blocks,  # Pass precomputed merged blocks
            num_blocks=num_blocks
        )
        
        if not success:
            print(f"Warning: Fusion failed for candidate solution {i+1}, skipping evaluation")
            # Use default values as objective function values
            continue
        
        model_paths.append(model_output_dir)
        
        # Create evaluation task configuration - Using bi-objective configuration, without ifeval
        task_cfg = config_manager.create_biojective_eval_task_config(model_output_dir, max_tokens)
        
        # Create task dictionary
        task = {
            'task_id': f'task_{model_id}',
            'model_path': model_output_dir,
            'params_dict': {'task_cfg': task_cfg},
            'func_handle': run_task_with_server
        }
        
        tasks.append(task)
        print(f"Created task: {task['task_id']}")

    if not tasks:
        print("Warning: No merged models were successfully created, cannot perform evaluation")
        return np.zeros((0, 2))
    
    # Use VllmServerManager to run tasks
    print(f"\nStarting to execute {len(tasks)} evaluation tasks using VllmServerManager...")
    start_time = time.time()
    print("GPU nums:",list[int](range(torch.cuda.device_count())))
    try:
        # If max_model_len is not specified, set to max_tokens + 3000
        if max_model_len is None:
            max_model_len = max_tokens + 3000
            print(f"max_model_len not specified, using default value: {max_model_len} (max_tokens + 3000)")
    
        with VllmServerManager(available_gpus=available_gpus_global, 
                                max_model_len=max_model_len) as server_manager:
            # Call run_series_tasks method to execute all tasks
            results = server_manager.run_series_tasks(tasks)
        
        print(f"Evaluation completed, time taken: {time.time() - start_time:.2f} seconds")
    
        # Use ResultProcessor to process results
        print("Processing evaluation results...")
        result_processor = ResultProcessor()
        res = result_processor.process_and_save(results)
    
        # Extract objective function values
        objectives = extract_objectives(res, base_model_results, expert_model_results)
    
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        # Return empty result array
        objectives = np.zeros((0, 2))
        res = []
    
    return objectives, res


def run_task_with_server(port, task_cfg):
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
    
    print(f"Executing task on port {port}: model={task_cfg.model}, datasets={task_cfg.datasets}")
    
    try:
        # Execute task
        result = run_task(task_cfg=task_cfg)
        print(f"Task execution completed on port {port}")
        return result
    except Exception as e:
        print(f"Error executing task on port {port}: {e}")
        return {"error": str(e)}


def extract_objectives(results, base_model_results, expert_model_results):
    """
    Extract objective function values from evaluation results (using dynamic normalization)
    
    Args:
        results: Evaluation results dictionary
        base_model_results: List of evaluation results for base models
        expert_model_results: List of evaluation results for expert models
    
    Returns:
        numpy array: Shape (n, 2), each row contains two objective function values
    """
    objectives = []
    
    # Check results structure - According to ResultProcessor return format
    if isinstance(results, dict) and 'processed_results' in results:
        results_list = results['processed_results']
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = [results]
    
    for result in results_list:
        try:
            # Extract various metrics - Adapt to new TaskConfig format, without ifeval
            aime25_acc = result['aime25'].get('mean_acc', 0) if 'aime25' in result else 0
            aime25_tokens_num = result['aime25'].get('mean_tokens_num', 0) if 'aime25' in result else 0
            gpqa_diamond_acc = result['gpqa_diamond'].get('mean_acc', 0) if 'gpqa_diamond' in result else 0
            gpqa_diamond_tokens_num = result['gpqa_diamond'].get('mean_tokens_num', 0) if 'gpqa_diamond' in result else 0
            
            # Ensure base_model_results and expert_model_results are valid
            if not base_model_results or not expert_model_results:
                print("Warning: base_model_results or expert_model_results not initialized, using default values")
                # Use default values as backup
                f1 = np.mean([(aime25_acc-0.45)/(0.8-0.45),(gpqa_diamond_acc-0.3)/(0.7-0.3)])
                f2 = -np.mean([(aime25_tokens_num-9000)/(22000-9000),(gpqa_diamond_tokens_num-1000)/(9000-1000)])
            else:
                # Use dynamically calculated normalization values
                # Get base model metrics
                base_aime25_acc = base_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.45)
                base_gpqa_diamond_acc = base_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.3)
                base_aime25_tokens = base_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 9000)
                base_gpqa_diamond_tokens = base_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 1000)
                
                # Get expert model metrics (take first expert model)
                expert_aime25_acc = expert_model_results[0].get('metrics', {}).get('aime25', {}).get('mean_acc', 0.8)
                expert_gpqa_diamond_acc = expert_model_results[0].get('metrics', {}).get('gpqa_diamond', {}).get('mean_acc', 0.7)
                expert_aime25_tokens = expert_model_results[1].get('metrics', {}).get('aime25', {}).get('mean_tokens_num', 22000)
                expert_gpqa_diamond_tokens = expert_model_results[1].get('metrics', {}).get('gpqa_diamond', {}).get('mean_tokens_num', 9000)
                
                # Calculate f1: Normalize using accuracy of aime25 and gpqa_diamond
                # Avoid division by zero
                aime25_denominator = expert_aime25_acc - base_aime25_acc
                gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
                aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
                gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
                f1 = np.mean([aime25_norm, gpqa_diamond_norm])
                
                # Calculate f2: Normalize using token count, no longer considering ifeval
                aime25_tokens_denominator = expert_aime25_tokens - base_aime25_tokens
                gpqa_diamond_tokens_denominator = expert_gpqa_diamond_tokens - base_gpqa_diamond_tokens
                aime25_tokens_norm = (aime25_tokens_num - base_aime25_tokens) / aime25_tokens_denominator
                gpqa_diamond_tokens_norm = (gpqa_diamond_tokens_num - base_gpqa_diamond_tokens) / gpqa_diamond_tokens_denominator
                f2 = np.mean([aime25_tokens_norm, gpqa_diamond_tokens_norm])
            
            objectives.append([f1, f2])
        except Exception as e:
            print(f"Error extracting objective function values: {e}")
            # Use default values
            objectives.append([-0.2, -0.2])
    
    return np.array(objectives)


def create_optimizer_config(
    custom_initial_solutions=None,
    num_blocks=8,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=50,
    verbose=True,
    device='cpu',
    dtype=torch.double,
    initial_samples=8,
    noise_level=0.0001,
    run_id="blcok_test0",
    checkpoint_dir="./checkpoints",
    optimize_density=1
) -> dict:
    """
    Create configuration for saasbo_qnehvi_optimizer function
    
    Args:
        custom_initial_solutions: list, optional
            User-defined initial solution list, e.g., [0.55, 0.8]. Two initial solutions will be created with all values 0.55 and all values 0.8, and the remaining N-2 solutions will be generated by the original algorithm
        num_blocks: int, optional
            Number of blocks, decision variable dimension: (num_blocks+1) (weights only)
        num_objectives: int, optional
            Objective function dimension: 2 objectives
        BATCH_SIZE: int, optional
            Number of samples per batch evaluation
        NUM_RESTARTS: int, optional
            Number of optimization restarts
        RAW_SAMPLES: int, optional
            Number of initial sampling points
        MC_SAMPLES: int, optional
            Number of MC samples
        N_BATCH: int, optional
            Number of iterations
        verbose: bool, optional
            Detailed output
        device: str, optional
            Computing device
        dtype: torch.dtype, optional
            Data type
        initial_samples: int, optional
            Number of initial sampling points
        noise_level: float, optional
            Noise level
        run_id: str, optional
            Run ID
        checkpoint_dir: str, optional
            Checkpoint save directory, default same as saasbo_qnehvi_optimizer
        optimize_density: int, optional
            Density optimization mode: Only supports 1 - weights only

    Returns:
        dict: Optimizer configuration dictionary
    """
    # Calculate decision variable dimensions and bounds
    # Only optimize weights, decision variable dimension: (num_blocks + 1)
    dim = num_blocks + 1
    # Variable bounds: All variables are weights, 0-1
    bounds = torch.tensor([[0.0]*dim, [1.0]*dim]).to(dtype)
    
    # Configuration parameters
    config = {
        'dim': dim,                  # Decision variable dimension
        'num_objectives': num_objectives,        # Objective function dimension: 2 objectives
        'bounds': bounds,            # Variable bounds
        'BATCH_SIZE': BATCH_SIZE,            # Number of samples per batch
        'NUM_RESTARTS': NUM_RESTARTS,         # Number of optimization restarts
        'RAW_SAMPLES': RAW_SAMPLES,         # Number of initial sampling points
        'MC_SAMPLES': MC_SAMPLES,          # Number of MC samples
        'N_BATCH': N_BATCH,              # Number of iterations
        'verbose': verbose,            # Verbose output
        'device': device,
        'dtype': dtype,
        'initial_samples': initial_samples,      # Number of initial samples
        'noise_level': noise_level,       # Noise level
        'ref_point': torch.tensor([-0.2, -0.2]).to(dtype),          # Reference point
        'run_id': run_id,     # Run ID
        'checkpoint_dir': checkpoint_dir,  # Checkpoint save directory
        'custom_initial_solutions': custom_initial_solutions  # User-defined initial solutions
    }
    
    return config

# Global variable for storing evaluation metrics
metrics_history = []

def model_merge_optimization_function(x: np.ndarray, merged_blocks: list = None, num_blocks: int = 8, cache_dir: str = None, 
                                      base_model_path="models/Qwen3-4B", task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"],
                                      fusion_method="breadcrumbs", base_model_results=None, expert_model_results=None, 
                                      optimize_density=1) -> tuple:
    """
    Model merge optimization objective function
    Maps decision variables to objective function values
    
    Args:
        x: Decision variables with shape (n_samples, dim), dim=num_blocks+1 (weights only)
        merged_blocks: Precomputed list of merged blocks to avoid redundant calculations
        cache_dir: Cache directory location
        base_model_path: Base model path
        task_model_paths: List of task model paths
        fusion_method: Model fusion method
        base_model_results: List of evaluation results for base models
        expert_model_results: List of evaluation results for expert models
        optimize_density: int, optional
            Density optimization mode: Only supports 1 - No density optimization, only optimize existing weights
    
    Returns:
        tuple: Tuple containing objective function values and evaluation metrics
            - Objective function values: Shape (n_samples, n_obj), here n_obj=2
            - Evaluation metrics: Shape (n_samples,), each element is an evaluation result dictionary
    """
    n_samples = x.shape[0]
    
    print(f"\nProcessing {n_samples} samples...")
    
    # Create temporary output directory
    if cache_dir is None:
        cache_dir = os.path.join('output', 'mi_optimization_temp')
    output_dir = os.path.join(cache_dir, str(int(time.time())))
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default max_tokens value
    max_tokens = 35000
    max_model_len = None  # Will be automatically calculated as max_tokens + 3000 in process_decision_variables
    
    # Process all decision variables at once
    objectives, metrics = process_decision_variables(
        decision_matrix=x,
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        base_output_dir=output_dir,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        merged_blocks=merged_blocks,  # Pass precomputed merged blocks
        num_blocks=num_blocks,  # Pass number of blocks
        fusion_method=fusion_method,
        base_model_results=base_model_results,
        expert_model_results=expert_model_results,
        optimize_density=optimize_density  # Pass density optimization parameter
    )
    
    # Save evaluation metrics to global variable
    metrics_history.extend(metrics)
    
    # Ensure returned result shape is correct
    expected_shape = (n_samples, 2)
    if objectives.shape != expected_shape:
        print(f"Warning: process_decision_variables returned incorrect result shape: {objectives.shape}, expected {expected_shape}")
        # Create result array with correct shape
        result = np.zeros(expected_shape)
        # Copy available results
        min_samples = min(n_samples, objectives.shape[0])
        if min_samples > 0:
            result[:min_samples] = objectives[:min_samples]
        return torch.from_numpy(result), metrics
    
    return torch.from_numpy(objectives), metrics


def create_iteration_callback(cache_dir: str, cleanup_interval: int = 1, 
                             keep_dirs: list = None, exclude_patterns: list = None):
    """
    Create iteration callback function for cleaning cache during optimization
    
    Args:
        cache_dir: Cache directory to clean
        cleanup_interval: Cleaning interval (number of iterations)
        keep_dirs: List of directories to keep
        exclude_patterns: List of patterns to exclude
    
    Returns:
        Callable: Callback function
    """
    if keep_dirs is None:
        keep_dirs = ['important', 'keep']
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    def callback(iteration, x, y, hypervolume):
        """Optimization iteration callback function"""
        # Print current iteration information
        print(f"\n===== Iteration {iteration+1} ====")
        print(f"Current hypervolume: {hypervolume[-1]:.6f}")
        
        # Clean cache periodically
        if (iteration + 1) % cleanup_interval == 0:
            print(f"\nCleaning cache directory: {cache_dir}")
            clean_temp_files(cache_dir, keep_dirs, exclude_patterns)
    
    return callback


def clean_temp_files(directory, keep_dirs=None, exclude_patterns=None):
    """
    Clean temporary files and directories
    
    Args:
        directory: Directory to clean
        keep_dirs: List of directories to keep
        exclude_patterns: List of patterns to exclude
    
    Returns:
        dict: Cleaning statistics
    """
    if keep_dirs is None:
        keep_dirs = []
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    deleted_files = 0
    deleted_dirs = 0
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return {'deleted_files': 0, 'deleted_dirs': 0}
    
    try:
        # Get all items in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            should_keep = False
            
            # Check if in keep_dirs list
            if any(keep_dir in item_path for keep_dir in keep_dirs):
                should_keep = True
            
            # Check if matches exclude patterns
            if not should_keep:
                for pattern in exclude_patterns:
                    if pattern in item:
                        should_keep = True
                        break
            
            # Delete if should not keep
            if not should_keep:
                if os.path.isdir(item_path):
                    try:
                        import shutil
                        shutil.rmtree(item_path)
                        deleted_dirs += 1
                        print(f"Deleted directory: {item_path}")
                    except Exception as e:
                        print(f"Failed to delete directory {item_path}: {e}")
                elif os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        deleted_files += 1
                        print(f"Deleted file: {item_path}")
                    except Exception as e:
                        print(f"Failed to delete file {item_path}: {e}")
    
    except Exception as e:
        print(f"Error cleaning cache: {e}")
    
    stats = {'deleted_files': deleted_files, 'deleted_dirs': deleted_dirs}
    print(f"Cache cleaning completed: Deleted {deleted_files} files and {deleted_dirs} directories")
    return stats


def visualize_optimization_results(result_dict: dict, output_dir: str):
    """
    Visualize optimization results (using VisualizationTool)
    
    Args:
        result_dict: Optimization result dictionary
        output_dir: Output directory
    """
    # Use visualization tool class for plotting
    visualizer.visualize_optimization_results(result_dict, output_dir)
    print(f"Visualization results saved to {output_dir}")


def save_optimization_results(result_dict: dict, output_dir: str):
    """
    Save optimization results to files
    
    Parameters:
        result_dict: Optimization results dictionary
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get Pareto front
    pareto_x = result_dict.get('pareto_x', np.array([]))
    pareto_y = result_dict.get('pareto_y', np.array([]))
    
    # Save Pareto decision variables
    np.save(os.path.join(output_dir, 'pareto_decision_variables.npy'), pareto_x)
    
    # Save Pareto objective values
    np.save(os.path.join(output_dir, 'pareto_objectives.npy'), pareto_y)
    
    # Save all evaluated points
    np.save(os.path.join(output_dir, 'all_evaluated_variables.npy'), result_dict.get('all_x', np.array([])))
    np.save(os.path.join(output_dir, 'all_evaluated_objectives.npy'), result_dict.get('all_y', np.array([])))
    
    # Save as JSON format (human-readable), without parameters
    import json
    results = {
        'pareto_solutions': [
            {
                'decision_variables': x.tolist(),
                'objectives': y.tolist()
            }
            for x, y in zip(pareto_x, pareto_y)
        ],
        'hypervolume_history': result_dict.get('hypervolume_history', []),
        'total_evaluations': len(result_dict.get('all_y', [])),
        'best_hypervolume': max(result_dict.get('hypervolume_history', [0])) if result_dict.get('hypervolume_history') else 0
    }
    
    with open(os.path.join(output_dir, 'optimization_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Optimization results saved to {output_dir}")


def save_settings(params: dict, output_dir: str):
    """
    Save optimization settings to setting.json file
    
    Parameters:
        params: Optimization parameter dictionary containing all configuration parameters
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    settings_path = os.path.join(output_dir, 'setting.json')
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"Optimization settings saved to {settings_path}")


def compute_layer_importance(merged_blocks, num_blocks, optimize_density=1):
    """
    Compute layer importance based on differences between task model layers
    
    Parameters:
        merged_blocks: List of merged blocks, each containing layer list and difference value
        num_blocks: Number of blocks
        optimize_density: Density optimization mode (only supports 1)
    
    Returns:
        torch.Tensor: Layer importance weights, normalized to 0-1 range, shape (num_blocks+1,)
    """
    print("\n===== Computing Layer Importance =====")
    
    # Extract layer differences
    layer_differences = []
    for block_info in merged_blocks:
        # calculate_merged_blocks returns a list of tuples, each containing (layer index list, difference score)
        # block_info[0] is layer index list, block_info[1] is difference score
        diff = abs(block_info[1])
        layer_differences.append(diff)
    
    # Add difference for the last dimension (norm and lm_head), using average
    layer_differences.append(np.mean(layer_differences))
    
    print(f"Layer difference raw values: {layer_differences}")
    
    # Normalize to 0-1 range
    diff_array = np.array(layer_differences)
    min_diff = diff_array.min()
    max_diff = diff_array.max()
    
    # Avoid division by zero
    if max_diff - min_diff < 1e-8:
        importance = np.ones_like(diff_array) * 0.5
    else:
        importance = (diff_array - min_diff) / (max_diff - min_diff)
    
    print(f"Raw layer importance: {importance}")
    
    # Only optimize weights, return original importance array
    final_importance = importance
    print(f"Layer importance: {final_importance}")
    print(f"Importance shape: {final_importance.shape}")
    
    print("===== Layer Importance Computation Complete =====")
    
    return torch.tensor(final_importance, dtype=torch.float64)


def main_optimization(
    custom_initial_solutions=None,
    num_blocks=8,
    num_objectives=2,
    BATCH_SIZE=4,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=50,
    verbose=True,
    device='cpu',
    dtype=torch.double,
    initial_samples=8,
    noise_level=0.0001,
    run_id="blcok_test0",
    cache_dir="output/mi_optimization_temp",
    alpha=1.0,
    beta=0.005,
    # Model path parameters
    base_model=['models/Qwen3-4B','models/Qwen3-4B-thinking-2507'],
    expert_model=['models/Qwen3-4B-thinking-2507', 'models/Qwen3-4B'],
    base_model_path="models/Qwen3-4B",
    task_model_paths=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B"],
    # Algorithm parameters
    algorithm="saasbo_qnehvi",  # Optimization algorithm selection, only supports "saasbo_qnehvi"
    use_saas=True,
    enable_importance_prior=True,
    fusion_method="breadcrumbs",
    # New parameters
    optimize_density=1,
    # Distance metric and partition method parameters
    metric="L2-norm",
    partition_method="hybrid"
):
    """
    Main function - Execute model merging optimization process
    
    Parameters:
        custom_initial_solutions: list, optional
            User-defined initial solutions list. For example, [0.55, 0.8] will set two initial solutions: all 0.55s and all 0.8s, then generate N-2 initial solutions using the original algorithm
        num_blocks: int, optional
            Number of blocks
        num_objectives: int, optional
            Objective function dimension: 2 objectives
        BATCH_SIZE: int, optional
            Number of samples per evaluation batch
        NUM_RESTARTS: int, optional
            Number of optimization restarts
        RAW_SAMPLES: int, optional
            Number of initial sampling points
        MC_SAMPLES: int, optional
            Number of MC samples
        N_BATCH: int, optional
            Number of iterations
        verbose: bool, optional
            Verbose output
        device: str, optional
            Computing device
        dtype: torch.dtype, optional
            Data type
        initial_samples: int, optional
            Number of initial samples
        noise_level: float, optional
            Noise level
        run_id: str, optional
            Run ID
        # Model path parameters
        base_model: list, optional
            List of base model paths
        expert_model: list, optional
            List of expert model paths
        base_model_path: str, optional
            Base model path
        task_model_paths: list, optional
            List of task model paths
        # Algorithm parameters
        algorithm: str, optional
            Optimization algorithm selection, only supports "saasbo_qnehvi": uses SAASBO+qNEHVI algorithm
        use_saas: bool, optional
            Whether to use SAAS prior
        enable_importance_prior: bool, optional
            Whether to enable importance prior
        fusion_method: str, optional
            Model fusion method
        optimize_density: int, optional
            Density optimization mode: only supports 1 - no density optimization, only optimize existing weights
    """
    # Set checkpoint directory, consistent with saasbo_qnehvi_optimizer
    checkpoint_dir = "./checkpoints"
    
    # Generate run_id if it's None
    if run_id is None:
        run_id = time.strftime('%Y%m%d_%H%M%S')
    
    # Build complete checkpoint path: ./checkpoints/[run_id]
    checkpoint_run_dir = os.path.join(checkpoint_dir, run_id)
    
    # Create output directory, placed under checkpoint directory
    output_root = os.path.join(checkpoint_run_dir, 'output')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting model merging optimization process, output directory: {output_dir}")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Build parameter dictionary, save to setting.json file
    params_dict = {
        'custom_initial_solutions': custom_initial_solutions,
        'num_blocks': num_blocks,
        'num_objectives': num_objectives,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_RESTARTS': NUM_RESTARTS,
        'RAW_SAMPLES': RAW_SAMPLES,
        'MC_SAMPLES': MC_SAMPLES,
        'N_BATCH': N_BATCH,
        'verbose': verbose,
        'device': device,
        'initial_samples': initial_samples,
        'noise_level': noise_level,
        'run_id': run_id,
        'checkpoint_dir': checkpoint_dir,
        'cache_dir': cache_dir,
        'alpha': alpha,
        'beta': beta,
        'base_model': base_model,
        'expert_model': expert_model,
        'base_model_path': base_model_path,
        'task_model_paths': task_model_paths,
        'algorithm': algorithm,
        'use_saas': use_saas,
        'enable_importance_prior': enable_importance_prior,
        'fusion_method': fusion_method,
        'dtype': str(dtype),
        'optimize_density': optimize_density,
        'metric': metric,
        'partition_method': partition_method
    }
    
    # 保存设置到saasbo_qnehvi_optimizer的checkpoint目录中
    print("\n保存设置到checkpoint目录...")
    save_settings(params_dict, checkpoint_run_dir)
    
    # 初始化时计算所有需要的合并块，并保存图形到output_dir
    print("\n初始化时计算所有需要的合并块...")
    from src.sipbmm.block_fusion import calculate_merged_blocks
    
    # 收集所有需要的块数
    required_block_counts = set()
    
    # 只使用固定的num_blocks
    required_block_counts.add(num_blocks)
    
    # 确保36（真实层数）也被计算，用于精细搜索
    required_block_counts.add(36)
    
    # 计算并存储每个块数对应的merged_blocks
    merged_blocks_dict = {}
    initial_importance_dict = {}
    
    # 转换为列表并按升序排列
    sorted_block_counts = sorted(required_block_counts)
    
    # 逐个计算每个块数
    print(f"\n逐个计算所有块数: {sorted_block_counts}")
    
    for n_blocks in sorted_block_counts:
        print(f"\n计算 {n_blocks} 个块的合并结果...")
        merged_blocks = calculate_merged_blocks(
            task_model_paths=task_model_paths,
            num_blocks=n_blocks,
            alpha=alpha,
            beta=beta,
            checkpoint_dir=output_dir,  # 保存图形到输出目录
            metric=metric,
            partition_method=partition_method
        )
        merged_blocks_dict[n_blocks] = merged_blocks
    
    # 计算每个块数对应的层重要性
    for n_blocks in sorted_block_counts:
        merged_blocks = merged_blocks_dict[n_blocks]
        initial_importance = compute_layer_importance(
            merged_blocks=merged_blocks,
            num_blocks=n_blocks,
            optimize_density=optimize_density
        )
        initial_importance_dict[n_blocks] = initial_importance
    
    # 默认使用固定num_blocks
    default_n_blocks = num_blocks
    initial_importance = initial_importance_dict[default_n_blocks]
    
    # 创建优化器配置，添加checkpoint_dir参数
    print("\n初始化优化配置...")
    config = create_optimizer_config(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=num_blocks,
        num_objectives=num_objectives,
        BATCH_SIZE=BATCH_SIZE,
        NUM_RESTARTS=NUM_RESTARTS,
        RAW_SAMPLES=RAW_SAMPLES,
        MC_SAMPLES=MC_SAMPLES,
        N_BATCH=N_BATCH,
        verbose=verbose,
        device=device,
        dtype=dtype,
        initial_samples=initial_samples,
        noise_level=noise_level,
        run_id=run_id,
        checkpoint_dir=checkpoint_dir,
        optimize_density=optimize_density
    )
    

    
    # 运行优化
    print("\n开始优化过程...")
    start_time = time.time()
    
    # 创建缓存清理回调函数，每5轮迭代清理一次
    exclude_patterns = ['pareto', 'best', 'important']  # 需要排除的模式
    iteration_callback = create_iteration_callback(cache_dir, cleanup_interval=1, exclude_patterns=exclude_patterns)
    
    # 初始化模型评测结果
    base_model_results, expert_model_results = initialize_model_evaluations(base_model, expert_model)
    
    # 创建包装函数，将所有必要参数传递给model_merge_optimization_function
    def wrapped_optimization_function(x):
        # 根据x的维度动态调整num_blocks
        # 仅优化权重: num_blocks = x.shape[1] - 1
        if len(x.shape) > 1:
            current_dim = x.shape[1]
            current_num_blocks = current_dim - 1
        else:
            current_num_blocks = num_blocks  # 使用默认值
        
        # 确保current_num_blocks与传入的块数匹配
        print(f"wrapped_optimization_function: x.shape[1]={x.shape[1]}, current_num_blocks={current_num_blocks}")
        
        # 从merged_blocks_dict中获取对应的merged_blocks
        if current_num_blocks in merged_blocks_dict:
            current_merged_blocks = merged_blocks_dict[current_num_blocks]
            print(f"使用块数 {current_num_blocks} 对应的merged_blocks，包含 {len(current_merged_blocks)} 个块")
            
            # 检查merged_blocks包含的块数是否与current_num_blocks匹配
            if len(current_merged_blocks) != current_num_blocks:
                print(f"警告：merged_blocks包含的块数 {len(current_merged_blocks)} 与current_num_blocks {current_num_blocks} 不匹配，重新计算")
                # 重新计算merged_blocks
                from src.evoMI.mi_block_fusion import calculate_merged_blocks
                current_merged_blocks = calculate_merged_blocks(
                    task_model_paths=task_model_paths,
                    num_blocks=current_num_blocks,
                    alpha=alpha,
                    beta=beta,
                    checkpoint_dir=cache_dir  # 保存图形到缓存目录
                )
                # 更新merged_blocks_dict
                merged_blocks_dict[current_num_blocks] = current_merged_blocks
                print(f"重新计算后merged_blocks包含 {len(current_merged_blocks)} 个块")
        else:
            # 如果没有对应的merged_blocks，使用默认的
            default_n_blocks = num_blocks
            current_merged_blocks = merged_blocks_dict[default_n_blocks]
            print(f"使用默认块数 {default_n_blocks} 对应的merged_blocks，包含 {len(current_merged_blocks)} 个块")
            
            # 检查merged_blocks包含的块数是否与current_num_blocks匹配
            if len(current_merged_blocks) != current_num_blocks:
                print(f"警告：merged_blocks包含的块数 {len(current_merged_blocks)} 与current_num_blocks {current_num_blocks} 不匹配，重新计算")
                # 重新计算merged_blocks
                from src.evoMI.mi_block_fusion import calculate_merged_blocks
                current_merged_blocks = calculate_merged_blocks(
                    task_model_paths=task_model_paths,
                    num_blocks=current_num_blocks,
                    alpha=alpha,
                    beta=beta,
                    checkpoint_dir=cache_dir  # 保存图形到缓存目录
                )
                # 更新merged_blocks_dict
                merged_blocks_dict[current_num_blocks] = current_merged_blocks
                print(f"重新计算后merged_blocks包含 {len(current_merged_blocks)} 个块")
        
        return model_merge_optimization_function(
            x, 
            merged_blocks=current_merged_blocks, 
            num_blocks=current_num_blocks, 
            cache_dir=cache_dir,
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            fusion_method=fusion_method,
            base_model_results=base_model_results,
            expert_model_results=expert_model_results,
            optimize_density=optimize_density
        )
    
    # 调用saasbo_qnehvi_optimizer
    result = saasbo_qnehvi_optimizer(
        wrapped_optimization_function, 
        iteration_callback=iteration_callback, 
        initial_importance=initial_importance,  # 传入层重要性
        use_saas=use_saas,  # 使用SAAS先验
        enable_importance_prior=enable_importance_prior,  # 启用重要性先验
        **config
    )
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\n优化完成！总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/3600:.2f} 小时)")
    
    # 构建结果字典
    # saasbo_qnehvi_optimizer返回值：train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    result_dict = {
        'pareto_x': result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        'pareto_y': result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        'all_x': result[0].cpu().numpy() if isinstance(result[0], torch.Tensor) else result[0],
        'all_y': result[1].cpu().numpy() if isinstance(result[1], torch.Tensor) else result[1],
        'all_metrics': result[2],  # 保存所有评测指标
        'hypervolume_history': result[3] if len(result) > 3 else [],
        'problem_ref_point': result[4].tolist() if isinstance(result[4], torch.Tensor) else result[4],
        'run_id': result[5] if len(result) > 5 else None
    }
    

    
    # 保存结果，不再包含参数
    print("\n保存优化结果...")
    save_optimization_results(result_dict, output_dir)
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_optimization_results(result_dict, output_dir)
    
    # 获取帕累托前沿
    pareto_x = result_dict.get('pareto_x', np.array([]))
    pareto_y = result_dict.get('pareto_y', np.array([]))
    print(f"\nFound {len(pareto_x)} Pareto optimal solutions")
    
    # Print optimization statistics
    print("\n=== Optimization Statistics ===")
    print(f"总评估次数: {len(result_dict.get('all_x', []))}")
    print(f"初始样本数: {config['initial_samples']}")
    print(f"迭代次数: {config['N_BATCH']}")
    print(f"批次大小: {config['BATCH_SIZE']}")
    try:
        hypervolume_history = result_dict.get('hypervolume_history', [0])
        best_hypervolume = max(hypervolume_history) if hypervolume_history else 0
        print(f"最佳超体积: {best_hypervolume}")
    except:
        print("最佳超体积: 计算失败")
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="模型合并优化工具 - 使用SAASBO+qNEHVI算法 - 版本2")
    
    # Add command line arguments
    parser.add_argument('--custom-initial-solutions', type=str, default=None,
                        help='User-defined initial solutions list, e.g., "0.55,0.8" will set two initial solutions as all 0.55s and all 0.8s, then generate N-2 initial solutions using the original algorithm')
    parser.add_argument('--num-blocks', type=int, default=36,
                        help='Number of blocks, the number of optimization parameters is (number of blocks + 1) * 2 (first (block+1) are weight parameters, next (block+1) are density parameters)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Variance weight, default: 1.0')
    parser.add_argument('--beta', type=float, default=0.000,
                        help='Balance weight, default: 0.005')
    parser.add_argument('--num-objectives', type=int, default=2,
                        help='Objective function dimension: 2 objectives')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of samples per evaluation batch')
    parser.add_argument('--num-restarts', type=int, default=10,
                        help='Number of optimization restarts')
    parser.add_argument('--raw-samples', type=int, default=512,
                        help='Number of initial sampling points')
    parser.add_argument('--mc-samples', type=int, default=128,
                        help='Number of MC samples')
    parser.add_argument('--n-batch', type=int, default=50,
                        help='Number of iterations')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device')
    parser.add_argument('--initial-samples', type=int, default=8,
                        help='Number of initial samples')
    parser.add_argument('--noise-level', type=float, default=0.0001,
                        help='Noise level')
    parser.add_argument('--run-id', type=str, default="instruct_saasbo_qnehvi_prior_grassmann_wasserstein_block_36",
                        help='Run ID')
    parser.add_argument('--cache-dir', type=str, default="output/mi_optimization_temp",
                        help='Cache directory location /mnt/data/output/mi_optimization_temp')
    parser.add_argument('--available-gpus', type=str, default="0,1,2,3",
                        help='Available GPU list, separated by commas, e.g.: 0,1,2')
    
    # 模型路径参数
    # 模型路径参数
    parser.add_argument('--base-model', type=str, default="models/Qwen3-4B-Instruct-2507,models/Qwen3-4B-thinking-2507",
                        help='Base model path list, separated by commas, e.g.: models/Qwen3-4B-Instruct-2507,models/Qwen3-4B-thinking-2507')
    parser.add_argument('--expert-model', type=str, default="models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507",
                        help='Expert model path list, separated by commas, e.g.: models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507')
    parser.add_argument('--base-model-path', type=str, default="models/Qwen3-4B-Instruct-2507",
                        help='Base model path')
    parser.add_argument('--task-model-paths', type=str, default="models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507",
                        help='Task model path list, separated by commas, e.g.: models/Qwen3-4B-thinking-2507,models/Qwen3-4B-Instruct-2507')
    
    
    # Algorithm selection parameters
    parser.add_argument('--algorithm', type=str, default="saasbo_qnehvi",
                        help='Optimization algorithm selection, only supports "saasbo_qnehvi"')
    
    # Algorithm parameters
    parser.add_argument('--use-saas', type=bool, default=True,
                        help='Whether to use SAAS prior')
    parser.add_argument('--enable-importance-prior', type=bool, default=True,
                        help='Whether to enable importance prior')
    parser.add_argument('--fusion-method', type=str, default="task_arithmetic",
                        help='Model fusion method, e.g.: breadcrumbs, task_arithmetic, etc.')
    
    # Distance metric and partition method parameters
    parser.add_argument('--metric', type=str, default='L2-norm',
                        help='Distance metric method, default: "Grassmann", options: "Fisher", "Grassmann", "L2-norm", "Block", "Grassmann-Wasserstein", "LayerNorm-Wasserstein"')
    parser.add_argument('--partition-method', type=str, default='hybrid',
                        help='Partition method, default: "hybrid", options: "hybrid", "balance", "variance"')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置可用GPU列表
    available_gpus_global = [int(gpu.strip()) for gpu in args.available_gpus.split(',')]
    
    # 处理自定义初始解参数
    custom_initial_solutions = None
    if args.custom_initial_solutions:
        try:
            custom_initial_solutions = [float(val.strip()) for val in args.custom_initial_solutions.split(',')]
            print(f"使用自定义初始解: {custom_initial_solutions}")
        except ValueError:
            print(f"警告: 自定义初始解参数格式错误: {args.custom_initial_solutions}，将使用默认初始解生成方式")
    
    # 解析模型路径列表
    base_model = [model.strip() for model in args.base_model.split(',')]
    expert_model = [model.strip() for model in args.expert_model.split(',')]
    task_model_paths = [model.strip() for model in args.task_model_paths.split(',')]
    

    
    # Call main function, passing all parameters
    main_optimization(
        custom_initial_solutions=custom_initial_solutions,
        num_blocks=args.num_blocks,
        num_objectives=args.num_objectives,
        BATCH_SIZE=args.batch_size,
        NUM_RESTARTS=args.num_restarts,
        RAW_SAMPLES=args.raw_samples,
        MC_SAMPLES=args.mc_samples,
        N_BATCH=args.n_batch,
        verbose=args.verbose,
        device=args.device,
        initial_samples=args.initial_samples,
        cache_dir=args.cache_dir,
        noise_level=args.noise_level,
        run_id=args.run_id,
        alpha=args.alpha,
        beta=args.beta,
        base_model=base_model,
        expert_model=expert_model,
        base_model_path=args.base_model_path,
        task_model_paths=task_model_paths,
        algorithm=args.algorithm,
        use_saas=args.use_saas,
        enable_importance_prior=args.enable_importance_prior,
        fusion_method=args.fusion_method,
        metric=args.metric,
        partition_method=args.partition_method
    )
