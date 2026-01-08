#!/usr/bin/env python3
"""
Model-level Fusion Test Script
Tests model-level fusion effects under different weights [0.1-0.9], supporting multiple fusion methods
"""

import os
import sys
import time
import uuid
import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple
import tempfile

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Import required modules
from src.vllm_server_manager import VllmServerManager
from src.config_manager import config_manager
from src.sipbmm.block_fusion import mi_block_fusion, calculate_merged_blocks
from src.sipbmm.tool import ResultProcessor, generate_model_cache_key, get_model_cache_path, load_cached_results, save_results_to_cache

# Global variables
base_model = config_manager._base_model_bi
expert_model = config_manager._expert_model_bi
checkpoint_dir = config_manager.checkpoint_dir
available_gpus_global = [0,1,2,3]

def initialize_model_evaluations(max_tokens: int = 35000, max_model_len: int = None):
    """
    Initialize evaluation results for base and expert models
    
    Args:
        max_tokens: Maximum number of generated tokens
        max_model_len: Maximum model length
    """
    global base_model_results, expert_model_results
    
    print("\n===== Initializing Model Evaluation Results =====")
    
    # Evaluate base models
    print(f"\nEvaluating base models: {base_model}")
    base_model_results = []
    
    for model_path in base_model:
        model_key = generate_model_cache_key(model_path)
        cache_path = get_model_cache_path(checkpoint_dir, model_key, 'original')
        cached_result = load_cached_results(cache_path)
        
        if cached_result:
            print(f"Using cached result for base model: {model_path}")
            base_model_results.append(cached_result)
        else:
            print(f"No cached result found for base model, starting evaluation: {model_path}")
            
            # Implement evaluation logic directly, using bi-objective configuration
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            # Create evaluation task configuration - using bi-objective configuration
            task_cfg = config_manager.create_biojective_eval_task_config(model_path, max_tokens)
            
            # Create task dictionary
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            # Use VllmServerManager to run the task
            print(f"Starting evaluation for model: {model_path}")
            start_time = time.time()
            try:
                # If max_model_len is not specified, set to max_tokens + 3000
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"max_model_len not specified, using default: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=max_model_len) as server_manager:
                    # Call run_series_tasks method to execute the task
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
                        # Simply extract all metrics from all datasets, use key as metric name
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
                print(f"Error during evaluation: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
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
            print(f"Using cached result for expert model: {model_path}")
            expert_model_results.append(cached_result)
        else:
            print(f"No cached result found for expert model, starting evaluation: {model_path}")
            
            # Implement evaluation logic directly, using bi-objective configuration
            model_id = f"original_{os.path.basename(model_path)}_{uuid.uuid4().hex[:8]}"
            
            # Create evaluation task configuration - using bi-objective configuration
            task_cfg = config_manager.create_biojective_eval_task_config(model_path, max_tokens)
            
            # Create task dictionary
            task = {
                'task_id': f'task_{model_id}',
                'model_path': model_path,
                'params_dict': {'task_cfg': task_cfg},
                'func_handle': run_task_with_server
            }
            
            # Use VllmServerManager to run the task
            print(f"Starting evaluation for model: {model_path}")
            start_time = time.time()
            try:
                # If max_model_len is not specified, set to max_tokens + 3000
                if max_model_len is None:
                    max_model_len = max_tokens + 3000
                    print(f"max_model_len not specified, using default: {max_model_len} (max_tokens + 3000)")
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=max_model_len) as server_manager:
                    # Call run_series_tasks method to execute the task
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
                        # Simply extract all metrics from all datasets, use key as metric name
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
                print(f"Error during evaluation: {e}")
                metrics = {"error": str(e)}
            
            model_result = {
                'model_type': 'thinking' if 'thinking' in model_path.lower() else 'instruct',
                'model_name': os.path.basename(model_path),
                'model_path': model_path,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            expert_model_results.append(model_result)
            # Save to cache
            save_results_to_cache(cache_path, model_result)
    
    print("\n===== Model Evaluation Initialization Completed =====")
    return base_model_results, expert_model_results

def run_task_with_server(port, task_cfg):
    """
    Execute task on server at specified port
    
    Args:
        port: Server port
        task_cfg: Task configuration object
    
    Returns:
        Task execution results
    """
    # Update API URL to use correct port
    task_cfg.api_url = f'http://127.0.0.1:{port}/v1/chat/completions'
    
    print(f"Executing task on port {port}: model={task_cfg.model}, datasets={task_cfg.datasets}")
    
    try:
        # Execute task
        from evalscope import run_task
        result = run_task(task_cfg=task_cfg)
        print(f"Task execution completed on port {port}")
        return result
    except Exception as e:
        print(f"Task execution error on port {port}: {e}")
        return {"error": str(e)}

def extract_objectives(results, base_model_results=None, expert_model_results=None):
    """
    Extract objective function values from evaluation results (using dynamic normalization)
    
    Args:
        results: Evaluation results dictionary
        base_model_results: Base model evaluation results
        expert_model_results: Expert model evaluation results
    
    Returns:
        numpy array: Array of shape (n, 2), each row contains two objective function values
    """
    objectives = []
    
    # Check results structure
    if isinstance(results, dict) and 'processed_results' in results:
        results_list = results['processed_results']
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = [results]
    
    for result in results_list:
        try:
            # Extract various metrics
            aime25_acc = result['aime25'].get('mean_acc', 0) if 'aime25' in result else 0
            aime25_tokens_num = result['aime25'].get('mean_tokens_num', 0) if 'aime25' in result else 0
            gpqa_diamond_acc = result['gpqa_diamond'].get('mean_acc', 0) if 'gpqa_diamond' in result else 0
            gpqa_diamond_tokens_num = result['gpqa_diamond'].get('mean_tokens_num', 0) if 'gpqa_diamond' in result else 0
            
            # Ensure base_model_results and expert_model_results are valid
            if not base_model_results or not expert_model_results:
                print("Warning: base_model_results or expert_model_results not initialized, using default values")
                # Use default values as fallback
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
                
                # Calculate f1: Normalize using aime25 and gpqa_diamond accuracy
                # Avoid division by zero
                aime25_denominator = expert_aime25_acc - base_aime25_acc
                gpqa_diamond_denominator = expert_gpqa_diamond_acc - base_gpqa_diamond_acc
                aime25_norm = (aime25_acc - base_aime25_acc) / aime25_denominator
                gpqa_diamond_norm = (gpqa_diamond_acc - base_gpqa_diamond_acc) / gpqa_diamond_denominator
                f1 = np.mean([aime25_norm, gpqa_diamond_norm])
                
                # Calculate f2: Normalize using token count, no longer consider ifeval
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

def test_model_level_fusion(
    fusion_method: str = "task_arithmetic",
    num_blocks: int = 8,
    max_tokens: int = 35000,
    max_model_len: int = None,
    run_id: str = "model_level_test",
    batch_size: int = 3,
    weight_min: float = 0.1,
    weight_max: float = 0.9,
    density_min: float = 0.1,
    density_max: float = 0.9,
    budget: int = 10
):
    """
    Test model-level fusion effects using different weight and density parameters
    
    Args:
        fusion_method: Fusion method
        num_blocks: Number of blocks
        max_tokens: Maximum number of generated tokens
        max_model_len: Maximum model length
        run_id: Run ID
        batch_size: Batch size, number of models processed each time
        weight_min: Minimum weight value
        weight_max: Maximum weight value
        density_min: Minimum density value (used by non-task_arithmetic methods)
        density_max: Maximum density value (used by non-task_arithmetic methods)
        budget: Total computational budget count
    """
    # Model path settings
    base_model_path = "models/Qwen3-4B-Instruct-2507"
    task_model_paths = ["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"]
    
    # Output directory, including fusion_method identifier
    output_root = f"checkpoints/{run_id}_{fusion_method}"
    os.makedirs(output_root, exist_ok=True)
    
    # Create temporary model storage directory
    temp_model_dir = "output/model_level_temp"
    os.makedirs(temp_model_dir, exist_ok=True)
    
    # Calculate merged_blocks once during initialization
    print(f"\n===== Calculating Automatically Merged Blocks (Run Once) =====")
    merged_blocks = calculate_merged_blocks(
        task_model_paths=task_model_paths,
        num_blocks=num_blocks,
        checkpoint_dir=output_root
    )
    
    # Initialize base and expert model evaluation results
    base_results, expert_results = initialize_model_evaluations(max_tokens, max_model_len)
    
    # Generate grid search parameters
    print(f"\n===== Generating Grid Search Parameters =====")
    print(f"Fusion method: {fusion_method}")
    print(f"Weight range: [{weight_min}, {weight_max}]")
    if fusion_method != 'task_arithmetic':
        print(f"Density range: [{density_min}, {density_max}]")
    print(f"Computational budget: {budget}")
    
    # Generate parameter combinations
    param_combinations = []
    
    if fusion_method == 'task_arithmetic':
        # Only needs weight parameters
        num_weight_points = budget
        weights = np.linspace(weight_min, weight_max, num_weight_points)
        for weight in weights:
            param_combinations.append({
                'weight': weight,
                'density': None
            })
    else:
        # Needs weight and density parameters
        # Calculate weight and density ranges
        weight_range = weight_max - weight_min
        density_range = density_max - density_min

        print(f"Weight range size: {weight_range:.4f}")
        print(f"Density range size: {density_range:.4f}")
        
        # Calculate range ratio
        if weight_range == 0:
            weight_range = 1.0
        if density_range == 0:
            density_range = 1.0
        
        range_ratio = weight_range / density_range
        print(f"Range ratio (weight:density): {range_ratio:.4f}")
        
        # Allocate grid points according to range ratio, ensuring total combinations are close to budget
        # Use geometric allocation method, so that product of weight and density grid points is close to budget
        total_grid_points = max(4, budget)  # Ensure at least 4 points (2x2)
        
        # Calculate number of points for each parameter based on range ratio
        if range_ratio >= 1.0:
            # Weight range is larger, allocate more points
            num_density_points = max(2, int(np.sqrt(total_grid_points / range_ratio)))
            num_weight_points = max(2, int(total_grid_points / num_density_points))
        else:
            # Density range is larger, allocate more points
            num_weight_points = max(2, int(np.sqrt(total_grid_points * range_ratio)))
            num_density_points = max(2, int(total_grid_points / num_weight_points))
        
        # Adjust to ensure product does not exceed budget and is close to budget
        actual_total = num_weight_points * num_density_points
        while actual_total > budget and (num_weight_points > 2 or num_density_points > 2):
            if num_weight_points >= num_density_points:
                num_weight_points -= 1
            else:
                num_density_points -= 1
            actual_total = num_weight_points * num_density_points
        
        print(f"Allocated grid points: weight={num_weight_points}, Density={num_density_points}")
        print(f"Expected total combinations: {num_weight_points * num_density_points}")
        
        # Generate weight and density grids
        weights = np.linspace(weight_min, weight_max, num_weight_points)
        densities = np.linspace(density_min, density_max, num_density_points)
        
        # Create grid and flatten
        weight_grid, density_grid = np.meshgrid(weights, densities)
        weight_flat = weight_grid.flatten()
        density_flat = density_grid.flatten()
        
        # Add all combinations (now total is close to budget)
        for weight, density in zip(weight_flat, density_flat):
            param_combinations.append({
                'weight': weight,
                'density': density
            })
    
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    # Store all results
    all_decision_variables = []
    all_objectives = []
    all_metrics = []
    skipped_params = []
    
    # Split parameter combinations into multiple batches
    param_batches = [param_combinations[i:i+batch_size] for i in range(0, len(param_combinations), batch_size)]
    
    # Process each batch
    for batch_idx, batch_params in enumerate(param_batches):
        print(f"\n===== Processing Batch {batch_idx+1}/{len(param_batches)}, containing {len(batch_params)} parameter combinations =====")
        
        # Create task list and related mappings
        batch_tasks = []
        batch_task_info_map = {}
        
        # Phase 1: Create models and tasks for current batch
        for params in batch_params:
            weight = params['weight']
            density = params['density']
            
            # Generate model-level decision variables: all blocks use the same weight
            # Decision variable format: num_blocks block weights + 1 embedding weight + 1 norm/lm_head weight
            # For model-level fusion, all blocks use the same weight
            decision_vars = [weight] * (num_blocks + 1)
            
            print(f"\n===== Preparing Parameters ====")
            print(f"Weight: {weight:.3f}")
            if density is not None:
                print(f"Density: {density:.3f}")
            print(f"Decision variables: {decision_vars}")
            
            # Generate unique model ID containing weight and density parameters
            if density is None:
                model_id = f"model_level_{fusion_method}_w{weight:.3f}_{uuid.uuid4().hex[:8]}"
            else:
                model_id = f"model_level_{fusion_method}_w{weight:.3f}_d{density:.3f}_{uuid.uuid4().hex[:8]}"
            
            # Save model to temporary directory
            model_output_dir = os.path.join(temp_model_dir, model_id)
            
            # Check if evaluation result cache exists
            model_key = generate_model_cache_key(model_output_dir)
            general_cache_path = get_model_cache_path(checkpoint_dir, model_key, 'solution')
            cached_result = load_cached_results(general_cache_path)
            
            if cached_result:
                print(f"Using cached evaluation result: {model_id}")
                # Extract objective function values
                metrics = cached_result['metrics']
                # Convert to results format
                results = {'processed_results': [metrics]}
                objectives = extract_objectives(results, base_results, expert_results)
                
                # Save results
                all_decision_variables.append(decision_vars)
                all_objectives.append(objectives[0])  # Only one result
                all_metrics.append(metrics)
                skipped_weights.append(weight)
                print(f"Weight {weight} evaluation completed, objective value: {objectives[0]}")
            else:
                # Call mi_block_fusion method for model fusion
                success = mi_block_fusion(
                    base_model_path=base_model_path,
                    task_model_paths=task_model_paths,
                    block_weights=decision_vars,
                    output_dir=model_output_dir,
                    fusion_method=fusion_method,
                    copy_from_base=True,
                    merged_blocks=merged_blocks,
                    num_blocks=num_blocks
                )
                
                if not success:
                    print(f"Warning: Weight {weight} fusion failed, skipping evaluation")
                    skipped_weights.append(weight)
                    continue
                
                # Create evaluation task configuration - using bi-objective configuration
                task_cfg = config_manager.create_biojective_eval_task_config(model_output_dir, max_tokens)
                
                # Create task dictionary
                task = {
                    'task_id': f'task_{model_id}',
                    'model_path': model_output_dir,
                    'params_dict': {'task_cfg': task_cfg},
                    'func_handle': run_task_with_server
                }
                
                # Add task to list
                batch_tasks.append(task)
                
                # Save task-related information
            batch_task_info_map[task['task_id']] = {
                'weight': weight,
                'density': density,
                'decision_vars': decision_vars,
                'model_id': model_id,
                'model_output_dir': model_output_dir,
                'general_cache_path': general_cache_path
            }
            
            if density is None:
                print(f"Created task: {task['task_id']} for weight: {weight:.3f}")
            else:
                print(f"Created task: {task['task_id']} for weight: {weight:.3f}, Density: {density:.3f}")
        
        # Phase 2: Execute all tasks in the current batch
        if batch_tasks:
            print(f"\n===== Starting Evaluation Tasks for Batch {batch_idx+1}/{len(batch_tasks)} =====")
            batch_start_time = time.time()
            
            try:
                # If max_model_len is not specified, set to max_tokens + 3000
                if max_model_len is None:
                    batch_max_model_len = max_tokens + 3000
                    print(f"max_model_len not specified, using default: {batch_max_model_len} (max_tokens + 3000)")
                else:
                    batch_max_model_len = max_model_len
                
                with VllmServerManager(available_gpus=available_gpus_global, 
                                     max_model_len=batch_max_model_len) as server_manager:
                    # Call run_series_tasks method to execute all tasks in the current batch
                    batch_results = server_manager.run_series_tasks(batch_tasks)
                    
                print(f"All evaluation tasks for batch {batch_idx+1} completed in: {time.time() - batch_start_time:.2f} seconds")
                
                # Use ResultProcessor to process results
                print(f"Processing evaluation results for batch {batch_idx+1}...")
                result_processor = ResultProcessor()
                processed_batch_results = result_processor.process_and_save(batch_results)
                
                # Check structure of processed_results
                if isinstance(processed_batch_results, dict) and 'processed_results' in processed_batch_results:
                    batch_results_list = processed_batch_results['processed_results']
                elif isinstance(processed_batch_results, list):
                    batch_results_list = processed_batch_results
                else:
                    batch_results_list = [processed_batch_results]
                
                # Process each result
                for i, result in enumerate(batch_results_list):
                    if i >= len(batch_tasks):
                        break
                    
                    task = batch_tasks[i]
                    task_id = task['task_id']
                    task_info = batch_task_info_map[task_id]
                    weight = task_info['weight']
                    decision_vars = task_info['decision_vars']
                    general_cache_path = task_info['general_cache_path']
                    
                    try:
                        # Extract objective function values
                        objectives = extract_objectives([result], base_results, expert_results)
                        
                        # Extract detailed metrics
                        metrics = {}
                        for dataset_name in ['aime25', 'gpqa_diamond']:
                            if dataset_name in result and isinstance(result[dataset_name], dict):
                                metrics[dataset_name] = result[dataset_name]
                        
                        # Save results to cache
                        model_result = {
                            'model_type': f'model_level_{fusion_method}',
                            'model_name': f'model_level_{fusion_method}_{weight:.1f}',
                            'model_path': task_info['model_output_dir'],
                            'metrics': metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                        save_results_to_cache(general_cache_path, model_result)
                        
                        # Save results
                        all_decision_variables.append(decision_vars)
                        all_objectives.append(objectives[0])  # Only one result
                        all_metrics.append(metrics)
                        
                        print(f"Batch {batch_idx+1} - Weight {weight} evaluation completed, objective value: {objectives[0]}")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx+1} - task {task_id} results: {e}")
                        continue
            
            except Exception as e:
                print(f"Error during batch {batch_idx+1} evaluation: {e}")
        
        # Clear temporary model directory for current batch to free up storage
        print(f"\n===== Clearing Temporary Model Directory {temp_model_dir} for Batch {batch_idx+1} =====")
        import shutil
        try:
            shutil.rmtree(temp_model_dir)
            # Recreate empty temporary directory
            os.makedirs(temp_model_dir, exist_ok=True)
            print(f"Temporary model directory cleared: {temp_model_dir}")
        except Exception as e:
            print(f"Error clearing temporary model directory: {e}")
    
    # Save all results to files, compatible with checkpoint_analyzer
    print(f"\n===== Saving Results to {output_root} =====")
    
    # Save as numpy format
    np.save(os.path.join(output_root, "decision_variables.npy"), np.array(all_decision_variables))
    np.save(os.path.join(output_root, "objectives.npy"), np.array(all_objectives))
    
    # Save as JSON format
    results_json = {
        "fusion_method": fusion_method,
        "run_id": run_id,
        "weights": weights.tolist(),
        "results": [
            {
                "weight": float(weights[i]),
                "decision_variables": all_decision_variables[i] if i < len(all_decision_variables) else [],
                "objectives": all_objectives[i].tolist() if i < len(all_objectives) else [],
                "metrics": all_metrics[i] if i < len(all_metrics) else {}
            }
            for i in range(len(weights))
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(output_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # Save in checkpoint-like format, compatible with checkpoint_analyzer
    checkpoint_data = {
        'train_x': torch.tensor(np.array(all_decision_variables)),
        'train_obj_true': torch.tensor(np.array(all_objectives)),
        'train_info': all_metrics,
        'fusion_method': fusion_method,
        'run_id': run_id,
        'weights': weights.tolist()
    }
    
    # Save as PyTorch checkpoint file
    torch.save(checkpoint_data, os.path.join(output_root, "checkpoint_iter_0.pt"))
    torch.save(checkpoint_data, os.path.join(output_root, "checkpoint_latest.pt"))
    
    # Clear temporary model directory
    import shutil
    try:
        shutil.rmtree(temp_model_dir)
        print(f"Temporary model directory cleared: {temp_model_dir}")
    except Exception as e:
        print(f"Error clearing temporary model directory: {e}")
    
    print(f"All results saved to: {output_root}")
    return output_root

def main():
    """
    Main function
    """
    import argparse
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description="Model-level Fusion Test Tool")
    
    # Add command line arguments
    parser.add_argument('--fusion_method', type=str, default="dare_linear",
                        choices=["task_arithmetic", "ties", "dare_ties", "dare_linear", 
                                 "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"],
                        help='Fusion method')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='Number of blocks')
    parser.add_argument('--max_tokens', type=int, default=35000,
                        help='Maximum generated tokens')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Maximum model length')
    parser.add_argument('--run_id', type=str, default="model_level_test_ins_88",
                        help='Run ID')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size, number of models processed each time')
    
    # Grid search parameters
    parser.add_argument('--weight_min', type=float, default=0.05,
                        help='Minimum weight value')
    parser.add_argument('--weight_max', type=float, default=0.95,
                        help='Maximum weight value')
    parser.add_argument('--density_min', type=float, default=0.7,
                        help='Minimum density value (used by non-task_arithmetic methods)')
    parser.add_argument('--density_max', type=float, default=0.9,
                        help='Maximum density value (used by non-task_arithmetic methods)')
    parser.add_argument('--budget', type=int, default=88,
                        help='Total computational budget count')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Call test function
    test_model_level_fusion(
        fusion_method=args.fusion_method,
        num_blocks=args.num_blocks,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        run_id=args.run_id,
        batch_size=args.batch_size,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        density_min=args.density_min,
        density_max=args.density_max,
        budget=args.budget
    )

if __name__ == "__main__":
    main()
