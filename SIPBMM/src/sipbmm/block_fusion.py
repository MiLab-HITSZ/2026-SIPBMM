#!/usr/bin/env python3
"""
Model Block Fusion Module
Provides model fusion functionality based on automatic block merging
"""

import os
import sys
import json
import re
import uuid
from typing import List, Dict, Tuple
import numpy as np

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from src.sipbmm.task_diff_visualizer import TaskDiffVisualizer
from src.ta_methods.model_fusion_layer import LayerwiseModelFusion, LayerFusionConfig


def calculate_merged_blocks(task_model_paths: List[str], num_blocks: int = 8, alpha: float = 1.0, beta: float = 0.005, checkpoint_dir: str = None, block_numbers: List[int] = None, metric: str = "L2-norm", partition_method: str = "hybrid"):
    """
    Calculate automatically merged blocks, only run once during initialization
    
    Args:
        task_model_paths: List of task model paths
        num_blocks: Number of blocks to merge, default is 8
        alpha: Variance weight, default is 1.0
        beta: Balance weight, default is 0.005
        checkpoint_dir: Checkpoint directory for saving visualization results
        block_numbers: Optional, used to generate block configurations from fine to coarse (e.g., [6, 12, 24, 36])
        metric: Distance metric method, default is "L2-norm"
        partition_method: Partition method, options: "hybrid", "balance", "variance", default is "hybrid"
    
    Returns:
        When block_numbers is None: List[Tuple[List[int], float]] - List of merged blocks
        When block_numbers is not None: Dict[int, List[Tuple[List[int], float]]] - Dictionary of merged blocks for different block counts
    """
    # Initialize TaskDiffVisualizer to get automatically merged blocks
    print("\nInitializing TaskDiffVisualizer to get automatically merged blocks...")
    visualizer = TaskDiffVisualizer(device="cpu", alpha=alpha, beta=beta)
    
    # Use the first two task models to determine block merging
    if len(task_model_paths) < 2:
        raise ValueError("At least two task models are required to determine block merging")
    
    model1_path = task_model_paths[0]
    model2_path = task_model_paths[1]
    
    if block_numbers is None:
        # Get a single automatically merged block configuration
        merged_blocks = visualizer.run(
            model1_path, 
            model2_path, 
            output_dir=checkpoint_dir,  # Save visualization to checkpoint directory
            num_blocks=num_blocks,
            metric=metric,
            partition_method=partition_method
        )
        
        print(f"\nAutomatic merging result: Generated {len(merged_blocks)} blocks in total")
        for i, (block_layers, block_diff) in enumerate(merged_blocks):
            print(f"Block {i+1}: Layers {block_layers[0]}-{block_layers[-1]} (difference: {block_diff:.6f})")
        
        return merged_blocks
    else:
        # Generate multiple block configurations from fine to coarse
        print(f"\nGenerating block configurations from fine to coarse: {block_numbers}")
        block_configs = visualizer.generate_multiple_block_configs(
            model1_path, 
            model2_path, 
            block_numbers=block_numbers, 
            output_dir=checkpoint_dir,
            metric=metric,
            partition_method=partition_method
        )
        
        # Print all block configurations
        for block_count, merged_blocks in block_configs.items():
            print(f"\nAutomatic merging result ({block_count} blocks): Generated {len(merged_blocks)} blocks in total")
            for i, (block_layers, block_diff) in enumerate(merged_blocks):
                print(f"Block {i+1}: Layers {block_layers[0]}-{block_layers[-1]} (difference: {block_diff:.6f})")
        
        return block_configs


def create_fusion_configs(merged_blocks: List[Tuple[List[int], float]], block_weights: List[float], 
                         block_densities: List[float] = None, fusion_method: str = "ties", num_task_models: int = 2, block_gammas: List[float] = None, num_blocks: int = None):
    """
    Create fusion configurations based on merged blocks and weights
    
    Args:
        merged_blocks: List of merged blocks
        block_weights: List of block weights, containing num_blocks+1 decision variables
        block_densities: List of block densities, containing num_blocks+1 decision variables
        fusion_method: Fusion method
        num_task_models: Number of task models, default is 2
        num_blocks: Number of blocks, used to calculate expected_length, if None uses len(merged_blocks)
    
    Returns:
        List[LayerFusionConfig]: List of fusion configurations
    """
    # Ensure block_weights is a list
    if not isinstance(block_weights, list):
        raise ValueError("block_weights must be a list type")
    
    # Determine the number of blocks to use, prefer the passed num_blocks parameter
    if num_blocks is None:
        num_blocks = len(merged_blocks)
    
    # block_weights length must be num_blocks + 1 (num_blocks block weights + 1 norm/lm_head weight)
    expected_length = num_blocks + 1
    if len(block_weights) != expected_length:
        raise ValueError(f"block_weights must contain {expected_length} weight values, current length is {len(block_weights)}")
    
    # If block_densities is provided, ensure its length is correct
    if block_densities is not None:
        if len(block_densities) != expected_length:
            raise ValueError(f"block_densities must contain {expected_length} density values, current length is {len(block_densities)}")
    
    print(f"\nCreating fusion configurations...")
    configs = []
    
    # Create configuration for each automatically merged block
    for i, (block_layers, _) in enumerate(merged_blocks):
        # Generate layer pattern regular expression
        if len(block_layers) == 1:
            # Single layer
            layer_pattern = rf"layers\.{block_layers[0]}"
        else:
            # Layer range
            start_layer = block_layers[0]
            end_layer = block_layers[-1]
            # Create regex to match this range
            layer_pattern = rf"layers\.({start_layer}|{end_layer}|({start_layer+1}-{end_layer-1}))"
        
        # Create model_weights for current block: [x, 1-x], where x is block_weights[i]
        # For each block, use independent model_weights, corresponding to decision variable x as [x, 1-x]
        current_model_weights = [block_weights[i], 1 - block_weights[i]]
        
        # Get density value for current block from block_densities, default 0.8
        current_density = 0.8
        if block_densities is not None:
            current_density = block_densities[i]
        
        # Get gamma value for current block, default 0.0
        current_gamma = 0.0
        if block_gammas is not None:
            current_gamma = block_gammas[i]
        
        # Create configuration for current block
        config = LayerFusionConfig(
            method=fusion_method,
            params={
                "density": current_density, 
                "layer_weight": 1.0,
                "normalize": True,
                "gamma": current_gamma
            },
            layer_pattern=layer_pattern,
            apply_to_embeddings=False,
            apply_to_norm=False,
            apply_to_lm_head=False,
            model_weights=current_model_weights  # Configure independent model_weights for each block
        )
        configs.append(config)
        print(f"Created configuration for block {i+1}: Layers {block_layers[0]}-{block_layers[-1]}, weight: {1.0}, model_weights: {current_model_weights}, density: {current_density}, gamma: {current_gamma}")
    
    # Handle embedding layer with fixed weight of 1.0
    embedding_weight = 1.0  # Fixed embedding weight as 1.0
    embedding_model_weights = [embedding_weight, 1 - embedding_weight]
    
    # Handle embeddings layer
    emb_config = LayerFusionConfig(
        method="task_arithmetic",
        params={
            "layer_weight": 1.0
        },
        apply_to_embeddings=True,
        apply_to_norm=False,
        apply_to_lm_head=False,
        model_weights=embedding_model_weights  # Configure independent model_weights for embeddings layer
    )
    configs.append(emb_config)
    print(f"Created embeddings layer configuration: weight: 1.0, model_weights: {embedding_model_weights}")
    
    # Handle norm layer and lm_head layer using the num_blocks-th parameter
    norm_lm_head_weight = block_weights[num_blocks]  # The num_blocks-th parameter is used for both norm and lm_head layers
    norm_lm_head_model_weights = [norm_lm_head_weight, 1 - norm_lm_head_weight]
    
    # Get density value for norm layer, default is 1.0
    norm_density = 1.0
    if block_densities is not None:
        norm_density = block_densities[num_blocks]
    
    # Handle norm layer
    norm_config = LayerFusionConfig(
        method=fusion_method,
        params={
            "density": norm_density, 
            "layer_weight": 1.0
        },
        apply_to_embeddings=False,
        apply_to_norm=True,
        apply_to_lm_head=False,
        model_weights=norm_lm_head_model_weights  # norm layer uses shared model_weights
    )
    configs.append(norm_config)
    print(f"Created norm layer configuration: weight: {1.0}, model_weights: {norm_lm_head_model_weights}, density: {norm_density}")
    
    # Handle lm_head layer, get density value, default is 1.0
    lm_head_density = 1.0
    if block_densities is not None:
        lm_head_density = block_densities[num_blocks]
    
    lm_head_config = LayerFusionConfig(
        method="task_arithmetic",
        params={
            "density": lm_head_density, 
            "layer_weight": 1.0
        },
        apply_to_embeddings=False,
        apply_to_norm=False,
        apply_to_lm_head=True,
        model_weights=norm_lm_head_model_weights  # lm_head layer uses shared model_weights
    )
    configs.append(lm_head_config)
    print(f"Created lm_head layer configuration: weight: {1.0}, model_weights: {norm_lm_head_model_weights}, density: {lm_head_density}")
    
    return configs


def mi_block_fusion(base_model_path: str, task_model_paths: List[str], 
                   block_weights: List[float], output_dir: str, 
                   fusion_method: str = "task_arithmetic", 
                   copy_from_base: bool = True, 
                   merged_blocks: List[Tuple[List[int], float]] = None, 
                   num_blocks: int = 8,
                   block_densities: List[float] = None,
                   block_gammas: List[float] = None):
    """
    Model fusion method based on automatic block merging
    
    Args:
        base_model_path: Base model path
        task_model_paths: List of task model paths
        block_weights: List of block weights, containing num_blocks + 1 decision variables:
                      - First num_blocks values for num_blocks automatically merged transformer blocks
                      - Last 1 value shared for norm layers and output layer (lm_head)
        output_dir: Output directory
        fusion_method: Fusion method, default is "task_arithmetic"
        copy_from_base: Whether to copy non-weight files from the base model
        merged_blocks: Precomputed list of merged blocks, automatically computed if None
        num_blocks: Number of blocks to merge, default is 8
        block_densities: List of block densities, containing num_blocks + 1 decision variables:
                        - First num_blocks values for num_blocks automatically merged transformer blocks
                        - Last 1 value shared for norm layers and output layer (lm_head)
        block_gammas: List of block gamma parameters, containing num_blocks + 1 decision variables:
                     - First num_blocks values for num_blocks automatically merged transformer blocks
                     - Last 1 value shared for norm layers and output layer (lm_head)
    
    Returns:
        bool: Whether fusion was successful
    """
    # Ensure block_weights is a list
    if not isinstance(block_weights, list):
        raise ValueError("block_weights must be a list type")
    
    print(f"Number of block weights used: {len(block_weights)}")
    if block_densities is not None:
        print(f"Number of block densities used: {len(block_densities)}")
    print(f"Fusion strategy: {num_blocks} automatically merged transformer blocks, one group for norm layers and output layer")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")
    
    # If precomputed blocks are not provided, calculate block merging
    if merged_blocks is None:
        merged_blocks = calculate_merged_blocks(task_model_paths, num_blocks)
    
    # Create fusion configurations, passing task model count, block densities, and gamma parameters
    configs = create_fusion_configs(merged_blocks, block_weights, block_densities, fusion_method, len(task_model_paths), block_gammas, num_blocks)
    
    # Initialize fusion engine
    print("\nInitializing LayerwiseModelFusion...")
    fusion = LayerwiseModelFusion()
    
    # Execute fusion, no longer passing global model_weights, each configuration has its own model_weights
    print("\nStarting layer fusion execution...")
    success = fusion.layer_fusion(
        base_model_path=base_model_path,
        task_model_paths=task_model_paths,
        output_path=output_dir,
        layer_configs=configs,
        device="cuda"  # Use specified GPU
    )
    
    if success:
        print(f"\nModel fusion completed successfully!")
        print(f"Fused model saved to: {output_dir}")
        print(f"Fusion method used: {fusion_method}")
        print(f"Number of automatically merged blocks: {len(merged_blocks)}")
    else:
        print(f"\nModel fusion failed!")
        return False
    return True


def process_decision_variables(decision_matrix, base_model_path, 
                              task_model_paths, base_output_dir,
                              fusion_method="ties",
                              num_blocks=8):
    """
    Process decision variable matrix, create fusion model for each decision variable
    
    Args:
        decision_matrix: numpy array with shape (n, num_blocks+1), each row represents num_blocks+1 decision variables for a candidate solution
        base_model_path: Base model path
        task_model_paths: List of task model paths
        base_output_dir: Base output directory
        fusion_method: Fusion method
        num_blocks: Number of automatically merged blocks
    
    Returns:
        list: List of successfully created model paths
    """
    # Ensure decision matrix is a numpy array
    if isinstance(decision_matrix, list):
        decision_matrix = np.array(decision_matrix)
    
    # Validate decision matrix dimensions
    expected_dim = num_blocks + 1
    if decision_matrix.ndim != 2 or decision_matrix.shape[1] != expected_dim:
        raise ValueError(f"Decision variable matrix must be a 2D array, each row containing {expected_dim} decision variables")
    
    num_candidates = decision_matrix.shape[0]
    print(f"Starting to process {num_candidates} candidate solutions")
    
    # Ensure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create task list
    model_paths = []
    
    # Create fusion model for each candidate solution
    for i in range(num_candidates):
        # Extract decision variables from current row
        block_weights = decision_matrix[i].tolist()
        print(f"\nProcessing candidate solution {i+1}/{num_candidates}: {block_weights}")
        
        # Generate unique model ID
        model_id = f"merged_model_{i}_{uuid.uuid4().hex[:8]}"
        model_output_dir = os.path.join(base_output_dir, model_id)
        
        # Call mi_block_fusion method for model fusion
        success = mi_block_fusion(
            base_model_path=base_model_path,
            task_model_paths=task_model_paths,
            block_weights=block_weights,
            output_dir=model_output_dir,
            fusion_method=fusion_method,
            copy_from_base=True,
            num_blocks=num_blocks
        )
        
        if success:
            model_paths.append(model_output_dir)
            print(f"Fusion model created successfully: {model_output_dir}")
        else:
            print(f"Warning: Fusion failed for candidate solution {i+1}")
    
    return model_paths


def main():
    """
    Command line interface example
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Model fusion tool based on automatic block merging")
    parser.add_argument("--base_model", type=str, default="models/Qwen3-4B-Base", help="Base model path")
    parser.add_argument("--task_models", type=str, nargs='+', 
                        default=["models/Qwen3-4B-thinking-2507", "models/Qwen3-4B-Instruct-2507"], 
                        help="List of task model paths")
    parser.add_argument("--block_weights", type=str, 
                        default='[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]', 
                        help="JSON string of block weights list, format like '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]'")
    parser.add_argument("--output_dir", type=str, default="models/test-merged-block", help="Output directory")
    parser.add_argument("--fusion_method", type=str, default="ties", 
                        choices=["task_arithmetic", "ties", "dare_ties", "dare_linear", 
                                 "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"],
                        help="Fusion method")
    parser.add_argument("--copy_from_base", type=bool, default=True, help="Whether to copy non-weight files from base model")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of blocks to merge")
    
    args = parser.parse_args()
    
    # Parse block weights
    try:
        block_weights = json.loads(args.block_weights)
        print(f"Loaded {len(block_weights)} block weights")
    except Exception as e:
        print(f"Error: Failed to parse block_weights: {e}")
        return
    
    # Execute fusion
    success = mi_block_fusion(
        base_model_path=args.base_model,
        task_model_paths=args.task_models,
        block_weights=block_weights,
        output_dir=args.output_dir,
        fusion_method=args.fusion_method,
        copy_from_base=args.copy_from_base,
        num_blocks=args.num_blocks
    )
    
    if success:
        print(f"\nFusion completed! The fused model can be found at {args.output_dir}.")
    else:
        print(f"\nAn error occurred during fusion.")


if __name__ == "__main__":
    main()
