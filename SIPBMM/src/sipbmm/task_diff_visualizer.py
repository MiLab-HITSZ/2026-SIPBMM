import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from safetensors.torch import load_file
import re
from tqdm import tqdm
from scipy.stats import wasserstein_distance

class LayerNormDifferenceAnalyzer:
    """
    LayerNorm Parameter Difference Analyzer
    Specialized for comparing LayerNorm differences between two Transformer blocks
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize the analyzer
        
        Args: 
        device: Computing device
        """
        self.device = device
        
    def extract_layernorm_parameters_from_tensors(self, tensors, param_type='both'):
        """
        Extract all LayerNorm parameters from the model tensor dictionary
        
        Args:
        tensors: Model tensor dictionary
        param_type: 'gamma'(weight), 'beta'(bias), 'both'(both)
        
        Returns:
        Parameter dictionary: {layer_name: {'gamma': numpy array, 'beta': numpy array}}
        """
        params_dict = {}
        
        for name, tensor in tensors.items():
            if 'norm' in name.lower() and ('weight' in name or 'bias' in name):
                # Parse layer name and parameter type
                layer_name_parts = name.split('.')
                param_type_in_name = 'gamma' if 'weight' in name else 'beta'
                
                # Construct layer name (remove the last parameter type part)
                layer_name = '.'.join(layer_name_parts[:-1])
                
                if layer_name not in params_dict:
                    params_dict[layer_name] = {'gamma': None, 'beta': None}
                
                # Extract parameters, ensure conversion to float32 to avoid type issues with BFloat16 etc.
                if param_type_in_name == 'gamma' and param_type in ['gamma', 'both']:
                    params_dict[layer_name]['gamma'] = tensor.to(dtype=torch.float32).detach().cpu().numpy()
                
                if param_type_in_name == 'beta' and param_type in ['beta', 'both']:
                    params_dict[layer_name]['beta'] = tensor.to(dtype=torch.float32).detach().cpu().numpy()
        
        # Filter out layers without parameters
        params_dict = {k: v for k, v in params_dict.items() if v['gamma'] is not None or v['beta'] is not None}
        
        return params_dict
    
    def compute_wasserstein_distance(self, params1, params2):
        """
        Calculate Wasserstein distance between two parameter vectors
        Suitable for comparing distribution shapes
        """
        # Flatten
        p1_flat = params1.flatten()
        p2_flat = params2.flatten()
        
        # Calculate 1D Wasserstein distance
        w_dist = wasserstein_distance(p1_flat, p2_flat)
        
        return w_dist
    
    def compute_kl_divergence_estimate(self, params1, params2, bins=50, eps=1e-10):
        """
        Estimate KL divergence (need to handle zero probabilities carefully)
        Estimate probability distribution via histogram
        """
        # Flatten
        p1_flat = params1.flatten()
        p2_flat = params2.flatten()
        
        # Determine common value range
        min_val = min(p1_flat.min(), p2_flat.min())
        max_val = max(p1_flat.max(), p2_flat.max())
        
        # Calculate histogram
        hist1, _ = np.histogram(p1_flat, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(p2_flat, bins=bins, range=(min_val, max_val), density=True)
        
        # Add small constant to avoid zeros
        hist1 = hist1 + eps
        hist2 = hist2 + eps
        
        # Calculate KL divergence
        kl_12 = np.sum(hist1 * np.log(hist1 / hist2))
        kl_21 = np.sum(hist2 * np.log(hist2 / hist1))
        
        # Return symmetric KL divergence
        return (kl_12 + kl_21) / 2
    
    def calculate_layernorm_diffs(self, model1_tensors, model2_tensors):
        """
        Calculate LayerNorm differences between two model tensors
        
        Args:
        model1_tensors, model2_tensors: Tensor dictionaries of two models

        Returns:
        Difference dictionary: {layer_num: {'layernorm_kl': float, 'layernorm_wasserstein': float}}
        """
        # Extract LayerNorm parameters
        ln_params1 = self.extract_layernorm_parameters_from_tensors(model1_tensors)
        ln_params2 = self.extract_layernorm_parameters_from_tensors(model2_tensors)
        
        # Get common LayerNorm layers
        common_ln_layers = set(ln_params1.keys()) & set(ln_params2.keys())
        print(f"Found {len(common_ln_layers)} common LayerNorm layers")
        
        # Calculate differences
        layer_diffs = {}
        
        for layer_name in common_ln_layers:
            # Get layer number
            layer_match = re.search(r"layers\.([0-9]+)", layer_name)
            if layer_match:
                layer_num = int(layer_match.group(1))
            else:
                # Skip LayerNorm from non-Transformer layers
                continue
            
            # Get current layer parameters
            params1 = ln_params1[layer_name]
            params2 = ln_params2[layer_name]
            
            # Calculate differences between weights and biases
            gamma_kl = gamma_wasserstein = beta_kl = beta_wasserstein = 0.0
            
            if params1['gamma'] is not None and params2['gamma'] is not None:
                gamma_kl = self.compute_kl_divergence_estimate(params1['gamma'], params2['gamma'])
                gamma_wasserstein = self.compute_wasserstein_distance(params1['gamma'], params2['gamma'])
            
            if params1['beta'] is not None and params2['beta'] is not None:
                beta_kl = self.compute_kl_divergence_estimate(params1['beta'], params2['beta'])
                beta_wasserstein = self.compute_wasserstein_distance(params1['beta'], params2['beta'])
            
            # Combine differences from weights and biases (take average)
            avg_kl = (gamma_kl + beta_kl) / 2
            avg_wasserstein = (gamma_wasserstein + beta_wasserstein) / 2
            
            # Store differences
            if layer_num not in layer_diffs:
                layer_diffs[layer_num] = {
                    'layernorm_kl': avg_kl,
                    'layernorm_wasserstein': avg_wasserstein
                }
            else:
                # If multiple LayerNorm in the same layer, take average
                layer_diffs[layer_num]['layernorm_kl'] = (layer_diffs[layer_num]['layernorm_kl'] + avg_kl) / 2
                layer_diffs[layer_num]['layernorm_wasserstein'] = (layer_diffs[layer_num]['layernorm_wasserstein'] + avg_wasserstein) / 2
        
        return layer_diffs

class TaskDiffVisualizer:
    def __init__(self, device=None, alpha=1.0, beta=0.005):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_functions = {
            "Fisher": self.fisher_information_distance,
            "Grassmann": self.geodesic_distance_on_grassmann,
            "L2-norm": torch.norm,
            "Block": torch.norm
        }
        self.norm_params = {
            "Fisher": {},
            "Grassmann": {},
            "L2-norm": {"p": 2},
            "Block": {"p": float("inf")}
        }
        self.alpha = alpha  # Variance weight, default 1.0
        self.beta = beta  # Balance weight, default 1.0
    
    def fisher_information_distance(self, model1_layer, model2_layer, **kwargs):
        """
        Distance based on Fisher information
        Measures differences between two probability distributions
        
        Args:
            model1_layer: Layer tensor from first model
            model2_layer: Layer tensor from second model
            **kwargs: Additional parameters (for interface compatibility)

        Returns:
            float: Fisher-Rao distance
        """
        # Calculate mean and standard deviation
        mu1 = model1_layer.mean().item()
        sigma1 = model1_layer.std().item()
        
        mu2 = model2_layer.mean().item()
        sigma2 = model2_layer.std().item()
        
        # Calculate Fisher-Rao distance
        if sigma1 > 0 and sigma2 > 0:
            distance = np.sqrt(2 * np.log((sigma1 + sigma2) / (2 * np.sqrt(sigma1 * sigma2))) +
                              (mu1 - mu2)**2 / (sigma1 + sigma2))
        else:
            # If standard deviation is 0, use absolute distance
            distance = np.abs(mu1 - mu2)
        
        return distance
    
    def geodesic_distance_on_grassmann(self, model1_layer, model2_layer, **kwargs):
        """
        Calculate geodesic distance on Grassmann manifold
        Suitable for comparing subspace structures
        
        Args:
            model1_layer: Layer tensor from first model
            model2_layer: Layer tensor from second model
            **kwargs: Additional parameters (for interface compatibility)

        Returns:
            float: Geodesic distance on Grassmann manifold
        """
        import numpy as np
        from scipy.linalg import svd
        
        # Convert PyTorch tensors to NumPy arrays
        W1 = model1_layer.cpu().numpy()
        W2 = model2_layer.cpu().numpy()
        
        # Check if it's a 2D matrix
        if W1.ndim != 2 or W2.ndim != 2:
            # If not a matrix, return 0 or skip (return 0 as default here)
            return 0.0
        
        try:
            # Singular Value Decomposition
            U1, S1, V1 = svd(W1, full_matrices=False)
            U2, S2, V2 = svd(W2, full_matrices=False)
            
            # Calculate principal angles
            cos_theta = svd(U1.T @ U2, compute_uv=False)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            # Geodesic distance
            geodesic_dist = np.sqrt(np.sum(theta**2))
            
            return geodesic_dist
        except Exception as e:
            # Handle potential exceptions during SVD calculation
            print(f"Warning: Failed to compute Grassmann distance: {e}")
            return 0.0
    
    def load_model_tensors(self, model_path):
        """
        Load all tensors from a model directory
        """
        model_files = glob(os.path.join(model_path, "*.safetensors"))
        if not model_files:
            raise ValueError(f"No safetensors files found in {model_path}")
        
        model_tensors = {}
        for file_path in model_files:
            try:
                print(f"Loading {file_path}...")
                # Load to CPU first to save GPU memory
                tensors = load_file(file_path, device="cpu")
                model_tensors.update(tensors)
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(model_tensors)} tensors from {model_path}")
        return model_tensors
    
    def get_layer_key_info(self, key):
        """
        Extract layer information from tensor key
        """
        # Check for special layer types
        if "embed_tokens" in key or "rotary_emb" in key:
            return "embeddings", -1
        if "norm" in key and ".layers." not in key:
            return "norm", -1
        if "lm_head" in key:
            return "lm_head", -1
        
        # Extract layer number from transformer layers
        layer_match = re.search(r"layers\.([0-9]+)", key)
        if layer_match:
            layer_num = int(layer_match.group(1))
            return "transformer", layer_num
        
        return "other", -1
    
    def calculate_task_diffs(self, model1_tensors, model2_tensors, metric="L2-norm"):
        """
        Calculate differences between two task models
        """
        layer_diffs = {}
        
        with torch.no_grad():
            # Get common keys between both models
            common_keys = set(model1_tensors.keys()) & set(model2_tensors.keys())
            print(f"Found {len(common_keys)} common keys between models")
            
            # Process each common key
            for key in tqdm(common_keys, desc="Processing tensors"):
                # Get layer info
                layer_type, layer_num = self.get_layer_key_info(key)
                
                # Skip non-transformer layers
                if layer_type != "transformer":
                    continue
                
                try:
                    # Get tensors
                    tensor1 = model1_tensors[key].to(dtype=torch.float32, device=self.device)
                    tensor2 = model2_tensors[key].to(dtype=torch.float32, device=self.device)
                    
                    # Check shape match
                    if tensor1.shape != tensor2.shape:
                        if "embed_tokens" in key:
                            try:
                                tensor2 = tensor2[:tensor1.shape[0], :tensor1.shape[1]]
                            except:
                                torch.cuda.empty_cache()
                                continue
                        else:
                            torch.cuda.empty_cache()
                            continue
                    
                    # Calculate norms based on the specified metric
                    layer_diff = {}
                    
                    # Calculate base metrics to support Grassmann-Wasserstein hybrid metric
                    if metric in ["Fisher", "Grassmann-Wasserstein"]:
                        fisher_diff = self.fisher_information_distance(tensor1, tensor2)
                        layer_diff["Fisher"] = fisher_diff
                    
                    if metric in ["Grassmann", "Grassmann-Wasserstein"]:
                        grassmann_diff = self.geodesic_distance_on_grassmann(tensor1, tensor2)
                        layer_diff["Grassmann"] = grassmann_diff
                    
                    if metric in ["L2-norm"]:
                        l2_diff = torch.norm(tensor1 - tensor2, p=2).item()
                        layer_diff["L2-norm"] = l2_diff
                    
                    if metric in ["Block"]:
                        linf_diff = torch.norm(tensor1 - tensor2, p=float("inf")).item()
                        layer_diff["Block"] = linf_diff
                    
                    # LayerNorm-Wasserstein is calculated directly in subsequent processing, no need to handle here
                    # For other unspecified metrics, default to L2-norm
                    if not layer_diff:
                        l2_diff = torch.norm(tensor1 - tensor2, p=2).item()
                        layer_diff["L2-norm"] = l2_diff
                    
                    # Store in layer_diffs
                    if layer_num not in layer_diffs:
                        layer_diffs[layer_num] = {}
                    
                    if key not in layer_diffs[layer_num]:
                        layer_diffs[layer_num][key] = layer_diff
                    
                    # Free memory
                    del tensor1, tensor2
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: CUDA OOM for key {key}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        # Average over all tensors in each layer
        avg_layer_diffs = {}
        for layer_num, tensor_diffs in layer_diffs.items():
            # Calculate average for each metric
            avg_metrics = {}
            for metric_key in list(tensor_diffs.values())[0].keys():
                avg_metrics[metric_key] = np.mean([tensor_diff[metric_key] for tensor_diff in tensor_diffs.values()])
            
            avg_layer_diffs[layer_num] = avg_metrics
        
        # Add LayerNorm difference calculation
        print(f"\nCalculating LayerNorm differences...")
        ln_analyzer = LayerNormDifferenceAnalyzer(self.device)
        ln_diffs = ln_analyzer.calculate_layernorm_diffs(model1_tensors, model2_tensors)
        
        # Merge LayerNorm differences into avg_layer_diffs
        for layer_num, ln_diff in ln_diffs.items():
            if layer_num in avg_layer_diffs:
                avg_layer_diffs[layer_num].update(ln_diff)
            else:
                # If the layer has no other differences, only add LayerNorm differences
                avg_layer_diffs[layer_num] = ln_diff
        
        # Calculate Grassmann-Wasserstein hybrid metric
        if metric == "Grassmann-Wasserstein":
            print(f"\nCalculating Grassmann-Wasserstein hybrid metric...")
            # Collect Grassmann and Wasserstein values for normalization
            grassmann_values = []
            wasserstein_values = []
            
            for layer_num, diffs in avg_layer_diffs.items():
                if "Grassmann" in diffs and "layernorm_wasserstein" in diffs:
                    grassmann_values.append(diffs["Grassmann"])
                    wasserstein_values.append(diffs["layernorm_wasserstein"])
            
            # Normalize and calculate hybrid metric
            if grassmann_values and wasserstein_values:
                # Normalize Grassmann value
                grassmann_min, grassmann_max = min(grassmann_values), max(grassmann_values)
                # Normalize Wasserstein value
                wasserstein_min, wasserstein_max = min(wasserstein_values), max(wasserstein_values)
                
                for layer_num, diffs in avg_layer_diffs.items():
                    if "Grassmann" in diffs and "layernorm_wasserstein" in diffs:
                        # Normalize to [0, 1]
                        norm_grassmann = (diffs["Grassmann"] - grassmann_min) / (grassmann_max - grassmann_min) if grassmann_max > grassmann_min else 0.5
                        norm_wasserstein = (diffs["layernorm_wasserstein"] - wasserstein_min) / (wasserstein_max - wasserstein_min) if wasserstein_max > wasserstein_min else 0.5
                        
                        # Weighted hybrid: 0.9 * Grassmann + 0.1 * Wasserstein
                        grassmann_wasserstein = 0.5 * norm_grassmann + 0.5 * norm_wasserstein
                        
                        # Add to difference dictionary
                        diffs["Grassmann-Wasserstein"] = grassmann_wasserstein
        
        # Handle LayerNorm-Wasserstein as a separate metric
        if metric == "LayerNorm-Wasserstein":
            print(f"\nUsing LayerNorm-Wasserstein as primary metric...")
            # Ensure each layer has layernorm_wasserstein value
            for layer_num, diffs in avg_layer_diffs.items():
                if "layernorm_wasserstein" in diffs:
                    # Copy LayerNorm-Wasserstein value to main metric position
                    diffs["LayerNorm-Wasserstein"] = diffs["layernorm_wasserstein"]
        
        return avg_layer_diffs
    
    def hybrid_optimization(self, values, k, alpha=1.0, beta=1.0):
        """
        General Optimization Strategy using DP.
        Cost = alpha * Variance_Cost + beta * Balance_Cost
        
        - alpha=0, beta=1: Pure Balance (Equal Info)
        - alpha=1, beta=0: Pure Variance (Fisher-Jenks / Homogeneity)
        - alpha=1, beta=2: Hybrid
        """
        n = len(values)
        if k >= n: return list(range(n + 1))
        
        prefix_sum = np.concatenate(([0], np.cumsum(values)))
        target_block_sum = prefix_sum[n] / k
        
        # Precompute variance cost (SSE)
        var_cost = np.zeros((n, n))
        for i in range(n):
            s1 = s2 = 0
            for j in range(i, n):
                val = values[j]
                s1 += val
                s2 += val * val
                var = s2 - (s1 * s1) / (j - i + 1)
                var_cost[i, j] = var

        dp = np.full((n + 1, k + 1), np.inf)
        path = np.zeros((n + 1, k + 1), dtype=int)
        dp[0, 0] = 0

        for j in range(1, k + 1):
            for i in range(1, n + 1):
                for m in range(i):
                    # Cost components
                    c_var = var_cost[m, i - 1]
                    block_sum = prefix_sum[i] - prefix_sum[m]
                    c_balance = (block_sum - target_block_sum) ** 2
                    
                    # Weighted total cost
                    total_cost = (alpha * c_var) + (beta * c_balance)
                    
                    if dp[m, j - 1] + total_cost < dp[i, j]:
                        dp[i, j] = dp[m, j - 1] + total_cost
                        path[i, j] = m

        cuts = [n]
        curr = n
        for j in range(k, 0, -1):
            curr = path[curr, j]
            cuts.append(curr)
        return sorted(cuts)
    
    def prepare_visualization_data(self, layer_diffs, num_blocks=8):
        """
        Prepare data for visualization with current metric
        """
        all_layers = sorted(layer_diffs.keys())
        
        # Get currently used metric (assuming only one metric in layer_diffs)
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Define row labels
        norm_types = [current_metric, "Balance", "Variance", "Hybrid"]
        
        num_layers = len(all_layers)
        data_matrix = np.zeros((4, num_layers))
        
        # 1. Fill current metric (Row 0)
        for j, layer_num in enumerate(all_layers):
            data_matrix[0, j] = layer_diffs[layer_num][current_metric]
        
        # Normalize current metric
        normalized_matrix = np.zeros((1, num_layers))
        row = data_matrix[0, :]
        row_min, row_max = np.min(row), np.max(row)
        if row_max > row_min:
            normalized_matrix[0, :] = (row - row_min) / (row_max - row_min)
        else:
            normalized_matrix[0, :] = row
        data_matrix[0:1, :] = normalized_matrix
        
        # Input for optimization algorithms
        total_normalized_diffs = normalized_matrix.sum(axis=0)
        
        # 2. Run Strategies
        print(f"Running Optimization Strategies for {num_blocks} blocks...")
        
        # Strategy 1: Pure Balance (Row 1) - alpha=0, beta=1
        cuts_balance = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=0.0, beta=1.0)
        
        # Strategy 2: Pure Variance (Row 2) - alpha=1, beta=0
        cuts_variance = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=1.0, beta=0.0)
        
        # Strategy 3: Hybrid (Row 3) - Use instance's alpha and beta values
        cuts_hybrid = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=self.alpha, beta=self.beta)
        
        # 3. Fill Matrix Blocks
        strategies = [
            (1, cuts_balance),
            (2, cuts_variance),
            (3, cuts_hybrid)
        ]
        
        for row_idx, cuts in strategies:
            for i in range(len(cuts) - 1):
                start, end = cuts[i], cuts[i+1]
                # Fill with block index (for coloring)
                for k in range(start, end):
                    data_matrix[row_idx, k] = i
        
        # 4. Prepare return value (Hybrid blocks for downstream usage)
        merged_blocks = []
        for i in range(len(cuts_hybrid) - 1):
            start_idx, end_idx = cuts_hybrid[i], cuts_hybrid[i+1]
            block_layers = [all_layers[k] for k in range(start_idx, end_idx)]
            block_diff = np.sum(total_normalized_diffs[start_idx:end_idx])
            merged_blocks.append((block_layers, block_diff))
            
        return data_matrix, norm_types, all_layers, merged_blocks
    
    def save_layer_sorted_result(self, layer_diffs, output_path, num_blocks=8):
        """
        Save layer sorted result to txt file, sorted by merged block differences
        """
        # Get all unique layer numbers (sorted by layer number)
        all_layers = sorted(layer_diffs.keys())
        
        # Get currently used metric (assuming only one metric in layer_diffs)
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Only use current metric
        norm_types = [current_metric]  # Only use current metric
        
        # Calculate total difference for each layer (sum of current metric)
        layer_total_diffs = {}
        for layer_num, diffs in layer_diffs.items():
            total_diff = sum(diffs[norm] for norm in norm_types)
            layer_total_diffs[layer_num] = total_diff
        
        # Calculate total normalized difference for each layer
        # First calculate normalized values
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer][norm] for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        
        # Calculate total normalized difference for each layer
        total_normalized_diffs = [sum(normalized_diffs[norm][i] for norm in norm_types) for i in range(len(all_layers))]
        
        # Run Hybrid Optimization for the report, using instance's alpha and beta values
        cuts = self.hybrid_optimization(total_normalized_diffs, num_blocks, alpha=self.alpha, beta=self.beta)
        
        merged_blocks = []
        for i in range(len(cuts) - 1):
            start_idx, end_idx = cuts[i], cuts[i+1]
            block_layers = [all_layers[k] for k in range(start_idx, end_idx)]
            block_diff = sum(total_normalized_diffs[start_idx:end_idx])
            block_total_raw_diff = sum(sum(layer_diffs[l][norm] for norm in norm_types) for l in block_layers)
            merged_blocks.append((block_layers, block_diff, block_total_raw_diff))
        
        merged_blocks.sort(key=lambda x: x[2], reverse=True)
        
        # Write to txt file
        with open(output_path, 'w') as f:
            f.write("Layer Blocks (Hybrid Strategy) Sorted by Difference\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Block':<8} {'Layers':<20} {'Total Difference':<20} {'Block Size':<12}\n")
            f.write("-" * 60 + "\n")
            for block_idx, (block_layers, block_diff, block_total_diff) in enumerate(merged_blocks):
                layer_range = f"{block_layers[0]}-{block_layers[-1]}" if len(block_layers) > 1 else f"{block_layers[0]}"
                f.write(f"{block_idx+1:<8} {layer_range:<20} {block_total_diff:<20.6f} {len(block_layers):<12}\n")
        
        print(f"Layer sorted result saved to: {output_path}")
    
    def plot_horizontal_heatmap(self, data_matrix, norm_types, layer_numbers, output_path=None):
        """
        Plot horizontal heatmap with variable number of rows
        """
        num_rows = data_matrix.shape[0]
        
        # Adjust figure height based on number of rows (approx 0.7 inch per row + margins)
        plt.figure(figsize=(6, 0.7 * num_rows))
        
        # GridSpec with dynamic rows
        gs = plt.GridSpec(num_rows, 2, width_ratios=[0.95, 0.05], 
                         height_ratios=[1] * num_rows, wspace=0.03, hspace=0.1)
        
        axes = []
        # Create axes sharing x-axis
        ax0 = plt.subplot(gs[0, 0])
        axes.append(ax0)
        for i in range(1, num_rows):
            axes.append(plt.subplot(gs[i, 0], sharex=ax0))
            
        # Colorbar next to L2 row (index 1) if it exists
        cbar_ax = plt.subplot(gs[1, 1]) if num_rows > 1 else None
        
        plt.style.use('seaborn-v0_8-paper')
        
        for i, row_label in enumerate(norm_types):
            # Determine if this is a Block row (Indices 2+)
            is_block_row = i >= 2
            
            if is_block_row:
                # Discrete colormap for blocks
                unique_blocks = np.unique(data_matrix[i, :])
                # Use high-contrast discrete colormap
                color_palettes = ["tab20", "tab20b", "tab20c"]
                # Combine multiple colormaps as needed to get enough colors
                all_colors = []
                for palette in color_palettes:
                    all_colors.extend(sns.color_palette(palette))
                # Ensure enough colors
                while len(all_colors) < len(unique_blocks):
                    # If still not enough, cycle through existing colors
                    all_colors.extend(all_colors[:len(unique_blocks) - len(all_colors)])
                # Create custom colormap
                custom_cmap = sns.color_palette(all_colors[:len(unique_blocks)], as_cmap=True)
                
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap=custom_cmap,
                    xticklabels=layer_numbers,
                    yticklabels=[row_label],
                    cbar=False,
                    square=False,
                    linewidths=0,  # Remove separators between blocks
                    ax=axes[i],
                    vmin=0, vmax=max(unique_blocks) if len(unique_blocks) > 0 else 1
                )
            elif i == 0:  # L1 row (No colorbar)
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=[row_label],
                    cbar=False,
                    square=False,
                    linewidths=0,
                    ax=axes[i],
                    vmin=0, vmax=1
                )
            elif i == 1:  # L2 row (With colorbar)
                sns.heatmap(
                    data_matrix[i:i+1, :],
                    annot=False,
                    cmap="viridis",
                    xticklabels=False,
                    yticklabels=[row_label],
                    cbar=True,
                    cbar_ax=cbar_ax,
                    cbar_kws={"label": "Norm Diff", "shrink": 1.0}, # Adjusted shrink
                    square=False,
                    linewidths=0,
                    ax=axes[i],
                    vmin=0, vmax=1
                )
        
        # Colorbar styling
        if cbar_ax is not None:
            cbar_ax.set_ylabel('Norm Diff', fontsize=7, labelpad=5)
            cbar_ax.tick_params(labelsize=6)
        
        # Axis styling
        for ax in axes:
            ax.tick_params(axis='y', labelsize=7, rotation=0) # Ensure y-labels are horizontal
            ax.set_aspect("auto")
            # Hide x-ticks for all but the last row
            if ax != axes[-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        # Bottom label
        axes[-1].set_xlabel("Transformer Layer", fontsize=9)
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
        
        plt.tight_layout()
        # Adjust margins
        plt.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.92)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()
        plt.close()
    
    def visualize_task_diffs(self, model1_path, model2_path, output_dir=None, num_blocks=8, metric="L2-norm"):
        """
        Visualize differences between two task models with automatic layer merging
        Returns: merged_blocks - list of tuples containing (layer_numbers, block_diff)
        """
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load models
        print(f"Loading model 1 from {model1_path}...")
        model1_tensors = self.load_model_tensors(model1_path)
        
        print(f"\nLoading model 2 from {model2_path}...")
        model2_tensors = self.load_model_tensors(model2_path)
        
        # Calculate task differences
        print(f"\nCalculating differences between models using {metric} metric...")
        layer_diffs = self.calculate_task_diffs(model1_tensors, model2_tensors, metric=metric)
        
        # Save layer sorted result to txt file, sorted by merged block differences
        print(f"\nSaving layer sorted result...")
        sorted_result_path = os.path.join(output_dir, "task_diff_layer_sorted.txt") if output_dir else "task_diff_layer_sorted.txt"
        self.save_layer_sorted_result(layer_diffs, sorted_result_path, num_blocks)
        
        # Prepare visualization data with automatic layer merging
        print(f"\nPreparing visualization data with {num_blocks} merged blocks...")
        data_matrix, norm_types, layer_numbers, merged_blocks = self.prepare_visualization_data(layer_diffs, num_blocks)
        
        # Plot heatmap
        print(f"\nPlotting heatmap...")
        output_path = os.path.join(output_dir, "task_diff_heatmap.png") if output_dir else None
        self.plot_horizontal_heatmap(data_matrix, norm_types, layer_numbers, output_path)
        
        print(f"\nAutomatic layer merging completed. {num_blocks} blocks created.")
        return merged_blocks
    
    def generate_multiple_block_configs(self, model1_path, model2_path, block_numbers, output_dir=None, metric="L2-norm", partition_method="hybrid"):
        """
        Generate multiple block configurations from fine to coarse, loading models only once.
        
        Args:
            model1_path: Path to the first model
            model2_path: Path to the second model
            block_numbers: List of block numbers in ascending order (e.g., [6, 12, 24, 36])
            output_dir: Directory to save visualization results
            metric: Distance metric to use for calculating layer differences
            partition_method: Partition method to use ("hybrid", "balance", "variance")
            
        Returns:
            Dict[int, List[Tuple[List[int], float]]]: Dictionary mapping block numbers to their merged blocks
        """
        # Validate inputs
        if not os.path.exists(model1_path):
            raise ValueError(f"Model 1 path does not exist: {model1_path}")
        
        if not os.path.exists(model2_path):
            raise ValueError(f"Model 2 path does not exist: {model2_path}")
        
        if not isinstance(block_numbers, list) or len(block_numbers) < 2:
            raise ValueError("block_numbers must be a list with at least 2 elements")
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load models once
        print(f"Loading model 1 from {model1_path}...")
        model1_tensors = self.load_model_tensors(model1_path)
        
        print(f"\nLoading model 2 from {model2_path}...")
        model2_tensors = self.load_model_tensors(model2_path)
        
        # Calculate task differences once with specified metric
        print(f"\nCalculating differences between models using {metric} metric...")
        layer_diffs = self.calculate_task_diffs(model1_tensors, model2_tensors, metric=metric)
        
        # Get all unique layer numbers (sorted by layer number)
        all_layers = sorted(layer_diffs.keys())
        num_layers = len(all_layers)
        
        # Calculate total normalized difference for each layer
        # Only use specified metric to calculate differences
        norm_types = [metric]  # Only use specified metric
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer].get(norm, 0.0) for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        total_normalized_diffs = [sum(normalized_diffs[norm][i] for norm in norm_types) for i in range(len(all_layers))]
        
        # Generate block configurations from fine to coarse
        block_configs = {}
        
        # Sort block numbers in ascending order (fine to coarse)
        sorted_block_numbers = sorted(block_numbers)
        
        # Generate the finest partition first
        finest_blocks = sorted_block_numbers[-1]
        print(f"\nGenerating finest partition: {finest_blocks} blocks...")
        
        # Get data for finest partition
        data_matrix, _, _, merged_blocks = self.prepare_visualization_data(layer_diffs, finest_blocks)
        block_configs[finest_blocks] = merged_blocks
        
        # Generate coarser partitions using the previous partition as input
        for i in range(len(sorted_block_numbers) - 2, -1, -1):
            current_blocks = sorted_block_numbers[i]
            previous_blocks = sorted_block_numbers[i + 1]
            
            print(f"\nGenerating partition: {current_blocks} blocks (from {previous_blocks} blocks)...")
            
            # Get the previous partition
            prev_merged_blocks = block_configs[previous_blocks]
            
            # Convert previous merged blocks to block boundaries
            prev_block_boundaries = []
            current_layer_idx = 0
            for block_layers, _ in prev_merged_blocks:
                prev_block_boundaries.append(current_layer_idx)
                current_layer_idx += len(block_layers)
            prev_block_boundaries.append(current_layer_idx)
            
            # Calculate block-level differences
            block_diff_values = []
            for block_idx in range(len(prev_merged_blocks)):
                start = prev_block_boundaries[block_idx]
                end = prev_block_boundaries[block_idx + 1]
                block_diff = sum(total_normalized_diffs[start:end])
                block_diff_values.append(block_diff)
            
            # Apply optimization based on partition method
            if partition_method == "balance":
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=0.0, beta=1.0)
            elif partition_method == "variance":
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=1.0, beta=0.0)
            else:  # hybrid
                cuts = self.hybrid_optimization(block_diff_values, current_blocks, alpha=self.alpha, beta=self.beta)
            
            # Convert cuts from block indices to layer indices
            new_block_boundaries = [prev_block_boundaries[cut] for cut in cuts]
            
            # Generate new merged blocks based on the new boundaries
            new_merged_blocks = []
            for j in range(len(new_block_boundaries) - 1):
                start_idx, end_idx = new_block_boundaries[j], new_block_boundaries[j + 1]
                block_layers = all_layers[start_idx:end_idx]
                block_diff = sum(total_normalized_diffs[start_idx:end_idx])
                new_merged_blocks.append((block_layers, block_diff))
            
            block_configs[current_blocks] = new_merged_blocks
        
        # Generate combined heatmap with L1, L2, and all block configurations
        self.generate_combined_block_heatmap(
            layer_diffs, 
            block_configs, 
            all_layers, 
            output_dir=output_dir
        )
        
        # Save individual visualizations for each block configuration
        for num_blocks in block_configs.keys():
            if output_dir:
                # Prepare data matrix for visualization
                data_matrix, norm_types, layer_numbers, _ = self.prepare_visualization_data(layer_diffs, num_blocks)
                output_path = os.path.join(output_dir, f"task_diff_heatmap_{num_blocks}.png")
                self.plot_horizontal_heatmap(data_matrix, norm_types, layer_numbers, output_path)
                
                sorted_result_path = os.path.join(output_dir, f"task_diff_layer_sorted_{num_blocks}.txt")
                self.save_layer_sorted_result(layer_diffs, sorted_result_path, num_blocks)
        
        return block_configs
    
    def generate_combined_block_heatmap(self, layer_diffs, block_configs, all_layers, output_dir=None):
        """
        Generate a combined heatmap with dynamic rows:
        Row 1: Current metric
        Row 2: Block configuration 1 (finest)
        Row 3: Block configuration 2
        Row 4: Block configuration 3
        Row 5: Block configuration 4 (coarsest)
        
        Args:
            layer_diffs: Layer differences dictionary
            block_configs: Dictionary mapping block numbers to merged blocks
            all_layers: List of all layer numbers
            output_dir: Directory to save the visualization
        """
        print("\nGenerating combined block heatmap...")
        
        # Get current metric from layer_diffs
        current_metric = list(layer_diffs[all_layers[0]].keys())[0]
        
        # Sort block numbers in descending order (finest to coarsest)
        sorted_block_nums = sorted(block_configs.keys(), reverse=True)
        
        # Calculate total normalized difference for each layer using current metric
        norm_types = [current_metric]
        normalized_diffs = {}
        for norm in norm_types:
            norm_values = [layer_diffs[layer][norm] for layer in all_layers]
            norm_min = min(norm_values)
            norm_max = max(norm_values)
            if norm_max > norm_min:
                normalized_diffs[norm] = [(v - norm_min) / (norm_max - norm_min) for v in norm_values]
            else:
                normalized_diffs[norm] = [0.0 for _ in norm_values]
        
        # Create combined data matrix with 5 rows (1 metric + 4 block configs)
        num_layers = len(all_layers)
        combined_data = np.zeros((5, num_layers))
        
        # Row 0: Current metric
        combined_data[0, :] = normalized_diffs[current_metric]
        
        # Rows 1-4: Block configurations (finest to coarsest)
        for i, num_blocks in enumerate(sorted_block_nums[:4]):
            merged_blocks = block_configs[num_blocks]
            
            # Create layer-to-block mapping
            layer_block_map = {}
            for block_idx, (block_layers, _) in enumerate(merged_blocks):
                for layer in block_layers:
                    layer_idx = all_layers.index(layer)
                    layer_block_map[layer_idx] = block_idx
            
            # Fill in the data for this block configuration
            for layer_idx in range(num_layers):
                combined_data[1 + i, layer_idx] = layer_block_map.get(layer_idx, 0)
        
        # Create row labels with dynamic block configurations
        row_labels = [
            current_metric
        ]
        
        # Add labels only for the actual number of block configurations
        for num_blocks in sorted_block_nums[:4]:
            row_labels.append(f"Blocks-{num_blocks}")
        
        # Fill remaining rows with empty labels if needed
        while len(row_labels) < 5:
            row_labels.append("")
        
        # Plot the combined heatmap
        if output_dir:
            output_path = os.path.join(output_dir, "task_diff_heatmap_combined.png")
            self.plot_horizontal_heatmap(combined_data, row_labels, all_layers, output_path)
        else:
            self.plot_horizontal_heatmap(combined_data, row_labels, all_layers)
    
    def run(self, model1_path, model2_path, output_dir=None, num_blocks=8, metric="L2-norm", partition_method="hybrid"):
        """
        Run the complete visualization pipeline with automatic layer merging
        Returns: merged_blocks - list of tuples containing (layer_numbers, block_diff)
        """
        # Validate inputs
        if not os.path.exists(model1_path):
            raise ValueError(f"Model 1 path does not exist: {model1_path}")
        
        if not os.path.exists(model2_path):
            raise ValueError(f"Model 2 path does not exist: {model2_path}")
        
        # Run visualization with automatic layer merging and return merged blocks
        merged_blocks = self.visualize_task_diffs(model1_path, model2_path, output_dir, num_blocks, metric=metric)
        return merged_blocks


def main():
    """
    Main function to run the visualizer
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Model Difference Visualizer")
    parser.add_argument("--model1", required=True, help="Path to first task model directory")
    parser.add_argument("--model2", required=True, help="Path to second task model directory")
    parser.add_argument("--output", default=None, help="Output directory for heatmap")
    parser.add_argument("--device", default=None, help="Device to use (cpu or cuda)")
    parser.add_argument("--metric", type=str, default="L2-norm", help="Distance metric to use: Fisher, Grassmann, L2-norm, Block")
    parser.add_argument("--partition-method", type=str, default="hybrid", help="Partition method to use: hybrid, balance, variance")
    parser.add_argument("--num-blocks", type=int, default=8, help="Number of blocks to merge into")
    
    args = parser.parse_args()
    
    # Create visualizer instance
    visualizer = TaskDiffVisualizer(device=args.device)
    
    # Run visualization
    visualizer.run(args.model1, args.model2, args.output, args.num_blocks, args.metric, args.partition_method)

if __name__ == "__main__":
    main()
