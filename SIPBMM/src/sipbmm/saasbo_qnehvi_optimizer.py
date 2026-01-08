import os
import gc
import torch
import numpy as np
import datetime
import json
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.settings import fast_computations
from botorch.fit import fit_gpytorch_mll as botorch_fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior, HalfCauchyPrior
from botorch.models.transforms import Normalize as BoTorchNormalize, Standardize

def saasbo_qnehvi_optimizer(
    objective_function,
    dim=3,
    num_objectives=2,
    bounds=None,
    BATCH_SIZE=5,
    NUM_RESTARTS=20,
    RAW_SAMPLES=512,
    MC_SAMPLES=128,
    N_BATCH=40,
    verbose=True,
    device="cpu",
    dtype=torch.double,  # Force use of double precision
    initial_samples=10,
    noise_level=0.01,
    iteration_callback=None,
    ref_point=-1.1,
    run_id=None,
    checkpoint_dir="./checkpoints",
    custom_initial_solutions=None,
    # SAASBO and importance parameters
    use_saas=True,
    initial_importance=None,
    enable_importance_prior=True,
    # Random seed parameter for reproducible results
    seed=42
):
    """
    Encapsulation function for optimizing multi-objective problems using SAASBO+qNEHVI algorithm
    
    Args:
    ----------
    objective_function : callable
        Objective function that takes a tensor of shape (batch_size, dim) as input and returns a tensor of shape (batch_size, num_objectives)
    dim : int, optional
        Dimension of decision variables, default is 3
    num_objectives : int, optional
        Number of objective functions, default is 2
    bounds : torch.Tensor, optional
        Bounds of decision variables, shape is (2, dim). If None, default bounds [0, 1] are used
    BATCH_SIZE : int, optional
        Number of candidate points selected per iteration, default is 5
    NUM_RESTARTS : int, optional
        Number of restarts when optimizing the acquisition function, default is 20
    RAW_SAMPLES : int, optional
        Number of raw samples used to initialize optimization, default is 512
    MC_SAMPLES : int, optional
        Number of Monte Carlo samples, default is 128
    N_BATCH : int, optional
        Total number of optimization rounds, default is 40
    verbose : bool, optional
        Whether to print optimization progress, default is True
    device : str, optional
        Computing device, default is "cpu"
    dtype : torch.dtype, optional
        Data type, default is torch.double
    initial_samples : int, optional
        Number of initial sampling points, default is 10
    noise_level : float, optional
        Observation noise standard deviation, default is 0.01
    iteration_callback : callable, optional
        Callback function called after each iteration completes, with signature callback(iteration, train_x, train_obj_true, hvs), default is None
    ref_point : float or torch.Tensor, optional
        Reference point for hypervolume calculation, default is -1.1
    run_id : str, optional
        Unique identifier for the run, if None an automatically generated timestamp format ID is used
    checkpoint_dir : str, optional
        Root directory for saving checkpoints, default is "./checkpoints"
    custom_initial_solutions : list, optional
        User-defined initial solution list
    use_saas : bool, optional
        Whether to use SAAS prior, default is True
    initial_importance : torch.Tensor, optional
        Initial importance weights, shape is (dim,), default is None (uniform distribution)
    enable_importance_prior : bool, optional
        Whether to integrate importance prior in the model, default is True
    
    Returns:
    -------
    train_x : torch.Tensor
        Decision variable values for all evaluation points
    train_obj_true : torch.Tensor
        True objective function values for all evaluation points
    hvs : list
        Hypervolume values for each iteration
    problem_ref_point : torch.Tensor
        Reference point for the problem
    run_id : str
        Unique identifier for this run
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set tkwargs
    tkwargs = {
        "dtype": dtype,
        "device": torch.device(device),
    }
    
    # Generate or use provided run_id
    if run_id is None:
        # Generate unique ID using current time
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set bounds
    if bounds is None:
        bounds = torch.zeros(2, dim, **tkwargs)
        bounds[1] = 1
    else:
        # Ensure bounds are on the correct device and dtype
        bounds = bounds.to(**tkwargs)
    
    # Standardize bounds
    standard_bounds = torch.zeros(2, dim, **tkwargs)
    standard_bounds[1] = 1
    
    # Set noise
    NOISE_SE = torch.full((num_objectives,), noise_level, **tkwargs)
    
    # Determine reference point (according to botorch requirements, reference point should be a point in objective space where all true objectives should be greater than this point)
    # Here we assume the objective is maximization, so set reference point to a smaller value
    if type(ref_point) == torch.Tensor:
        problem_ref_point = ref_point
    else:
        problem_ref_point = torch.full((num_objectives,), ref_point, **tkwargs)
    
    # Initialize importance
    if initial_importance is None:
        importance = torch.ones(dim, **tkwargs) * 0.5
    else:
        importance = initial_importance.clone().to(**tkwargs)
    
    def uniform_lhs(n_samples: int) -> torch.Tensor:
        """Uniform Latin Hypercube Sampling"""
        samples = draw_sobol_samples(bounds, n_samples, 1).squeeze(1)
        return samples.to(**tkwargs)
    
    def generate_initial_data(n=initial_samples, custom_solutions=None):
        """Generate initial training data"""
        # Process custom initial solutions
        if custom_solutions is not None and len(custom_solutions) > 0:
            # Calculate number of custom solutions to generate
            num_custom = len(custom_solutions)
            # Ensure not exceeding total sample count
            num_custom = min(num_custom, n)
            
            # Generate custom solutions
            custom_x = []
            for val in custom_solutions:
                # Generate solution with all val, length dim
                custom_sol = torch.full((1, dim), val, **tkwargs)
                custom_x.append(custom_sol)
            
            # Merge custom solutions
            custom_x = torch.cat(custom_x, dim=0)
            
            # Calculate remaining samples needed
            remaining = n - num_custom
            
            if remaining > 0:
                # Generate remaining samples using uniform LHS
                random_x = uniform_lhs(remaining)
                # Merge custom and random solutions
                train_x = torch.cat([custom_x, random_x], dim=0)
            else:
                # If custom solutions >= n, take first n
                train_x = custom_x[:n]
        else:
            # No custom solutions, generate using uniform LHS
            train_x = uniform_lhs(n)
        
        # Call objective function to get objective values and evaluation results
        result = objective_function(train_x)
        
        # Process return result, supporting two formats: only objective values, or objective values and evaluation results
        if isinstance(result, tuple) and len(result) == 2:
            train_obj_true, train_info = result
        else:
            train_obj_true = result
            train_info = [{} for _ in range(train_x.shape[0])]  # Empty evaluation results
        
        # Ensure train_obj_true is on the correct device and dtype
        train_obj_true = train_obj_true.to(**tkwargs)
        # Generate random noise on the same device and dtype
        train_obj = train_obj_true + torch.randn_like(train_obj_true, **tkwargs) * NOISE_SE
        
        return train_x, train_obj, train_obj_true, train_info
    
    def create_saas_model(train_x, train_obj):
        """Create SAAS model"""
        # Ensure train_x and train_obj use double precision
        train_x = train_x.to(dtype=torch.double, device=train_obj.device)
        train_y = train_obj.to(dtype=torch.double, device=train_obj.device)
        models = []
        
        for i in range(train_obj.shape[-1]):
            train_y_i = train_y[..., i : i + 1]
            
            # Force use of double precision for all model components
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 10.0)
                )
            )
            
            if use_saas and enable_importance_prior:
                # Set importance-based length scale initial values
                with torch.no_grad():
                    # High importance → small length scale
                    # Interval mapping based on importance values
                    # Divide importance values into four intervals and map to specific length scales
                    # Interval 1: importance > 0.9 → length scale 1.0
                    # Interval 2: 0.8 < importance ≤ 0.9 → length scale 2.0
                    # Interval 3: 0.7 < importance ≤ 0.8 → length scale 3.0
                    # Interval 4: importance ≤ 0.7 → length scale 5.0
                    initial_lengthscales = torch.zeros_like(importance)
                    
                    # Interval 1: importance > 0.9 → 1.0
                    initial_lengthscales[importance > 0.9] = 0.5
                    
                    # Interval 2: 0.8 < importance ≤ 0.9 → 2.0
                    initial_lengthscales[(importance > 0.8) & (importance <= 0.9)] = 1.0
                    
                    # Interval 3: 0.7 < importance ≤ 0.8 → 3.0
                    initial_lengthscales[(importance > 0.7) & (importance <= 0.8)] = 1.0
                    
                    # Interval 4: importance ≤ 0.7 → 5.0
                    initial_lengthscales[importance <= 0.7] = 5.0
                    
                    initial_lengthscales = initial_lengthscales.view(1, dim).to(dtype=torch.double)
                    covar_module.base_kernel.lengthscale = initial_lengthscales
                
            # Create model
            model = SingleTaskGP(
                train_x,
                train_y_i,
                covar_module=covar_module,
                input_transform=BoTorchNormalize(d=dim),
                outcome_transform=Standardize(m=1)
            )
            
            if use_saas:
                # Set SAAS prior
                model.likelihood.noise_covar.register_prior(
                    "noise_prior",
                    GammaPrior(1.1, 0.05),
                    "raw_noise"
                )
                
                model.covar_module.outputscale_prior = GammaPrior(2.0, 0.15)
            
            # Ensure model parameters use double precision
            model = model.double()
            models.append(model)
        
        model = ModelListGP(*models)
        # Ensure all model components use double precision
        model = model.double()
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    
    def initialize_model(train_x, train_obj):
        """Initialize model"""
        return create_saas_model(train_x, train_obj)
    
    def fit_gpytorch_mll(mll):
        """Train model with retry mechanism"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Try using different optimizer parameters
                optimizer_kwargs = {
                    'options': {
                        'maxiter': 1000,
                        'maxfun': 1000
                    },
                    'method': 'L-BFGS-B'
                }
                botorch_fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)
                return  # Successfully fitted, exit function
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # Exceeded maximum retries, throw exception
                # Reduce learning rate and retry
                print(f"Model fitting failed, retrying {retry_count}/{max_retries}: {str(e)}")
    
    def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
        """Optimize qNEHVI acquisition function and get observations"""
        # Ensure all data is on the same device and dtype
        ref_point_device = train_x.device
        ref_point_dtype = train_x.dtype
        
        # Ensure bounds are on the correct device and dtype
        device_bounds = bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # Normalize training data
        train_x_normalized = normalize(train_x, device_bounds)
        train_x_normalized = train_x_normalized.to(dtype=ref_point_dtype)
        
        # Create acquisition function
        ref_point = problem_ref_point.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # Create acquisition function
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x_normalized,
            prune_baseline=True,
            sampler=sampler,
        )
        
        # Ensure standard_bounds are on the correct device and dtype
        device_standard_bounds = standard_bounds.to(device=ref_point_device, dtype=ref_point_dtype)
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=device_standard_bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 100},
        )
        
        # Observe new values
        with torch.no_grad():
            new_x = unnormalize(candidates.detach(), bounds=device_bounds)
            # Ensure new_x is on the correct device and dtype
            new_x = new_x.to(device=ref_point_device, dtype=ref_point_dtype)
            
            # Call objective function to get objective values and evaluation results
            result = objective_function(new_x)
            
            # Process return result, supporting two formats: only objective values, or objective values and evaluation results
            if isinstance(result, tuple) and len(result) == 2:
                new_obj_true, new_info = result
            else:
                new_obj_true = result
                new_info = [{} for _ in range(new_x.shape[0])]  # Empty evaluation results
            
            # Ensure new_obj_true is on the correct device and dtype
            new_obj_true = new_obj_true.to(device=ref_point_device, dtype=ref_point_dtype)
            # Ensure NOISE_SE is on the correct device and dtype
            device_noise_se = NOISE_SE.to(device=ref_point_device, dtype=ref_point_dtype)
            # Generate random noise on the same device and dtype
            new_obj = new_obj_true + torch.randn_like(new_obj_true, device=ref_point_device, dtype=ref_point_dtype) * device_noise_se
        
        # Explicitly release temporary variables
        del train_x_normalized, acq_func, candidates
        torch.cuda.empty_cache()
        
        return new_x, new_obj, new_obj_true, new_info
    
    def compute_pareto_hypervolume(objectives, ref_point):
        """Compute hypervolume of Pareto front"""
        # Ensure ref_point and objectives are on the same device
        ref_point = ref_point.to(device=objectives.device, dtype=objectives.dtype)
        bd = FastNondominatedPartitioning(ref_point=ref_point, Y=objectives)
        return bd.compute_hypervolume().item()

    def save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs):
        """
        Save optimization process checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'iteration': iteration,
            'hvs': hvs,
            'importance': importance.cpu().tolist(),
            'evaluated_solutions': {
                'decision_variables': train_x.cpu().tolist(),
                'objectives': train_obj_true.cpu().tolist(),
                'metrics': train_info
            }
        }
        
        # Save non-tensor data to JSON file
        json_path = os.path.join(run_dir, f'checkpoint_iter_{iteration}.json')
        with open(json_path, 'w') as f:
            json.dump(checkpoint, f)
        
        # Save tensor data
        torch.save({
            'train_x': train_x.cpu(),
            'train_obj': train_obj.cpu(),
            'train_obj_true': train_obj_true.cpu(),
            'train_info': train_info,
            'importance': importance.cpu()
        }, os.path.join(run_dir, f'checkpoint_iter_{iteration}.pt'))
        
        # Also save the latest checkpoint for easy restart
        torch.save({
            'train_x': train_x.cpu(),
            'train_obj': train_obj.cpu(),
            'train_obj_true': train_obj_true.cpu(),
            'train_info': train_info,
            'iteration': iteration,
            'hvs': hvs,
            'importance': importance.cpu()
        }, os.path.join(run_dir, 'checkpoint_latest.pt'))

    def load_checkpoint(run_dir, tkwargs):
        """
        Load the latest checkpoint
        """
        latest_checkpoint_path = os.path.join(run_dir, 'checkpoint_latest.pt')
        if not os.path.exists(latest_checkpoint_path):
            return None
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
        
        # Convert to specified device and dtype
        train_x = checkpoint['train_x'].to(**tkwargs)
        train_obj = checkpoint['train_obj'].to(**tkwargs)
        train_obj_true = checkpoint['train_obj_true'].to(**tkwargs)
        iteration = checkpoint['iteration']
        hvs = checkpoint['hvs']
        importance = checkpoint['importance'].to(**tkwargs)
        
        # Load evaluation information, create empty list if not exists
        train_info = checkpoint.get('train_info', [{} for _ in range(train_x.shape[0])])
        
        return train_x, train_obj, train_obj_true, train_info, iteration, hvs, importance
    
    def get_importance_report():
        """
        Get importance report
        
        Returns:
        -------
        dict: Dictionary containing importance report
        """
        import pandas as pd
        
        data = []
        for i, imp in enumerate(importance):
            # Determine importance category
            if imp > 0.8:
                category = "critical"
            elif imp > 0.6:
                category = "important"
            elif imp > 0.4:
                category = "medium"
            elif imp > 0.2:
                category = "minor"
            else:
                category = "negligible"
            
            data.append({
                'variable': i,
                'importance': imp.item(),
                'category': category
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('importance', ascending=False)
        
        # Convert to dictionary format and return
        return {
            'importance_values': importance.tolist(),
            'importance_report': df.to_dict(orient='records'),
            'average_importance': importance.mean().item(),
            'critical_variables': len(df[df['category'] == 'critical']),
            'important_variables': len(df[df['category'] == 'important']),
            'medium_variables': len(df[df['category'] == 'medium']),
            'minor_variables': len(df[df['category'] == 'minor']),
            'negligible_variables': len(df[df['category'] == 'negligible'])
        }
    
    def get_performance_summary(train_x, train_obj_true, hvs):
        """
        Get performance summary
        
        Args:
        ----------
        train_x : torch.Tensor
            Decision variable values for all evaluation points
        train_obj_true : torch.Tensor
            True objective function values
        hvs : list
            Hypervolume history
        
        Returns:
        -------
        dict: Dictionary containing performance summary
        """
        summary = {
            'total_evaluations': len(train_x),
            'best_hypervolume': hvs[-1] if hvs else 0.0,
            'average_importance': importance.mean().item(),
            'total_iterations': len(hvs) - 1 if hvs else 0,
            'dimensions': dim,
            'num_objectives': num_objectives
        }
        
        # Calculate statistics for objective functions
        if train_obj_true.numel() > 0:
            summary['objective_min'] = train_obj_true.min().item()
            summary['objective_max'] = train_obj_true.max().item()
            summary['objective_mean'] = train_obj_true.mean().item()
        
        return summary
    
    # Start optimization process
    if verbose:
        print(f"Optimizing multi-objective problem using SAASBO+qNEHVI algorithm (dim: {dim}, objectives: {num_objectives})")
        print(f"Device: {tkwargs['device']}, dtype: {tkwargs['dtype']}")
        print(f"Run ID: {run_id}, checkpoint dir: {run_dir}")
        print(f"SAAS config: use_saas={use_saas}")
        print(f"Importance config: enable_importance_prior={enable_importance_prior}")
    
    # Try to load checkpoint
    checkpoint = load_checkpoint(run_dir, tkwargs)
    if checkpoint is not None:
        train_x, train_obj, train_obj_true, train_info, start_iteration, hvs, importance = checkpoint
        if verbose:
            print(f"Successfully loaded checkpoint, continuing from iteration {start_iteration}")
            print(f"Current hypervolume: {hvs[-1]:.4f}")
    else:
        # Generate initial data
        train_x, train_obj, train_obj_true, train_info = generate_initial_data(custom_solutions=custom_initial_solutions)
        
        # Record hypervolume
        hvs = []
        initial_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
        hvs.append(initial_hv)
        if verbose:
            print(f"Initial hypervolume: {initial_hv:.4f}")
        
        start_iteration = 0
        # Save initial state
        save_checkpoint(0, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs)
    
    try:
        # Run remaining rounds of Bayesian optimization
        for iteration in range(start_iteration + 1, N_BATCH + 1):
            # Initialize and train model
            mll, model = initialize_model(train_x, train_obj)
            fit_gpytorch_mll(mll)
            
            # Define QMC sampler
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            
            # Optimize acquisition function and get new observations
            new_x, new_obj, new_obj_true, new_info = optimize_qnehvi_and_get_observation(
                model, train_x, train_obj, sampler
            )
            
            # Update training points, ensure all tensors are on the same device
            train_x = torch.cat([train_x, new_x.to(train_x.device)])
            train_obj = torch.cat([train_obj, new_obj.to(train_obj.device)])
            train_obj_true = torch.cat([train_obj_true, new_obj_true.to(train_obj_true.device)])
            train_info.extend(new_info)  # Update evaluation information list
            
            # Calculate new hypervolume
            with torch.no_grad():
                new_hv = compute_pareto_hypervolume(train_obj_true, problem_ref_point)
            hvs.append(new_hv)
            
            # Print progress
            if verbose:
                print(f"Iteration {iteration:>2}: Hypervolume = {new_hv:.4f}")
                # Print current GPU memory usage
                if tkwargs['device'].type == 'cuda':
                    print(f"    GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
            
            # Execute iteration callback function (if provided)
            if iteration_callback is not None:
                try:
                    iteration_callback(iteration, train_x, train_obj_true, hvs)
                except Exception as e:
                    print(f"Warning: Iteration callback execution failed: {e}")
            
            # Save current iteration state
            save_checkpoint(iteration, train_x, train_obj, train_obj_true, train_info, hvs, run_dir, tkwargs)
    except Exception as e:
        print(f"Error: Exception occurred during optimization: {e}")
        import traceback
        traceback.print_exc()
        # Return existing results even if an error occurs
        return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
    
    return train_x, train_obj_true, train_info, hvs, problem_ref_point, run_id
