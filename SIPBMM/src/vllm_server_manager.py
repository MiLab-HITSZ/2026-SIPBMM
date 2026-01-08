#!/usr/bin/env python3
"""
VLLM Server Manager Module

This module provides a manager class for centralized management of multiple vLLM server processes,
supporting automatic resource recycling, and providing functions such as adding and querying servers.
"""

import os
import signal
import subprocess
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import requests
import json

    # Built-in simple implementation to ensure functionality even if start_vllm_server cannot be imported
def set_gpu_exclusive_mode(gpus: List[int]) -> bool:
    """
    Set GPU to exclusive process mode
    
    Args:
        gpus: List of GPUs to be set to exclusive mode
        
    Returns:
        bool: Whether all GPUs were successfully set to exclusive mode
    """
    success = True
    for gpu_id in gpus:
        try:
            # Use nvidia-smi command to set GPU to exclusive process mode (value 3)
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-c", "3"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully set GPU {gpu_id} to exclusive process mode")
        except subprocess.CalledProcessError as e:
            print(f"Failed to set GPU {gpu_id} to exclusive process mode: {e.stderr}")
            success = False
        except Exception as e:
            print(f"Error occurred while setting GPU {gpu_id} to exclusive process mode: {str(e)}")
            success = False
    return success

def set_gpu_default_mode(gpus: List[int]) -> bool:
    """
    Set GPU to default mode
    
    Args:
        gpus: List of GPUs to be set to default mode
        
    Returns:
        bool: Whether all GPUs were successfully set to default mode
    """
    success = True
    for gpu_id in gpus:
        try:
            # Use nvidia-smi command to set GPU to default mode
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-c", "default"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully restored GPU {gpu_id} to default mode")
        except subprocess.CalledProcessError as e:
            print(f"Failed to restore GPU {gpu_id} to default mode: {e.stderr}")
            success = False
        except Exception as e:
            print(f"Error occurred while restoring GPU {gpu_id} to default mode: {str(e)}")
            success = False
    return success

def start_vllm_server(
    model_path: str = "models/Qwen3-4B-Instruct-2507",
    port: int = 8000,
    served_model_name: str = "Qwen3-4B-Instruct",
    tensor_parallel_size: int = 8,
    max_model_len: int = 25000,
    worker_multiproc_method: str = "spawn",
    use_modelscope: bool = True,
    gpus: Optional[List[int]] = None,
    additional_args: Optional[List[str]] = None,
    blocking: bool = False,
    log_file: Optional[str] = None,
    exclusive_gpu_mode: bool = False
) -> subprocess.Popen:
    """Implementation of the start_vllm_server function, including CUDA environment variable configuration and compiler flag settings"""
    import sys
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = worker_multiproc_method
    env["VLLM_USE_MODELSCOPE"] = "True" if use_modelscope else "False"
    
    
    # Determine the GPU list to use
    if gpus is not None:
        gpu_list = gpus
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    else:
        gpu_list = list(range(8))  # Default to using GPUs 0-7
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # If GPU exclusive mode is enabled, set GPUs to exclusive process mode
    if exclusive_gpu_mode:
        set_gpu_exclusive_mode(gpu_list)
    
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(0.85)
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    # Set output redirection
    stdout_stream = subprocess.PIPE
    stderr_stream = subprocess.PIPE
    
    # If a log file is specified, redirect output to the file
    if log_file:
        # Ensure the log file directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        # Open the log file (append mode)
        log_stream = open(log_file, 'a', buffering=1)  # Line buffering
        stdout_stream = log_stream
        stderr_stream = log_stream
    
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_stream,
        stderr=stderr_stream,
        text=True
    )


@dataclass
class Task:
    """Task data class"""
    task_id: str
    model_path: str
    params_dict: Dict[str, Any]
    func_handle: Callable
    gpu_count: int
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: float = 0.0  # Task start time


@dataclass
class VllmServerInfo:
    """Data class for storing vLLM server information"""
    process: subprocess.Popen
    model_path: str
    port: int
    served_model_name: str
    tensor_parallel_size: int
    max_model_len: int
    worker_multiproc_method: str
    use_modelscope: bool
    gpus: Optional[List[int]]
    additional_args: Optional[List[str]]
    log_file: Optional[str] = None
    current_task: Optional[Task] = None
    start_time: float = field(default_factory=time.time)
    exclusive_gpu_mode: bool = False
    
    def is_running(self) -> bool:
        """Check if the process is still running"""
        return self.process.poll() is None
    
    def is_ready(self, timeout: int = 2) -> bool:
        """
        Check if the vLLM server is ready to accept requests
        
        Args:
            timeout: API call timeout (seconds)
            
        Returns:
            bool: True if the server is ready, False otherwise
        """
        try:
            # vLLM servers typically provide a /health or similar endpoint to check status
            # Here we use the /v1/models endpoint as a health check since it usually responds when the server is ready
            response = requests.get(
                f"http://localhost:{self.port}/v1/models",
                timeout=timeout
            )
            # If the server responds normally with status code 200, consider it ready
            return response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # If connection fails or times out, the server is likely not ready yet
            return False
        except Exception:
            # Catch any other possible exceptions
            return False
            
    def is_idle(self, timeout: int = 5) -> Optional[bool]:
        """
        Check if the vLLM server is idle (no ongoing requests)
        
        Args:
            timeout: HTTP request timeout (seconds)
            
        Returns:
            Optional[bool]: True if the server is idle, False if there are active tasks,
                          None if unable to connect to the server
        """
        if not self.is_running():
            return None
        
        try:
            # vLLM's API interface typically provides a stats endpoint to view server status
            url = f"http://localhost:{self.port}/v1/internal/stats"
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                stats = response.json()
                # Check if there are active generation tasks
                # Depending on vLLM implementation, active tasks count might be in different fields
                active_requests = stats.get('active_requests', 0)
                return active_requests == 0
            else:
                # If there's no stats endpoint, try using the health check endpoint
                health_url = f"http://localhost:{self.port}/health"
                health_response = requests.get(health_url, timeout=timeout)
                # Note: Health check only confirms the server is running, but cannot directly determine if it's idle
                # This is a fallback, returning True assumes the server is idle
                # In actual use, it's best to use the stats endpoint
                return health_response.status_code == 200
        except Exception as e:
            print(f"Error checking idle status of server {self.port}: {e}")
            return None
    
    def get_pid(self) -> int:
        """Get process ID"""
        return self.process.pid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert server information to dictionary format"""
        return {
            "pid": self.get_pid(),
            "model_path": self.model_path,
            "port": self.port,
            "served_model_name": self.served_model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "worker_multiproc_method": self.worker_multiproc_method,
            "use_modelscope": self.use_modelscope,
            "gpus": self.gpus,
            "additional_args": self.additional_args,
            "log_file": self.log_file,
            "start_time": self.start_time,
            "is_running": self.is_running(),
            "exclusive_gpu_mode": self.exclusive_gpu_mode
        }


class VllmServerManager:
    """
    vLLM Server Manager
    
    Features:
    1. Manage multiple vLLM server processes
    2. Automatically clean up resources (processes)
    3. Provide interfaces for adding and querying servers
    4. Support context manager mode
    5. Support task queue and GPU resource management
    """
    
    def __init__(self, available_gpus: Optional[List[int]] = None, max_model_len: int = 30000):
        """Initialize the manager"""
        # Dictionary to store server information, using port as key
        self._servers: Dict[int, VllmServerInfo] = {}
        # Thread lock to ensure thread safety
        self._lock = threading.RLock()
        # Flag indicating whether the manager has been destroyed
        self._destroyed = False
        # List of available GPUs
        self._available_gpus = available_gpus if available_gpus is not None else list(range(8))  # Default 8 GPUs
        # Mapping of GPUs in use
        self._used_gpus: Dict[int, List[int]] = {}
        # Task queue
        self._task_queue = queue.Queue()
        # Task results dictionary
        self._task_results: Dict[str, Any] = {}
        # Task thread
        self._task_thread: Optional[threading.Thread] = None
        # Whether task processing is running
        self._tasks_running = False
        # Result events for notifying task completion
        self._result_events: Dict[str, threading.Event] = {}
        # List of all tasks
        self._all_tasks: Dict[str, Task] = {}
        # Progress log file path
        self._progress_log_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output'), 'vllm_servers_logs.log')
        # Progress update thread
        self._progress_thread: Optional[threading.Thread] = None
        # Whether progress updates are running
        self._progress_running = False
        # Maximum model length
        self._max_model_len = max_model_len
        
        # Clear any existing task logs
        self._clear_progress_log()
        
        print(f"vLLM Server Manager initialized, available GPUs: {self._available_gpus}, max model length: {self._max_model_len}")
    
    def __del__(self):
        """Destructor, ensures all processes are cleaned up"""
        self.destroy()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, ensures resources are cleaned up"""
        self.destroy()
        return False  # Do not suppress exceptions
    
    def add_server(
        self,
        model_path: str = "models/Qwen3-4B-Instruct-2507",
        port: int = 8000,
        served_model_name: str = "Qwen3-4B-Instruct",
        tensor_parallel_size: int = 8,
        max_model_len: Optional[int] = None,
        worker_multiproc_method: str = "spawn",
        use_modelscope: bool = True,
        gpus: Optional[List[int]] = None,
        additional_args: Optional[List[str]] = None,
        wait_for_start: bool = False,
        wait_seconds: int = 5,
        log_file: Optional[str] = None,
        exclusive_gpu_mode: bool = False
    ) -> VllmServerInfo:
        """
        Add and start a new vLLM server
        
        Args:
            model_path: Model path
            port: Server port (must be unique)
            served_model_name: Served model name
            tensor_parallel_size: Tensor parallel size
            max_model_len: Maximum model length
            worker_multiproc_method: Worker multiprocessing method
            use_modelscope: Whether to use ModelScope
            gpus: List of visible GPUs
            additional_args: List of additional command-line arguments
            wait_for_start: Whether to wait for the server to start (check if process is still running)
            wait_seconds: Seconds to wait for server startup
            log_file: Log file path
            exclusive_gpu_mode: Whether to use exclusive GPU mode
            
        Returns:
            VllmServerInfo: Server information object
            
        Raises:
            ValueError: If the port is already in use
            RuntimeError: If the manager has been destroyed
        """
        with self._lock:
            # Check if the manager has been destroyed
            if self._destroyed:
                raise RuntimeError("Manager has been destroyed, cannot add new servers")
            
            # Check if the port is already in use
            if port in self._servers:
                existing_server = self._servers[port]
                if existing_server.is_running():
                    raise ValueError(f"Port {port} is already in use")
                else:
                    print(f"Note: Server on port {port} has stopped, will replace with new server")
            
            # Start the server
            print(f"Starting vLLM server, port: {port}, model: {model_path}")
            # Generate default log filename if none specified
            if log_file is None:
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output/vllm_manager_logs')
                model_name = os.path.basename(model_path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"vllm_server_{model_name}_{port}_{timestamp}.log")
                print(f"No log file specified, will use default log file: {log_file}")
            
            # Use instance variable value if max_model_len not specified
            if max_model_len is None:
                max_model_len = self._max_model_len
                print(f"max_model_len not specified, will use manager default: {max_model_len}")
            
            process = start_vllm_server(
                model_path=model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                blocking=False,
                log_file=log_file
            )
            
            # Create server information object
            server_info = VllmServerInfo(
                process=process,
                model_path=model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                log_file=log_file,
                exclusive_gpu_mode=exclusive_gpu_mode
            )
            
            # Store server information
            self._servers[port] = server_info
            
            # Wait for server startup
            if wait_for_start:
                print(f"Waiting for server to start and become ready, port: {port}...")
                start_time = time.time()
                ready = False
                
                # Loop to check if server is ready until timeout
                while time.time() - start_time < wait_seconds:
                    # First check if process is still running
                    if not server_info.is_running():
                        print(f"Warning: Server process exited, port: {port}")
                        # Try to get error output
                        stdout, stderr = process.communicate(timeout=1)
                        if stderr:
                            print(f"Error output:\n{stderr[:500]}..." if len(stderr) > 500 else f"Error output:\n{stderr}")
                        break
                    
                    # Check if server is ready to accept requests
                    if server_info.is_ready():
                        ready = True
                        print(f"Server is ready, port: {port}")
                        break
                    
                    # Sleep briefly before checking again
                    time.sleep(0.5)
                
                # If timeout and server is still not ready
                if not ready:
                    if server_info.is_running():
                        print(f"Warning: Server not fully ready within {wait_seconds} seconds, but process is still running, port: {port}")
                    else:
                        print(f"Warning: Server startup failed, port: {port}")
            
            print(f"vLLM server added, port: {port}, process ID: {process.pid}")
            return server_info
    
    def get_server_info(self, port: int) -> Optional[VllmServerInfo]:
        """
        Get server information for the specified port
        
        Args:
            port: Server port
            
        Returns:
            Optional[VllmServerInfo]: Server information object, None if not exists
        """
        with self._lock:
            return self._servers.get(port)
    
    def list_servers(self) -> List[VllmServerInfo]:
        """
        List all server information
        
        Returns:
            List[VllmServerInfo]: List of server information
        """
        with self._lock:
            return list(self._servers.values())
    
    def list_active_servers(self) -> List[VllmServerInfo]:
        """
        List all active (running) servers
        
        Returns:
            List[VllmServerInfo]: List of active server information
        """
        with self._lock:
            # Clean up stopped servers first
            self._cleanup_stopped_servers()
            # Return still running servers
            return [server for server in self._servers.values() if server.is_running()]
    
    def get_available_processes(self) -> Dict[int, Dict[str, Any]]:
        """
        Get current available vLLM process information
        
        Returns:
            Dict[int, Dict[str, Any]]: Mapping with port as key and server info dict as value
        """
        with self._lock:
            # Clean up stopped servers
            self._cleanup_stopped_servers()
            # Return info dictionaries for all active servers
            return {
                port: server.to_dict() 
                for port, server in self._servers.items() 
                if server.is_running()
            }
    
    def stop_server(self, port: int, wait_for_idle: bool = True, idle_timeout: int = 300) -> bool:
        """
        Stop the server on the specified port
        
        Args:
            port: Server port
            wait_for_idle: Whether to wait for server to be idle before stopping
            idle_timeout: Maximum time to wait for server to be idle (seconds)
            
        Returns:
            bool: True if server was successfully stopped, False otherwise
        """
        with self._lock:
            if port not in self._servers:
                print(f"Warning: No server found on port {port}")
                return False
            
            server_info = self._servers[port]
            # Save GPU information and exclusive mode settings
            gpus_to_release = server_info.gpus
            is_exclusive_mode = server_info.exclusive_gpu_mode
            
            if not server_info.is_running():
                print(f"Note: Server on port {port} has already stopped")
                del self._servers[port]
                # If exclusive GPU mode is enabled, restore GPUs to default mode
                if is_exclusive_mode and gpus_to_release:
                    set_gpu_default_mode(gpus_to_release)
                return True
            
            try:
                print(f"Stopping server, port: {port}, process ID: {server_info.get_pid()}")
                # If need to wait for server to be idle
                if wait_for_idle:
                    print(f"Waiting for server {port} to be idle...")
                    start_wait_time = time.time()
                    while time.time() - start_wait_time < idle_timeout:
                        # Check if server is idle
                        is_idle_result = server_info.is_idle()
                        if is_idle_result is True:
                            print(f"Server {port} is idle, ready to stop")
                            break
                        elif is_idle_result is False:
                            # Server is processing tasks, continue waiting
                            print(f"Server {port} is processing tasks, waiting...")
                            time.sleep(5)  # Check every 5 seconds
                        else:
                            # Cannot determine server status, may have stopped or other issues
                            print(f"Cannot determine server {port} status, continuing stop process")
                            break
                    
                    # Check if timed out
                    if time.time() - start_wait_time >= idle_timeout:
                        print(f"Timeout waiting for server {port} to be idle, forcing stop")
                
                # Try graceful stop (first send SIGTERM)
                os.kill(server_info.get_pid(), signal.SIGTERM)
                
                # Wait for process to exit
                wait_timeout = 5
                start_time = time.time()
                while time.time() - start_time < wait_timeout:
                    if not server_info.is_running():
                        break
                    time.sleep(0.1)
                
                # If process is still running, force terminate
                if server_info.is_running():
                    print(f"Force terminating server, port: {port}")
                    os.kill(server_info.get_pid(), signal.SIGKILL)
                
                # Remove from records
                del self._servers[port]
                
                # If exclusive GPU mode is enabled, restore GPUs to default mode
                if is_exclusive_mode and gpus_to_release:
                    set_gpu_default_mode(gpus_to_release)
                    
                print(f"Server stopped, port: {port}")
                return True
            except Exception as e:
                print(f"Failed to stop server, port: {port}, error: {e}")
                # Even if error occurs, try to restore GPU mode
                if is_exclusive_mode and gpus_to_release:
                    set_gpu_default_mode(gpus_to_release)
                return False
    
    def stop_all_servers(self) -> int:
        """
        Stop all servers
        
        Returns:
            int: Number of servers successfully stopped
        """
        with self._lock:
            ports = list(self._servers.keys())
            success_count = 0
            
            for port in ports:
                if self.stop_server(port):
                    success_count += 1
            
            print(f"Stopped {success_count}/{len(ports)} servers")
            return success_count
    
    def _cleanup_stopped_servers(self):
        """
        Clean up records of stopped servers
        """
        stopped_ports = [
            port for port, server in self._servers.items() 
            if not server.is_running()
        ]
        
        for port in stopped_ports:
            print(f"Cleaning up record of stopped server, port: {port}")
            del self._servers[port]
    
    def update_server(
        self,
        port: int,
        model_path: str,
        served_model_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        worker_multiproc_method: Optional[str] = None,
        use_modelscope: Optional[bool] = None,
        gpus: Optional[List[int]] = None,
        additional_args: Optional[List[str]] = None,
        wait_for_start: bool = True,
        wait_seconds: int = 10,
        wait_for_idle: bool = True,
        idle_timeout: int = 300,
        log_file: Optional[str] = None,
        exclusive_gpu_mode: bool = False
    ) -> VllmServerInfo:
        """
        Update vLLM server model on specified port
        
        Args:
            port: Server port
            model_path: New model path
            served_model_name: New served model name (uses model_path if None)
            tensor_parallel_size: New tensor parallel size (uses original value if None)
            max_model_len: New maximum model length (uses original value if None)
            worker_multiproc_method: New worker multiprocessing method (uses original value if None)
            use_modelscope: Whether to use ModelScope (uses original value if None)
            gpus: New list of visible GPUs (uses original value if None)
            additional_args: New list of additional command-line arguments (uses original value if None)
            wait_for_start: Whether to wait for server startup
            wait_seconds: Seconds to wait for server startup
            wait_for_idle: Whether to wait for server to be idle before stopping
            idle_timeout: Maximum time to wait for server to be idle (seconds)
            log_file: Log file path
            exclusive_gpu_mode: Whether to use exclusive GPU mode
            
        Returns:
            VllmServerInfo: Updated server information object
            
        Raises:
            ValueError: If no server exists on specified port
            RuntimeError: If the manager has been destroyed
        """
        with self._lock:
            # Check if the manager has been destroyed
            if self._destroyed:
                raise RuntimeError("Manager has been destroyed, cannot update server")
            
            # Get existing server information
            existing_server = self.get_server_info(port)
            if not existing_server:
                raise ValueError(f"No server found on port {port}")
            
            # Use new parameters or keep original ones
            if served_model_name is None:
                served_model_name = model_path
            if tensor_parallel_size is None:
                tensor_parallel_size = existing_server.tensor_parallel_size
            if max_model_len is None:
                max_model_len = existing_server.max_model_len
            if worker_multiproc_method is None:
                worker_multiproc_method = existing_server.worker_multiproc_method
            if use_modelscope is None:
                use_modelscope = existing_server.use_modelscope
            if gpus is None:
                gpus = existing_server.gpus
            if additional_args is None:
                additional_args = existing_server.additional_args
            
            print(f"Preparing to update server on port {port}, from model {existing_server.model_path} to {model_path}")
            
            # Stop existing server
            if not self.stop_server(port, wait_for_idle=wait_for_idle, idle_timeout=idle_timeout):
                raise RuntimeError(f"Failed to stop server on port {port}")
            
            # Ensure port is released (brief wait)
            print(f"Waiting for port {port} to be released...")
            # Check if port is available
            start_time = time.time()
            max_wait = 5  # Wait up to 5 seconds
            while time.time() - start_time < max_wait:
                # Check if any process is using the port
                try:
                    import socket
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        # Try to bind port, if successful, port is available
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind(("localhost", port))
                        print(f"Port {port} released and available")
                        break
                except socket.error:
                    # Port is occupied, continue waiting
                    time.sleep(0.5)
            else:
                print(f"Warning: Port {port} may not be fully released within {max_wait} seconds, but proceeding with startup")
            
            # Start new server on the same port
            print(f"Starting new server on port {port}, model: {model_path}")
            # Generate new default log filename if no new log file is specified
            if log_file is None:
                log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'vllm_logs')
                model_name = os.path.basename(model_path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"vllm_server_{model_name}_{port}_{timestamp}_updated.log")
                print(f"No log file specified when updating server, will use new default log file: {log_file}")
            
            new_server_info = self.add_server(
                model_path=model_path,
                port=port,
                served_model_name=served_model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                worker_multiproc_method=worker_multiproc_method,
                use_modelscope=use_modelscope,
                gpus=gpus,
                additional_args=additional_args,
                wait_for_start=False,  # Don't wait yet, we'll check ourselves
                log_file=log_file,
                exclusive_gpu_mode=exclusive_gpu_mode
            )
            
            # Wait for server to start and become ready
            if wait_for_start:
                print(f"Waiting for server to start and become ready, port: {port}...")
                start_time = time.time()
                ready = False
                
                # Loop to check if server is ready until timeout
                while time.time() - start_time < wait_seconds:
                    # First check if process is still running
                    if not new_server_info.is_running():
                        print(f"Warning: Server process exited, port: {port}")
                        # Try to get error output
                        stdout, stderr = new_server_info.process.communicate(timeout=1)
                        if stderr:
                            print(f"Error output:\n{stderr[:500]}..." if len(stderr) > 500 else f"Error output:\n{stderr}")
                        break
                    
                    # Check if server is ready to accept requests
                    if new_server_info.is_ready():
                        ready = True
                        print(f"Server is ready, port: {port}")
                        break
                    
                    # Sleep briefly before checking again
                    time.sleep(0.5)
                
                # If timeout and server is still not ready
                if not ready:
                    if new_server_info.is_running():
                        print(f"Warning: Server not fully ready within {wait_seconds} seconds, but process is still running, port: {port}")
                    else:
                        print(f"Warning: Server startup failed, port: {port}")
            
            return new_server_info
    
    def _find_available_gpus(self, required_count: int) -> Optional[List[int]]:
        """
        Find available GPUs
        
        Args:
            required_count: Number of GPUs needed
            
        Returns:
            Optional[List[int]]: List of available GPUs, None if not enough GPUs available
        """
        available_gpus = []
        
        # Collect all used GPUs
        used_gpus_set = set()
        for gpus in self._used_gpus.values():
            used_gpus_set.update(gpus)
        
        # Find unused GPUs
        for gpu in self._available_gpus:
            if gpu not in used_gpus_set:
                available_gpus.append(gpu)
                if len(available_gpus) == required_count:
                    break
        
        return available_gpus if len(available_gpus) >= required_count else None
    
    def _allocate_gpus(self, port: int, gpu_count: int) -> Optional[List[int]]:
        """
        Allocate GPUs for server on specified port
        
        Args:
            port: Server port
            gpu_count: Number of GPUs needed
            
        Returns:
            Optional[List[int]]: Allocated GPU list, None if allocation failed
        """
        gpus = self._find_available_gpus(gpu_count)
        if gpus:
            self._used_gpus[port] = gpus
            print(f"Allocated GPUs for port {port}: {gpus}")
        return gpus
    
    def _release_gpus(self, port: int):
        """
        Release GPUs allocated for specified port
        
        Args:
            port: Server port
        """
        if port in self._used_gpus:
            released_gpus = self._used_gpus.pop(port)
            print(f"Released GPUs for port {port}: {released_gpus}")
    
    def _process_tasks(self):
        """Process task queue"""
        while self._tasks_running:
            try:
                # Try to get task with 1 second timeout
                task = self._task_queue.get(timeout=1)
                
                # Try to allocate GPUs
                gpus = self._allocate_gpus(-1, task.gpu_count)  # Use temporary port -1 for allocation
                
                if gpus:
                    # Allocation successful, start executing task
                    task.status = "running"
                    task.start_time = time.time()  # Record task start time
                    
                    # Find available port
                    port = 8000
                    while port in self._servers:
                        port += 1
                    
                    # Update temporary allocation to actual port
                    if -1 in self._used_gpus:
                        self._used_gpus[port] = self._used_gpus.pop(-1)
                    
                    try:
                        # Start vLLM server
                        print(f"Starting vLLM server for task {task.task_id}, port: {port}, model: {task.model_path}, GPU: {gpus}")
                        
                        # Write port to params dictionary
                        task.params_dict['port'] = port
                        
                        # Start server
                        server_info = self.add_server(
                            model_path=task.model_path,
                            port=port,
                            served_model_name=task.model_path,
                            tensor_parallel_size=task.gpu_count,
                            max_model_len=self._max_model_len,
                            gpus=gpus,
                            wait_for_start=True,
                            wait_seconds=60000
                        )
                        
                        # Associate task with server
                        server_info.current_task = task
                        
                        # Execute task function asynchronously
                        def execute_task(task=task, server_info=server_info):
                            try:
                                result = task.func_handle(**task.params_dict)

                                # After restoring output, main program's print statements will display normally
                                task.result = result
                                task.status = "completed"
                                print(f"Task {task.task_id} completed")
                            except Exception as e:
                                # Exception information will also be displayed after restoring output
                                task.result = str(e)
                                task.status = "failed"
                                print(f"Task {task.task_id} failed: {e}")
                            finally:
                                # Stop server and release resources after execution
                                with self._lock:
                                    # Use port from server_info to ensure correct server operation
                                    server_port = server_info.port
                                    # Stop server
                                    self.stop_server(server_port, wait_for_idle=True)
                                    # Release GPUs
                                    self._release_gpus(server_port)
                                    # Save result
                                    self._task_results[task.task_id] = task.result
                                    # Set result event
                                    if task.task_id in self._result_events:
                                        self._result_events[task.task_id].set()
                                    # Mark task as done
                                    self._task_queue.task_done()
                        
                        # Create and start task thread
                        task_thread = threading.Thread(target=execute_task)
                        task_thread.daemon = True
                        task_thread.start()
                        
                    except Exception as e:
                        print(f"Failed to start server for task {task.task_id}: {e}")
                        task.result = str(e)
                        task.status = "failed"
                        self._release_gpus(port)
                        self._task_results[task.task_id] = task.result
                        if task.task_id in self._result_events:
                            self._result_events[task.task_id].set()
                        self._task_queue.task_done()
                else:
                    # No GPUs available, put task back into queue
                    print(f"Task {task.task_id} waiting for GPU resources...")
                    self._task_queue.put(task)
                    time.sleep(5)  # Wait 5 seconds before retry
                    
            except queue.Empty:
                # Queue is empty, continue looping
                pass
            except Exception as e:
                print(f"Error processing task: {e}")
    
    def _clear_progress_log(self):
        """Clear progress log file"""
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(self._progress_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # If file exists, delete it
            if os.path.exists(self._progress_log_file):
                os.remove(self._progress_log_file)
            
            # Create new log file and write header
            with open(self._progress_log_file, 'w', encoding='utf-8') as f:
                f.write("========== vLLM Server Progress Log ==========\n")
                f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            print(f"Error clearing progress log file: {e}")
    
    def _get_progress_info(self) -> str:
        """Get current progress information"""
        with self._lock:
            # Stat GPU usage
            total_gpus = len(self._available_gpus)
            used_gpus_count = sum(len(gpus) for gpus in self._used_gpus.values() if gpus)
            available_gpus_count = total_gpus - used_gpus_count
            
            # Stat task statuses
            pending_tasks = []
            running_tasks = []
            completed_tasks = []
            failed_tasks = []
            
            # Get statuses from all tasks list
            for task_id, task in self._all_tasks.items():
                if task.status == "pending":
                    pending_tasks.append(task)
                elif task.status == "running":
                    running_tasks.append(task)
                elif task.status == "completed":
                    completed_tasks.append(task)
                elif task.status == "failed":
                    failed_tasks.append(task)
            
            # Also need to consider tasks in queue
            queue_size = self._task_queue.qsize()
            
            # Build progress information
            progress_info = []
            progress_info.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Progress update")
            progress_info.append("-" * 50)
            
            # GPU usage
            progress_info.append(f"GPU usage:")
            progress_info.append(f"  Total: {total_gpus}")
            progress_info.append(f"  Used: {used_gpus_count}")
            progress_info.append(f"  Available: {available_gpus_count}")
            
            # Task statistics
            total_tasks = len(self._all_tasks)
            progress_info.append(f"\nTask statistics:")
            progress_info.append(f"  Total: {total_tasks}")
            progress_info.append(f"  Pending: {len(pending_tasks)}")
            progress_info.append(f"  Running: {len(running_tasks)}")
            progress_info.append(f"  Completed: {len(completed_tasks)}")
            progress_info.append(f"  Failed: {len(failed_tasks)}")
            
            # Remaining tasks count
            remaining_tasks = len(pending_tasks) + len(running_tasks)
            progress_info.append(f"  Remaining: {remaining_tasks}")
            
            # Completion percentage
            if total_tasks > 0:
                completion_percentage = (len(completed_tasks) / total_tasks) * 100
                progress_info.append(f"  Completion rate: {completion_percentage:.1f}%")
            
            # Details of running tasks
            if running_tasks:
                progress_info.append("\nRunning tasks:")
                for task in running_tasks:
                    runtime = time.time() - task.start_time
                    hours, remainder = divmod(runtime, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    progress_info.append(f"  - {task.task_id}")
                    progress_info.append(f"    Model: {task.model_path}")
                    progress_info.append(f"    GPU count: {task.gpu_count}")
                    progress_info.append(f"    Runtime: {runtime_str}")
            
            # Completed tasks
            if completed_tasks:
                progress_info.append("\nRecently completed tasks:")
                # Only show the last 5 completed tasks
                for task in completed_tasks[-5:]:
                    progress_info.append(f"  - {task.task_id} (success)")
            
            # Failed tasks
            if failed_tasks:
                progress_info.append("\nFailed tasks:")
                for task in failed_tasks:
                    progress_info.append(f"  - {task.task_id} (failed)")
                    # Only show part of the error information
                    error_info = str(task.result)[:100] + "..." if len(str(task.result)) > 100 else str(task.result)
                    progress_info.append(f"    Error: {error_info}")
            
            progress_info.append("=" * 50 + "\n")
            
            return "\n".join(progress_info)
    
    def _update_progress_log(self):
        """Update progress log"""
        try:
            progress_info = self._get_progress_info()
            # Write to file
            with open(self._progress_log_file, 'w', encoding='utf-8') as f:
                f.write(progress_info)
            # Print to console as well
            #print(progress_info)
        except Exception as e:
            print(f"Error updating progress log: {e}")
    
    def _progress_monitor(self):
        """Progress monitoring thread function"""
        while self._progress_running:
            try:
                self._update_progress_log()
                time.sleep(5)  # Update progress every 5 seconds
            except Exception as e:
                print(f"Progress monitoring thread error: {e}")
                time.sleep(1)
    
    def run_series_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a series of tasks, automatically allocating GPU resources
        
        Args:
            tasks: List of tasks, each task contains:
                - task_id: Task ID
                - model_path: Model path
                - params_dict: Parameter dictionary
                - func_handle: Function handle
                
        Returns:
            Dict[str, Any]: Execution results of all tasks
        """
        # Clear previous tasks and progress
        with self._lock:
            # Clear task-related data
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                    self._task_queue.task_done()
                except queue.Empty:
                    break
            
            self._all_tasks.clear()
            self._task_results.clear()
            self._result_events.clear()
            
            # Clear progress log
            self._clear_progress_log()
            
            # Start progress monitoring thread
            if not self._progress_running:
                self._progress_running = True
                self._progress_thread = threading.Thread(target=self._progress_monitor)
                self._progress_thread.daemon = True
                self._progress_thread.start()
            
            if self._destroyed:
                raise RuntimeError("Manager has been destroyed, cannot run tasks")
            
            # Calculate GPU allocation
            total_gpus = len(self._available_gpus)
            task_count = len(tasks)
            
            # Basic GPU allocation logic: evenly distribute, first task uses remaining GPUs
            gpu_allocation = []
            if task_count > 0:
                # GPU count for other tasks (floor division), minimum 1
                base_gpu_per_task = max(1, total_gpus // task_count)
                
                # Calculate additional GPUs for first task
                remaining_gpus = max(1, total_gpus - (base_gpu_per_task * (task_count - 1)))
                
                # First task uses remaining GPUs
                gpu_allocation.append(remaining_gpus)
                
                # Other tasks use base GPU count
                for i in range(1, task_count):
                    gpu_allocation.append(base_gpu_per_task)
            
            print(f"GPU allocation plan: total available GPUs={total_gpus}, total tasks={task_count}, GPU allocation per task={gpu_allocation}")
            
            # Validate task format
            for i, task_info in enumerate(tasks):
                required_fields = ['task_id', 'model_path', 'params_dict', 'func_handle']
                for field in required_fields:
                    if field not in task_info:
                        raise ValueError(f"Task missing required field: {field}")
            
            # Start task processing thread if not started
            if not self._tasks_running:
                self._tasks_running = True
                self._task_thread = threading.Thread(target=self._process_tasks)
                self._task_thread.daemon = True
                self._task_thread.start()
            
            # Create task objects and add to queue and task list
            task_ids = []
            for i, task_info in enumerate(tasks):
                # Use calculated GPU allocation
                task_gpu_count = gpu_allocation[i] if i < len(gpu_allocation) else 1
                
                task = Task(
                    task_id=task_info['task_id'],
                    model_path=task_info['model_path'],
                    params_dict=task_info['params_dict'],
                    func_handle=task_info['func_handle'],
                    gpu_count=task_gpu_count
                )
                print(f"Task {task.task_id} allocated {task.gpu_count} GPUs")
                self._task_queue.put(task)
                self._all_tasks[task.task_id] = task
                task_ids.append(task.task_id)
                # Create result event
                self._result_events[task.task_id] = threading.Event()
        
        print(f"Added {len(tasks)} tasks to queue")
        
        # Wait for all tasks to complete
        results = {}
        for task_id in task_ids:
            # Wait for task completion
            event = self._result_events[task_id]
            event.wait()  # Wait for task completion
            results[task_id] = self._task_results[task_id]
            # Clean up event
            del self._result_events[task_id]
        
        # Update final progress after all tasks are completed
        self._update_progress_log()
        
        return results
    
    def destroy(self):
        """
        Destroy the manager, stop all servers and clean up resources
        """
        with self._lock:
            if not self._destroyed:
                print("Destroying vLLM Server Manager, stopping all servers...")
                # Stop progress monitoring thread
                if self._progress_running:
                    self._progress_running = False
                    if self._progress_thread:
                        self._progress_thread.join(timeout=5)
                # Stop task processing
                self._tasks_running = False
                if self._task_thread:
                    self._task_thread.join(timeout=5)
                # Stop all servers
                self.stop_all_servers()
                # Clean up GPU usage records
                self._used_gpus.clear()
                
                # Clean up all log files
                for server_info in list(self._servers.values()):
                    # Note: Don't directly close log files here, as they are handled in subprocess.Popen
                    # When the process ends, related file handles will be automatically closed
                    pass
                    
                self._destroyed = True
                print("vLLM Server Manager destroyed")
                
                # Clear data structures
                self._servers.clear()
                self._task_results.clear()
                self._result_events.clear()
                self._all_tasks.clear()


if __name__ == "__main__":
    # Run example
    example_usage()