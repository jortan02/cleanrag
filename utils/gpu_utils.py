import torch
import subprocess
import platform

def get_cuda_info():
    """
    Get CUDA information including version and availability.
    Returns a dictionary with CUDA information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory": None,
        "system_info": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
        }
    }
    
    if info["cuda_available"]:
        # Get CUDA version from torch
        info["cuda_version"] = torch.version.cuda
        
        # Get GPU name
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Get GPU memory info
        try:
            info["gpu_memory"] = {
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # Convert to GB
                "free": torch.cuda.memory_reserved(0) / 1024**3,  # Convert to GB
            }
        except Exception:
            info["gpu_memory"] = None
            
        # Try to get more detailed CUDA info using nvidia-smi
        try:
            nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
            info["nvidia_smi_output"] = nvidia_smi
        except Exception:
            info["nvidia_smi_output"] = None
    
    return info

def format_gpu_info(info):
    """
    Format GPU information into a readable string.
    """
    if not info["cuda_available"]:
        return "CUDA is not available on this system."
    
    output = []
    output.append(f"CUDA Version: {info['cuda_version']}")
    output.append(f"GPU: {info['gpu_name']}")
    
    if info["gpu_memory"]:
        output.append(f"GPU Memory: {info['gpu_memory']['total']:.2f} GB total")
        output.append(f"GPU Memory Free: {info['gpu_memory']['free']:.2f} GB")
    
    return "\n".join(output) 