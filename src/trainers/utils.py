import torch
from collections import OrderedDict

def get_gpu_device():
    """
    Returns:
        - None if no GPU is available
        - torch.device object if only one GPU is available
        - dict[int, torch.device] if multiple GPUs are available
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return torch.device("cuda:0")
        else:
            return OrderedDict({i: torch.device(f"cuda:{i}") for i in range(num_gpus)})
    elif torch.backends.mps.is_available():
        # MPS backend (Apple Silicon) doesn't have multi-GPU concept
        return torch.device("mps")
    else:
        return None

def get_cpu_device():
    return torch.device("cpu")



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min', verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (str): 'min' to minimize metric (e.g., loss), 'max' to maximize (e.g., accuracy).
            verbose (bool): Whether to print early stopping messages.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.best_score = float('inf')
            self.monitor_condition = lambda metric, best: metric < best - self.min_delta
        else:  # mode == 'max'
            self.best_score = float('-inf')
            self.monitor_condition = lambda metric, best: metric > best + self.min_delta

        self.counter = 0
        self.early_stop = False

    def __call__(self, metric_value):
        if self.monitor_condition(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0  # Reset counter if there is improvement
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
