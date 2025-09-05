import torch
import random, numpy as np
from collections import OrderedDict, deque
import pickle
import os

import torch.distributed as torch_distributed

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
        # apple silicon
        return torch.device("mps")
    else:
        return None

def get_cpu_device():
    return torch.device("cpu")

def setup_distributed():
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    if is_distributed:
        if not torch_distributed.is_initialized():
            torch_distributed.init_process_group(backend="nccl")
            print(f'DDP is initialized for rank {torch_distributed.get_rank()}')
            
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = torch_distributed.get_rank()
        torch.cuda.set_device(local_rank)
        return {
            'rank': rank,
            'local_rank': local_rank
        }
        
    else:
        return {
            'rank': 0,
            'local_rank': 0
        }
    

def seed_everything(base_seed: int, rank: int = 0) -> dict:
    """Seed python, numpy, torch CPU+CUDA."""
    seed = base_seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

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



class LossRankedSampleStorage:
    
    def __init__(self):
        pass
    
    def activate_low_loss_samples_buffer(self, consistency_window: int = 5, consistency_threshold: float = 0.8):
        if not hasattr(self, 'train_dataloader'):
            raise RuntimeError("Please call `setup_data_loaders` before activating the buffers.")
        
        self.accumulate_low_loss = True
        self.low_loss_consistency_window = consistency_window
        self.low_loss_consistency_threshold = consistency_threshold
        
        # Set up dynamic percentage targets
        self.low_loss_percentages = [p / 100.0 for p in range(5, 100, 5)] # 0.05, 0.10, ... 0.95
        self.current_low_loss_perc_index = 0
        
        available_classes = self.dataset.get_available_classes()
        self.num_classes = len(available_classes)
        self.num_train_samples = len(self.train_dataloader.dataset)
        print(len(self.train_dataloader.dataset))
        
        self.low_loss_sample_indices = {i: set() for i in available_classes}
        self.low_loss_history = {i: deque(maxlen=self.low_loss_consistency_window) for i in range(self.num_train_samples)}
        print("Low-loss sample buffer activated. Will save indices at 5% increments.")

    def activate_high_loss_samples_buffer(self, consistency_window: int = 5, consistency_threshold: float = 0.8):
        if not hasattr(self, 'train_dataloader'):
            raise RuntimeError("Please call `setup_data_loaders` before activating the buffers.")
            
        self.accumulate_high_loss = True
        self.high_loss_consistency_window = consistency_window
        self.high_loss_consistency_threshold = consistency_threshold

        # Set up dynamic percentage targets
        self.high_loss_percentages = [p / 100.0 for p in range(5, 100, 5)]
        self.current_high_loss_perc_index = 0
        
        available_classes = self.dataset.get_available_classes()
        self.num_classes = len(available_classes)
        self.num_train_samples = len(self.train_dataloader.dataset)
        
        self.high_loss_sample_indices = {i: set() for i in available_classes}
        self.high_loss_history = {i: deque(maxlen=self.high_loss_consistency_window) for i in range(self.num_train_samples)}
        print("High-loss sample buffer activated. Will save indices at 5% increments.")

    def check_and_save_low_loss_buffers(self):
        # Loop as long as we might be able to save the next tier in the same epoch
        while self.current_low_loss_perc_index < len(self.low_loss_percentages):
            current_target_perc = self.low_loss_percentages[self.current_low_loss_perc_index]
            target_size_per_class = int((self.num_train_samples * current_target_perc) / self.num_classes)

            # Check if the current target is met
            all_classes_met = True
            for class_idx in self.low_loss_sample_indices:
                if len(self.low_loss_sample_indices[class_idx]) < target_size_per_class:
                    all_classes_met = False
                    break
            
            if all_classes_met:
                # If met, save the indices for this percentage and move to the next target
                self.save_low_loss_indices(current_target_perc)
                self.current_low_loss_perc_index += 1
            else:
                # If not met, stop checking for this epoch
                break
        
        # Deactivate if all targets are completed
        if self.current_low_loss_perc_index >= len(self.low_loss_percentages):
            print("All low-loss percentage targets have been met and saved.")
            self.accumulate_low_loss = False

    def check_and_save_high_loss_buffers(self):
        while self.current_high_loss_perc_index < len(self.high_loss_percentages):
            current_target_perc = self.high_loss_percentages[self.current_high_loss_perc_index]
            target_size_per_class = int((self.num_train_samples * current_target_perc) / self.num_classes)
            
            all_classes_met = True
            for class_idx in self.high_loss_sample_indices:
                if len(self.high_loss_sample_indices[class_idx]) < target_size_per_class:
                    all_classes_met = False
                    break
            
            if all_classes_met:
                self.save_high_loss_indices(current_target_perc)
                self.current_high_loss_perc_index += 1
            else:
                break
                
        if self.current_high_loss_perc_index >= len(self.high_loss_percentages):
            print("All high-loss percentage targets have been met and saved.")
            self.accumulate_high_loss = False

    def save_low_loss_indices(self, percentage: float):
        if not hasattr(self, 'low_loss_sample_indices'):
            print("Low-loss buffer was not activated. Nothing to save.")
            return
        
        output_path = self.log_dir / f'low_loss_indices_{percentage:.2f}.pkl'
        
        # Calculate the exact number of samples needed per class and slice the list.
        target_size_per_class = int((self.num_train_samples * percentage) / self.num_classes)

        indices_to_save = {
            class_idx: sorted(list(idx_set))[:target_size_per_class] # Slice the list here
            for class_idx, idx_set in self.low_loss_sample_indices.items()
        }

        with open(output_path, 'wb') as f:
            pickle.dump(indices_to_save, f)
            
        total_saved = sum(len(v) for v in indices_to_save.values())
        # Improved percentage formatting in the print statement
        # print(f"✅ Saved low-loss indices for {total_saved} samples ({percentage:.2%}) to: {output_path}")


    def save_high_loss_indices(self, percentage: float):
        if not hasattr(self, 'high_loss_sample_indices'):
            print("High-loss buffer was not activated. Nothing to save.")
            return

        output_path = self.log_dir / f'high_loss_indices_{percentage:.2f}.pkl'

        # Calculate the exact number of samples needed per class and slice the list.
        target_size_per_class = int((self.num_train_samples * percentage) / self.num_classes)
        
        indices_to_save = {
            class_idx: sorted(list(idx_set))[:target_size_per_class] # Slice the list here
            for class_idx, idx_set in self.high_loss_sample_indices.items()
        }

        with open(output_path, 'wb') as f:
            pickle.dump(indices_to_save, f)

        total_saved = sum(len(v) for v in indices_to_save.values())
        # Improved percentage formatting in the print statement
        # print(f"✅ Saved high-loss indices for {total_saved} samples ({percentage:.2%}) to: {output_path}")

