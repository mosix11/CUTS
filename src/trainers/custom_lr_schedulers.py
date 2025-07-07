from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings

# From https://arxiv.org/pdf/1912.02292        
class InverseSquareRootLR(_LRScheduler):
    """
    Implements the Inverse Square Root learning rate schedule by subclassing _LRScheduler.

    The learning rate for step t (where t = self.last_epoch) is calculated as:
        lr = initial_lr / sqrt(1 + floor(t / L))

    This scheduler is intended to be stepped after each optimizer update (gradient step).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        L (int): The frequency parameter from the formula. The learning rate value
                 changes every L steps. Must be a positive integer.
        last_epoch (int): The index of the last step. Used when resuming training.
                          Default: -1 (indicates the start).
    """
    def __init__(self, optimizer, L, last_epoch=-1):
        if not isinstance(L, int) or L <= 0:
            raise ValueError(f"L must be a positive integer, but got {L}")
        self.L = L
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using the inverse square root formula for the current step.
        This method is called by step() in the base class.
        """
        current_step = self.last_epoch # Corresponds to 't' in the formula
        factor = (1.0 + current_step // self.L) ** -0.5
        return [base_lr * factor for base_lr in self.base_lrs]
    
    
    
class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Scheduler that combines a linear warm-up phase with a cosine annealing decay phase.
    This implementation is inspired by the official PyTorch CosineAnnealingLR scheduler
    and uses a recursive formula for the cosine phase to be robust to external
    changes to the learning rate.

    Args:
        optimizer (Optimizer): The optimizer wrapped by the scheduler.
        warmup_steps (int): The number of steps for the linear warm-up phase.
        T_max (int): The total number of steps for the entire schedule. After this
                     many steps, the learning rate will reach its minimum value.
        eta_min (float, optional): The minimum learning rate. Defaults to 0.0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0.0, last_epoch=-1):
        if warmup_steps >= T_max:
            raise ValueError("T_max must be greater than warmup_steps.")
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.T_cosine = T_max - warmup_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculates the learning rate for the current step.
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        current_step = self.last_epoch

        # 1. Linear Warm-up Phase
        if current_step < self.warmup_steps:
            warmup_factor = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # 2. First step of Cosine Annealing
        # On the first step after warm-up, the LR should be the base_lr.
        # The recursive formula needs a starting point.
        elif current_step == self.warmup_steps:
            return self.base_lrs

        # 3. Recursive Cosine Annealing Phase
        else:
            # The step number within the cosine phase
            t_cosine = current_step - self.warmup_steps
            
            # This is the recursive formula from the official PyTorch implementation
            # It calculates the new LR based on the *previous* LR
            numerator = 1 + math.cos(math.pi * t_cosine / self.T_cosine)
            denominator = 1 + math.cos(math.pi * (t_cosine - 1) / self.T_cosine)
            
            # For each parameter group, calculate the new LR
            return [
                self.eta_min + (group['lr'] - self.eta_min) * numerator / denominator
                for group in self.optimizer.param_groups
            ]