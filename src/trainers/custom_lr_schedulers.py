from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings
import numpy as np

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
            
            
            

def cosine_warmup_lr(epoch, base_lr, warmup_epochs, total_epochs):
    """
    Compute learning rate with a linear warmup followed by cosine decay.

    Args:
        epoch (int): Current epoch.
        base_lr (float): Base learning rate.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of epochs.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * base_lr * (1 + np.cos(np.pi * progress))


def exponential_warmup_lr(epoch, base_lr, warmup_epochs):
    """
    Compute learning rate with exponential warmup.

    Args:
        epoch (int): Current epoch.
        base_lr (float): Base learning rate.
        warmup_epochs (int): Number of warmup epochs.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch < warmup_epochs:
        return base_lr * (np.exp(epoch / warmup_epochs) - 1) / (np.e - 1)
    return base_lr


def linear_warmup_lr(epoch, base_lr, warmup_epochs):
    """
    Compute learning rate with linear warmup.

    Args:
        epoch (int): Current epoch.
        base_lr (float): Base learning rate.
        warmup_epochs (int): Number of warmup epochs.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

class CustomWarmupLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        base_scheduler,
        warmup_strategy="lin",
        warmup_epochs=10,
        total_epochs=100,
        last_epoch=-1,
    ):
        """
        Custom learning rate scheduler with warm-up strategies and integration with a PyTorch scheduler.

        Args:
            optimizer (Optimizer): Optimizer to be wrapped.
            base_scheduler (_LRScheduler): Base PyTorch scheduler (e.g., StepLR, CosineAnnealingLR).
            warmup_strategy (str): Warm-up strategy to use (`lin`, `exp`, `cos`). Defaults to `lin`.
            warmup_epochs (int): Number of warmup epochs. Defaults to 10.
            total_epochs (int): Total number of epochs. Defaults to 100.
            last_epoch (int): The index of the last epoch. Defaults to -1.
        """
        self.base_scheduler = base_scheduler
        self.warmup_strategy = warmup_strategy
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(CustomWarmupLRScheduler, self).__init__(optimizer, last_epoch)

    def _get_warmup_lr(self, epoch, base_lr):
        if self.warmup_strategy == "cos":
            return cosine_warmup_lr(
                epoch, base_lr, self.warmup_epochs, self.total_epochs
            )
        elif self.warmup_strategy == "exp":
            return exponential_warmup_lr(epoch, base_lr, self.warmup_epochs)
        else:  # Default to linear warm-up
            return linear_warmup_lr(epoch, base_lr, self.warmup_epochs)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch < self.warmup_epochs:
            return [self._get_warmup_lr(epoch, base_lr) for base_lr in self.base_lrs]
        else:
            # Step into the base scheduler after warmup
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        super(CustomWarmupLRScheduler, self).step(epoch)
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step(
                epoch - self.warmup_epochs if epoch is not None else None
            )
