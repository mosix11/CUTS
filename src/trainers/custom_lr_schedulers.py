from torch.optim.lr_scheduler import _LRScheduler
import math

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