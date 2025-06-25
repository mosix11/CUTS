import torch
import torch.nn as nn
import torch.nn.functional as F

class CompoundLoss(nn.Module):
    """
    A compound loss function combining CrossEntropyLoss for multi-class classification
    and BCEWithLogitsLoss for binary classification (e.g., noisy sample detection).

    Attributes:
        ce_loss (nn.CrossEntropyLoss): The CrossEntropyLoss instance.
        binary_loss (nn.BCEWithLogitsLoss): The BCEWithLogitsLoss instance.
        ce_weight (float): Coefficient for the CrossEntropyLoss.
        binary_weight (float): Coefficient for the BCEWithLogitsLoss.
    """
    def __init__(self, ce_weight: float = 1.0, binary_weight: float = 1.0):
        """
        Initializes the CompoundLoss.

        Args:
            ce_weight (float): The weight/coefficient to apply to the CrossEntropyLoss.
                               Defaults to 1.0.
            binary_weight (float): The weight/coefficient to apply to the BCEWithLogitsLoss.
                                   Defaults to 1.0.
        """
        super().__init__()
        # Initialize CrossEntropyLoss for multi-class classification
        # For CE loss, targets are class indices (long type), outputs are raw logits
        self.reduction = 'mean'
        self.ce_loss = nn.CrossEntropyLoss()

        # Initialize BCEWithLogitsLoss for binary classification
        # For BCEWithLogitsLoss, targets are float (0.0 or 1.0), outputs are raw logits
        self.binary_loss = nn.BCEWithLogitsLoss()

        # Store the coefficients for each loss component
        if not isinstance(ce_weight, (int, float)) or ce_weight < 0:
            raise ValueError("ce_weight must be a non-negative float or int.")
        if not isinstance(binary_weight, (int, float)) or binary_weight < 0:
            raise ValueError("binary_weight must be a non-negative float or int.")

        self.ce_weight = ce_weight
        self.binary_weight = binary_weight

    def forward(self, ce_output: torch.Tensor, ce_target: torch.Tensor,
                binary_output: torch.Tensor, binary_target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted sum of CrossEntropyLoss and BCEWithLogitsLoss.

        Args:
            ce_output (torch.Tensor): The raw logits from the classification model
                                      (before softmax/sigmoid) for the CrossEntropyLoss.
                                      Shape: (batch_size, num_classes).
            ce_target (torch.Tensor): The ground truth class labels for CrossEntropyLoss.
                                      Should be of type torch.long.
                                      Shape: (batch_size,).
            binary_output (torch.Tensor): The raw logits from the binary classification model
                                          (before sigmoid) for the BCEWithLogitsLoss.
                                          Shape: (batch_size, 1) or (batch_size,).
            binary_target (torch.Tensor): The ground truth binary labels for
                                          BCEWithLogitsLoss (0 or 1).
                                          Should be of type torch.float.
                                          Shape: (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: The scalar compound loss value.
        """
        # Calculate the CrossEntropyLoss
        ce_calculated_loss = self.ce_loss(ce_output, ce_target)

        # Calculate the BCEWithLogitsLoss
        # Ensure binary_target has the same shape as binary_output
        binary_target = binary_target.float() # BCEWithLogitsLoss expects float targets
        binary_calculated_loss = self.binary_loss(binary_output.squeeze(), binary_target.squeeze())

        # Return the weighted sum of the two losses
        compound_loss = (self.ce_weight * ce_calculated_loss +
                         self.binary_weight * binary_calculated_loss)
        return compound_loss
    
