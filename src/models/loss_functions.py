import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    For each anchor i, the loss averages the log-probability of selecting any
    same-label sample j (j != i) among all non-self samples in the batch:

        L_i = - (1/|P(i)|) * sum_{j in P(i)}
              log( exp(sim(i,j)/tau) / sum_{a != i} exp(sim(i,a)/tau) )

    where P(i) = { j != i : y_j == y_i } and sim is cosine (if normalize=True).

    Args:
        tau (float): temperature scaling.
        normalize (bool): if True, L2-normalize embeddings before similarity.
        reduction (str): 'mean' | 'sum' | 'none' (averaging over valid anchors).
        ignore_index (int | None): label to ignore (anchors/positives with this label are skipped).

    Notes:
        - Requires at least two samples of the same label in a batch to form positives.
        - If an anchor has no positives in the batch, it is skipped in the reduction.
        - Uses O(B^2) memory/time due to pairwise similarities.

    Example:
        loss_fn = SupervisedContrastiveLoss(tau=0.07, normalize=True)
        z = encoder(x)                       # (B, d)
        y = labels.long()                    # (B,)
        loss = loss_fn(z, y)
        loss.backward()
    """
    def __init__(self, tau: float = 0.07, normalize: bool = True,
                 reduction: str = "mean", ignore_index: int | None = None):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'")
        self.tau = tau
        self.normalize = normalize
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be (B, d), got {tuple(z.shape)}")
        if y.ndim != 1 or y.shape[0] != z.shape[0]:
            raise ValueError("y must be (B,) and match z.shape[0]")
        if not torch.isfinite(z).all():
            raise ValueError("Input embeddings contain NaN/Inf")

        B = z.size(0)
        if B < 2:
            return z.new_zeros((), requires_grad=True)

        if self.normalize:
            z = F.normalize(z, dim=1, eps=1e-12)

        # Pairwise logits
        logits = (z @ z.t()) / self.tau  # (B, B)

        # Masks
        device = z.device
        eye = torch.eye(B, dtype=torch.bool, device=device)
        y_col = y.view(-1, 1)
        mask_pos = (y_col == y_col.t())
        mask_pos.fill_diagonal_(False)  # j != i

        if self.ignore_index is not None:
            valid = (y != self.ignore_index)
            # positives require both sides valid
            mask_pos &= valid.view(-1, 1) & valid.view(1, -1)
            # denominator: for anchor i, allow only columns j that are valid and j != i
            denom_mask = (~eye) & valid.view(1, -1)
        else:
            denom_mask = ~eye  # all non-self

        # If an anchor has no admissible denominator entries, skip it later
        has_den = denom_mask.any(dim=1)

        # Mask out disallowed columns in the denominator
        masked_logits = logits.masked_fill(~denom_mask, float("-inf"))

        # Row-wise logsumexp (finite if has_den[i] is True; -inf otherwise)
        log_den = torch.logsumexp(masked_logits, dim=1, keepdim=True)
        log_prob = masked_logits - log_den

        # For rows with no denominator, zero them out to avoid NaNs propagating
        if not has_den.all():
            log_prob = torch.where(has_den.view(-1, 1), log_prob, torch.zeros_like(log_prob))

        # Positives must be part of the denominator set too
        valid_pos_mask = mask_pos & denom_mask

        # Count positives per anchor and mark anchors that actually have positives
        pos_counts = valid_pos_mask.sum(dim=1)  # (B,)
        has_pos = pos_counts > 0

        # Sum log-probs over positives (avoid 0 * -inf → NaN)
        sum_pos = log_prob.masked_fill(~valid_pos_mask, 0.0).sum(dim=1)
        per_anchor = -(sum_pos / pos_counts.clamp_min(1))

        if self.reduction == "none":
            out = per_anchor
            # zero-out anchors without denom or positives
            out = out.masked_fill(~(has_den & has_pos), 0.0)
            return out

        valid_anchors = has_den & has_pos
        if not valid_anchors.any():
            return z.new_zeros((), requires_grad=True)

        if self.reduction == "mean":
            return per_anchor[valid_anchors].mean()
        else:
            return per_anchor[valid_anchors].sum()


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
    
