"""
    Code from https://github.com/if-loops/towards_poison_unlearning/blob/main/src/utils.py
    with minor changes.
"""

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions and classes for the unlearning framework.

Adapted from "Corrective Unlearning" and "Potion:Towards Poison Unlearning"
GitHub repos. https://github.com/drimpossible/corrective-unlearning-bench
https://github.com/if-loops/towards_poison_unlearning
"""

import copy
import functools
import os
import pickle
import random
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import sampler
import tqdm

Sampler = sampler.Sampler
DataLoader = data.DataLoader
_LRScheduler = lr_scheduler.LRScheduler
partial = functools.partial


def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch

class ParameterPerturber:
  """Perturbs model parameters based on calculated importances.

  This class implements methods for calculating parameter importances from
  datasets and modifying model weights based on these importances, following
  principles from SSD (Selective Synaptic Dampening) and its adaptive and
  label-free variants.
  """

  def __init__(
      self,
      model,
      opt,
      device_="cuda" if torch.cuda.is_available() else "mps",
      parameters=None,
      adaptive=False,
      label_free=False,
      x_d=False,
  ):
    self.model = model.to(device_)
    self.opt = opt
    self.alpha = parameters["selection_weighting"]
    self.xmin = None
    self.device = device_
    self.adaptive = adaptive
    self.label_free = label_free
    self.x_d = x_d
    self.lower_bound = 1
    self.exponent = 1
    self.magnitude_diff = None  # unused
    self.min_layer = -1
    self.max_layer = -1
    self.forget_threshold = 1  # unused
    self.dampening_constant = "Adaptive"  # parameters["dampening_constant"]
    self.selection_weighting = "Adaptive"  # parameters["selection_weighting"]

  def get_layer_num(self, layer_name: str) -> int:
    layer_id = layer_name.split(".")[1]
    if layer_id.isnumeric():
      return int(layer_id)
    else:
      return -1

  def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
    """Taken from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Returns a dict like
    named_parameters(), with zeroed-out parameter valuse

    Args:
      model (torch.nn): model to get param dict from

    Returns:
      dict(str,torch.Tensor): dict of zero-like params
    """
    return dict([
        (k, torch.zeros_like(p, device=p.device))
        for k, p in model.named_parameters()
    ])

  def calc_importance(
      self, dataloader: DataLoader, extra_noise=False
  ) -> Dict[str, torch.Tensor]:
    """Adapated from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Calculate per-parameter, importance

        returns a dictionary [param_name: list(importance per parameter)]
    Args:
      dataloader (DataLoader): DataLoader to be iterated over.
      extra_noise (bool): Whether to add extra noise to the input.

    Returns:
      importances (dict(str, torch.Tensor([]))): named_params-like dict
        containing list of importances for each parameter.
    """
    criterion = nn.CrossEntropyLoss()
    importances = self.zerolike_params_dict(self.model)
    for batch in dataloader:
      batch = prepare_batch(batch, self.device)
      x, y = batch[0], batch[1]
      if extra_noise:
        print("NOISE IS ON")
        # add torch rand x% noise
        x += torch.randn_like(x) * 0.01
      self.opt.zero_grad()
      out = self.model(x)
      if self.label_free:
        if self.x_d:
          # loss = torch.abs(out).sum(dim=1).mean() # L1
          # loss = torch.norm(out, p="fro", dim=1).abs().mean() # prev
          loss = torch.norm(out, p="fro", dim=1).abs().mean()
        else:
          # loss = torch.norm(out, p="fro", dim=1).abs().mean()
          loss = torch.norm(out, p="fro", dim=1).pow(2).mean()  # actual one
          # loss = (torch.norm(out, p="fro", dim=1).pow(2).mean())
          # # original SSD is pow(2) not abs -> l2 vs l1 norm
      else:
        loss = criterion(out, y)
      loss.backward()
      for (_, p), (_, imp) in zip(
          self.model.named_parameters(), importances.items()
      ):
        if p.grad is not None:
          if self.label_free:
            if self.x_d:
              # print("Using x_d")
              imp.data += p.grad.data.clone().abs()
              # imp.data += (p.grad.data.clone().abs()) # prev
            else:  # original is abs
              # print("Using original LF")
              imp.data += p.grad.data.clone().abs()
          else:
            # print("Using original SSD")
            imp.data += p.grad.data.clone().pow(2)
    # average over mini batch length
    for _, imp in importances.items():
      imp.data /= float(len(dataloader))
    return importances

  def modify_weight(
      self,
      original_importance: Dict[str, torch.Tensor],
      forget_importance: Dict[str, torch.Tensor],
      percentile_val="PERCENTILE NOT AUTOMATICALLY SET BUT ADAPTIVE SELECTED - ERROR",
      x_d=False,
  ) -> None:
    """Perturb weights based on the SSD equations given in the paper.

    Args:
      original_importance (Dict[str, torch.Tensor]): dict of importances for
        original dataset.
      forget_importance (Dict[str, torch.Tensor]): dictionary of importances for
        forget sample.
      percentile_val (str): The percentile value used for adaptive selection.
      x_d (bool): Whether to use the x_d variant of the loss.

    Returns:
      None
    """
    self.x_d = x_d
    if self.adaptive:  # adaptive SSD
      rel_list = list()
      # Get the indices of the fully connected layers
      fully_connected_layer_indices = list()
      for idx, layer in enumerate(self.model.children()):
        if isinstance(layer, nn.Linear):
          fully_connected_layer_indices.append(idx + 1)
      all_relative_values = []
      with torch.no_grad():
        for (p_name, p), (_, oimp), (_, fimp) in zip(
            self.model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
          layer_size_cutoff = 0  # overrride to do all layers

          if p.ndim == 0:
            continue
          
          if p.shape[0] >= layer_size_cutoff:  # only look at large layers
            divs_ = fimp.div(oimp)
            # select only the non nan values of divs_ to avoid errors
            divs_ = divs_[~torch.isnan(divs_)]
            # remove inf
            divs_ = divs_[~torch.isinf(divs_)]
            all_relative_values.append(divs_.reshape(-1).cpu().numpy())

      all_relative_values = np.concatenate(
          all_relative_values
      )  # flatten the array
      # PERCENTILE = NONE
      # print("USED Percentile: ", percentile_val)
      print("percentile:", percentile_val)
      print(all_relative_values)
      percentile = np.nanpercentile(all_relative_values, percentile_val)
      # print("USED cutoff value: ", percentile)
      percentile = percentile.item()
      print("percentile value:", percentile)
      # Main part after finding cutoff value
      with torch.no_grad():
        for (_, p), (_, oimp), (_, fimp) in zip(
            self.model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
          if p.ndim == 0:
            continue
          # divs_ = fimp.div(oimp)
          # select only the non nan values of divs_
          # divs_ = divs_[~torch.isnan(divs_)]
          # print(fimp, "XXX", oimp)
          # relative = torch.mean(divs_)
          # rel_std = torch.std(divs_)
          # rel_median = torch.median(divs_)
          # calculate absolute difference between median and mean
          # abs_diff = torch.abs(rel_median - relative)
          # Always adaptive
          self.selection_weighting = percentile
          self.dampening_constant = 1  # constant from ASSD paper
          rel_list.append(self.selection_weighting)
          # Synapse Selection with parameter alpha
          oimp_norm = oimp.mul(self.selection_weighting)
          locations = torch.where(fimp > oimp_norm)
          # Synapse Dampening with parameter lambda
          weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
              self.exponent
          )
          update = weight[locations]
          # Bound by 1 to prevent parameter values to increase.
          min_locs = torch.where(update > self.lower_bound)
          # for update take the update value where update > 0.1,
          # otherwise set update 0.1
          # We do not use this in the paper but this can be used
          # to avoid dead nerons for extra robustness
          # if x_d:
          #    dampen_limit = 0.01
          # else:
          #    dampen_limit = 0
          dampen_limit = 0
          update[update < dampen_limit] = dampen_limit
          update[min_locs] = self.lower_bound
          p[locations] = p[locations].mul(update)




# placeholder for now reusing assd
def alfssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device_,
    frac_dl,
    filtered_loader,
    x_d=False,
    optimizer=None,
    original_importances=None,
    sample_importances=None,
):
  """Performs Adaptive Label-Free SSD (ALFSSD) tuning on the model.

  This function calculates importances for a filtered dataset and the forget set
  using a label-free approach, and then adaptively modifies the model weights
  based on these importances. It supports an option for an 'x_d' variant.
  Pre-calculated importances can be provided to speed up the process.

  Args:
    model: The PyTorch model to be tuned.
    forget_train_dl: DataLoader for the forget set.
    dampening_constant: Unused in ALFSSD, as it's adaptive.
    selection_weighting: Unused in ALFSSD, as it's adaptive.
    full_train_dl: Unused in this function.
    device_: The device to run the computations on ('cuda' or 'mps').
    frac_dl: The fraction of the dataset used for adaptive percentile calc.
    filtered_loader: DataLoader for the filtered dataset used to calculate
      original importances.
    x_d (bool): Whether to use the x_d variant of the loss.
    optimizer: Optional optimizer. If None, a new SGD optimizer is created.
    original_importances: Optional pre-calculated importances for the original
      dataset.
    sample_importances: Optional pre-calculated importances for the forget set.

  Returns:
    A tuple containing:
      - model: The modified PyTorch model.
      - original_importances: The calculated or provided original importances.
      - sample_importances: The calculated or provided sample importances.
  """
  parameters = {
      "lower_bound": 1,
      "exponent": 1,
      "magnitude_diff": None,
      "min_layer": -1,
      "max_layer": -1,
      "forget_threshold": 1,
      "dampening_constant": None,  # adaptive
      "selection_weighting": None,
  }
  _ = (dampening_constant,)  # unused in alfssd
  _ = selection_weighting
  _ = full_train_dl
  # Sweep loop (ASSD extension)
  if x_d:
    sweeps_n = 1
    frac_dl = frac_dl * (1 / sweeps_n)
  else:
    sweeps_n = 1  # i.e. original
    frac_dl = frac_dl * (1 / sweeps_n)
  for sweep_i in range(sweeps_n):
    sweep_i += 1
    print("Sweep #", sweep_i)
    # load the trained model
    if optimizer is None:
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ParameterPerturber(
        model,
        optimizer,
        device_,
        parameters,
        adaptive=True,
        label_free=True,
        x_d=x_d,
    )
    model = model.eval()
    if original_importances:
      pass
    else:
      # ----
      # Calculate the importances of D (see paper)
      original_importances = pdr.calc_importance(
          filtered_loader, extra_noise=False
      )
      # safe the importances locally for reuse
      # with open("original_importances.pkl", "wb") as f:
      #    pickle.dump(original_importances, f)
      # Calculation of the forget set importances
      sample_importances = pdr.calc_importance(
          forget_train_dl, extra_noise=False
      )
    if x_d:
      share_off = np.log(1 + frac_dl * 100)  # orig 100
      percentile = 100 - share_off
    else:  # original
      share_off = np.log(1 + frac_dl * 100)
      percentile = 100 - share_off
    print("###### ----- Length based percentile: ", percentile)
    # Dampen selected parameters
    _ = pdr.modify_weight(
        original_importances,
        sample_importances,
        percentile_val=percentile,
        x_d=x_d,
    )
  return model, original_importances, sample_importances


class LinearLR(_LRScheduler):
  r"""Sets the learning rate of each parameter group with a linear schedule.

  The schedule follows :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0`
  is the initial lr, :math:`t` is the current epoch or iteration (zero-based)
  and :math:`T` is the total training epochs or iterations. It is recommended to
  use the iteration based calculation if the total number of epochs is small.
  When last_epoch=-1, sets initial lr as lr. This scheduler is studied in:
  `Budgeted Training: Rethinking Deep Neural Network Training Under Resource

   Constraints`_.
  .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
  Resource Constraints:
      https://arxiv.org/abs/1905.04753
  """

  def __init__(self, optimizer, t, warmup_epochs=100, last_epoch=-1):
    """Initializes the LinearLR scheduler.

    Args:
      optimizer (Optimizer): Wrapped optimizer.
      t (int): Total number of training epochs or iterations.
      warmup_epochs (int): Number of epochs for the linear warmup phase.
      last_epoch (int): The index of last epoch or iteration. Default: -1.
    """
    self.t = float(t)
    self.warm_ep = warmup_epochs
    super(LinearLR, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    if self.last_epoch - self.warm_ep >= 0:
      rate = 1 - ((self.last_epoch - self.warm_ep) / self.t)
    else:
      rate = (self.last_epoch + 1) / (self.warm_ep + 1)
    return [rate * base_lr for base_lr in self.base_lrs]

  def _get_closed_form_lr(self):
    return self.get_lr()







def unlearn_func(
    model,
    method,
    factor=0.1,
    device_="cuda" if torch.cuda.is_available() else "mps",
):
  """Applies unlearning to the model based on the specified method.

  This function creates a copy of the model and modifies its weights based on
  the chosen unlearning method ("EU" or "Mixed").

  Args:
    model: The PyTorch model to be unlearned.
    method (str): The unlearning method to apply. Can be "EU" (Erasure
      Unlearning) or "Mixed".
    factor (float): The scaling factor used in the "Mixed" method.
    device_ (str): The device to move the model to after modification. Defaults
      to "cuda" if available, otherwise "mps".

  Returns:
    torch.nn.Module: The modified PyTorch model.
  """
  model = copy.deepcopy(model)
  model = model.cpu()
  if method == "EU":
    model.apply(initialize_weights)
  elif method == "Mixed":
    partialfunc = partial(modify_weights, factor=factor)
    model.apply(partialfunc)
  else:
    pass
  model.to(device_)
  return model


def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d):
    m.reset_parameters()
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
    m.reset_parameters()
  elif isinstance(m, torch.nn.Linear):
    m.reset_parameters()
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)


def modify_weights(m, factor=0.1):
  """Modifies the weights and biases of a given module by a scaling factor.

  This function is used within the "Mixed" unlearning method to scale down the
  parameters of Conv2d, BatchNorm2d, and Linear layers.

  Args:
    m: The torch.nn.Module to modify.
    factor (float): The scaling factor to apply to the weights and biases.
  """
  if isinstance(m, torch.nn.Conv2d):
    m.weight.data = m.weight.data * factor
    if m.bias is not None:
      m.bias.data = m.bias.data * factor
  elif isinstance(m, torch.nn.BatchNorm2d):
    if m.affine:
      m.weight.data = m.weight.data * factor
      m.bias.data = m.bias.data * factor
  elif isinstance(m, torch.nn.Linear):
    m.weight.data = m.weight.data * factor
    if m.bias is not None:
      m.bias.data = m.bias.data * factor


def distill_kl_loss(y_s, y_t, t, reduction="sum"):
  p_s = torch.nn.functional.log_softmax(y_s / t, dim=1)
  p_t = torch.nn.functional.softmax(y_t / t, dim=1)
  loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
  if reduction == "none":
    loss = torch.sum(loss, dim=1)
  loss = loss * (t**2) / y_s.shape[0]
  return loss


def compute_accuracy(preds, y):
  return np.equal(np.argmax(preds, axis=1), y).mean()


class SubsetSequentialSampler(Sampler):

  def __init__(self, indices):
    self.indices = indices

  def __iter__(self):
    return (self.indices[i] for i in range(len(self.indices)))

  def __len__(self):
    return len(self.indices)