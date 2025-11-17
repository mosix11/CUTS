"""
    Code from https://github.com/if-loops/towards_poison_unlearning/blob/main/src/methods.py
    with some cleaning up and changes.
"""

import torch, torchmetrics, tqdm, copy, time
from .potion_utils import (
    LinearLR,
    unlearn_func,
    distill_kl_loss,
    compute_accuracy,
    alfssd_tuning,
)
from torch.amp import autocast, GradScaler
import numpy as np
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from torch.utils.data import DataLoader
import os

from src.utils import nn_utils, misc_utils


ITERATIVE_SEARCH = True
STEP_MULT = 1.1
MAX_TRY = 500  # easily enough
MIN_ACC = "OVERRIDEN"

# Specify the file names for the importances
file_name_1 = "original_importances.pkl"
file_name_2 = "sample_importances.pkl"


# FYI: We left some additional experiment code chunks in the code for people to potentially build upon/experiment with


def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def evaluate_model(
    model:torch.nn.Module,
    dataloader:DataLoader,
    device:torch.device
):
    """
    Evaluates the given model on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: A tuple containing (all_predictions, all_targets, metrics_dict).
    """
    loss_met = misc_utils.AverageMeter()
    model.reset_metrics()
    all_preds = []
    all_targets = []
    
    if dataloader == None:
        return None, None, None
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            input_batch, target_batch = batch[:2]
            
            loss, preds = model.validation_step(input_batch, target_batch, use_amp=True, return_preds=True)
            if model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            predictions = torch.argmax(preds, dim=-1)
            all_preds.extend(predictions.cpu())
            all_targets.extend(target_batch.cpu())
            
            
    metric_results = model.compute_metrics()
    metric_results['Loss'] = loss_met.avg
    model.reset_metrics()
    
    return metric_results, torch.tensor(all_preds), torch.tensor(all_targets) 

class Naive:
    def __init__(self, opt, model, prenet=None, device='cpu'):
        self.device = device
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {"train_top1": [], "val_top1": [], "train_time_taken": 0}
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.opt.max_lr,
            momentum=0.9,
            weight_decay=self.opt.wd,
        )
        self.scheduler = LinearLR(
            self.optimizer,
            T=self.opt.train_iters * 1.25,
            warmup_epochs=self.opt.train_iters // 100,
        )  # Spend 1% time in warmup, and stop 66% of the way through training
        # self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        self.top1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.opt.num_classes
        ).to(
            self.device
        )  # MAC
        self.scaler = GradScaler()

    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        self.model.to(self.device)
        

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss

    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for images, target, infgt in tqdm.tqdm(loader):
            # images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            images, target, infgt = (
                images.to(self.device),
                target.to(self.device),
                infgt.to(self.device),
            )
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files["train_top1"].append(top1)
        print(f"Step: {self.curr_step} Train Top1: {top1:.3f}")
        return

    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for images, target in tqdm.tqdm(loader):
                with autocast():
                    # images, target = images.cuda(), target.cuda()
                    images, target = images.to(self.device), target.to(self.device)  # MAC
                    output = (
                        self.model(images)
                        if self.prenet is None
                        else self.model(self.prenet(images))
                    )
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds:
            print(f"Step: {self.curr_step} Val Top1: {top1*100:.2f}")

        if save_model:
            self.save_files["val_top1"].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files["train_time_taken"] += time.process_time() - time_start
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        if self.opt.unlearn_method != "Naive":
            self.unlearn_file_prefix = (
                self.opt.pretrain_file_prefix
                + "/"
                + str(self.opt.deletion_size)
                + "_"
                + self.opt.unlearn_method
                + "_"
                + self.opt.exp_name
            )
        return

    def compute_and_save_results(
        self,
        train_test_loader,
        test_loader,
        adversarial_train_loader,
        adversarial_test_loader,
    ):
        print("==> Compute And Save Results In Progress")

        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        torch.save(
            self.best_model.state_dict(), self.unlearn_file_prefix + "/model.pth"
        )
        np.save(
            self.unlearn_file_prefix + "/train_top1.npy", self.save_files["train_top1"]
        )
        np.save(self.unlearn_file_prefix + "/val_top1.npy", self.save_files["val_top1"])
        np.save(
            self.unlearn_file_prefix + "/unlearn_time.npy",
            self.save_files["train_time_taken"],
        )
        # self.model = self.best_model.cuda()
        self.model = self.best_model.to(self.device)  # MAC

        print(
            "==> Completed! Unlearning Time: [{0:.3f}]\t".format(
                self.save_files["train_time_taken"]
            )
        )

        for loader, name in [
            (train_test_loader, "train"),
            (test_loader, "test"),
            (adversarial_train_loader, "adv_train"),
            (adversarial_test_loader, "adv_test"),
        ]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                np.save(self.unlearn_file_prefix + "/preds_" + name + ".npy", preds)
                np.save(self.unlearn_file_prefix + "/targets" + name + ".npy", targets)
        return


class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None, device='cpu'):
        super().__init__(opt, model, prenet, device)

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(
            model, k=self.opt.k, model_name=self.opt.model
        )
        model = unlearn_func(
            model=model,
            method=self.opt.unlearn_method,
            factor=self.opt.factor,
            device=self.device,
        )
        self.model = model
        self.prenet = prenet
        self.model.to(self.device)
        if self.prenet is not None:
            self.prenet.to(self.device).eval() 

    def divide_model(self, model, k, model_name):
        if k == -1:  # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == "resnet9":
            assert k in [1, 2, 4, 5, 7, 8]
            mapping = {1: 6, 2: 5, 4: 4, 5: 3, 7: 2, 8: 1}
            dividing_part = mapping[k]
            all_mods = [
                model.conv1,
                model.conv2,
                model.res1,
                model.conv3,
                model.res2,
                model.conv4,
                model.fc,
            ]
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == "resnetwide28x10":
            assert k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
            all_mods = [
                model.conv1,
                model.layer1,
                model.layer2,
                model.layer3,
                model.norm,
                model.fc,
            ]
            mapping = {1: 5, 9: 3, 17: 2, 25: 1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(
                    all_mods[layer][: int(4 - (((k - 1) // 2) % 4))]
                )
                prenet = torch.nn.Sequential(*(prenet_list + prenet_additions))
                net_list = list(all_mods[layer][int(4 - (((k - 1) // 2) % 4)) :])
                net_additions = all_mods[layer + 1 :]
                net = torch.nn.Sequential(*(net_list + net_additions))

        elif model_name == "vitb16":
            assert k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1: 3, 13: 1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13 - k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)

        prenet.to(self.device)
        net.to(self.device)
        return prenet, net

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )

        return


class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None, device='cpu'):
        super().__init__(opt, model, prenet, device)
        self.og_model = copy.deepcopy(model)
        self.og_model.to(self.device).eval()

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        with torch.no_grad():
            logit_t = self.og_model(images)

        loss = F.cross_entropy(output, target)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)

        if self.maximize:
            loss = -loss

        self.top1(output, target)
        return loss

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize = False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize = True
                time_start = time.process_time()
                self.train_one_epoch(loader=forget_loader)
                self.save_files["train_time_taken"] += time.process_time() - time_start
                self.eval(loader=test_loader)

            self.maximize = False
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.save_files["train_time_taken"] += time.process_time() - time_start
            self.eval(loader=test_loader)
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += (
            "_"
            + str(self.opt.kd_T)
            + "_"
            + str(self.opt.alpha)
            + "_"
            + str(self.opt.msteps)
        )
        return





class XALFSSD:
  """Implements the Potion method, also known as XALFSSD.

  This method is an extension of the ALFSSD, incorporating
  modifications/alternative calculations for better performance
  """

  def __init__(self, opt, model, device='cpu'):
    self.opt = opt
    self.device = device
    self.model = model.to(device)
    self.best_model = None

    # Track metrics and timing (similar to Naive.save_files)
    self.save_files = {
        "train_top1": [],
        "val_top1": [],
        "train_time_taken": [0.0],
    }


  def unlearn(
      self,
      train_loader,
      forget_loader,
      frac_dl=None,
      min_acc_val=None,
  ):
    print("--- Unlearning with XALFSSD ---")
    self.opt.train_iters = len(train_loader) + len(forget_loader)
    clean_retain_loader = train_loader
    # -----------------------------
    if ITERATIVE_SEARCH:  # iterative
      # -----------------------------
      # start the frac iterations
      original_model = copy.deepcopy(
          self.model
      )  # do not forget to pass to device after reassigning
      max_tries = MAX_TRY
      min_acc = min_acc_val  # MIN_ACC
      # calculate the accuracy of the model on the train_loader
      org_model_acc = evaluate_model(original_model, forget_loader, self.device)[0]['ACC']
      print("Original Corrupted Model Proxy Set Accuracy: ", org_model_acc)
      # Remove the old files
      try:
        os.remove(file_name_1)
        os.remove(file_name_2)
      except FileNotFoundError:
        print("No previous importance files")
      original_importances, sample_importances = None, None
      for try_i in range(max_tries):
        print("----------------> Attempt #", try_i)
        self.best_model, original_importances, sample_importances = (
            alfssd_tuning(
                self.model,
                forget_loader,
                None,
                None,
                train_loader,  # train_loader,
                self.device,
                frac_dl,
                clean_retain_loader,
                x_d=True,  # turn on alternative D calc
                original_importances=original_importances,
                sample_importances=sample_importances,
            )
        )
        # calculate the accuracy of the model on the train_loader
        new_model_acc = evaluate_model(self.model, forget_loader, self.device)[0]['ACC']
        print("Attempted Corrected Model Proxy Set Accuracy: ", new_model_acc)
        # train_new_model_acc = self.get_acc(self.model, train_loader)
        # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
        # check if minimum acc reached
        if new_model_acc <= min_acc * org_model_acc:
          print("minimum accuracy reached")
          break
        elif (
            new_model_acc <= min_acc
        ):  # in case of already being good enough at the start
          print("minimum accuracy reached")
          break
        else:
          frac_dl = STEP_MULT * frac_dl
          self.model = copy.deepcopy(original_model)
          print("retry with new frac_dl: ", frac_dl)
    else:
      # --------
      self.best_model = alfssd_tuning(
          self.model,
          forget_loader,
          None,
          None,
          train_loader,  # train_loader,
          self.device,
          frac_dl,
          clean_retain_loader,
          x_d=True,  # turn on alternative D calc
      )
    
    
    return self.best_model
