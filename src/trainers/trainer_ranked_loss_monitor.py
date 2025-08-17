import torch

from torch.amp import GradScaler
from torch.amp import autocast

from typing import List, Tuple, Union
from tqdm import tqdm
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

from . import BaseClassificationTrainer

from . import utils
from ..utils import misc_utils


class TrainerRLS(BaseClassificationTrainer, utils.LossRankedSampleStorage):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.accumulate_low_loss = False
        self.accumulate_high_loss = False
        
        
    def unpack_batch(self, batch, default_value=None):
        """
        Unpacks a list/tuple into four variables, assigning default_value
        to any variables that don't have corresponding items.
        """
        x = batch[0]
        y = batch[1]
        idx = batch[2]
        is_noisy = batch[3] if len(batch) == 4 else torch.zeros_like(y)
        return x, y, idx, is_noisy
    


    def _fit_epoch(self) -> dict:
        """
        Implementation of the training loop for a single epoch.
        """
        epoch_train_loss = misc_utils.AverageMeter()

        inner_iter = enumerate(self.train_dataloader)
        if self.iteration_mode:
            inner_iter = tqdm(
                enumerate(self.train_dataloader),
                total=self.num_train_batches,
                desc=f"Epoch {self.epoch + 1} — steps",
            )

        for i, batch in inner_iter:
            batch = self.prepare_batch(batch)
            input_batch, target_batch, idxs, is_noisy  = self.unpack_batch(batch)
            
            self.optim.zero_grad()
                    
            if self.accumulate_low_loss or self.accumulate_high_loss:
                loss, predictions = self.model.training_step(input_batch, target_batch, self.use_amp, return_preds=True)
                
                is_correct = (predictions.argmax(dim=-1) == target_batch)

                for j in range(len(idxs)):
                    sample_idx = idxs[j].item()
                    sample_target = target_batch[j].item()
                    sample_was_correct = is_correct[j].item()


                    # Handle low loss (correctly classified) samples
                    if self.accumulate_low_loss:
                        self.low_loss_history[sample_idx].append(sample_was_correct)
                        history = self.low_loss_history[sample_idx]
                        if len(history) == self.low_loss_consistency_window:
                            consistency_score = sum(history) / len(history)
                            if consistency_score >= self.low_loss_consistency_threshold:
                                self.low_loss_sample_indices[sample_target].add(sample_idx)

                    # Handle low loss (correctly classified) samples
                    if self.accumulate_high_loss:
                        self.high_loss_history[sample_idx].append(not sample_was_correct)
                        history = self.high_loss_history[sample_idx]
                        if len(history) == self.high_loss_consistency_window:
                            consistency_score = sum(history) / len(history)
                            if consistency_score >= self.high_loss_consistency_threshold:
                                self.high_loss_sample_indices[sample_target].add(sample_idx)
            
            else:
                loss = self.model.training_step(input_batch, target_batch, self.use_amp)

            if self.model.loss_fn.reduction == 'none':
                loss = loss.mean()
            
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optim)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optim.step()
                

            epoch_train_loss.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            # Tell the base: a gradient step happened (centralized per-step logic)
            self.after_optimizer_step(
                step_loss=loss.detach().cpu().item(),
                train_snapshot={
                    'Train/Loss': epoch_train_loss.avg if epoch_train_loss.count > 0 else float(loss.detach().cpu().item()),
                    'Train/LR': self.optim.param_groups[0]['lr'],
                }
            )
            
            if self.iteration_mode and isinstance(inner_iter, tqdm):
                inner_iter.set_postfix_str(f"step={self.global_step}, loss={epoch_train_loss.avg:.4f}")


            if self.iteration_mode and self.global_step >= self.max_iterations:
                break
            
            
        if self.accumulate_low_loss:
                self.check_and_save_low_loss_buffers()
        if self.accumulate_high_loss:
            self.check_and_save_high_loss_buffers()

        metrics_results = self.model.compute_metrics()
        self.model.reset_metrics()
        
        metrics_results = {f"Train/{k}": v for k, v in metrics_results.items()}
        metrics_results['Train/Loss'] = epoch_train_loss.avg
        metrics_results['Train/LR'] = self.optim.param_groups[0]['lr']
        return metrics_results


    def _evaluate_set(self, dataloader) -> dict:
        """
        Implementation of the evaluation loop for a given dataset.
        """
        loss_met = misc_utils.AverageMeter()
        
        for batch in dataloader:
            batch = self.prepare_batch(batch)
            input_batch, target_batch, _, _ = self.unpack_batch(batch)
            
            # Model-specific validation logic
            loss = self.model.validation_step(input_batch, target_batch, self.use_amp)
            if self.model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
        
        metrics_results = self.model.compute_metrics()
        self.model.reset_metrics()
            
        metrics_results['Loss'] = loss_met.avg
        return metrics_results
    
    
    def confmat(self, set='Val'):
        num_classes = self.dataset.get_num_classes()
        self.model.eval()
        
        dataloader = None
        if set == 'Train':
            dataloader = self.train_dataloader
        elif set == 'Val':
            dataloader = self.val_dataloader
        elif set == 'Test':
            dataloader = self.test_dataloader
        else:
            raise ValueError("Invalid set specified. Choose 'Train', 'Val', or 'Test'.")
        
        
        cm_metric = MulticlassConfusionMatrix(num_classes=num_classes)
        if self.run_on_gpu:
            cm_metric.to(self.gpu)
        
        for i, batch in enumerate(dataloader):
            batch = self.prepare_batch(batch)
            input_batch, target_batch, idxs, is_noisy = self.unpack_batch(batch)
            
            model_output = self.model.predict(input_batch) # Get raw model output (logits)
            predictions = torch.argmax(model_output, dim=-1) # Get predicted class labels
            
            cm_metric.update(predictions.detach(), target_batch.detach())
        
        cm = cm_metric.compute().cpu().numpy()
        return cm


# class TrainerRLS(BaseClassificationTrainer, utils.LossRankedSampleStorage):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         self.accumulate_low_loss = False
#         self.accumulate_high_loss = False
        
        
        
#     def unpack_batch(self, batch, default_value=None):
#         """
#         Unpacks a list/tuple into four variables, assigning default_value
#         to any variables that don't have corresponding items.
#         """
#         x = batch[0]
#         y = batch[1]
#         idx = batch[2]
#         is_noisy = batch[3] if len(batch) == 4 else torch.zeros_like(y)
#         return x, y, idx, is_noisy
    


#     def _fit_epoch(self) -> dict:
#         """
#         Implementation of the training loop for a single epoch.
#         """
#         if self.model.loss_fn.reduction != 'none' and (self.accumulate_low_loss or self.accumulate_high_loss):
#             raise RuntimeError('In order to accumulate samples with low or high loss in a subset, the reduction type of the loss function should be set to `none`.')
        
#         epoch_train_loss = misc_utils.AverageMeter()

        
#         if self.batch_prog:
#             pbar = tqdm(
#                 enumerate(self.train_dataloader),
#                 total=self.num_train_batches,
#                 desc="Processing Training Epoch {}".format(self.epoch + 1),
#             )
#         else:
#             pbar = enumerate(self.train_dataloader)
        
#         for i, batch in pbar:
#             batch = self.prepare_batch(batch)
#             input_batch, target_batch, idxs, is_noisy  = self.unpack_batch(batch)
            
#             self.optim.zero_grad()
                    
                
#             if self.accumulate_low_loss or self.accumulate_high_loss:
#                 loss, predictions = self.model.training_step(input_batch, target_batch, self.use_amp, return_preds=True)
                
#                 is_correct = (predictions.argmax(dim=-1) == target_batch)

#                 for j in range(len(idxs)):
#                     sample_idx = idxs[j].item()
#                     sample_target = target_batch[j].item()
#                     sample_was_correct = is_correct[j].item()


#                     # Handle low loss (correctly classified) samples
#                     if self.accumulate_low_loss:
#                         self.low_loss_history[sample_idx].append(sample_was_correct)
#                         history = self.low_loss_history[sample_idx]
#                         if len(history) == self.low_loss_consistency_window:
#                             consistency_score = sum(history) / len(history)
#                             if consistency_score >= self.low_loss_consistency_threshold:
#                                 self.low_loss_sample_indices[sample_target].add(sample_idx)

#                     # Handle low loss (correctly classified) samples
#                     if self.accumulate_high_loss:
#                         self.high_loss_history[sample_idx].append(not sample_was_correct)
#                         history = self.high_loss_history[sample_idx]
#                         if len(history) == self.high_loss_consistency_window:
#                             consistency_score = sum(history) / len(history)
#                             if consistency_score >= self.high_loss_consistency_threshold:
#                                 self.high_loss_sample_indices[sample_target].add(sample_idx)
            
#             else:
#                 loss = self.model.training_step(input_batch, target_batch, self.use_amp)

#             if self.model.loss_fn.reduction == 'none':
#                 loss = loss.mean()
            
#             if self.use_amp:
#                 self.grad_scaler.scale(loss).backward()
#                 self.grad_scaler.step(self.optim)
#                 self.grad_scaler.update()
#             else:
#                 loss.backward()
#                 self.optim.step()
                
#             if self.lr_scheduler and self.lr_sch_step_on_batch:
#                 self.lr_scheduler.step()
                
#             epoch_train_loss.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
        
            
#         if self.lr_scheduler and not self.lr_sch_step_on_batch:
#             self.lr_scheduler.step()
            
#         if self.accumulate_low_loss:
#                 self.check_and_save_low_loss_buffers()
#         if self.accumulate_high_loss:
#             self.check_and_save_high_loss_buffers()

#         metrics_results = self.model.compute_metrics()
#         self.model.reset_metrics()
        
#         metrics_results = {f"Train/{k}": v for k, v in metrics_results.items()}
#         metrics_results['Train/Loss'] = epoch_train_loss.avg
#         metrics_results['Train/LR'] = self.optim.param_groups[0]['lr']
#         return metrics_results


#     def _evaluate_set(self, dataloader) -> dict:
#         """
#         Implementation of the evaluation loop for a given dataset.
#         """
#         loss_met = misc_utils.AverageMeter()
        
#         for batch in dataloader:
#             batch = self.prepare_batch(batch)
#             input_batch, target_batch, _, _ = self.unpack_batch(batch)
            
#             # Model-specific validation logic
#             loss = self.model.validation_step(input_batch, target_batch, self.use_amp)
#             if self.model.loss_fn.reduction == 'none':
#                 loss = loss.mean()
#             loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
        
#         metrics_results = self.model.compute_metrics()
#         self.model.reset_metrics()
            
#         metrics_results['Loss'] = loss_met.avg
#         return metrics_results
    
    
#     def confmat(self, set='Val'):
#         num_classes = self.dataset.get_num_classes()
#         self.model.eval()
        
#         dataloader = None
#         if set == 'Train':
#             dataloader = self.train_dataloader
#         elif set == 'Val':
#             dataloader = self.val_dataloader
#         elif set == 'Test':
#             dataloader = self.test_dataloader
#         else:
#             raise ValueError("Invalid set specified. Choose 'Train', 'Val', or 'Test'.")
        
        
#         cm_metric = MulticlassConfusionMatrix(num_classes=num_classes)
#         if self.run_on_gpu:
#             cm_metric.to(self.gpu)
        
#         for i, batch in enumerate(dataloader):
#             batch = self.prepare_batch(batch)
#             input_batch, target_batch, idxs, is_noisy = self.unpack_batch(batch)
            
#             model_output = self.model.predict(input_batch) # Get raw model output (logits)
#             predictions = torch.argmax(model_output, dim=-1) # Get predicted class labels
            
#             cm_metric.update(predictions.detach(), target_batch.detach())
        
#         cm = cm_metric.compute().cpu().numpy()
#         return cm
    
    
