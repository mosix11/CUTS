import comet_ml
import torch
from torch.optim import AdamW, Adam, SGD
from torch.amp import GradScaler
from torch.amp import autocast

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from .custom_lr_schedulers import InverseSquareRootLR

from torchmetrics import ConfusionMatrix

import os
from pathlib import Path
import time
from tqdm import tqdm
import random
import numpy as np
import dotenv
import copy
import json
import collections
import pickle

from typing import List, Tuple, Union


from ..utils import nn_utils, misc_utils


# Trainer based on epochs
class TrainerEp:

    def __init__(
        self,
        outputs_dir: Path = Path("./outputs"),
        dotenv_path: Path = Path("./.env"),
        max_epochs: int = 400,
        optimizer_cfg: dict = {
                'type': 'adamw',
                'lr': 1e-4,
                'betas': (0.9, 0.999)
            },
        lr_schedule_cfg: dict = None,
        validation_freq: int = -1,
        save_best_model: bool = True,
        checkpoint_freq: int = -1,
        early_stopping: bool = False,
        run_on_gpu: bool = True,
        use_amp: bool = True,
        batch_prog: bool = False,
        log_comet: bool = False,
        comet_api_key: str = "",
        comet_project_name: str = None,
        exp_name: str = None,
        exp_tags: List[str] = None,
        model_log_call: bool = False,
        seed: int = None
    ):
        outputs_dir.mkdir(exist_ok=True)
        self.outputs_dir = outputs_dir
        self.log_dir = outputs_dir / Path(exp_name) / Path('log')
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = outputs_dir / Path(exp_name) / Path('checkpoint')
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # if dotenv_path.exists():
        #     dotenv.load_dotenv('.env')
            
        if seed:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True) 
        torch.set_float32_matmul_precision("high")
        
        
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""GPU device not found!""")
        self.run_on_gpu = run_on_gpu
        self.use_amp = use_amp

        

        self.max_epochs = max_epochs
        self.optimizer_cfg = optimizer_cfg
        self.lr_schedule_cfg = lr_schedule_cfg

        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_best_model = save_best_model
        
        if save_best_model:
            if validation_freq < 1:
                raise RuntimeError('In order to save the best model the validation phase needs to be done. Sepcify `validation_freq`.')
            self.best_model_perf = {
                'Train/Loss': torch.inf,
                'Train/ACC': 0,
                'Val/Loss': torch.inf,
                'Val/ACC': 0
            }
            
        self.early_stopping = early_stopping
        

        self.batch_prog = batch_prog
        self.log_comet = log_comet
        self.comet_api_key = comet_api_key
        if log_comet and not comet_api_key:
            raise ValueError('When `log_comet` is set to `True`, `comet_api_key` should be provided.\n Please put your comet api key in a file called `.env` in the root directory of the project with the variable name `COMET_API_KEY`')
        self.comet_project_name = comet_project_name
        self.exp_name = exp_name
        self.exp_tags = exp_tags
        if log_comet and comet_project_name is None:
            raise RuntimeError('When CometML logging is active, the `comet_project_name` must be specified.')
        self.model_log_call = model_log_call
            

        self.accumulate_low_loss = False
        self.accumulate_high_loss = False
        

    def setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.val_dataloader = dataset.get_val_dataloader()
        self.test_dataloader = dataset.get_test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )
        self.num_test_batches = len(self.test_dataloader)
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [tens.to(self.gpu) for tens in batch]
            return batch
        else: return batch
        
    def unpack_batch(self, batch, default_value=None):
        """
        Unpacks a list/tuple into four variables, assigning default_value
        to any variables that don't have corresponding items.
        """
        # x = batch[0] if len(batch) > 0 else default_value
        # y = batch[1] if len(batch) > 1 else default_value
        # is_noisy = batch[2] if len(batch) > 2 else default_value
        # idx = batch[3] if len(batch) > 3 else default_value
        x = batch[0]
        y = batch[1]
        is_noisy = batch[2] if len(batch) == 4 else torch.zeros_like(y)
        idx = batch[3] if len(batch) == 4 else batch[2]
        return x, y, is_noisy, idx

    def prepare_model(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if self.run_on_gpu:
            self.model.to(self.gpu)

    def configure_optimizers(self, optim_state_dict=None, last_epoch=-1, last_gradient_step=-1):
        optim_cfg = copy.deepcopy(self.optimizer_cfg)
        del optim_cfg['type']
        if self.optimizer_cfg['type'] == "adamw":
            
            optim = AdamW(
                params=self.model.parameters(),
                **optim_cfg
            )

        elif self.optimizer_cfg['type'] == "adam":
            optim = Adam(
                params=self.model.parameters(),
                **optim_cfg
            )
        elif self.optimizer_cfg['type'] == "sgd":
            optim = SGD(
                params=self.model.parameters(),
                **optim_cfg
            )
        else:
            raise RuntimeError("Invalide optimizer type")
        if optim_state_dict:
            optim.load_state_dict(optim_state_dict)
        

        if self.lr_schedule_cfg:
            lr_sch_cfg = copy.deepcopy(self.lr_schedule_cfg)
            del lr_sch_cfg['type']
            if self.lr_schedule_cfg['type'] == 'step':
                self.lr_scheduler = MultiStepLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=last_epoch
                )
                
            elif self.lr_schedule_cfg['type'] == 'isqrt':
                self.lr_scheduler = InverseSquareRootLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=last_gradient_step
                )
            elif self.lr_schedule_cfg['type'] == 'plat':
                self.lr_scheduler = ReduceLROnPlateau(
                    optim,
                    **lr_sch_cfg,
                )
            elif self.lr_schedule_cfg['type'] == 'cosann':
                self.lr_schedule_cfg = CosineAnnealingLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=last_epoch
                )
        else: self.lr_scheduler = None

        # if self.early_stopping:
        #     self.early_stopping = nn_utils.EarlyStopping(patience=8, min_delta=0.001, mode='max', verbose=False)
        self.optim = optim

    def configure_logger(self, experiment_key=None):
        experiment_config = comet_ml.ExperimentConfig(
            name=self.exp_name,
            tags=self.exp_tags
        )
        self.comet_experiment = comet_ml.start(
            api_key=self.comet_api_key,
            workspace="mosix",
            project_name=self.comet_project_name,
            experiment_key=experiment_key,
            online=True,
            experiment_config=experiment_config
        )
        with open(self.log_dir / Path('comet_exp_key'), 'w') as mfile:
            mfile.write(self.comet_experiment.get_key()) 

    def fit(self, model, dataset, resume=False):
        self.setup_data_loaders(dataset)
        self.model = model
        if resume:
            ckp_path = self.checkpoint_dir / Path('resume_ckp.pth')
            if not ckp_path.exists():
                raise RuntimeError(
                    "There is no checkpoint saved! Set the `resume` flag to False."
                )
            self.load_full_checkpoint(ckp_path)
        else:
            self.prepare_model()
            self.configure_optimizers()
            if self.log_comet:
                self.configure_logger()
            self.epoch = 0


        self.grad_scaler = GradScaler("cuda", enabled=self.use_amp)

        self.early_stop = False
        
        if model.loss_fn.reduction != 'none' and (self.accumulate_low_loss or self.accumulate_high_loss):
            raise RuntimeError('In order to accumulate samples with low or high loss in a subset, the reduction type of the loss function should be set to `none`.')
            
        
        # Whether to log the progress for each batch or for each epoch
        if self.batch_prog:
            for self.epoch in range(self.epoch, self.max_epochs):
                if self.early_stop: break
                self.fit_epoch()
        else:
            pbar = tqdm(range(self.epoch, self.max_epochs), total=self.max_epochs)
            for self.epoch in pbar:
                pbar.set_description(f"Processing Training Epoch {self.epoch + 1}/{self.max_epochs}")
                if self.early_stopping and self.early_stop:
                    break
                self.fit_epoch()

        

        final_results = {}
        final_results.update(self.evaluate(set='Train'))
        final_results.update(self.evaluate(set='Test'))
        results = {
            'final': final_results,
        }

        if self.save_best_model:
            results['best'] = self.best_model_perf
        
        if self.log_comet:
            self.comet_experiment.log_parameters(results, nested_support=True)
            self.comet_experiment.end()
        
        final_ckp_path = self.checkpoint_dir / Path('final_ckp.pth')
        self.save_full_checkpoint(final_ckp_path)
        results_path = self.log_dir / Path('results.json')
        with open(results_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        
        return results


    def fit_epoch(self):

        # ******** Training Part ********
        self.model.train()

        # epoch_start_time = time.time()
        epoch_train_loss = misc_utils.AverageMeter()

        
        if self.batch_prog:
            pbar = tqdm(
                enumerate(self.train_dataloader),
                total=self.num_train_batches,
                desc="Processing Training Epoch {}".format(self.epoch + 1),
            )
        else:
            pbar = enumerate(self.train_dataloader)
        
        for i, batch in pbar:
            batch = self.prepare_batch(batch)
            input_batch, target_batch, is_noisy, idxs = self.unpack_batch(batch)
            
            self.optim.zero_grad()
                    
                
            if self.accumulate_low_loss or self.accumulate_high_loss:
                loss, predictions = self.model.training_step(input_batch, target_batch, self.use_amp, return_preds=True)
                
                is_correct = (predictions.argmax(dim=-1) == target_batch)

                # Update history and check consistency for each sample in the batch
                for j in range(len(idxs)):
                    sample_idx = idxs[j].item()
                    sample_target = target_batch[j].item()
                    sample_was_correct = is_correct[j].item()

                    # Handle low loss (correctly classified) samples
                    if self.accumulate_low_loss and not self.is_low_loss_buffer_full():
                        self.low_loss_history[sample_idx].append(sample_was_correct)
                        history = self.low_loss_history[sample_idx]
                        if len(history) == self.low_loss_consistency_window:
                            consistency_score = sum(history) / len(history)
                            if consistency_score >= self.low_loss_consistency_threshold:
                                if len(self.low_loss_sample_indices[sample_target]) < self.target_buffer_size_per_class_low:
                                    self.low_loss_sample_indices[sample_target].add(sample_idx)

                    # Handle high loss (wrongly classified) samples
                    if self.accumulate_high_loss and not self.is_high_loss_buffer_full():
                        self.high_loss_history[sample_idx].append(not sample_was_correct) # Note: append opposite
                        history = self.high_loss_history[sample_idx]
                        if len(history) == self.high_loss_consistency_window:
                            consistency_score = sum(history) / len(history)
                            if consistency_score >= self.high_loss_consistency_threshold:
                                if len(self.high_loss_sample_indices[sample_target]) < self.target_buffer_size_per_class_high:
                                    self.high_loss_sample_indices[sample_target].add(sample_idx)
            
            else:
                # TODO remove is_noisy
                # loss = self.model.training_step(input_batch, (target_batch, is_noisy), self.use_amp)
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
            
        
            
        if self.lr_scheduler:
            self.lr_scheduler.step()
            

                
        # print( 
        #     f"Epoch {self.epoch + 1}/{self.max_epochs}, "
        #     f"Training Loss: {epoch_train_loss.avg}, "
        #     f"Average Acc: {epoch_train_acc.avg}, "
        #     f"Time taken: {int((time.time() - epoch_start_time)//60)}:"
        #     f"{int((time.time() - epoch_start_time)%60)} minutes"
        # )
        
        metrics_results = self.model.compute_metrics()
        self.model.reset_metrics()
        
        statistics = {
            'Train/Loss': epoch_train_loss.avg,
            'Train/LR': self.optim.param_groups[0]['lr']
        }
        for met_name, met_val in metrics_results.items():
            stat_key = f"Train/{met_name}"
            
            statistics[stat_key] = met_val
          
          
        # TODO fix the problem of early stopping  
        # if epoch_train_loss.avg == 0.0 or metrics_results['ACC'] == 1.0:
        #     if self.early_stopping: self.early_stop = True
        
        if self.validation_freq > 0:
            if (self.epoch+1) % self.validation_freq == 0:
                res = self.evaluate(set='Val')
                for met, val in res.items():
                    statistics[met] = val
                if self.save_best_model:
                    if self.best_model_perf['Val/ACC'] < statistics['Val/ACC']:
                        self.best_model_perf = copy.deepcopy(statistics)
                        self.best_model_perf['epoch'] = self.epoch
                        ckp_path = self.checkpoint_dir / Path('best_ckp.pth')
                        self.save_full_checkpoint(ckp_path)
        
        if self.checkpoint_freq > 0 and (self.epoch+1) % self.checkpoint_freq == 0:
            ckp_path = self.checkpoint_dir / Path('resume_ckp.pth')
            self.save_full_checkpoint(ckp_path)
                
        if self.model_log_call:
            model_logs = self.model.log_stats()
            statistics.update(model_logs)
        if self.log_comet:
            self.comet_experiment.log_metrics(statistics, step=self.epoch)



    def evaluate(self, set='Val'):
        self.model.eval()
        self.model.reset_metrics()
        loss_met = misc_utils.AverageMeter()
        
        dataloader = None
        if set == 'Train':
            dataloader = self.train_dataloader
        elif set == 'Val':
            dataloader = self.val_dataloader
        elif set == 'Test':
            dataloader = self.test_dataloader
        else:
            raise ValueError("Invalid set specified. Choose 'train', 'val', or 'test'.")

        
        for i, batch in enumerate(dataloader):
            batch = self.prepare_batch(batch)
            input_batch, target_batch, is_noisy, idxs = self.unpack_batch(batch)
            # loss = self.model.validation_step(input_batch, (target_batch, is_noisy), self.use_amp)
            loss = self.model.validation_step(input_batch, target_batch, self.use_amp)
            if self.model.loss_fn.reduction == 'none':
                loss = loss.mean()
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
        
        metrics_results = self.model.compute_metrics()
        self.model.reset_metrics()
            
        results = {
            f"{set}/Loss": loss_met.avg,
        }
        for met_name, met_val in metrics_results.items():
            stat_key = f"{set}/{met_name}"
            results[stat_key] = met_val
            
        return results
    
    
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
        
        all_preds = []
        all_targets = []
        
        for i, batch in enumerate(dataloader):
            batch = self.prepare_batch(batch)
            input_batch, target_batch, is_noisy, idxs = self.unpack_batch(batch)
            
            model_output = self.model.predict(input_batch) # Get raw model output (logits)
            predictions = torch.argmax(model_output, dim=-1) # Get predicted class labels
            
            all_preds.extend(predictions.detach().cpu())
            all_targets.extend(target_batch.detach().cpu())
            
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        cm = confmat(torch.tensor(all_preds), torch.tensor(all_targets))
        return cm
        
    
    
    def save_full_checkpoint(self, path):
        save_dict = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'epoch': self.epoch+1,
        }
        if self.log_comet:
            save_dict['exp_key'] = self.comet_experiment.get_key()
        if self.save_best_model:
            save_dict['best_prf'] = self.best_model_perf
        torch.save(save_dict, path)
        
    def load_full_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.prepare_model(checkpoint["model_state"])
        self.configure_optimizers(
            checkpoint["optim_state"], last_epoch=checkpoint["epoch"]
        )
        self.epoch = checkpoint["epoch"]
        if self.log_comet:
            self.configure_logger(checkpoint["exp_key"])
            
        if 'best_prf' in checkpoint:
            self.best_model_perf = checkpoint['best_prf']
            

    
    def activate_low_loss_samples_buffer(self, percentage: float = 0.3, consistency_window: int = 5, consistency_threshold: float = 0.8):
        if not hasattr(self, 'train_dataloader'):
            raise RuntimeError("Please call `setup_data_loaders` before activating the buffers.")
        self.accumulate_low_loss = True
        self.low_loss_perc = percentage
        self.low_loss_consistency_window = consistency_window
        self.low_loss_consistency_threshold = consistency_threshold
        
        available_classes = self.dataset.get_available_classes()
        num_classes = len(available_classes)
        num_train_samples = len(self.train_dataloader.dataset)
        
        self.target_buffer_size_per_class_low = int((num_train_samples * self.low_loss_perc) / num_classes)
        self.low_loss_sample_indices = {i: set() for i in available_classes}
        self.low_loss_history = {i: collections.deque(maxlen=self.low_loss_consistency_window) for i in range(num_train_samples)}
    
    def activate_high_loss_samples_buffer(self, percentage: float = 0.3, consistency_window: int = 5, consistency_threshold: float = 0.8):
        if not hasattr(self, 'train_dataloader'):
            raise RuntimeError("Please call `setup_data_loaders` before activating the buffers.")
        self.accumulate_high_loss = True
        self.high_loss_perc = percentage
        self.high_loss_consistency_window = consistency_window
        self.high_loss_consistency_threshold = consistency_threshold

        available_classes = self.dataset.get_available_classes()
        num_classes = len(available_classes)
        num_train_samples = len(self.train_dataloader.dataset)
        
        self.target_buffer_size_per_class_high = int((num_train_samples * self.high_loss_perc) / num_classes)
        self.high_loss_sample_indices = {i: set() for i in available_classes}
        self.high_loss_history = {i: collections.deque(maxlen=self.high_loss_consistency_window) for i in range(num_train_samples)}

    def is_low_loss_buffer_full(self):
        if not self.accumulate_low_loss: return True
        for class_idx in self.low_loss_sample_indices:
            if len(self.low_loss_sample_indices[class_idx]) < self.target_buffer_size_per_class_low:
                return False
        print("Low-loss buffer is full and balanced.")
        self.save_low_loss_indices()
        self.accumulate_low_loss = False
        return True

    def is_high_loss_buffer_full(self):
        if not self.accumulate_high_loss: return True
        for class_idx in self.high_loss_sample_indices:
            if len(self.high_loss_sample_indices[class_idx]) < self.target_buffer_size_per_class_high:
                return False
        print("High-loss buffer is full and balanced.")
        self.save_high_loss_indices()
        self.accumulate_high_loss = False
        return True
    
    
    
    def save_low_loss_indices(self):
        if not hasattr(self, 'low_loss_sample_indices'):
            print("Low-loss buffer was not activated. Nothing to save.")
            return
        
        
        output_path = self.log_dir / f'low_loss_indices_{self.low_loss_perc}.pkl'

        # Convert sets to sorted lists for clean, deterministic output
        indices_to_save = {
            class_idx: sorted(list(idx_set))
            for class_idx, idx_set in self.low_loss_sample_indices.items()
        }

        with open(output_path, 'wb') as f:
            pickle.dump(indices_to_save, f)
            
        print(f"Successfully saved low-loss indices for {sum(len(v) for v in indices_to_save.values())} samples to: {output_path}")


    def save_high_loss_indices(self):
        if not hasattr(self, 'high_loss_sample_indices'):
            print("High-loss buffer was not activated. Nothing to save.")
            return

        output_path = self.log_dir / f'high_loss_indices_{self.high_loss_perc}.pkl'

        # Convert sets to sorted lists for clean, deterministic output
        indices_to_save = {
            class_idx: sorted(list(idx_set))
            for class_idx, idx_set in self.high_loss_sample_indices.items()
        }

        with open(output_path, 'wb') as f:
            pickle.dump(indices_to_save, f)

        print(f"Successfully saved high-loss indices for {sum(len(v) for v in indices_to_save.values())} samples to: {output_path}")
