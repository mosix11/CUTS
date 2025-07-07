import comet_ml
import torch
from torch.optim import AdamW, Adam, SGD
from torch.amp import GradScaler
from torch.amp import autocast

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from .custom_lr_schedulers import InverseSquareRootLR, CosineAnnealingWithWarmup



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

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.
    Implements the Template Method design pattern.
    """
    
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
        
        if dotenv_path.exists():
            dotenv.load_dotenv('.env')
            
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
        
        
    def prepare_model(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if self.run_on_gpu:
            self.model.to(self.gpu)
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [tens.to(self.gpu) for tens in batch]
            return batch
        else: return batch
        
        
    
    
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
            self.lr_sch_step_on_batch = False
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
                self.lr_sch_step_on_batch = True
            elif self.lr_schedule_cfg['type'] == 'plat':
                self.lr_scheduler = ReduceLROnPlateau(
                    optim,
                    **lr_sch_cfg,
                )
            elif self.lr_schedule_cfg['type'] == 'cosann':
                self.lr_scheduler = CosineAnnealingLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=last_epoch
                )
            elif self.lr_schedule_cfg['type'] == 'cosann_warmup':
                self.lr_scheduler = CosineAnnealingWithWarmup(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=last_epoch
                )
            elif self.lr_schedule_cfg['type'] == 'onecycle':
                self.lr_scheduler = OneCycleLR(
                    optim,
                    total_steps=self.max_epochs*len(self.train_dataloader),
                    **lr_sch_cfg,
                    last_epoch=last_epoch
                )
                self.lr_sch_step_on_batch = True
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
            
            
            
            
            
            
    @abstractmethod
    def _fit_epoch(self) -> dict:
        """
        Runs a single training epoch.
        This method MUST be implemented by subclasses.
        
        Returns:
            dict: A dictionary of training metrics for the epoch (e.g., {'Train/Loss': 0.1, 'Train/ACC': 0.95}).
        """
        pass
            
            
            
    def fit(self, model, dataset, resume=False):
        """
        This is the main "template method". It orchestrates the training process.
        DO NOT OVERRIDE THIS METHOD.
        """
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
            pbar = range(self.epoch, self.max_epochs)
        else:
            pbar = tqdm(range(self.epoch, self.max_epochs), total=self.max_epochs)
            
        for self.epoch in pbar:
            if isinstance(pbar, tqdm):
                pbar.set_description(f"Processing Training Epoch {self.epoch + 1}/{self.max_epochs}")
                
            if self.early_stopping and self.early_stop: break

            # 1. Call the abstract training method (implemented by subclass)
            self.model.train()
            statistics = self._fit_epoch()
            
            if self.accumulate_low_loss:
                self.check_and_save_low_loss_buffers()
            if self.accumulate_high_loss:
                self.check_and_save_high_loss_buffers()
            
            if self.model_log_call:
                model_logs = self.model.log_stats()
                statistics.update(model_logs)
            
            # 2. Call the abstract evaluation method (implemented by subclass)
            if self.validation_freq > 0 and (self.epoch + 1) % self.validation_freq == 0:
                val_stats = self.evaluate(set_name='Val')
                statistics.update(val_stats)

                # 3. Handle saving best model (generic logic)
                if self.save_best_model and val_stats['Val/ACC'] > self.best_model_perf.get('Val/ACC', 0):
                    self.best_model_perf = {**statistics, 'epoch': self.epoch}
                    self.save_full_checkpoint(self.checkpoint_dir / 'best_ckp.pth')
            
            # 4. Handle logging, checkpointing, etc. (generic logic)
            if self.checkpoint_freq > 0 and (self.epoch+1) % self.checkpoint_freq == 0:
                self.save_full_checkpoint(self.checkpoint_dir / 'resume_ckp.pth')
            
            
            
            if self.log_comet:
                self.comet_experiment.log_metrics(statistics, step=self.epoch)

        # Final evaluation, saving, etc.
        final_results = {}
        final_results.update(self.evaluate(set='Train'))
        final_results.update(self.evaluate(set='Test'))
        for key, value in final_results.items():
            if isinstance(value, torch.Tensor):
                final_results[key] = value.cpu().item()
        results = {
            'final': final_results,
        }

        if self.save_best_model:
            for key, value in self.best_model_perf.items():
                if isinstance(value, torch.Tensor):
                    self.best_model_perf[key] = value.cpu().item()
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
            
            
            
            
    @abstractmethod
    def _evaluate_set(self, dataloader) -> dict:
        """
        Evaluates the model on a given dataloader (e.g., validation or test).
        This method MUST be implemented by subclasses.

        Args:
            dataloader (DataLoader): The dataloader to evaluate on.

        Returns:
            dict: A dictionary of evaluation metrics (e.g., {'Val/Loss': 0.2, 'Val/ACC': 0.9}).
        """
        pass
        
    def evaluate(self, set: str = 'Val') -> dict:
        """
        Public-facing evaluation method.
        """
        self.model.eval()
        self.model.reset_metrics()
        
        if set == 'Train':
            dataloader = self.train_dataloader
        elif set == 'Val':
            dataloader = self.val_dataloader
        elif set == 'Test':
            dataloader = self.test_dataloader
        else:
            raise ValueError("Invalid set specified. Choose 'Train', 'Val', or 'Test'.")
        
        metrics = self._evaluate_set(dataloader)
        
        # Add the set name prefix to the metrics
        return {f"{set}/{k}": v for k, v in metrics.items()}
            
            
            
            
            
            
            
            
            
            
            
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
        
        self.low_loss_sample_indices = {i: set() for i in available_classes}
        self.low_loss_history = {i: collections.deque(maxlen=self.low_loss_consistency_window) for i in range(self.num_train_samples)}
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
        self.high_loss_history = {i: collections.deque(maxlen=self.high_loss_consistency_window) for i in range(self.num_train_samples)}
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