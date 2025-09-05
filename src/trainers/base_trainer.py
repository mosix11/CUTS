import comet_ml
import torch
from torch.optim import AdamW, Adam, SGD
from torch.amp import GradScaler
from torch.amp import autocast

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from .custom_lr_schedulers import InverseSquareRootLR, CosineAnnealingWithWarmup

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import warnings
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


from . import utils

from abc import ABC, abstractmethod


class BaseClassificationTrainer(ABC):
    """
    Abstract base class for trainers.
    Implements the Template Method design pattern.
    """
    
    def __init__(
        self,
        outputs_dir: Path = Path("./outputs"),
        max_epochs: int = None,
        max_iterations: int = None,
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
        log_comet: bool = False,
        comet_api_key: str = None,
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
        
            
        if seed:
            self.seed = seed
            self.generator = torch.Generator().manual_seed(self.seed)

        
        self._run_on_gpu = run_on_gpu
        self.use_amp = use_amp


        self.max_epochs = max_epochs
        self.max_iterations = max_iterations 
        
        if (self.max_epochs is not None and self.max_epochs > 0) and (self.max_iterations is not None and self.max_iterations > 0):
            raise ValueError("Only one of max_epochs or max_iterations can be set at a time.")
        if (self.max_epochs is None or self.max_epochs <= 0) and (self.max_iterations is None or self.max_iterations <= 0):
            raise ValueError("Set either max_epochs (>0) or max_iterations (>0).")
        
        self.iteration_mode = self.max_iterations is not None and self.max_iterations > 0  

        
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
          
          
                
    @abstractmethod
    def _fit_epoch(self) -> dict:
        """
        Runs a single training epoch.
        This method MUST be implemented by subclasses.
        
        Returns:
            dict: A dictionary of training metrics for the epoch (e.g., {'Train/Loss': 0.1, 'Train/ACC': 0.95}).
        """
        pass
            
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
      
    
    
    def _setup_device(self):
        if self._run_on_gpu:
            if self._is_distributed():
                self.device = torch.device(f"cuda:{self._local_rank}")
            else:
                devices = utils.get_gpu_device()
                if devices == None:
                    raise RuntimeError('No GPU devices detected. Set `run_on_gpu` to False.')
                elif isinstance(devices, dict):
                    warnings.warn(f'Multiple GPU devices where found: {str(devices)}. Using device:0.')
                    self.device = devices['0']
                else:
                    self.device = devices
        else:
            self.device = utils.get_cpu_device()

        
    def _setup_data_loaders(self, dataset):
        self.dataset = dataset
        self.train_dataloader = dataset.get_train_dataloader()
        self.val_dataloader = dataset.get_val_dataloader()
        self.test_dataloader = dataset.get_test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )
        self.num_test_batches = len(self.test_dataloader)
        
        
    def _prepare_model(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)

        if self._is_distributed():
            # IMPORTANT: wrap AFTER .to(device) and BEFORE creating optimizer
            self.model = DDP(self.model, device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=False)
        
    def _prepare_batch(self, batch):
        batch = [tens.to(self.device, non_blocking=True) for tens in batch]
        return batch

        
        
    
    
    def _configure_optimizers(self, optim_state_dict=None, last_epoch=-1, last_gradient_step=-1):
        optim_cfg = copy.deepcopy(self.optimizer_cfg)
        del optim_cfg['type']

        opt_type = self.optimizer_cfg['type'].lower()
        if opt_type == "adamw":
            optim = AdamW(params=self.model.parameters(), **optim_cfg)
        elif opt_type == "adam":
            optim = Adam(params=self.model.parameters(), **optim_cfg)
        elif opt_type == "sgd":
            optim = SGD(params=self.model.parameters(), **optim_cfg)
        else:
            raise RuntimeError("Invalid optimizer type")
        if optim_state_dict:
            optim.load_state_dict(optim_state_dict)
        
        
        self.lr_scheduler = None
        self.lr_sch_step_on_batch = False 

        if self.lr_schedule_cfg:
            lr_sch_cfg = copy.deepcopy(self.lr_schedule_cfg)
            sch_type = lr_sch_cfg.pop('type')
            update_on = lr_sch_cfg.pop('update_on', 'epoch')  # 'gradient_step' or 'epoch'
            self.lr_sch_step_on_batch = (update_on == 'gradient_step')

            # Choose which 'last_*' marker to use
            restore_pos = last_gradient_step if self.lr_sch_step_on_batch else last_epoch

            if sch_type == 'step':
                self.lr_scheduler = MultiStepLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=restore_pos
                )
            elif sch_type == 'isqrt':
                self.lr_scheduler = InverseSquareRootLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=restore_pos
                )
            elif sch_type == 'plat':
                self.lr_scheduler = ReduceLROnPlateau(
                    optim,
                    **lr_sch_cfg,
                )
            elif sch_type == 'cosann':
                self.lr_scheduler = CosineAnnealingLR(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=restore_pos
                )
            elif sch_type == 'cosann_warmup':
                self.lr_scheduler = CosineAnnealingWithWarmup(
                    optim,
                    **lr_sch_cfg,
                    last_epoch=restore_pos
                )
            elif sch_type == 'onecycle':
                # OneCycleLR is *designed* to step per batch.
                if self.lr_sch_step_on_batch:
                    total_steps = self._compute_total_steps()
                    self.lr_scheduler = OneCycleLR(
                        optim,
                        total_steps=total_steps,
                        **lr_sch_cfg,
                        last_epoch=restore_pos
                    )
                else:
                    raise ValueError('OneCycleLR is designed to be updated every gradient step. So set `update_on=\'gradient_step\'`.')
            else:
                raise ValueError(f"Unknown scheduler type: {sch_type}")

        self.optim = optim

        # if self.early_stopping:
        #     self.early_stopping = nn_utils.EarlyStopping(patience=8, min_delta=0.001, mode='max', verbose=False)
        
        
        
        
    def _configure_logger(self, experiment_key=None):
        if not self._is_main_process:
            self.log_comet = False
            return
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


    def _save_full_checkpoint(self, path):
        if not self._is_main_process:
            return
        save_dict = {
            'model_state': self.model.module.state_dict() if self._is_distributed() else self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'epoch': self.epoch+1,
            'global_step': self.global_step,
            'iteration_mode': self.iteration_mode,
        }
        if self.log_comet:
            save_dict['exp_key'] = self.comet_experiment.get_key()
        if self.save_best_model:
            save_dict['best_prf'] = self.best_model_perf
        torch.save(save_dict, path)
        
        
    def _load_full_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self._prepare_model(checkpoint["model_state"])
        self.iteration_mode = checkpoint.get('iteration_mode', False)
        self.global_step = checkpoint.get('global_step', 0)
        
        self._configure_optimizers(
            checkpoint["optim_state"], last_epoch=checkpoint["epoch"], last_gradient_step=self.global_step 
        )
        self.epoch = checkpoint["epoch"]
        if self.log_comet:
            self._configure_logger(checkpoint["exp_key"])
            
        if 'best_prf' in checkpoint:
            self.best_model_perf = checkpoint['best_prf']
            
            
    def fit(self, model, dataset, resume=False):
        """
        This is the main "template method". It orchestrates the training process.
        DO NOT OVERRIDE THIS METHOD.
        """
        self._setup_distributed()
        self._setup_device()
        
        self._setup_data_loaders(dataset)
        self.model = model
        if resume:
            ckp_path = self.checkpoint_dir / Path('resume_ckp.pth')
            if not ckp_path.exists():
                raise RuntimeError(
                    "There is no checkpoint saved! Set the `resume` flag to False."
                )
            self._load_full_checkpoint(ckp_path)
        else:
            self._prepare_model()
            self._configure_optimizers()
            if self.log_comet:
                self._configure_logger()
            self.epoch = 0
            self.global_step = 0   

        self.grad_scaler = GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping_activated = False
            
        if self.iteration_mode:
            outer_iterable = range(self.epoch, 10**12)  # effectively unbounded; we'll break on max_iterations
        else:
            outer_iterable = tqdm(range(self.epoch, self.max_epochs), total=self.max_epochs)


            
        for self.epoch in outer_iterable:
            if self._is_distributed():
                self.train_dataloader.sampler.set_epoch(self.epoch)
            
            if (not self.iteration_mode) and isinstance(outer_iterable, tqdm):
                outer_iterable.set_description(f"Processing Training Epoch {self.epoch + 1}/{self.max_epochs}")
                
            if self.early_stopping and self.early_stopping_activated: break

            # Call the abstract training method (implemented by subclass)
            statistics = self._fit_epoch()
            
            
            if self.model_log_call:
                model_logs = self.model.log_stats()
                statistics.update(model_logs)
            
            
            self.after_epoch_end(
                epoch_train_stats=statistics,
                epoch_train_loss=statistics.get('Train/Loss') if isinstance(statistics, dict) else None
            )
                
           
            if self.iteration_mode and self.global_step >= self.max_iterations:
                break

        print('Training is finished!')
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
        
        # final_ckp_path = self.checkpoint_dir / Path('final_ckp.pth')
        # self.save_full_checkpoint(final_ckp_path)
        results_path = self.log_dir / Path('results.json')
        
        if self._is_main_process:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
        
        return results
                   
    
    def after_optimizer_step(
        self,
        *,
        train_snapshot: dict | None = None
    ) -> None:
        """
        Call this exactly once after each optimizer.step().
        Handles:
        - Incrementing global_step
        - Stepping per-step schedulers (including ReduceLROnPlateau with step metric)
        - Iteration-mode validation and checkpoint triggers
        """
        # 1) Count step
        self.global_step += 1

        # 2) LR scheduler (per-step)
        if self.lr_scheduler and self.lr_sch_step_on_batch:
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                # If user chose per-step plateau (unusual), feed the current step loss
                metric = float(train_snapshot['Train/Loss'])
                if metric is not None:
                    self.lr_scheduler.step(metric)
            else:
                self.lr_scheduler.step()

        # 3) Iteration-mode triggers
        if self.iteration_mode:
            # Validation on step frequency
            if self._should_validate_now():
                val_stats = self.evaluate(set='Val')
                self._check_update_best_and_save(val_stats, train_snapshot)

            # Checkpoint on step frequency
            if self._should_checkpoint_now():
                self._save_full_checkpoint(self.checkpoint_dir / 'resume_ckp.pth')

        if self.log_comet and self.iteration_mode:
            self.comet_experiment.log_metrics(train_snapshot, step=self.global_step)
    
    def after_epoch_end(
        self,
        *,
        epoch_train_stats: dict | None = None,
        epoch_train_loss: float | None = None
    ) -> None:
        """
        Call this exactly once at the end of an epoch (base.fit does this).
        Handles:
        - Epoch-mode validation / best saving / checkpointing
        - Epoch-mode LR scheduler stepping (including ReduceLROnPlateau)
        """
        # Epoch-mode triggers only
        if not self.iteration_mode:
            # Validation on epoch frequency
            if self._should_validate_now():
                val_stats = self.evaluate(set='Val')
                self._check_update_best_and_save(val_stats, epoch_train_stats)

            # Checkpoint on epoch frequency
            if self._should_checkpoint_now():
                self._save_full_checkpoint(self.checkpoint_dir / 'resume_ckp.pth')

            # LR scheduler (per-epoch)
            if self.lr_scheduler and not self.lr_sch_step_on_batch:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    metric = None
                    if epoch_train_loss is not None:
                        metric = float(epoch_train_loss)
                    elif epoch_train_stats is not None:
                        metric = epoch_train_stats.get('Train/Loss', None)
                    if metric is not None:
                        self.lr_scheduler.step(metric)
                else:
                    self.lr_scheduler.step()
                    
        if self.log_comet:
            self.comet_experiment.log_metrics(epoch_train_stats, epoch=self.epoch)
    
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
        
        if dataloader == None or len(dataloader) == 0:
            raise ValueError(f'The {set} set chosen for validation steps is empty.')
        
        metrics = self._evaluate_set(dataloader)
        
        # Add the set name prefix to the metrics
        return {f"{set}/{k}": v for k, v in metrics.items()}    
    
    
    def _is_distributed(self) -> bool:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    
    def _get_local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))
    
    
    def _setup_distributed(self):
        if self._is_distributed():
            if not dist.is_initialized():
                raise RuntimeError('Initialize `torch.distributed` at the start of the program for distributed training.')
            self._local_rank = self._get_local_rank()
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._is_main_process = (self._rank == 0)
        else:
            self._local_rank = 0
            self._rank = 0
            self._world_size = 1
            self._is_main_process = True
    
    def _compute_total_steps(self) -> int:
        """
        Total optimizer steps the scheduler should expect.
        - Iteration mode: exactly max_iterations
        - Epoch mode: max_epochs * len(train_dataloader)
        """
        if self.iteration_mode:
            return self.max_iterations
        return self.max_epochs * self.num_train_batches

    def _should_validate_now(self) -> bool:
        """
        Decide when to run validation based on mode:
        - Epoch mode: unchanged — every `validation_freq` epochs
        - Iteration mode: interpret `validation_freq` as steps
        """
        if not (self.validation_freq and self.validation_freq > 0):
            return False

        if self.iteration_mode:
            return self.global_step > 0 and (self.global_step % self.validation_freq == 0)
        else:
            # epoch-based behavior (called once per epoch)
            return (self.epoch + 1) % self.validation_freq == 0

    def _should_checkpoint_now(self) -> bool:
        """
        - Epoch mode: unchanged — every `checkpoint_freq` epochs
        - Iteration mode: interpret `checkpoint_freq` as steps
        """
        if not (self.checkpoint_freq and self.checkpoint_freq > 0):
            return False

        if self.iteration_mode:
            return self.global_step > 0 and (self.global_step % self.checkpoint_freq == 0)
        else:
            return (self.epoch + 1) % self.checkpoint_freq == 0
        
        
        
    def _merge_train_snapshot(self, val_stats: dict, train_snapshot: dict | None) -> dict:
        merged = {}
        if train_snapshot:
            merged.update(train_snapshot)
        merged.update(val_stats)
        return merged
    
    def _check_update_best_and_save(self, val_stats: dict, train_snapshot: dict | None):
        """
        Update best snapshot on Val/ACC and save 'best_ckp.pth' if improved.
        """
        if not self.save_best_model:
            return
        new_acc = val_stats.get('Val/ACC', None)
        if new_acc is None:
            return
        old_acc = self.best_model_perf.get('Val/ACC', 0)
        if new_acc > old_acc:
            merged = self._merge_train_snapshot(val_stats, train_snapshot)
            # add identifiers
            merged['epoch'] = getattr(self, 'epoch', 0)
            merged['global_step'] = getattr(self, 'global_step', 0)
            self.best_model_perf = merged
            self._save_full_checkpoint(self.checkpoint_dir / 'best_ckp.pth')
                
            
            
