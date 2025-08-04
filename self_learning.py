import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
import torch
import torchvision
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, Subset, ConcatDataset
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv
import yaml
import pickle
import copy
import random
import numpy as np
from torchmetrics import ConfusionMatrix
import json
from tqdm import tqdm
from collections import OrderedDict
from src.datasets import data_utils


from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, prepare_batch

def change_dataset_labels_with_preds(model, dataset, device):
    current_train_set = dataset.get_trainset()
    train_dl = torch.utils.data.DataLoader(
        current_train_set,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    all_preds_sm = []
    all_preds = []
    all_targets = []
    all_indices = []
    all_is_noisy_flags = []
    
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in train_dl:
            batch = prepare_batch(batch, device)
            input_batch, target_batch, indices, is_noisy = batch
            preds = model.predict(input_batch)
            
            predictions = torch.argmax(preds, dim=-1) 
            
            all_preds_sm.extend(list(torch.unbind(preds, dim=0)))
            all_preds.extend(predictions.detach().cpu())
            all_targets.extend(target_batch.cpu())
            all_indices.extend(indices.cpu())
            all_is_noisy_flags.extend(is_noisy.cpu())
        
    all_preds_tensor = torch.tensor(all_preds)
    all_targets_tensor = torch.tensor(all_targets)
    all_indices_tensor = torch.tensor(all_indices)
        
    mismatch_mask = all_preds_tensor != all_targets_tensor
    match_mask = all_preds_tensor == all_targets_tensor
    mismatch_indices = all_indices_tensor[mismatch_mask]
    match_indices = all_indices_tensor[match_mask]
    
    mismatch_subset = torch.utils.data.Subset(current_train_set, mismatch_indices.tolist())
    match_subset = torch.utils.data.Subset(current_train_set, match_indices.tolist())
        
    dummy_instance = current_train_set
    while not isinstance(dummy_instance, data_utils.NoisyClassificationDataset):
        dummy_instance = dummy_instance.dataset
        
    original_clean_lbls = dummy_instance.get_original_labels()
    # This should be equal to the noise rate in first iteration
    print(f'From {len(current_train_set)}, {(original_clean_lbls == dummy_instance.noisy_labels).sum()} had original labels matching targets.')
    
    dummy_instance.replace_labels(all_preds_tensor.long())
    # This should match the forget and healing rate of the task vector
    print('After changing the targets with predicted labels:')
    print(f'From {len(current_train_set)}, {(original_clean_lbls == dummy_instance.noisy_labels).sum()} had original labels matching targets.')
    
    dataset.set_trainset(current_train_set, shuffle=True)
    
    return dataset, match_subset, mismatch_subset



def pt_ft_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()

    augmentations = None
    if cfg['dataset']['name'] == 'cifar10':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'cifar100':
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]
    elif cfg['dataset']['name'] == 'mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    elif cfg['dataset']['name'] == 'fashion_mnist':
        pass
        # augmentations = [
        #     transformsv2.RandomCrop(32, padding=4),
        #     transformsv2.RandomHorizontalFlip(),
        # ]
    
    
    base_dataset, num_classes = dataset_factory.create_dataset(cfg, augmentations)
    base_model = model_factory.create_model(cfg['model'], num_classes)
    strategy = cfg['strategy']
    base_dataset.inject_noise(**strategy['noise']['pretraining'])
    
    if not outputs_dir.joinpath(f"{cfg_name}/gold/weights/model_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/gold"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['pretraining'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)

        torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False,
        )
        

    if not outputs_dir.joinpath(f"{cfg_name}/pretrain/weights/model_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        experiment_name = f"{cfg_name}/pretrain"

        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg['trainer']['pretraining'],
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        if strategy['finetuning_set'] == 'LowLoss':
            trainer.setup_data_loaders(dataset)
            trainer.activate_low_loss_samples_buffer(
                consistency_window=5,
                consistency_threshold=0.8
            )

        results = trainer.fit(model, dataset, resume=False)
        
        # print(results)

        torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

        class_names = [f"Class {i}" for i in range(num_classes)]
        confmat = trainer.confmat("Test")
        misc_utils.plot_confusion_matrix(
            cm=confmat,
            class_names=class_names,
            filepath=str(plots_dir / Path("confmat.png")),
            show=False,
        )


    
    self_learnt_dataset = copy.deepcopy(base_dataset)
    
    for attempt in range(1, 5):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{attempt}/weights/model_weights.pth").exists():
            dataset = copy.deepcopy(self_learnt_dataset)
            model = copy.deepcopy(base_model)
            
            if attempt == 1:
                base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
                base_model_ckp = torch.load(base_model_ckp_path)
                model.load_state_dict(copy.deepcopy(base_model_ckp))
            else:
                base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain_{attempt-1}") / Path('weights/model_weights.pth')
                base_model_ckp = torch.load(base_model_ckp_path)
                model.load_state_dict(copy.deepcopy(base_model_ckp))
                
            experiment_name = f"{cfg_name}/finetune_{attempt}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            

            if strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{strategy['percentage']:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.subset_set(set='Train', indices=all_easy_samples)
                dataset.inject_noise(**strategy['noise']['finetuning'])
            elif strategy['finetuning_set'] == 'HighLoss':
                pass
            elif strategy['finetuning_set'] == 'Heldout':
                dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
                dataset.inject_noise(**strategy['noise']['finetuning'])
            
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['finetuning'],
                exp_name=experiment_name,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            print(results)

            torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

            class_names = [f"Class {i}" for i in range(num_classes)]
            confmat = trainer.confmat("Test")
            misc_utils.plot_confusion_matrix(
                cm=confmat,
                class_names=class_names,
                filepath=str(plots_dir / Path("confmat.png")),
                show=False
            )
            
            ###################################################################3
            
            ft_weights = model.state_dict()
            noise_tv = TaskVector(base_model_ckp, ft_weights)
            
            model.load_state_dict(base_model_ckp)
            
            # best_coef, best_results, best_cm = search_optimal_coefficient(
            #     base_model=model,
            #     task_vector=noise_tv,
            #     search_range=(-3.0, 0.0),
            #     dataset=base_dataset,
            #     num_classes=num_classes,
            #     device=gpu
            # )
            
            # print(f"Best scaling coefficient for TV at attempt {attempt}% = {best_coef}")
            # print(f"Metrics of the negated model is {best_results}")
            # noise_tv.apply_to(model, scaling_coef=best_coef)
            
        
            noise_tv.apply_to(model, scaling_coef=-1.0)
            
            print("Clean and noisy set performance after applying TV on the original mixed dataset:")
            print(eval_model_on_clean_noise_splits(model, cfg, base_dataset, gpu))
            
            self_learnt_dataset, match_subset, mismatch_subset = change_dataset_labels_with_preds(model, self_learnt_dataset, gpu)
            
            print("Clean and noisy set performance on self learnt dataset before training:")
            print(eval_model_on_clean_noise_splits(model, None, self_learnt_dataset, gpu))
            
            self_learnt_dataset.set_trainset(match_subset, shuffle=True)
            
            ###################################################################
            
            print("New matched set size : ", len(match_subset))
            print("New mismatched set size : ", len(mismatch_subset))
            dataset = copy.deepcopy(self_learnt_dataset)
            
            
            experiment_name = f"{cfg_name}/pretrain_{attempt}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['finetuning'],
                exp_name=experiment_name,
            )
            
            results = trainer.fit(model, dataset, resume=False)

            torch.save(model.state_dict(), weights_dir / Path("model_weights.pth"))

            class_names = [f"Class {i}" for i in range(num_classes)]
            confmat = trainer.confmat("Test")
            misc_utils.plot_confusion_matrix(
                cm=confmat,
                class_names=class_names,
                filepath=str(plots_dir / Path("confmat.png")),
                show=False
            )

            # print("Clean and noisy set performance on self learnt dataset after training:")
            # print(eval_model_on_clean_noise_splits(model, None, self_learnt_dataset, gpu))
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume training from the last checkpoint.",
        action="store_true",
    )
    
    parser.add_argument(
        "-t",
        "--taskvector",
        help="Compare the task vectors.",
        action="store_true",
    )
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/various_experiments') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/various_experiments").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/various_experiments").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)


    pt_ft_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)