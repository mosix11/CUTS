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



def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch



def evaluate_model(model, dataloader, device):
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


def search_optimal_coefficient(base_model, task_vector, search_range, dataset, num_classes, device):
    """
    Performs a search to find the optimal task vector scaling coefficient.

    Args:
        base_model (torch.nn.Module): The pre-trained model. A deepcopy is made for each evaluation.
        task_vector (TaskVector): The task vector object.
        dataset: The dataset object to get the test dataloader from.
        search_range (list or tuple): A list/tuple [min_val, max_val] for the search.
        device (torch.device): The device to run evaluation on.
        num_classes (int): The number of classes for the confusion matrix.

    Returns:
        tuple: (best_coefficient, best_performance_metrics, confusion_matrix_tensor)
    """
    test_dataloader = dataset.get_test_dataloader()
    
    best_coef = 0.0
    best_acc = -1.0
    best_results = {}
    
    print("--- Starting Coarse Search ---")
    coarse_search_grid = np.arange(search_range[0], search_range[1] + 0.1, 0.1)
    
    for scale_coef in tqdm(coarse_search_grid, desc="Coarse Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results
    
    # print(f"\nCoarse search best coefficient: {best_coef:.2f} with Accuracy: {best_acc:.4f}")

    print("\n--- Starting Fine Search ---")
    fine_search_start = max(search_range[0], best_coef - 0.1)
    fine_search_end = min(search_range[1], best_coef + 0.1)
    fine_search_grid = np.linspace(fine_search_start, fine_search_end, num=21)

    for scale_coef in tqdm(fine_search_grid, desc="Fine Search"):
        search_model = copy.deepcopy(base_model)
        task_vector.apply_to(search_model, scaling_coef=scale_coef)
        
        metric_results, _, _ = evaluate_model(search_model, test_dataloader, device)
        
        if metric_results['ACC'] > best_acc:
            best_acc = metric_results['ACC']
            best_coef = scale_coef
            best_results = metric_results

    # print(f"\nRecalculating metrics and confusion matrix for best coefficient: {best_coef:.2f}")
    final_model = copy.deepcopy(base_model)
    task_vector.apply_to(final_model, scaling_coef=best_coef)
    final_model.to(device)

    best_results, all_preds, all_targets = evaluate_model(final_model, test_dataloader, device)
    
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    best_cm_tensor = confmat_metric(all_preds, all_targets)

    return best_coef, best_results, best_cm_tensor


def pt_ft_model(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()

    if cfg['dataset']['name'] == 'cifar10':
        # augmentations = None
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
        # augmentations = None
        augmentations = [
            transformsv2.RandomCrop(32, padding=4),
            transformsv2.RandomHorizontalFlip(),
        ]

    
    if not outputs_dir.joinpath(f"{cfg_name}/gold/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
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
            **cfg_cpy['trainer']['pretraining'],
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
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        strategy = cfg_cpy['strategy']
        
        dataset.inject_noise(**strategy['noise']['pretraining'])
        
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
        
        if cfg['strategy']['finetuning_set'] == 'LowLoss':
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


    if not outputs_dir.joinpath(f"{cfg_name}/finetune_gold/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
        checkpoint = torch.load(base_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/finetune_gold"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['pretraining'],
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
        
        
    if not outputs_dir.joinpath(f"{cfg_name}/finetune_gt_noise/weights/model_weights.pth").exists():
        cfg_cpy = copy.deepcopy(cfg)
        dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
        
        model = model_factory.create_model(cfg_cpy['model'], num_classes)
        
        base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
        checkpoint = torch.load(base_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        strategy = cfg_cpy['strategy']
        dataset.inject_noise(**strategy['noise']['pretraining'])
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(noisy_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/finetune_gt_noise"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        

        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **cfg_cpy['trainer']['finetuning'],
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

    
    
    for idx, low_loss_percentage in enumerate(cfg['strategy']['percentage']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{low_loss_percentage}/weights/model_weights.pth").exists():
            cfg_cpy = copy.deepcopy(cfg)
            dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
            
            model = model_factory.create_model(cfg_cpy['model'], num_classes)
            
            
            if idx == 0:
                base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
                base_model_ckp = torch.load(base_model_ckp_path)
                model.load_state_dict(copy.deepcopy(base_model_ckp))
            else:
                base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain_{cfg['strategy']['percentage'][idx-1]}") / Path('weights/model_weights.pth')
                base_model_ckp = torch.load(base_model_ckp_path)
                model.load_state_dict(copy.deepcopy(base_model_ckp))
            
            experiment_name = f"{cfg_name}/finetune_{low_loss_percentage}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            
            strategy = cfg_cpy['strategy']
            
            if strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{low_loss_percentage:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.inject_noise(**strategy['noise']['pretraining'])
                dataset.subset_set(set='Train', indices=all_easy_samples)
                
                dataset.inject_noise(**strategy['noise']['finetuning'])
                
            elif strategy['finetuning_set'] == 'HighLoss':
                pass
            
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
            
            best_coef, best_results, best_cm = search_optimal_coefficient(
                base_model=model,
                task_vector=noise_tv,
                search_range=(-2.0, 0.0),
                dataset=dataset,
                num_classes=num_classes,
                device=gpu
            )
            
            print(f"Best scaling coefficient for TV {low_loss_percentage}% = {best_coef}")
            print(f"Metrics of the negated model is {best_results}")
            
            noise_tv.apply_to(model, scaling_coef=best_coef)
            
            ###################################################################
            
            
            cfg_cpy = copy.deepcopy(cfg)
            dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
            
            # model = model_factory.create_model(cfg_cpy['model'], num_classes)
            
            experiment_name = f"{cfg_name}/pretrain_{low_loss_percentage}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            strategy = cfg_cpy['strategy']
            
            if strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{low_loss_percentage:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.inject_noise(**strategy['noise']['pretraining'])
                dataset.subset_set(set='Train', indices=all_easy_samples)
                
            elif strategy['finetuning_set'] == 'HighLoss':
                pass
            
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


def compare_task_vectors(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) 
    torch.set_float32_matmul_precision("high")
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    dataset, num_classes = dataset_factory.create_dataset(cfg)
    
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
    base_expr_dir = outputs_dir / cfg_name
    
    init_weights = torch.load(base_expr_dir / "init_model_weights.pth", map_location=cpu)
    clean_weights = torch.load(base_expr_dir / "clean/weights/model_weights.pth", map_location=cpu)  
    mix_weights = torch.load(base_expr_dir / "mix/weights/model_weights.pth", map_location=cpu)
    noisy_wieghts = torch.load(base_expr_dir / "noisy/weights/model_weights.pth", map_location=cpu)  
    
    clean_tv = TaskVector(init_weights, clean_weights)
    mix_tv = TaskVector(init_weights, mix_weights)
    noisy_tv = TaskVector(init_weights, noisy_wieghts)
    
    temp = TaskVector(mix_weights, noisy_wieghts)
    
    tv_list = [clean_tv, mix_tv, noisy_tv, temp]
    tv_names = ['clean', 'mix', 'noise', 'temp']
    
    
    tv_sim = []
    for i in range(len(tv_list)):
        anchor_tv = tv_list[i]
        tv_sim.append([])
        for j in range(len(tv_list)):
            other_tv = tv_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            tv_sim[i].append(cos_sim)
    tv_sim = np.array(tv_sim)
    
    
    
    misc_utils.plot_confusion_matrix(cm=tv_sim, class_names=tv_names, filepath=None, show=True)
    
    base_model.load_state_dict(mix_weights)
    
    best_coef, best_results, best_cm = search_optimal_coefficient(
        base_model=base_model,
        task_vector=tv_list[3],
        search_range=(-2.0, 0.0),
        dataset=dataset,
        num_classes=num_classes,
        device=gpu
    )
    
    print(f"Best scaling coefficient for TV = {best_coef}")
    print(f"Metrics of the negated model is {best_results}")
    
    strategy = cfg['strategy']
    dataset.inject_noise(**strategy['noise'])
    clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
    dataset.set_trainset(clean_set, shuffle=False)
    metric, _, _ = evaluate_model(base_model, dataloader=dataset.get_train_dataloader(), device=gpu)
    print("Performance on clean set before task vector:", metric)
    dataset.set_trainset(noisy_set, shuffle=False)
    metric, _, _ = evaluate_model(base_model, dataloader=dataset.get_train_dataloader(), device=gpu)
    print("Performance on noisy set before task vector:", metric)
    
    base_model.to(cpu)
    tv_list[3].apply_to(base_model, scaling_coef=best_coef)
    
    dataset.set_trainset(clean_set, shuffle=False)
    metric, _, _ = evaluate_model(base_model, dataloader=dataset.get_train_dataloader(), device=gpu)
    print("Performance on clean set after task vector:", metric)
    dataset.set_trainset(noisy_set, shuffle=False)
    metric, _, _ = evaluate_model(base_model, dataloader=dataset.get_train_dataloader(), device=gpu)
    print("Performance on noisy set after task vector:", metric)
    

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

    if args.taskvector:
        compare_task_vectors(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    else:
        pt_ft_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)