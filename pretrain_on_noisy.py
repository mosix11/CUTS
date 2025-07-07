import comet_ml
from src.datasets import dataset_factory
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import nn_utils, misc_utils
import torch
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

    
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/model_weights.pth").exists():
            cfg_cpy = copy.deepcopy(cfg)
            dataset, num_classes = dataset_factory.create_dataset(cfg_cpy, augmentations)
            
            model = model_factory.create_model(cfg_cpy['model'], num_classes)
            
            base_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / Path('weights/model_weights.pth')
            checkpoint = torch.load(base_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            strategy = cfg_cpy['strategy']
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout':
                dataset.inject_noise(**strategy['noise']['finetuning'])
                dataset.replace_heldout_as_train_dl()
            elif strategy['finetuning_set'] == 'CleanNoiseSplit':
                dataset.inject_noise(**strategy['noise']['pretraining'])
                clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
                if strategy['noise']['finetuning']['set'] == 'TrainClean':
                    dataset.set_trainset(clean_set, shuffle=True)
                    strategy['noise']['finetuning']['set'] = 'Train'
                    dataset.inject_noise(**strategy['noise']['finetuning'])
                
                elif strategy['noise']['finetuning']['set'] == 'TrainNoise':
                    dataset.set_trainset(noisy_set, shuffle=True)
                    
            elif strategy['finetuning_set'] == 'LowLoss':
                low_loss_idxs_path = outputs_dir/ Path(f"{cfg_name}/pretrain") / f'log/low_loss_indices_{strategy['percentage']:.2f}.pkl'
                with open(low_loss_idxs_path, 'rb') as mfile:
                    low_loss_indices = pickle.load(mfile)
                all_easy_samples = [idx for class_list in low_loss_indices.values() for idx in class_list]
                
                dataset.inject_noise(**strategy['noise']['pretraining'])
                dataset.subset_set(set='Train', indices=all_easy_samples)
                
                dataset.inject_noise(**noise_tv)
                
            
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
            
            
def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
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
    gold_dir = base_expr_dir / 'gold'
    pretrain_dir = base_expr_dir / 'pretrain'
    finetune_dirs = OrderedDict()
    for idx, noise_tv in enumerate(cfg['strategy']['noise']['finetuning']):
        ft_expr_dir = base_expr_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        finetune_dirs[f"{noise_tv['noise_rate']}_{noise_tv['seed']}"] = ft_expr_dir
        
    gold_weights = torch.load(gold_dir / 'weights/model_weights.pth', map_location=cpu)
    pretrain_weights = torch.load(pretrain_dir / 'weights/model_weights.pth', map_location=cpu)
    finetune_weights = OrderedDict()
    for ft_expr, ft_dir in finetune_dirs.items():
        finetune_weights[ft_expr] = torch.load(ft_dir / 'weights/model_weights.pth', map_location=cpu)
    
    base_model.load_state_dict(pretrain_weights)
    
    
    finetune_tvs = OrderedDict()
    for ft_expr, ft_weight in finetune_weights.items():
        finetune_tvs[ft_expr] = TaskVector(pretrain_weights, ft_weight)
        
    ft_tvs_list = list(finetune_tvs.values())
    
    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
    
    otrh_tvs, shrd_tvs = TaskVector.decompose_task_vectors_SVD(ft_tvs_list)
    
    orth_task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = otrh_tvs[i]
        orth_task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = otrh_tvs[j]
            cos_sim = anchor_tv.cosine_similarity(other_tv)
            orth_task_sim[i].append(cos_sim)
    orth_task_sim = np.array(orth_task_sim)
    
    shrd_tvs_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = shrd_tvs[i]
        shrd_tvs_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = shrd_tvs[j]
            cos_sim = anchor_tv.cosine_similarity(other_tv)
            shrd_tvs_sim[i].append(cos_sim)
    shrd_tvs_sim = np.array(shrd_tvs_sim)
    
    
    for ft_name, ft_tv in finetune_tvs.items():
        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=ft_tv,
            search_range=(-1.5, 0.0),
            dataset=dataset,
            num_classes=num_classes,
            device=gpu
        )
        print(f"Best scaling coefficient for {ft_name} = {best_coef}")
        print(f"Metrics of the negated model is {best_results}")
            
    # class_names = list(finetune_tvs.keys())
    # misc_utils.plot_confusion_matrix(cm=task_sim, class_names=class_names, filepath=None, show=True)
    # misc_utils.plot_confusion_matrix(cm=orth_task_sim, class_names=class_names, filepath=None, show=True)
    # misc_utils.plot_confusion_matrix(cm=shrd_tvs_sim, class_names=class_names, filepath=None, show=True)
    
    


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
        "--tv",
        help="Apply task vectors to an already trained and finetuned experiment.",
        action="store_true",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/pretrain_on_noisy') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/pretrain_on_noisy").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment/pretrain_on_noisy").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    else:
        pt_ft_model(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)