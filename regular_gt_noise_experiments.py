import os
PYTHON_HASH_SEED = 0
os.environ["PYTHONHASHSEED"] = str(PYTHON_HASH_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import comet_ml
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True) 
torch.set_float32_matmul_precision("high")

from src.datasets import dataset_factory, dataset_wrappers
from src.models import model_factory, TaskVector
from src.trainers import StandardTrainer, GradientAscentTrainer, utils as trainer_utils
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import misc_utils


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
from collections import OrderedDict, defaultdict
import re

import imageio.v2 as imageio

from src.utils import embedding_space_analysis
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, search_optimal_coefficient, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis

def eval_model_on_tvs(model, taskvectors, results_dict, cfg, dataset, num_classes, device):
    
    results = results_dict
    
    
    for tv_name, tv in taskvectors.items():
        results[tv_name] = OrderedDict()
        
        base_model = copy.deepcopy(model)
        results[tv_name]["-1.0"] = OrderedDict()
        tv.apply_to(base_model, scaling_coef=-1.0)
        base_test_results, _, _ = evaluate_model(base_model, dataset.get_test_dataloader(), device)
        base_train_split_results = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name]["-1.0"]['test_results'] = base_test_results
        results[tv_name]["-1.0"]['train_results'] = base_train_split_results
        
        base_model = copy.deepcopy(model)

        best_coef, best_results, best_cm = search_optimal_coefficient(
            base_model=base_model,
            task_vector=tv,
            search_range=(-2.0, 0.0),
            dataset=dataset,
            num_classes=num_classes,
            device=device
        )
        
        results[tv_name][best_coef] = OrderedDict()
        results[tv_name][best_coef]['test_results'] = best_results
        
        tv.apply_to(base_model, scaling_coef=best_coef)
        
        after_tv_metrics = eval_model_on_clean_noise_splits(base_model, cfg, dataset, device)
        results[tv_name][best_coef]['train_results'] = after_tv_metrics
        
        
    return results    


import matplotlib.pyplot as plt


def show_poisoned_samples(dataset, n=9, unnormalize=False):
    """
    Show `n` poisoned samples from a DatasetWithIndex-wrapped dataset.
    dataset: your ds.get_trainset() or similar
    unnormalize: if True, try to undo CIFAR-10 normalization for visualization
    """

    # Collect poisoned samples
    poisoned_imgs = []
    poisoned_labels = []
    for idx in range(len(dataset)):
        x, y, *_rest = dataset[idx]
        # last element in your tuple is is_poison flag
        is_poison = _rest[-1].item() if torch.is_tensor(_rest[-1]) else bool(_rest[-1])
        if is_poison:
            poisoned_imgs.append(x)
            poisoned_labels.append(y)
            if len(poisoned_imgs) >= n:
                break

    if not poisoned_imgs:
        print("No poisoned samples found!")
        return

    # CIFAR-10 normalization parameters
    cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    cifar_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

    # Plot grid
    fig, axes = plt.subplots(3, 3, figsize=(8,8))
    for ax, img, label in zip(axes.flat, poisoned_imgs, poisoned_labels):
        if unnormalize:
            img = img * cifar_std + cifar_mean
            
        img = img.clamp(0,1)
        if img.shape[0] == 1:  # grayscale
            ax.imshow(img.squeeze(0).cpu(), cmap="gray")
        else:
            ax.imshow(img.permute(1,2,0).cpu())
        ax.set_title(f"Label={label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['pretraining']['comet_api_key'] = os.getenv("COMET_API_KEY")
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    augmentations = [
        transformsv2.RandomCrop(224, padding=4),
    ]
    
    
    base_dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'], augmentations)
    base_model = model_factory.create_model(cfg['model'], num_classes)
    
    
    strategy = cfg['strategy']
    base_dataset.inject_noise(**strategy['noise']['pretraining'])

    if not outputs_dir.joinpath(f"{cfg_name}/mix/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
            
        experiment_name = f"{cfg_name}/mix"
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
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
      
    if not outputs_dir.joinpath(f"{cfg_name}/clean/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(clean_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/clean"
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
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
        
    for discovery_rate in np.linspace(0.1, 1.0, 10):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{discovery_rate}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            experiment_name = f"{cfg_name}/finetune_{discovery_rate}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
            num_noisy = len(noisy_set)
            num_discover = int(discovery_rate * num_noisy)
            rand_indices = torch.randperm(num_noisy)[:num_discover]
            noisy_discovery_set = Subset(noisy_set, rand_indices)
            dataset.set_trainset(noisy_discovery_set, shuffle=True)


            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **cfg['trainer']['finetuning'],
                exp_name=experiment_name,
                exp_tags=None,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))  
            

def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
    
    outputs_dir = outputs_dir / cfg_name
    
    results_dir = results_dir / cfg_name
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_dirs = {}
    results_dirs['cms'] = results_dir / 'confusion_mats'
    results_dirs['Ts'] = results_dir / 'transition_mats'
    results_dirs['W_norms'] = results_dir / 'weight_norms'
    results_dirs['TV_norms'] = results_dir / 'TV_norms'
    results_dirs['embed_plots'] = results_dir / 'embedding_plots'
    results_dirs['metrics'] = results_dir / 'metrics'
    for dir in results_dirs.values():
        dir.mkdir(exist_ok=True, parents=True)
    
    
    dataset, num_classes = dataset_factory.create_dataset(cfg['dataset'])
    model = model_factory.create_model(cfg['model'], num_classes)
    
    
    dataset.reset_train_dl(shuffle=False)
    
    dataset_clean = copy.deepcopy(dataset)
    
    strategy = cfg['strategy']
    dataset.inject_poison(**strategy['poison']['pretraining'])
    
    poison_tv_cfg = strategy['poison']['finetuning'][0]
    poison_tv_cfg['set'] = 'Heldout'
    dataset.inject_poison(**poison_tv_cfg)



    # Load weights while removing classifier weights from the state dict
    mix_weights = torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    )
    
    gold_weights = torch.load(
        outputs_dir.joinpath(f"clean/weights/ft_weights.pth"),
        map_location='cpu'
    )
    
    ft_ho_clean_weights = torch.load(
        outputs_dir.joinpath(f"finetune_clean/weights/ft_weights.pth"),
        map_location='cpu'
    )
    
    
    poison_weights = OrderedDict()
    
    for poison_tv in cfg['strategy']['poison']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{poison_tv['rate']}_{poison_tv['seed']}"
        n_weights = torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        )
        poison_weights[f"{poison_tv['rate']*100:.0f}% Noise, {poison_tv['seed']} Seed"] = n_weights
        
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in poison_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average TV'] = only_tv
    else:
        task_vectors['Average TV'] = TaskVector.mean(task_vectors)
        
    
    task_vectors['Clean'] = TaskVector(mix_weights, ft_ho_clean_weights)
    
    task_vectors['Random Vector'] = task_vectors['Average TV'].generate_random_vector_with_same_layer_norms(seed=11)

    
    
    
    ft_tvs_list = list(task_vectors.values())
    tv_names = list(task_vectors.keys())
    
    task_sim = []
    for i in range(len(ft_tvs_list)):
        anchor_tv = ft_tvs_list[i]
        task_sim.append([])
        for j in range(len(ft_tvs_list)):
            other_tv = ft_tvs_list[j]
            cos_sim = anchor_tv.cosine_similarity_flatten(other_tv)
            task_sim[i].append(cos_sim)
    task_sim = np.array(task_sim)
    
    misc_utils.plot_confusion_matrix(
        title='Task Vector Similarity Matrix',
        cm=task_sim,
        class_names=tv_names,
        color_map='vlag',
        color_bar=True,
        vmin= -1.0,
        vmax= 1.0,
        x_label='Task Vectors',
        y_label='Task Vectors',
        tick_label_font_size=6,
        filepath=results_dir / 'task_similarities.png',
        show=False
    )


    

    model.load_state_dict(mix_weights, strict=False)
    mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    mix_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    
    model.load_state_dict(gold_weights, strict=False)
    gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    gold_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    gold_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    model.load_state_dict(ft_ho_clean_weights, strict=False)
    ft_ho_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    ft_ho_ho_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    ft_ho_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    
    results_dict = OrderedDict()
    
    results_dict['Mix'] = {'test_results': mix_test_results, 'ho_results': mix_ho_results, 'train_results': mix_train_results}
    results_dict['Gold'] = {'test_results': gold_test_results, 'ho_results': gold_ho_results, 'train_results': gold_train_results}
    results_dict['FT HO Clean'] = {'test_results': ft_ho_test_results, 'ho_results': ft_ho_ho_results, 'train_results': ft_ho_train_results}
    
    # results_dict = OrderedDict()
    # for alpha in tqdm(np.linspace(-0.05, -1.5, 30)):
    # for alpha in tqdm(np.linspace(-0.1, -2.0, 20)):
    for alpha in tqdm(np.linspace(-0.1, -1.5, 15)):
    
        model.load_state_dict(mix_weights, strict=False)
        task_vectors['Average TV'].apply_to(model, scaling_coef=alpha, strict=False)
        tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        tv_ho_resutls, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
        tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

        results_dict[alpha] = {'test_results': tv_test_results, 'ho_results': tv_ho_resutls, 'train_results': tv_train_results}
    
    with open(results_dir / 'metrics.json' , 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    # print(results_dict)
    
    # with open(results_dir / 'tv_metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)
    
    

from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration to used for model.",
        type=str,
    )
    
    
    
    parser.add_argument(
        "-f",
        "--finetune",
        help="Finetune the image encoder with forzen heads on poisoned datasets.",
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
    
    cfg_path = Path('configs/single_experiment/regular_gt_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/regular_gt_noise_TA").absolute()
    results_dir = Path("results/single_experiment/regular_gt_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])
        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)


if __name__ == "__main__":
    main()