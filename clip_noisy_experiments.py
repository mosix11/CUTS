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
from WD_analysis import apply_WD_analysis, apply_WD_antitask_analysis

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



def finetune_models_gt(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    dataset_cfg = cfg['datasets'][0]
    noise_cfg = dataset_cfg.pop('noise_cfg')
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    
    base_dataset.inject_noise(**noise_cfg)
    
    
    if not outputs_dir.joinpath(f"{cfg_name}/mix/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
            
        experiment_name = f"{cfg_name}/mix"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        
        finetuning_cfg = None
        if 'mix' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['mix']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
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
        
        finetuning_cfg = None
        if 'clean' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['clean']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
        
    if not outputs_dir.joinpath(f"{cfg_name}/noise/weights/ft_weights.pth").exists():
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)  
        
        clean_set, noisy_set = dataset.get_clean_noisy_subsets(set='Train')
        dataset.set_trainset(noisy_set, shuffle=True)
            
        experiment_name = f"{cfg_name}/noise"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        finetuning_cfg = None
        if 'noise' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['noise']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))        
  
def show_first_nine(dataset):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image

    def _find_classes(ds):
        # Walk through wrappers (e.g., Subset or your NoisyClassificationDataset)
        seen = set()
        cur = ds
        for _ in range(5):
            if id(cur) in seen:
                break
            seen.add(id(cur))
            if hasattr(cur, "classes") and isinstance(cur.classes, (list, tuple)):
                return list(cur.classes)
            cur = getattr(cur, "dataset", None)
            if cur is None:
                break
        return None

    classes = _find_classes(dataset)

    def _to_numpy_image(x):
        # Accepts PIL.Image or torch.Tensor
        if isinstance(x, Image.Image):
            arr = np.array(x)
        elif torch.is_tensor(x):
            t = x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):  # CHW -> HWC
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            arr = np.array(x)

        # For floats, just clip to [0,1] so imshow can display (no denorm)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)

        # If single-channel with trailing dim, squeeze it
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        return arr

    n = min(9, len(dataset))
    cols, rows = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
    axes = axes.ravel()

    for i in range(9):
        ax = axes[i]
        if i < n:
            sample = dataset[i]
            img, label = sample[:2]
            if torch.is_tensor(label):
                label = label.item()

            title = str(label)
            if classes is not None and isinstance(label, (int, np.integer)) and 0 <= label < len(classes):
                title = f"{label}: {classes[label]}"

            ax.imshow(_to_numpy_image(img))
            ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.show()  
    
def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    
    dataset_cfg = cfg['datasets'][0]
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
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
        
        
        finetuning_cfg = None
        if 'mix' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['mix']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
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
        
        finetuning_cfg = None
        if 'clean' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['clean']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
    # Finetune on the set we use for noise vectors but with uncorrupted labels.
    if not outputs_dir.joinpath(f"{cfg_name}/finetune_clean/weights/ft_weights.pth").exists() and strategy['finetuning_set'] == 'Heldout':
        dataset = copy.deepcopy(base_dataset)
        model = copy.deepcopy(base_model)
        
        mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
        checkpoint = torch.load(mix_model_ckp_path)
        model.load_state_dict(checkpoint)
        
        
        noise_tv = strategy['noise']['finetuning'][0]
        # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
        if noise_tv['noise_type'] == 'asymmetric':
            noise_tv['set'] = 'Heldout'
            dataset.inject_noise(**noise_tv)
            hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
            dataset.switch_labels_to_clean(hs_noisy)
            
            dataset.set_trainset(hs_noisy, shuffle=True)
        else:
            dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
        
        experiment_name = f"{cfg_name}/finetune_clean"
        experiment_dir = outputs_dir / Path(experiment_name)

        weights_dir = experiment_dir / Path("weights")
        weights_dir.mkdir(exist_ok=True, parents=True)

        plots_dir = experiment_dir / Path("plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        finetuning_cfg = None
        if 'heldout' in cfg['trainer']['finetuning']:
            finetuning_cfg = cfg['trainer']['finetuning']['heldout']
            finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
        else: finetuning_cfg = cfg['trainer']['finetuning']
        
        trainer = StandardTrainer(
            outputs_dir=outputs_dir,
            **finetuning_cfg,
            exp_name=experiment_name,
            exp_tags=None,
        )
        
        results = trainer.fit(model, dataset, resume=False)
        torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
    # # Gradient Ascent Baseline
    # if not outputs_dir.joinpath(f"{cfg_name}/gradient_ascent/weights/ft_weights.pth").exists():
    #     dataset = copy.deepcopy(base_dataset)
    #     model = copy.deepcopy(base_model)
        
    #     mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
    #     checkpoint = torch.load(mix_model_ckp_path)
    #     model.load_state_dict(checkpoint)
        
        
    #     dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
        
    #     experiment_name = f"{cfg_name}/gradient_ascent"
    #     experiment_dir = outputs_dir / Path(experiment_name)

    #     weights_dir = experiment_dir / Path("weights")
    #     weights_dir.mkdir(exist_ok=True, parents=True)

    #     plots_dir = experiment_dir / Path("plots")
    #     plots_dir.mkdir(exist_ok=True, parents=True)
        
    #     if strategy['finetuning_set'] == 'Heldout':
    #         dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
    #         dataset.inject_noise(**strategy['noise']['finetuning'][0])
            
    #     finetuning_cfg = None
    #     if 'gradient_ascent' in cfg['trainer']['finetuning']:
    #         finetuning_cfg = cfg['trainer']['finetuning']['gradient_ascent']
    #         finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
    #     else: finetuning_cfg = cfg['trainer']['finetuning']
        
    #     trainer = GradientAscentTrainer(
    #         outputs_dir=outputs_dir,
    #         **finetuning_cfg,
    #         exp_name=experiment_name,
    #         exp_tags=None,
    #     )
        
    #     results = trainer.fit(model, dataset, resume=False)
    #     torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))
        
        
    for idx, noise_tv in enumerate(strategy['noise']['finetuning']):
        if not outputs_dir.joinpath(f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}/weights/ft_weights.pth").exists():
            dataset = copy.deepcopy(base_dataset)
            model = copy.deepcopy(base_model)
            
            mix_model_ckp_path = outputs_dir/ Path(f"{cfg_name}/mix") / Path('weights/ft_weights.pth')
            checkpoint = torch.load(mix_model_ckp_path)
            model.load_state_dict(checkpoint)
            
            experiment_name = f"{cfg_name}/finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
            experiment_dir = outputs_dir / Path(experiment_name)

            weights_dir = experiment_dir / Path("weights")
            weights_dir.mkdir(exist_ok=True, parents=True)

            plots_dir = experiment_dir / Path("plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            if strategy['finetuning_set'] == 'Heldout':
                # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
                if noise_tv['noise_type'] == 'asymmetric':
                    noise_tv['set'] = 'Heldout'
                    dataset.inject_noise(**noise_tv)
                    hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
                    dataset.set_trainset(hs_noisy, shuffle=True)
                else:
                    dataset.set_trainset(dataset.get_heldoutset(), shuffle=True)
                    dataset.inject_noise(**noise_tv)
                
            finetuning_cfg = None
            if 'noise' in cfg['trainer']['finetuning']:
                finetuning_cfg = cfg['trainer']['finetuning']['noise']
                finetuning_cfg['comet_api_key'] =  os.getenv("COMET_API_KEY")
            else: finetuning_cfg = cfg['trainer']['finetuning']
            trainer = StandardTrainer(
                outputs_dir=outputs_dir,
                **finetuning_cfg,
                exp_name=experiment_name,
                exp_tags=None,
            )
            
            results = trainer.fit(model, dataset, resume=False)
            torch.save(model.state_dict(), weights_dir / Path("ft_weights.pth"))  
            


def apply_tv_gt(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    dataset_seed = cfg['dataset_seed']
    if training_seed:
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
    
    cpu = trainer_utils.get_cpu_device()
    gpu = trainer_utils.get_gpu_device()
    
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
        

    
    dataset_cfg = cfg['datasets'][0]
    noise_cfg = dataset_cfg.pop('noise_cfg')
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: dataset.get_class_names()} 
    model = model_factory.create_model(cfg['model'])
    model.freeze_all_heads()
    
    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
    dataset_cfg['train_transforms'] = model.get_val_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    dataset.reset_train_dl(shuffle=False)
    
    dataset_clean = copy.deepcopy(dataset)
    
    dataset.inject_noise(**noise_cfg)

    ft_weights = OrderedDict()


    # Load finetuned weights while removing classifier weights from the state dict
    ft_weights['mix'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_weights['clean'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_weights['noise'] = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"{cfg_name}/noise/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in ft_weights.items():
        task_vectors[task_name] = TaskVector(pt_weights, finetuend_weights)

    sum_clean_noise_TV = TaskVector.sum([task_vectors['clean'], task_vectors['noise']])
    avg_clean_noise_TV = TaskVector.mean([task_vectors['clean'], task_vectors['noise']])
    goal_TV = task_vectors['mix'] - (task_vectors['noise'])
    
    
    ft_tvs_list = list(task_vectors.values())
    tv_names = list(task_vectors.keys())
    ft_tvs_list.extend([sum_clean_noise_TV, avg_clean_noise_TV, goal_TV])
    tv_names.extend(['Sum TV', 'Avg TV', 'Goal TV'])
    
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
    
    
    

    # results_dict = OrderedDict()
    # for alpha in tqdm(np.round(np.linspace(0.1, 1.0, 10), 1)):
    
    #     model.load_state_dict(pt_weights, strict=False)
    #     task_vectors['mix'].apply_to(model, scaling_coef=alpha, strict=False)
    #     tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    #     tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

    #     results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    
    # with open(results_dirs['metrics'] / 'metrics_mtl.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)

    
    # clean_train_ds, noisy_train_ds = dataset.get_clean_noisy_subsets('Train')
    # subset_size  = 2048
    # def random_subset(ds, k, seed: int):
    #     k = min(k, len(ds))
    #     g = torch.Generator().manual_seed(seed)
    #     idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    #     return Subset(ds, idx)

    # clean_subset = random_subset(clean_train_ds, subset_size, dataset_seed)
    # noisy_subset = random_subset(noisy_train_ds, subset_size, dataset_seed + 1)
    
    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_analysis(
    #     model=model,
    #     taskvector1=task_vectors['clean'],
    #     support_tv1=clean_subset,
    #     taskvector2=task_vectors['noise'],
    #     support_tv2=noisy_subset,
    #     alhpa_range=(-2.0, 2.0),
    #     step=0.4,
    #     batch_size=512,
    #     device=gpu
    # )
    # with open(results_dir / "WD.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)

    subset_size  = 2048
    def random_subset(ds, k, seed: int):
        k = min(k, len(ds))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:k].tolist()
        return Subset(ds, idx)

    test_subset = random_subset(dataset.get_testset(), subset_size, dataset_seed)
    
    wd_results = apply_WD_antitask_analysis(
        model=model,
        clean_tv=task_vectors['clean'],
        noise_tv=task_vectors['noise'],
        testset=test_subset,
        alpha_range=(0, 2.8),
        step=0.4,
        batch_size=512,
        device=gpu,
        metric='loss',
    )
    with open(results_dir / "WD_AT2.pkl", "wb") as f:
        pickle.dump(wd_results, f)

    # model.load_state_dict(pt_weights, strict=False)
    # sum_clean_noise_TV.apply_to(model, scaling_coef=1.0, strict=False)
    # sum_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(sum_test_results)
    # sum_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    # print(sum_train_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # avg_clean_noise_TV.apply_to(model, scaling_coef=1.0, strict=False)
    # avg_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(avg_test_results)
    # avg_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    # print(avg_train_results)
    
    
    # model.load_state_dict(pt_weights, strict=False)
    # pt_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # pt_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    # model.load_state_dict(ft_weights['mix'], strict=False)
    # mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    # model.load_state_dict(ft_weights['noise'], strict=False)
    # noise_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # noise_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    # model.load_state_dict(ft_weights['clean'], strict=False)
    # clean_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # clean_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    
    # model.load_state_dict(pt_weights, strict=False)
    
    # results_dict = OrderedDict()
    
    # results_dict['Pretrain'] = {'test_results': pt_test_results, 'train_results': pt_train_results}
    # results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
    # results_dict['Clean'] = {'test_results': clean_test_results, 'train_results': clean_train_results}
    # results_dict['Noise'] = {'test_results': noise_test_results, 'train_results': noise_train_results}
    

    # for alpha in tqdm(np.round(np.linspace(-0.1, -1.0, 10), 1)):
    
    #     model.load_state_dict(ft_weights['mix'], strict=False)
    #     task_vectors['noise'].apply_to(model, scaling_coef=alpha, strict=False)
    #     tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    #     tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

    #     results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
    
    # with open(results_dirs['metrics'] / 'metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)
    
    # with open(results_dir / 'tv_metrics.json' , 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)


def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    training_seed = cfg['training_seed']
    dataset_seed = cfg['dataset_seed']
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
    
    
    dataset_cfg = cfg['datasets'][0]
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    

    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: dataset.get_class_names()} 
    model = model_factory.create_model(cfg['model'])
    model.freeze_all_heads()
    
    pt_weights = copy.deepcopy(model.state_dict())
    pt_weights = OrderedDict((k, v) for k, v in pt_weights.items() if "classifier_heads" not in k)
    
    dataset_cfg['train_transforms'] = model.get_val_transforms()
    dataset_cfg['val_transforms'] = model.get_val_transforms()
    dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    
    dataset.reset_train_dl(shuffle=False)
    
    
    strategy = cfg['strategy']
    noise_tv = strategy['noise']['finetuning'][0]
    noise_tv['set'] = 'Heldout'
    # For asymmetric noise, we only consider the noisy samples (only a subset of classes are swapped.)
    if noise_tv['noise_type'] == 'asymmetric':
        dataset.inject_noise(**noise_tv)
        hs_clean, hs_noisy = dataset.get_clean_noisy_subsets(set='Heldout')
        dataset.switch_labels_to_clean(hs_noisy)
        
        dataset.set_heldoutset(hs_noisy, shuffle=False)
    
        dataset_clean = copy.deepcopy(dataset)
    
        dataset.inject_noise(**strategy['noise']['pretraining'])
        ho_set = dataset.get_heldoutset()
        dataset.switch_labels_to_noisy(ho_set)
        dataset.set_heldoutset(ho_set)
    else:
        dataset_clean = copy.deepcopy(dataset)
        dataset.inject_noise(**strategy['noise']['pretraining'])
        dataset.inject_noise(**noise_tv)

    # Load weights while removing classifier weights from the state dict
    mix_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"mix/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    gold_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    ft_ho_clean_weights = OrderedDict(
    (k, v) for k, v in torch.load(
        outputs_dir.joinpath(f"finetune_clean/weights/ft_weights.pth"),
        map_location='cpu'
    ).items() if "classifier_heads" not in k)
    
    
    # ft_gradient_ascent_weights = OrderedDict(
    # (k, v) for k, v in torch.load(
    #     outputs_dir.joinpath(f"gradient_ascent/weights/ft_weights.pth"),
    #     map_location='cpu'
    # ).items() if "classifier_heads" not in k)
    
    noise_weights = OrderedDict()
    
    for noise_tv in strategy['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = OrderedDict(
        (k, v) for k, v in torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        ).items() if "classifier_heads" not in k)
        noise_weights[f"Seed {noise_tv['seed']}"] = n_weights
    
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average'] = only_tv
    else:
        task_vectors['Average'] = TaskVector.mean(task_vectors)
        
    # task_vectors['Average Pruned 0.4'] = task_vectors['Average'].prune_small_weights(rate=0.4)
    # task_vectors['Average Pruned 0.6'] = task_vectors['Average'].prune_small_weights(rate=0.6)
    # task_vectors['Average Pruned 0.8'] = task_vectors['Average'].prune_small_weights(rate=0.8)
    # task_vectors['Average Pruned 0.9'] = task_vectors['Average'].prune_small_weights(rate=0.9)
    # task_vectors['Average Pruned 0.95'] = task_vectors['Average'].prune_small_weights(rate=0.95)
    # task_vectors['Average Pruned 0.99'] = task_vectors['Average'].prune_small_weights(rate=0.99)
    
    task_vectors['Clean'] = TaskVector(mix_weights, ft_ho_clean_weights)
    task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    
    task_vectors['Random Vector'] = task_vectors['Average'].generate_random_vector_with_same_layer_norms(seed=training_seed)


    
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


    



    
    # model.load_state_dict(mix_weights, strict=False)
    # fig_comp_pt = embedding_space_analysis.all_plot_comp(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_comp_pt.savefig(results_dirs['embed_plots'] / "comp_pt.png", bbox_inches="tight")
    
    
    # task_vectors['Average'].apply_to(model, scaling_coef=-1.0, strict=False)
    # fig_comp_AVG_1 = embedding_space_analysis.all_plot_comp(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_comp_AVG_1.savefig(results_dirs['embed_plots'] / "comp_avg_tv.png", bbox_inches="tight")
    
    
    # model.load_state_dict(gold_weights, strict=False)
    # fig_comp_gold = embedding_space_analysis.all_plot_comp(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_comp_gold.savefig(results_dirs['embed_plots'] / "comp_gold.png", bbox_inches="tight")
    
    # model.load_state_dict(mix_weights, strict=False)
    # fig_umap_pt = embedding_space_analysis.umap_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    #     n_neighbors=5,
    #     min_dist=1.0
    # )
    
    # fig_umap_pt.savefig(results_dirs['embed_plots'] / "umap_pt.png", bbox_inches="tight")
    
    # task_vectors['Average'].apply_to(model, scaling_coef=-1.0, strict=False)
    # fig_umap_AVG_1 = embedding_space_analysis.umap_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    #     n_neighbors=5,
    #     min_dist=1.0
    # )
    
    # fig_umap_AVG_1.savefig(results_dirs['embed_plots'] / "umap_avg_tv.png", bbox_inches="tight")
    
    
    # model.load_state_dict(gold_weights, strict=False)
    # fig_umap_gold = embedding_space_analysis.umap_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    #     n_neighbors=5,
    #     min_dist=1.0
    # )
    
    # fig_umap_gold.savefig(results_dirs['embed_plots'] / "umap_gold.png", bbox_inches="tight")
    
    
    # model.load_state_dict(mix_weights, strict=False)
    # fig_tsne_pt = embedding_space_analysis.tsne_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_tsne_pt.savefig(results_dirs['embed_plots'] / "tsne_pt.png", bbox_inches="tight")
    
    # task_vectors['Average'].apply_to(model, scaling_coef=-1.0, strict=False)
    # fig_tsne_AVG_1 = embedding_space_analysis.tsne_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_tsne_AVG_1.savefig(results_dirs['embed_plots'] / "tsne_avg_tv.png", bbox_inches="tight")
    
    
    # model.load_state_dict(gold_weights, strict=False)
    # fig_tsne_gold = embedding_space_analysis.tsne_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    # )
    
    # fig_tsne_gold.savefig(results_dirs['embed_plots'] / "tsne_gold.png", bbox_inches="tight")
    
    
    # model.load_state_dict(mix_weights, strict=False)
    # fig_pca_pt = embedding_space_analysis.pca_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names()
    # )
    
    # fig_pca_pt.savefig(results_dirs['embed_plots'] / "pca_pt.png", bbox_inches="tight")
    
    # task_vectors['Average'].apply_to(model, scaling_coef=-1.0, strict=False)
    # fig_pca_AVG_1 = embedding_space_analysis.pca_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names()
    # )
    
    # fig_pca_AVG_1.savefig(results_dirs['embed_plots'] / "pca_avg_tv.png", bbox_inches="tight")
    
    
    # def fig_to_rgb(fig):
    #     """Return an (H, W, 3) uint8 array from a Matplotlib Figure for any backend."""
    #     fig.canvas.draw()
    #     w, h = fig.canvas.get_width_height()

    #     # Try backends that support RGB directly (Agg, etc.)
    #     try:
    #         buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #         return buf.reshape(h, w, 3)
    #     except AttributeError:
    #         # TkAgg gives ARGB; convert to RGB
    #         buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    #         # ARGB -> RGB by dropping alpha and reordering
    #         return buf[:, :, 1:4]
    
    # def combine_figures(figs, ncols=3, nrows=2, figsize=(15, 10)):
    #     fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    #     for ax, f in zip(axes.flat, figs):
    #         img = fig_to_rgb(f)
    #         ax.imshow(img)
    #         ax.axis("off")
    #     for ax in axes.flat[len(figs):]:
    #         ax.axis("off")
    #     plt.tight_layout()
    #     return fig

    # def make_gif(figs, out_path="progress.gif", duration=0.8):
    #     frames = [fig_to_rgb(f) for f in figs]
    #     # Per-frame durations in seconds
    #     with imageio.get_writer(out_path, mode="I", loop=0, duration=duration) as w:
    #         for fr in frames:
    #             w.append_data(fr)
            
    # figs_pca = []
    # for alpha in np.round(np.linspace(0.0, -4.0, 9), 1):
    #     model.load_state_dict(mix_weights, strict=False)
    #     if alpha != 0.0:
    #         task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
    #     fig_pca = embedding_space_analysis.pca_plot(
    #         feature_extractor=model.get_image_encoder(),
    #         dataloader=dataset_clean.get_train_dataloader(),
    #         device=gpu,
    #         class_names=dataset_clean.get_class_names(),
    #         dataset_name=dataset_clean.dataset_name
    #     )
    #     figs_pca.append(fig_pca)
        
    # big_fig = combine_figures(figs_pca, ncols=3, nrows=3)
    # big_fig.savefig(results_dirs['embed_plots'] / "pca_evol_tr.png", bbox_inches="tight")
    
    # make_gif(figs_pca, results_dirs['embed_plots'] / "pca_evol_tr.gif", duration=5.0)
    
    
    # figs_pca = []
    # for alpha in np.round(np.linspace(0.0, -4.0, 9), 1):
    #     model.load_state_dict(mix_weights, strict=False)
    #     if alpha != 0.0:
    #         task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
    #     fig_pca = embedding_space_analysis.pca_plot(
    #         feature_extractor=model.get_image_encoder(),
    #         dataloader=dataset_clean.get_heldout_dataloader(),
    #         device=gpu,
    #         class_names=dataset_clean.get_class_names(),
    #         dataset_name=dataset_clean.dataset_name
    #     )
    #     figs_pca.append(fig_pca)
        
    # big_fig = combine_figures(figs_pca, ncols=3, nrows=3)
    # big_fig.savefig(results_dirs['embed_plots'] / "pca_evol_ho.png", bbox_inches="tight")
    
    # make_gif(figs_pca, results_dirs['embed_plots'] / "pca_evol_ho.gif", duration=5.0)

    
    # model.load_state_dict(gold_weights, strict=False)
    # fig_pca_gold = embedding_space_analysis.pca_plot(
    #     feature_extractor=model.get_image_encoder(),
    #     dataloader=dataset_clean.get_train_dataloader(),
    #     device=gpu,
    #     class_names=dataset.get_class_names(),
    #     dataset_name=dataset_clean.dataset_name
    # )
    
    # fig_pca_gold.savefig(results_dirs['embed_plots'] / "pca_gold.png", bbox_inches="tight")

    # exit()
    
    


    
    
    results_dict = OrderedDict()
    if not results_dir.joinpath('metrics.json').exists():
        target_vectors = copy.deepcopy(task_vectors)
        target_vectors.pop('Random Vector')
        target_vectors.pop('Clean')
        target_vectors.pop('Mix')
        
        model.load_state_dict(mix_weights, strict=False)
        mix_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        mix_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        
        model.load_state_dict(gold_weights, strict=False)
        gold_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        gold_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        model.load_state_dict(ft_ho_clean_weights, strict=False)
        ft_ho_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        ft_ho_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        
        results_dict['Mix'] = {'test_results': mix_test_results, 'train_results': mix_train_results}
        results_dict['Gold'] = {'test_results': gold_test_results, 'train_results': gold_train_results}
        results_dict['FT HO Clean'] = {'test_results': ft_ho_test_results, 'train_results': ft_ho_train_results}
        
        
        for tv_name, target_vector in target_vectors.items():
            results_dict[tv_name] = OrderedDict()
            
            if strategy['noise']['finetuning'][0]['noise_type'] == 'asymmetric':
                alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
            else:
                # for alpha in tqdm(np.linspace(-0.05, -1.5, 30)):
                # for alpha in tqdm(np.linspace(-0.1, -2.0, 20)):
                # for alpha in tqdm(np.linspace(-0.1, -1.5, 15)):
                alphas = tqdm(np.round(np.linspace(-0.05, -3.0, 60), 2))
            for alpha in alphas:
                
                model.load_state_dict(mix_weights, strict=False)
                target_vector.apply_to(model, scaling_coef=alpha, strict=False)
                tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
                tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
                # tv_ho_results, _, _  = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)

                # results_dict[tv_name][alpha] = {'test_results': tv_test_results, 'heldout_resutls':tv_ho_results, 'train_results': tv_train_results}
                results_dict[tv_name][alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    if 'alpha_kNN' not in results_dict:        
        from test_alpha import select_alpha_star, plot_alpha_metrics
        best, records, alpha_best = select_alpha_star(
            model=model,
            feature_extractor=model.get_image_encoder(),
            classifier=model.get_active_head(),
            state0=mix_weights,
            taskvector=task_vectors['Average'],
            unlabeled_loader=dataset_clean.get_heldout_dataloader(),
            # K=dataset.get_num_classes(),
            alphas=np.round(np.linspace(-0.1, -2.0, 20), 1),
            device=gpu
        )
        alpha_kNN = alpha_best['alpha_kNN']
        alpha_s4 = alpha_best['alpha_S4']

        results_dict['alpha_KNN'] = alpha_kNN
        results_dict['alpha_s4'] = alpha_s4
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)

    
    exit()

    # Weight Space Disentanglemet Analysis
    # clean_train_ds, noisy_train_ds = dataset.get_clean_noisy_subsets('Train')
    # subset_size  = 2048
    # def random_subset(ds, k, seed: int):
    #     k = min(k, len(ds))
    #     g = torch.Generator().manual_seed(seed)
    #     idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    #     return Subset(ds, idx)

    # clean_subset = random_subset(clean_train_ds, subset_size, dataset_seed)
    # noisy_subset = random_subset(noisy_train_ds, subset_size, dataset_seed + 1)
    
    records = []
    for a_str, res in results_dict['Seed 10'].items():
        if a_str in ['Mix', 'Gold', 'FT HO Clean']: continue
        a = float(a_str) if not isinstance(a_str, (int, float)) else a_str
        test_acc  = res["test_results"]["ACC"]
        test_loss = res["test_results"]["Loss"]
        noisy_acc = res["train_results"]["noisy_set"]["ACC"]
        records.append((a, test_acc, test_loss, noisy_acc))
    alpha_max_test_acc = max(records, key=lambda x: x[1])[0]
    alpha_min_test_loss = min(records, key=lambda x: x[2])[0]

    forgetting_threshold = 0.08
    alpha_forgetting_thrsh = None
    for a, _, _, noisy_acc in sorted(records, key=lambda x: x[0], reverse=True):
        if noisy_acc <= forgetting_threshold:
            alpha_forgetting_thrsh = a
            break
        
    print(
        'Alpha Max Test ACC:', alpha_max_test_acc,
        'Apha Min Test Loss:', alpha_min_test_loss,
        'Alpha Forget Threshold:', alpha_forgetting_thrsh
        )
    mix_vector = TaskVector(pt_weights, mix_weights)
    noise_vector = task_vectors['Seed 10'] * alpha_forgetting_thrsh * -1 # alpha is negative
    clean_vector = mix_vector - noise_vector
    
    # model.load_state_dict(pt_weights, strict=False)
    # clean_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(tv_test_results)
    # tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    # print(tv_train_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # noise_vector.apply_to(model, scaling_coef=1.0, strict=False)
    # tv_hot_results, _, _ = evaluate_model(model, dataset.get_heldout_dataloader(), gpu)
    # print(tv_hot_results)
    # tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(tv_test_results)
    # tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    # print(tv_train_results)
    
    # model.load_state_dict(pt_weights, strict=False)
    # sum_tv = clean_vector + noise_vector
    # sum_tv.apply_to(model, scaling_coef=1.0, strict=False)
    # tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    # print(tv_test_results)
    # tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
    # print(tv_train_results)
    
    # exit()
    
    # model.load_state_dict(pt_weights, strict=False)
    # wd_results = apply_WD_analysis(
    #     model=model,
    #     taskvector1=clean_vector,
    #     support_tv1=clean_subset,
    #     taskvector2=noise_vector,
    #     support_tv2=noisy_subset,
    #     alhpa_range=(-3.0, 3.0),
    #     step=0.3,
    #     batch_size=512,
    #     device=gpu
    # )
    # with open(results_dir / "WD.pkl", "wb") as f:
    #     pickle.dump(wd_results, f)
    
    subset_size  = 2048
    def random_subset(ds, k, seed: int):
        k = min(k, len(ds))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:k].tolist()
        return Subset(ds, idx)

    test_subset = random_subset(dataset.get_testset(), subset_size, dataset_seed)
    
    clean_vector.apply_to(model, scaling_coef=1.0, strict=False)
    noise_vector.apply_to(model, scaling_coef=1.0, strict=False)
    tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
    print(tv_test_results)

    

    model.load_state_dict(pt_weights, strict=False)
    wd_results = apply_WD_antitask_analysis(
        model=model,
        clean_tv=clean_vector,
        noise_tv=noise_vector,
        testset=test_subset,
        alpha_range=(0, 2),
        step=0.1,
        batch_size=512,
        device=gpu,
        metric='error',
    )
    with open(results_dir / "WD_AT2_acc.pkl", "wb") as f:
        pickle.dump(wd_results, f)
    
    


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
        help="Finetune the image encoder with forzen heads on noisy datasets.",
        action="store_true",
    )
    
    parser.add_argument(
        "-g",
        "--groundtruth",
        help="Finetune the image encoder with forzen heads on noisy datasets using ground truth noise.",
        action="store_true",
    )
    
    parser.add_argument(
        "-t",
        "--tv",
        help="Apply task vectors to an already trained and finetuned model.",
        action="store_true",
    )
    
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment/clip_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/clip_noise_TA").absolute()
    results_dir = Path("results/single_experiment/clip_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])

        
    if args.finetune and args.groundtruth:
        finetune_models_gt(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv and args.groundtruth:
        apply_tv_gt(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
    elif args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

if __name__ == "__main__":
    main()