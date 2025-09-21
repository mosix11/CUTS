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
from helper_funcs import evaluate_model, eval_model_on_clean_noise_splits, get_confusion_matrix, row_normalize
from src.utils import weight_norm_analysis



    
def finetune_models(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str):
    cfg['trainer']['finetuning']['comet_api_key'] = os.getenv("COMET_API_KEY")
    
    base_model = model_factory.create_model(cfg['model'])
    
    dataset_cfg = cfg['dataset']
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
    
    
    
    model = model_factory.create_model(cfg['model'])
    pt_weights = copy.deepcopy(model.state_dict())
    
    dataset_cfg = cfg['dataset']
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
    
    
    noise_weights = OrderedDict()
    
    for noise_tv in cfg['strategy']['noise']['finetuning']:
        ft_expr_dir = outputs_dir / f"finetune_{noise_tv['noise_rate']}_{noise_tv['seed']}"
        n_weights = torch.load(
            ft_expr_dir.joinpath(f"weights/ft_weights.pth"),
            map_location='cpu'
        )
        noise_weights[f"{noise_tv['noise_rate']*100:.0f}% Noise, {noise_tv['seed']} Seed"] = n_weights
        
            
    task_vectors = OrderedDict()
    for task_name, finetuend_weights in noise_weights.items():
        task_vectors[task_name] = TaskVector(mix_weights, finetuend_weights)
        
    if len(task_vectors) == 1:
        only_tv = task_vectors.popitem(last=False)[1]
        task_vectors['Average TV'] = only_tv
    else:
        task_vectors['Average TV'] = TaskVector.mean(task_vectors)
        
    
    task_vectors['Clean'] = TaskVector(mix_weights, ft_ho_clean_weights)
    task_vectors['Mix'] = TaskVector(pt_weights, mix_weights)
    task_vectors['Random Vector'] = task_vectors['Average TV'].generate_random_vector_with_same_layer_norms(seed=training_seed)

    
    
    
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
    
    
    # task_vectors['Average TV'].apply_to(model, scaling_coef=-1.0, strict=False)
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
    
    # task_vectors['Average TV'].apply_to(model, scaling_coef=-1.0, strict=False)
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
    
    # task_vectors['Average TV'].apply_to(model, scaling_coef=-1.0, strict=False)
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
    
    # task_vectors['Average TV'].apply_to(model, scaling_coef=-1.0, strict=False)
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
    # for alpha in np.linspace(0.0, -2.0, 9):
    #     model.load_state_dict(mix_weights, strict=False)
    #     if alpha != 0.0:
    #         task_vectors['Average TV'].apply_to(model, scaling_coef=alpha, strict=False)
    #     fig_pca = embedding_space_analysis.pca_plot(
    #         feature_extractor=model.get_image_encoder(),
    #         dataloader=dataset_clean.get_train_dataloader(),
    #         device=gpu,
    #         class_names=dataset_clean.get_class_names(),
    #         dataset_name=dataset_clean.dataset_name
    #     )
    #     figs_pca.append(fig_pca)
        
    # big_fig = combine_figures(figs_pca, ncols=3, nrows=3)
    # big_fig.savefig(results_dirs['embed_plots'] / "pca_evol.png", bbox_inches="tight")
    
    # make_gif(figs_pca, results_dirs['embed_plots'] / "pca_evol.gif", duration=5.0)
    # exit()
    
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
        
        if strategy['noise']['finetuning'][0]['noise_type'] == 'asymmetric':
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        else:
            alphas = tqdm(np.round(np.linspace(-0.05, -2.0, 40), 2))
        for alpha in alphas:
            
            model.load_state_dict(mix_weights, strict=False)
            task_vectors['Average'].apply_to(model, scaling_coef=alpha, strict=False)
            tv_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
            tv_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)

            results_dict[alpha] = {'test_results': tv_test_results, 'train_results': tv_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    else:
        with open(results_dir / "metrics.json", "r") as json_file:
            results_dict = json.load(json_file, object_pairs_hook=OrderedDict)
            
            
    if 'alpha_KNN' not in results_dict:        
        from test_alpha import select_alpha_star, plot_alpha_metrics
        best, records, alpha_best = select_alpha_star(
            model=model,
            feature_extractor=model.get_feature_extractor(),
            classifier=model.get_classifier_head(),
            state0=mix_weights,
            taskvector=task_vectors['Average'],
            unlabeled_loader=dataset_clean.get_heldout_dataloader(),
            # K=dataset.get_num_classes(),
            alphas=np.round(np.linspace(-0.05, -2.0, 40), 2),
            device=gpu
        )
        alpha_kNN = alpha_best['alpha_kNN']
        alpha_s4 = alpha_best['alpha_S4']

        results_dict['alpha_KNN'] = alpha_kNN
        results_dict['alpha_s4'] = alpha_s4
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)

    if 'Random Vector' not in results_dict:
        model.load_state_dict(mix_weights, strict=False)
        alpha_kNN = results_dict['alpha_KNN']
        task_vectors['Random Vector'].apply_to(model, scaling_coef=alpha_kNN, strict=False)
        random_test_results, _, _ = evaluate_model(model, dataset.get_test_dataloader(), gpu)
        random_train_results = eval_model_on_clean_noise_splits(model, None, dataset, gpu)
        results_dict['Random Vector'] = {'test_results': random_test_results, 'train_results': random_train_results}
        with open(results_dir / 'metrics.json' , 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    
    


from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    ranks = trainer_utils.setup_distributed()
    
    dotenv.load_dotenv(".env")

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
        "-t",
        "--tv",
        help="Apply task vectors to an already trained and finetuned experiment.",
        action="store_true",
    )
    args = parser.parse_args()

    cfg_path = Path('configs/single_experiment/dino_noise_TA') / f"{args.config}.yaml"

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment/dino_noise_TA").absolute()
    results_dir = Path("results/single_experiment/dino_noise_TA").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)

    global_seed = cfg['global_seed']
    trainer_utils.seed_everything(base_seed=global_seed, rank=ranks['rank'])
        
    if args.finetune:
        finetune_models(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)

    if args.tv:
        apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)


if __name__ == "__main__":
    main()