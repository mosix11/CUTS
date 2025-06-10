import comet_ml
from src.datasets import MNIST, CIFAR10, FashionMNIST, MoGSynthetic
from src.models import FC1, CNN5, make_resnet18k, FCN, TaskVector
from src.trainers import TrainerEp, TrainerGS
import matplotlib.pyplot as plt
from src.utils import nn_utils, misc_utils
import torch
import torchmetrics
import torchvision.transforms.v2 as transformsv2
from functools import partial
from pathlib import Path
import pickle
import argparse
import os
import dotenv
import numpy as np
import random
from torchmetrics import ConfusionMatrix
from tqdm import tqdm
import yaml
from PIL import Image
import copy
import json
import pandas as pd


# def plot_multiple_confusion_matrices(
#     filepath1,
#     filepath2,
#     filepath3,
#     title1=None,
#     title2=None,
#     title3=None,
#     main_title='Combined Confusion Matrices',
#     save_filepath=None,
#     show=True
# ):
#     """
#     Combines and displays three confusion matrix plots vertically in a single figure.

#     Args:
#         filepath1 (str): Path to the first confusion matrix image.
#         filepath2 (str): Path to the second confusion matrix image.
#         filepath3 (str): Path to the third confusion matrix image.
#         title1 (str, optional): Title for the first confusion matrix.
#         title2 (str, optional): Title for the second confusion matrix.
#         title3 (str, optional): Title for the third confusion matrix.
#         main_title (str): Overall title for the combined figure. Defaults to 'Combined Confusion Matrices'.
#         save_filepath (str, optional): Path to save the combined figure. If None, the figure is not saved.
#         show (bool): If True, display the combined figure. Defaults to True.
#     """
#     try:
#         img1 = Image.open(filepath1)
#         img2 = Image.open(filepath2)
#         img3 = Image.open(filepath3)
#     except FileNotFoundError as e:
#         print(f"Error: One or more confusion matrix image files not found. {e}")
#         return

#     fig, axes = plt.subplots(3, 1, figsize=(10, 18)) # 3 rows, 1 column, adjust figsize as needed

#     # Plot the first confusion matrix
#     axes[0].imshow(img1)
#     axes[0].axis('off')  # Turn off axis labels and ticks for the image
#     if title1:
#         axes[0].text(-0.1, 0.5, title1, transform=axes[0].transAxes,
#                      fontsize=12, va='center', ha='right', rotation=90) # Vertical title on the left

#     # Plot the second confusion matrix
#     axes[1].imshow(img2)
#     axes[1].axis('off')
#     if title2:
#         axes[1].text(-0.1, 0.5, title2, transform=axes[1].transAxes,
#                      fontsize=12, va='center', ha='right', rotation=90)

#     # Plot the third confusion matrix
#     axes[2].imshow(img3)
#     axes[2].axis('off')
#     if title3:
#         axes[2].text(-0.1, 0.5, title3, transform=axes[2].transAxes,
#                      fontsize=12, va='center', ha='right', rotation=90)

#     fig.suptitle(main_title, fontsize=16, y=1.02) # Overall title at the top
#     plt.tight_layout(rect=[0.05, 0.03, 1, 0.98]) # Adjust layout to make space for main title and side titles

#     if save_filepath:
#         plt.savefig(save_filepath, bbox_inches='tight')

#     if show:
#         plt.show()
    
#     plt.close() # Close the figure to free memory

def plot_multiple_confusion_matrices(
    filepaths,
    titles=None,
    results=None,
    main_title='Combined Confusion Matrices',
    save_filepath=None,
    show=True
):
    """
    Combines and displays multiple confusion matrix plots vertically in a single figure,
    with an optional table of performance results at the bottom.

    Args:
        filepaths (list): A list of paths to the confusion matrix images.
        titles (list, optional): A list of titles for each confusion matrix.
                                 Must be the same length as filepaths.
        results (list, optional): A list of dictionaries, where each dictionary
                                  specifies the performance metrics for a model
                                  associated with a confusion matrix.
                                  Each dictionary should be in the format:
                                  {'ACC': float, 'Loss': float, 'F1': float}.
                                  Must be the same length as filepaths.
        main_title (str): Overall title for the combined figure. Defaults to 'Combined Confusion Matrices'.
        save_filepath (str, optional): Path to save the combined figure. If None, the figure is not saved.
        show (bool): If True, display the combined figure. Defaults to True.
    """
    num_matrices = len(filepaths)
    if titles and len(titles) != num_matrices:
        raise ValueError("The number of titles must match the number of filepaths.")
    if results and len(results) != num_matrices:
        raise ValueError("The number of results dictionaries must match the number of filepaths.")

    images = []
    try:
        for fp in filepaths:
            images.append(Image.open(fp))
    except FileNotFoundError as e:
        print(f"Error: One or more confusion matrix image files not found. {e}")
        return

    # Determine figure height based on number of matrices and if a table is included
    fig_height_per_matrix = 6
    table_height = 0
    if results:
        # Estimate height needed for the table (adjust as needed for more padding)
        table_height = 0.6 * len(results) + 1.8 # Increased basic estimate per row + header
        fig, axes = plt.subplots(num_matrices + 1, 1, figsize=(10, num_matrices * fig_height_per_matrix + table_height))
    else:
        fig, axes = plt.subplots(num_matrices, 1, figsize=(10, num_matrices * fig_height_per_matrix))


    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis labels and ticks for the image
        if titles and titles[i]:
            axes[i].text(-0.1, 0.5, titles[i], transform=axes[i].transAxes,
                         fontsize=12, va='center', ha='right', rotation=90) # Vertical title on the left

    if results:
        # Prepare data for the table
        df = pd.DataFrame(results)
        # Add a 'Model' column for row labels if titles are provided
        if titles:
            df.insert(0, 'Model', titles)
        else:
            df.insert(0, 'Model', [f'Model {j+1}' for j in range(num_matrices)])

        # Format numerical columns to 3 decimal places
        for col in df.columns:
            # Check if the column can be converted to numeric before formatting
            # This avoids errors if 'Model' column or other non-numeric columns are present
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f"{x:.3f}")

        # Create the table
        table_ax = axes[-1] # The last subplot for the table
        table_ax.axis('off') # Turn off axis for the table plot
        
        # Adjust column widths dynamically
        num_columns = len(df.columns)
        col_widths = [1.0 / num_columns] * num_columns

        table = table_ax.table(cellText=df.values,
                               colLabels=df.columns,
                               loc='center',
                               cellLoc='center',
                               colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5) # Adjust table scale for more vertical padding (increased from 1.2 to 1.5)

        # Corrected way to set specific cell properties for padding
        for key, cell in table.get_celld().items():
            cell.set_height(0.1)  # You can adjust this value to control vertical padding

    fig.suptitle(main_title, fontsize=16, y=0.99 if results else 1.02) # Overall title at the top, adjusted if table exists
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96 if results else 0.98]) # Adjust layout to make space for main title and side titles and potentially table

    if save_filepath:
        plt.savefig(save_filepath, bbox_inches='tight')

    if show:
        plt.show()
    
    plt.close() # Close the figure to free memory

def prepare_batch(batch, device):
    batch = [tens.to(device) for tens in batch]
    return batch


def process_dataset(cfg, augmentations=None):
    cfg['dataset']['batch_size'] = cfg['trainer']['finetuning']['batch_size']
    del cfg['trainer']['finetuning']['batch_size']
    dataset_name = cfg['dataset'].pop('name')
    cfg['dataset']['augmentations'] = augmentations if augmentations else []
    
    if dataset_name == 'mnist':
        pass
    elif dataset_name == 'cifar10':
        num_classes = cfg['dataset'].pop('num_classes')
        dataset = CIFAR10(
            **cfg['dataset']
        )
    elif dataset_name == 'cifar100':
        pass
    elif dataset_name == 'mog':
        pass
    else: raise ValueError(f"Invalid dataset {dataset_name}.")
    
    return dataset, num_classes



def process_model(cfg, num_classes):
    model_type = cfg['model'].pop('type')
    if cfg['model']['loss_fn'] == 'MSE':
        cfg['model']['loss_fn'] = torch.nn.MSELoss()
    elif cfg['model']['loss_fn'] == 'CE':
        cfg['model']['loss_fn'] = torch.nn.CrossEntropyLoss()
    else: raise ValueError(f"Invalid loss function {cfg['model']['loss_fn']}.")
    
    
    if cfg['model']['metrics']:
        metrics = {}
        for metric_name in cfg['model']['metrics']:
            if metric_name == 'ACC':
                metrics[metric_name] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
            elif metric_name == 'F1':
                metrics[metric_name] = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
            else: raise ValueError(f"Invalid metric {metric_name}.")
        cfg['model']['metrics'] = metrics

    if model_type == 'fc1':
        model = FC1(**cfg)
    elif model_type == 'fcN':
        model = FCN(**cfg)
    elif model_type == 'cnn5':
        model = CNN5(**cfg['model'])
    elif model_type == 'resnet18k':
        model = make_resnet18k(**cfg)
    else: raise ValueError(f"Invalid model type {model_type}.")
    
    return model



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
            
            loss = model.validation_step(input_batch, target_batch, use_amp=True)
            loss_met.update(loss.detach().cpu().item(), n=input_batch.shape[0])
            
            model_output = model.predict(input_batch)
            predictions = torch.argmax(model_output, dim=-1) 
            
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


def apply_tv(outputs_dir: Path, results_dir: Path, cfg: dict, cfg_name:str, search_range = [-1.5, 0.0]):
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
    
    dataset, num_classes = process_dataset(cfg)
    
    model_base = process_model(cfg, num_classes)
    
    cpu = nn_utils.get_cpu_device()
    gpu = nn_utils.get_gpu_device()
    
    
    pretrain_dir = outputs_dir/ Path(f"{cfg_name}_pretrain")
    finetune_dir = outputs_dir/ Path(f"{cfg_name}_finetune")
    
    base_model_ckp_path = pretrain_dir / Path('weights/model_weights.pth')
    base_model_stat_dict = torch.load(base_model_ckp_path, map_location=cpu)
    model_base.load_state_dict(base_model_stat_dict)
    
    
    ft_model_ckp_path = finetune_dir / Path('weights/model_weights.pth')
    ft_model_state_dict = torch.load(ft_model_ckp_path, map_location=cpu)
    
    task_vector = TaskVector(
        pretrained_state_dict=base_model_stat_dict,
        finetuned_state_dict=ft_model_state_dict
    )
    
    best_coef, best_results, best_cm = search_optimal_coefficient(
        base_model=model_base,
        task_vector=task_vector,
        dataset=dataset,
        search_range=search_range,
        device=gpu,
        num_classes=num_classes
    )
    
    
    with open(pretrain_dir / Path('log/results.json'), 'r') as file:
        base_model_results = json.load(file)
        
    with open(finetune_dir / Path('log/results.json'), 'r') as file:
        ft_model_results = json.load(file)
    
    
    
    class_names = [f'Class {i}' for i in range(10)]
    misc_utils.plot_confusion_matrix(cm=best_cm, class_names=class_names, filepath=results_dir / Path('confusion_matrix_tv.png'), show=False)
    
    
    results_list = [
        {
            'ACC': base_model_results['final']['Test/ACC'],
            'F1': base_model_results['final']['Test/F1'],
            'Loss': base_model_results['final']['Test/Loss'],
        },
        {
            'ACC': ft_model_results['final']['Test/ACC'],
            'F1': ft_model_results['final']['Test/F1'],
            'Loss': ft_model_results['final']['Test/Loss'],
        },
        {
            'ACC': best_results['ACC'],
            'F1': best_results['F1'],
            'Loss': best_results['Loss'],
        },   
    ]
    
    plot_multiple_confusion_matrices(
        filepaths=[pretrain_dir / Path('plots/confmat.png'), finetune_dir / Path('plots/confmat.png'), results_dir / Path('confusion_matrix_tv.png')],
        titles=["Pretrained", "Finetuned", "TV"],
        results=results_list,
        save_filepath=results_dir / Path(f"{cfg_name}_confmat_combined.png"),
        show=True
    )
    
    print(f"Best scaling coefficient = {best_coef}")
    print(f"Metrics of the negated model is {best_results}")
    
    print('pretrained results : \n', base_model_results)
    
    

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
        "-s",
        "--scale",
        help="Scale coefficient for the task vector.",
        type=float,
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    
    cfg_path = Path('configs/single_experiment').joinpath(args.config)

    if not cfg_path.exists(): raise RuntimeError('The specified config file does not exist.')
    with open(cfg_path, 'r') as file:
        cfg = yaml.full_load(file)

    outputs_dir = Path("outputs/single_experiment").absolute()
    outputs_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results/single_experiment").absolute()
    results_dir.mkdir(exist_ok=True, parents=True)
    
    apply_tv(outputs_dir, results_dir, cfg, cfg_name=cfg_path.stem)
