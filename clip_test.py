import torch
from src.models.openai_clip import clip
from collections import OrderedDict

from src.datasets import dataset_factory
from src.models import model_factory, TaskVector

# model, preprocess_train, preprocess_val = clip.load("ViT-B/16")

openclip_weights_path = 'outputs/single_experiment/clip_noise_TA/config26/mix/weights/ft_weights.pth'

open_clip_conf = OrderedDict({
    'dataset': {
        'name': 'cifar10',
        'num_classes': 10,
        'img_size': [224, 224],
        'batch_size': 256,
        'heldout_conf': 0.02,
        'valset_ratio': 0.0,
        'num_workers': 12,
        'seed': 11
    },
    'model': {
        'type': 'open_clip_multihead_ViT-B-16',
        'pt_weights': 'openai',
        'loss_fn': {
            'type': 'CE'
        },
        'metrics': ['ACC', 'F1']
    }
})

opnenai_clip_conf = OrderedDict({
    'dataset': {
        'name': 'cifar10',
        'num_classes': 10,
        'img_size': [224, 224],
        'batch_size': 256,
        'heldout_conf': 0.02,
        'valset_ratio': 0.0,
        'num_workers': 12,
        'seed': 11
    },
    'model': {
        'type': 'clip_multihead_ViT-B/16',
        'loss_fn': {
            'type': 'CE'
        },
        'metrics': ['ACC', 'F1']
    }
})

def initialize_model_dataset(cfg: dict):
    dataset_cfg = cfg['dataset']
    
    base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
    cfg['model']['datasets_cfgs'] = {dataset_cfg['name']: base_dataset.get_class_names()} 
    base_model = model_factory.create_model(cfg['model'])
    base_model.freeze_all_heads()
    
    dataset_cfg['train_transforms'] = base_model.get_train_transforms()
    dataset_cfg['val_transforms'] = base_model.get_val_transforms()
    # base_dataset, num_classes = dataset_factory.create_dataset(dataset_cfg)
        

    return base_model, base_dataset, cfg


oc_clip, dataset, _ = initialize_model_dataset(open_clip_conf)
oa_clip, _, _ = initialize_model_dataset(opnenai_clip_conf)


torch.manual_seed(0)

# Dummy input image
x = torch.randn(2, 3, 224, 224).to('cuda:0')

# Ensure both in eval mode
oc_clip.to('cuda:0').eval()
oa_clip.to('cuda:0').eval()

with torch.no_grad():
    # Encode images
    out_oc = oc_clip(x)
    out_oa = oa_clip(x)

# Compare numerically
diff = (out_oc - out_oa).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
relative_error = diff.norm() / out_oc.norm()


print(f"Max absolute difference: {max_diff:.6e}")
print(f"Mean absolute difference: {mean_diff:.6e}")
print(f"Relative error: {relative_error:.6e}")
print(f"Sum error: {diff.sum()}")
# Optional: check if they are (almost) identical
if torch.allclose(out_oc, out_oa, atol=1e-6, rtol=1e-5):
    print("✅ The models produce identical outputs (within tolerance).")
else:
    print("⚠️ Outputs differ — investigate potential mismatch (e.g., QuickGELU or preprocessing).")

exit()


# mix_weights = OrderedDict(
#     (k, v) for k, v in torch.load(
#         openclip_weights_path,
#         map_location='cpu'
#     ).items() if "classifier_heads" not in k)

# print(mix_weights.keys())

# print('\n\n\n\n')
# print(model.state_dict().keys())

# for key in model.state_dict().keys():
#     if 'logit_scale' in key:
#         print(key)

# num_mismatch = 0
# target_list = list(model.visual.state_dict().keys())
# for key in list(mix_weights.keys()):
#     key = key.replace('image_encoder.model.visual.', '')
#     if key not in target_list:
#         print(key)
        
        
# num_mismatch = 0
# target_list = list(mix_weights.keys())
# for key in list(model.visual.state_dict().keys()):
#     key = 'image_encoder.model.visual.' + key
#     if key not in target_list:
#         print(key)












# import torch
# import open_clip


# def run_diagnostics():
#     """
#     Compares open_clip and openai-clip outputs to find discrepancies.
#     """
    
#     # --- 1. Setup ---
#     MODEL_NAME = 'ViT-B/16'
#     # Use 'cpu' to ensure a fair comparison without GPU-specific optimizations
#     # or mixed precision defaults.
#     DEVICE = "cpu" 
    
#     # A sample text list, similar to what your templates would generate
#     SAMPLE_TEXTS = [
#         "a photo of a dog",
#         "a picture of a cat",
#         "a drawing of a bird"
#     ]
    
#     print(f"--- Running Diagnostics for {MODEL_NAME} on {DEVICE} ---")

#     # --- 2. Load Models ---
#     print("\nLoading models...")
    
#     # Load OpenClip model
#     (
#         oc_model,
#         _,
#         _,
#     ) = open_clip.create_model_and_transforms(
#         'ViT-B-16', 
#         pretrained='openai', # This is key
#         device=DEVICE
#     )
    
#     # Load OpenAI model
#     (
#         oai_model, 
#         _,
#         _
#     ) = clip.load(MODEL_NAME, device=DEVICE)

#     # **CRITICAL: Ensure both models are in FP32 for a fair comparison**
#     # open_clip might default to FP16. oai_model is already FP32.
#     oc_model = oc_model.float()
#     oai_model = oai_model.float() # Already is, but good practice
    
#     oc_model.eval()
#     oai_model.eval()

#     print(f"OpenClip model dtype: {oc_model.visual.conv1.weight.dtype}")
#     print(f"OpenAI model dtype: {oai_model.dtype}")

#     # --- 3. Compare logit_scale ---
#     print("\n--- 3. Logit Scale Check ---")
#     oc_logit = oc_model.logit_scale.data
#     oai_logit = oai_model.logit_scale.data
    
#     print(f"OpenClip logit_scale (raw): {oc_logit.item():.6f}")
#     print(f"OpenAI logit_scale (raw):   {oai_logit.item():.6f}")
#     print(f"Are they close? {torch.allclose(oc_logit, oai_logit)}")

#     # --- 4. Compare Tokenizers ---
#     print("\n--- 4. Tokenizer Check ---")
#     oc_tokens = open_clip.tokenize(SAMPLE_TEXTS).to(DEVICE)
#     oai_tokens = clip.tokenize(SAMPLE_TEXTS).to(DEVICE)
    
#     print(f"OpenClip Tokens:\n{oc_tokens}")
#     print(f"OpenAI Tokens:\n{oai_tokens}")
    
#     # This is the most important check
#     are_tokens_identical = (oc_tokens == oai_tokens).all().item()
#     print(f"Are tokens identical? --> {are_tokens_identical}")

#     if not are_tokens_identical:
#         print("!! DISCREPANCY FOUND: Tokenizers are producing different token IDs.")
#         print("   This is the most likely source of your problem.")

#     # --- 5. Compare Text Encoder Weights (Spot Check) ---
#     print("\n--- 5. Weight Spot Check (text_projection) ---")
#     # We compare text_projection as it's the final layer before normalization.
#     # Note: open_clip stores it as a tensor, oai_model as nn.Parameter
#     oc_tp = oc_model.text_projection
#     oai_tp = oai_model.text_projection
    
#     print(f"OpenClip text_projection mean: {oc_tp.mean().item():.6f}")
#     print(f"OpenAI text_projection mean:   {oai_tp.mean().item():.6f}")
#     print(f"Are text_projection weights close? {torch.allclose(oc_tp, oai_tp, atol=1e-6)}")

#     # --- 6. Compare Text Encoder Output ---
#     print("\n--- 6. Text Encoder Output Check ---")
    
#     with torch.no_grad():
#         # Test 6a: Encode using THEIR OWN tokens (what your code does)
#         oc_embed_own = oc_model.encode_text(oc_tokens)
#         oai_embed_own = oai_model.encode_text(oai_tokens)
        
#         # Test 6b: Encode using the SAME tokens (using OpenAI's tokens for both)
#         oc_embed_shared = oc_model.encode_text(oai_tokens)
#         oai_embed_shared = oai_model.encode_text(oai_tokens)

#     # Normalize as your code does
#     oc_embed_own_norm = oc_embed_own / oc_embed_own.norm(dim=-1, keepdim=True)
#     oai_embed_own_norm = oai_embed_own / oai_embed_own.norm(dim=-1, keepdim=True)
    
#     oc_embed_shared_norm = oc_embed_shared / oc_embed_shared.norm(dim=-1, keepdim=True)
#     oai_embed_shared_norm = oai_embed_shared / oai_embed_shared.norm(dim=-1, keepdim=True)

#     print("\nCheck 6a: Using their own (potentially different) tokens")
#     print(f"OpenClip Normed Mean: {oc_embed_own_norm.mean().item():.6f}")
#     print(f"OpenAI Normed Mean:   {oai_embed_own_norm.mean().item():.6f}")
#     print(f"Are outputs close? {torch.allclose(oc_embed_own_norm, oai_embed_own_norm, atol=1e-5)}")

#     print("\nCheck 6b: Using shared (OpenAI) tokens")
#     print(f"OpenClip Normed Mean: {oc_embed_shared_norm.mean().item():.6f}")
#     print(f"OpenAI Normed Mean:   {oai_embed_shared_norm.mean().item():.6f}")
#     print(f"Are outputs close? {torch.allclose(oc_embed_shared_norm, oai_embed_shared_norm, atol=1e-5)}")


# if __name__ == '__main__':
#     # You might need to adjust the 'from .openai_clip import clip'
#     # to 'import clip' if you've pip installed it.
#     run_diagnostics()