path1 = 'outputs/single_experiment/config8_pretrain/log/low_loss_indices_0.4.pkl'
path2 = 'low_loss_indices_0.4.pkl'

import pickle

# Load the saved low-loss indices
try:
    with open(path1, 'rb') as f:
        low_loss_indices1 = pickle.load(f)
        
    with open(path2, 'rb') as f:
        low_loss_indices2 = pickle.load(f)

    # low_loss_indices is a dictionary like: {0: [12, 45, 101, ...], 1: [3, 88, 92, ...], ...}
    print(f"Loaded {len(low_loss_indices1)} classes of low-loss indices.")

    # Get a flat list of all low-loss indices
    all_easy_samples1 = [idx for class_list in low_loss_indices1.values() for idx in class_list]
    all_easy_samples2 = [idx for class_list in low_loss_indices2.values() for idx in class_list]
    # print(f"Total number of easy samples: {len(all_easy_samples)}")
    print(all_easy_samples1 == all_easy_samples2)

except FileNotFoundError:
    print("Could not find the low-loss indices file.")