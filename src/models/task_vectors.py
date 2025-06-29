import torch

class TaskVector:
    def __init__(self, pretrained_state_dict=None, finetuned_state_dict=None, vector=None):
        """
        Initialize a TaskVector using either:
        - pretrained_state_dict and finetuned_state_dict, OR
        - an existing vector (dictionary of parameter deltas).
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None, \
                "Provide either vector or both state_dicts."
            self.vector = {}
            with torch.no_grad():
                for key in pretrained_state_dict:
                    if key not in finetuned_state_dict:
                        print(f"Warning: key {key} missing in finetuned_state_dict.")
                        continue
                    if pretrained_state_dict[key].dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        continue  # Skip non-float entries
                    # # TODO maybe remove the BN exclusion?
                    # if 'running_mean' in key or 'running_var' in key:
                    #     continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors."""
        new_vector = {}
        with torch.no_grad():
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning: key {key} not found in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        return self if other in (None, 0) else self.__add__(other)

    def __neg__(self):
        """Negate the task vector."""
        with torch.no_grad():
            neg_vector = {key: -val for key, val in self.vector.items()}
        return TaskVector(vector=neg_vector)

    def apply_to(self, model, scaling_coef=1.0, strict=False):
        """
        Applies the task vector to the weights of an existing model.
        
        Parameters:
        - model: a PyTorch model with pre-loaded weights (state_dict already applied)
        - scaling_coef: float multiplier for the task vector (default: 1.0)
        - strict: if True, will raise errors for missing/unmatched keys
        """
        with torch.no_grad():
            updated_state_dict = model.state_dict()
            for key in self.vector:
                if key not in updated_state_dict:
                    if strict:
                        raise KeyError(f"Key {key} not found in model state_dict.")
                    else:
                        print(f"Warning: key {key} not found in model. Skipping.")
                        continue
                updated_state_dict[key] = updated_state_dict[key] + scaling_coef * self.vector[key]
            model.load_state_dict(updated_state_dict, strict=strict)
        return model
    
    
    
    
    def cosine_similarity(self, other_task_vector):
        """
        Computes the cosine similarity between the current task vector and another.

        Parameters:
        - other_task_vector: Another TaskVector instance.

        Returns:
        - float: The cosine similarity value, or 0.0 if either vector has zero magnitude.
        """
        dot_product_sum = 0.0
        norm_self_sq_sum = 0.0
        norm_other_sq_sum = 0.0

        with torch.no_grad():
            # Iterate over common keys to compute dot product and squared norms
            common_keys = set(self.vector.keys()).intersection(set(other_task_vector.vector.keys()))
            
            if not common_keys:
                print("Warning: No common keys found between task vectors. Cosine similarity will be 0.")
                return 0.0

            for key in common_keys:
                tensor_self = self.vector[key]
                tensor_other = other_task_vector.vector[key]

                # Ensure tensors are float types for calculation
                if tensor_self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    print(f"Skipping key {key} in self.vector due to non-float dtype: {tensor_self.dtype}")
                    continue
                if tensor_other.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    print(f"Skipping key {key} in other_task_vector due to non-float dtype: {tensor_other.dtype}")
                    continue
                
                # Flatten tensors for dot product and norm calculations
                flattened_self = tensor_self.flatten()
                flattened_other = tensor_other.flatten()

                # Dot product (sum of element-wise products)
                dot_product_sum += torch.dot(flattened_self, flattened_other).item()

                # Squared L2 norm (sum of squared elements)
                norm_self_sq_sum += torch.sum(flattened_self ** 2).item()
                norm_other_sq_sum += torch.sum(flattened_other ** 2).item()

            # Calculate L2 norms
            norm_self = norm_self_sq_sum ** 0.5
            norm_other = norm_other_sq_sum ** 0.5

            # Handle cases where one or both norms are zero to avoid division by zero
            if norm_self == 0 or norm_other == 0:
                print("Warning: One or both task vectors have zero magnitude. Cosine similarity is 0.")
                return 0.0
            
            cosine_sim = dot_product_sum / (norm_self * norm_other)
            return cosine_sim
