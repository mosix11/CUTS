import torch
from typing import List, Dict

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
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    
    # def __init__(self, pretrained_state_dict=None, finetuned_state_dict=None, vector=None):
    #     """
    #     Initialize a TaskVector using either:
    #     - pretrained_state_dict and finetuned_state_dict, OR
    #     - an existing vector (dictionary of parameter deltas).
    #     Skips ALL Batch Normalization layer parameters (weight, bias, running_mean, running_var).
    #     """
    #     if vector is not None:
    #         self.vector = vector
    #     else:
    #         assert pretrained_state_dict is not None and finetuned_state_dict is not None, \
    #             "Provide either vector or both state_dicts."
    #         self.vector = {}
    #         with torch.no_grad():
    #             # Identify all module prefixes that belong to BatchNorm layers
    #             # We can do this by looking for 'running_mean' or 'running_var'
    #             bn_module_prefixes = set()
    #             for key in pretrained_state_dict:
    #                 if '.running_mean' in key or '.running_var' in key:
    #                     # Extract the module prefix, e.g., 'net.1', 'net.4'
    #                     # This assumes the pattern 'module_prefix.parameter_name'
    #                     parts = key.rsplit('.', 1) # Split from the right, at most once
    #                     if len(parts) > 1:
    #                         bn_module_prefixes.add(parts[0])

    #             for key in pretrained_state_dict:
    #                 if key not in finetuned_state_dict:
    #                     print(f"Warning: key {key} missing in finetuned_state_dict.")
    #                     continue
    #                 if pretrained_state_dict[key].dtype not in [torch.float32, torch.float16, torch.bfloat16]:
    #                     continue  # Skip non-float entries

    #                 # Check if the current key belongs to a identified BN module
    #                 # by checking if its prefix is in our set of bn_module_prefixes
    #                 is_bn_parameter = False
    #                 for prefix in bn_module_prefixes:
    #                     if key.startswith(prefix + '.'):
    #                         is_bn_parameter = True
    #                         break

    #                 if is_bn_parameter:
    #                     continue # Skip all parameters of the identified BN module

    #                 self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]


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
    
    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        return self if other in (None, 0) else self.__add__(other)

    def __neg__(self):
        """Negate the task vector."""
        with torch.no_grad():
            neg_vector = {key: -val for key, val in self.vector.items()}
        return TaskVector(vector=neg_vector)
    
    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return TaskVector(vector=new_vector)
    
    
    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return TaskVector(vector=new_vector)
    
    
    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product
    
    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))
    

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
    
    

    
    
    
    def flatten_vector(self):
        """
        Flatten a task vector dictionary into a single 1D vector.
        """
        return torch.cat([v.flatten() for k, v in sorted(self.vector.items())])
    
    def unflatten_vector(self, flat_vector):
        """
        Unflatten a 1D tensor into a dictionary using the shapes in reference_dict.
        """
        new_dict = {}
        pointer = 0
        for k, v in sorted(self.vector.items()):
            numel = v.numel()
            new_dict[k] = flat_vector[pointer:pointer+numel].reshape_as(v)
            pointer += numel
        return TaskVector(vector=new_dict)
    
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



    def compute_tall_mask(self, multi_task_vector: "TaskVector", lambda_t: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute a binary TALL mask for the task vectore give a multi-task vector (sum of other task vectors with this task vector.)
        
        Args:
            multi_task_vector (TaskVector): τ_MTL = sum(τ_t's)
            lambda_t (float): scaling factor λ for thresholding.
        
        Returns:
            mask (Dict[str, Tensor]): binary mask for each parameter.
        """
        mask = {}
        with torch.no_grad():
            for key in self.vector:
                τt = self.vector[key]
                τ_MTL = multi_task_vector.vector[key]
                diff = (τ_MTL - τt).abs()
                mask[key] = (τt.abs() >= lambda_t * diff).to(dtype=torch.bool)
        return mask
    
    
    @staticmethod
    def compute_all_tall_masks(task_vectors: List["TaskVector"], lambda_t: float = 1.0) -> List[Dict[str, torch.Tensor]]:
        multi_task_vector = sum(task_vectors)
        return [tv.compute_tall_mask(multi_task_vector, lambda_t) for tv in task_vectors]
    
    
    def masked(self, mask: Dict[str, torch.Tensor]) -> "TaskVector":
        """
        Apply a binary mask to this TaskVector.
        
        Args:
            mask: Dict[str, torch.Tensor], same keys as self.vector.
        
        Returns:
            TaskVector: masked task vector.
        """
        masked_vector = {
            key: self.vector[key] * mask[key].to(self.vector[key].dtype)
            for key in self.vector if key in mask
        }
        return TaskVector(vector=masked_vector)
    
    


    @staticmethod
    def orthogonalize_task_vectors_GSP(task_vectors: List["TaskVector"]):
        """
        Given a list of task vectors, return:
        - orthogonalized task vectors
        - residual task vectors (original - orthogonalized)
        with Gram-Schmidt process

        Args:
            task_vectors (List[TaskVector]): List of task vectors.

        Returns:
            Tuple:
                - List[TaskVector]: Orthogonalized task vectors.
                - List[TaskVector]: Residual components removed to achieve orthogonality.
        """
        flat_vectors = [tv.flatten_vector() for tv in task_vectors]
        flat_matrix = torch.stack(flat_vectors)  # Shape: (N, D)

        orthogonalized = []
        residuals = []

        for i in range(flat_matrix.size(0)):
            vi = flat_matrix[i].clone()
            for uj in orthogonalized:
                proj = torch.dot(vi, uj) / uj.norm()**2 * uj
                vi = vi - proj
            orthogonalized.append(vi)
            residuals.append(flat_matrix[i] - vi)

        orth_tvs = []
        residual_tvs = []
        for orth_tv, res_tv, orig_tv in zip(orthogonalized, residuals, task_vectors):
            orth_tvs.append(orig_tv.unflatten_vector(orth_tv))
            residual_tvs.append(orig_tv.unflatten_vector(res_tv))

        return orth_tvs, residual_tvs
    
    
    
    @staticmethod
    def decompose_task_vectors_SVD(task_vectors: List["TaskVector"], variance_threshold: float = 0.90):
        """
        Perform order-invariant SVD-based decomposition of task vectors into shared and residual components.

        Args:
            task_vectors (List[TaskVector]): List of TaskVector objects (same structure).
            variance_threshold (float): Fraction of variance to keep for shared space (default: 0.90).

        Returns:
            shared_components (List[TaskVector]): Projected vectors in shared subspace.
            residual_components (List[TaskVector]): Remaining orthogonal (class-specific) parts.
        """
        assert len(task_vectors) > 1, "Need at least 2 task vectors for decomposition."

        # Flatten all task vectors into a matrix (C x D)
        flat_vectors = [tv.flatten_vector() for tv in task_vectors]
        T = torch.stack(flat_vectors)  # (C, D)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)  # Vh: (D, D)

        # Decide how many components to keep
        explained_var = S ** 2
        explained_var_ratio = explained_var / explained_var.sum()
        cumulative = torch.cumsum(explained_var_ratio, dim=0)
        k = (cumulative < variance_threshold).sum().item() + 1  # Smallest k s.t. ≥ threshold

        # Shared subspace basis (k leading singular vectors)
        Vk = Vh[:k, :]  # Shape: (k, D)

        # Projection onto shared space: T_shared = T @ Vk.T @ Vk
        T_shared = T @ Vk.T @ Vk  # (C, D)
        T_residual = T - T_shared  # (C, D)

        # Unflatten back to TaskVector structure
        shared_components = []
        residual_components = []
        for i in range(len(task_vectors)):
            shared_tv = task_vectors[i].unflatten_vector(T_shared[i])
            residual_tv = task_vectors[i].unflatten_vector(T_residual[i])
            shared_components.append(shared_tv)
            residual_components.append(residual_tv)

        return residual_components, shared_components 
    
    