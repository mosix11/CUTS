import random
from collections import defaultdict
from typing import List, Iterator, Optional
import torch
from torch.utils.data import Sampler, BatchSampler


class ClassBalancedBatchSampler(BatchSampler):
    """
    Yields batches with exactly `classes_per_batch` distinct classes and
    `samples_per_class` examples per class.

    Args:
        labels: list/1D tensor of integer class labels aligned with dataset indices [0..N-1]
        classes_per_batch (int): number of distinct classes per batch (C)
        samples_per_class (int): number of samples from each class per batch (M)
        num_batches (Optional[int]): how many batches to generate per epoch (recommended).
                                     If None, computed as floor(N / (C*M)).
        drop_last (bool): whether to drop last incomplete batch if computed automatically
        replacement (bool): sample with replacement within each class bucket (useful if some
                            classes have very few examples and you want consistent batches)
        generator (Optional[torch.Generator]): for reproducible shuffles
    """
    def __init__(
        self,
        labels,
        classes_per_batch: int,
        samples_per_class: int,
        num_batches: Optional[int] = None,
        drop_last: bool = True,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.labels = torch.as_tensor(labels).clone()
        self.C = int(classes_per_batch)
        self.M = int(samples_per_class)
        self.drop_last = drop_last
        self.replacement = replacement
        self.gen = generator

        # build index lists per class
        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.labels.tolist()):
            self.class_to_indices[int(y)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.N = len(labels)
        self.batch_size = self.C * self.M

        # conservative default for num_batches
        if num_batches is None:
            max_full_batches = self.N // self.batch_size
            if max_full_batches == 0 and not drop_last:
                max_full_batches = 1
            self.num_batches = max_full_batches
        else:
            self.num_batches = int(num_batches)

        # pre-shuffle class buckets
        self._reset_class_iters()

    def _shuffle_inplace(self, lst):
        if len(lst) <= 1:
            return
        perm = torch.randperm(len(lst), generator=self.gen).tolist()
        lst[:] = [lst[i] for i in perm]

    def _choice(self, seq):
        idx = torch.randint(len(seq), (1,), generator=self.gen).item()
        return seq[idx]

    def _sample(self, seq, k):
        if k >= len(seq):
            # fall back to sampling with replacement
            idxs = torch.randint(len(seq), (k,), generator=self.gen).tolist()
        else:
            idxs = torch.randperm(len(seq), generator=self.gen)[:k].tolist()
        return [seq[i] for i in idxs]
    # ------------------------------------

    def _reset_class_iters(self):
        self.buckets = {}
        for c, idxs in self.class_to_indices.items():
            idxs = idxs.copy()
            self._shuffle_inplace(idxs)
            self.buckets[c] = {"idxs": idxs, "ptr": 0}

    def _draw_from_class(self, c, m):
        b = self.buckets[c]
        out = []
        for _ in range(m):
            if b["ptr"] >= len(b["idxs"]):
                if self.replacement:
                    self._shuffle_inplace(b["idxs"])
                    b["ptr"] = 0
                else:
                    self._shuffle_inplace(b["idxs"])
                    b["ptr"] = 0
            out.append(b["idxs"][b["ptr"]])
            b["ptr"] += 1
        return out

    def __iter__(self):
        self._reset_class_iters()
        for _ in range(self.num_batches):
            if len(self.classes) >= self.C:
                chosen = self._sample(self.classes, self.C)
            else:
                chosen = [self._choice(self.classes) for _ in range(self.C)]
            batch = []
            for c in chosen:
                batch.extend(self._draw_from_class(c, self.M))
            self._shuffle_inplace(batch)
            yield batch

    def __len__(self):
        return self.num_batches