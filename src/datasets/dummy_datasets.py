import torch
from torch.utils.data import Dataset
from .base_classification_dataset import BaseClassificationDataset


class MyCustomDataset(Dataset):
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                x = self.data[idx]
                y = self.targets[idx]
                return x, y

class DummyClassificationDataset(BaseClassificationDataset):
    def __init__(
        self,
        **kwargs
    ) -> None:

        super().__init__(
            dataset_name='DummyClassificationDataset',
            num_classes=10,
            **kwargs,  
        )
        
    def load_train_set(self):
        data = torch.arange(50)
        targets = torch.repeat_interleave(torch.arange(10), repeats=5)
        
        return MyCustomDataset(data, targets)
    
    def load_validation_set(self):
        return None
    
    def load_test_set(self):
        data = torch.arange(50)
        targets = torch.repeat_interleave(torch.arange(10), repeats=5)
        
        return MyCustomDataset(data, targets) 


    def get_identifier(self):
        return 'dummy_classification_dataset'