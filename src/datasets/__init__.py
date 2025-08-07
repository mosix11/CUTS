from .base_classification_dataset import BaseClassificationDataset
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .clothing1M import Clothing1M
from .mog_synthetic import MoGSynthetic
from .dummy_datasets import DummyClassificationDataset
from . import dataset_wrappers as data_utils
from . import dataset_factory
