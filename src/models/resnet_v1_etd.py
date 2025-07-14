# Post Activation ResNet or ResNet V1 with Example Tied Dropout
# https://arxiv.org/pdf/2307.09542
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from .modules import ExampleTiedDropout

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True)
    )

class PostActResNet9_ETD(nn.Module):
    def __init__(self, init_channels=64, num_classes=10, img_size=(32, 32), grayscale:bool=False,
                 weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics: dict = None, dropout: dict = None):
        super(PostActResNet9_ETD, self).__init__()
        self.k = init_channels
        dims = [1 if grayscale else 3, 64, 128, 128, 128, 256, 256, 256, 128]
        dims = [int(d * (init_channels / 64)) for d in dims]

        self.conv1 = conv_bn(dims[0], dims[1], kernel_size=3, stride=1, padding=1)
        self.dropout1 = ExampleTiedDropout(**dropout, num_channels=dims[1]) if dropout else None

        self.conv2 = conv_bn(dims[1], dims[2], kernel_size=5, stride=2, padding=2)
        self.dropout2 = ExampleTiedDropout(**dropout, num_channels=dims[2]) if dropout else None

        self.res1 = Residual(nn.Sequential(conv_bn(dims[2], dims[3]), conv_bn(dims[3], dims[4])))
        # The output of res1 has dims[4] channels
        self.dropout3 = ExampleTiedDropout(**dropout, num_channels=dims[4]) if dropout else None

        self.conv3 = conv_bn(dims[4], dims[5], kernel_size=3, stride=1, padding=1)
        self.dropout4 = ExampleTiedDropout(**dropout, num_channels=dims[5]) if dropout else None

        self.res2 = Residual(nn.Sequential(conv_bn(dims[5], dims[6]), conv_bn(dims[6], dims[7])))
        # The output of res2 has dims[7] channels
        self.dropout5 = ExampleTiedDropout(**dropout, num_channels=dims[7]) if dropout else None

        self.conv4 = conv_bn(dims[7], dims[8], kernel_size=3, stride=1, padding=0)
        self.dropout6 = ExampleTiedDropout(**dropout, num_channels=dims[8]) if dropout else None
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dims[8], num_classes, bias=False)

        if weight_init:
            self.apply(weight_init)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn

        self.metrics = nn.ModuleDict(metrics or {})

    def forward(self, x, idx=None):
        x = self.conv1(x)
        if self.dropout1:
            x = self.dropout1(x, idx)
        x = self.conv2(x)
        if self.dropout2:
            x = self.dropout2(x, idx)

        x = self.res1(x)
        # if self.dropout3:
        #     x = self.dropout3(x, idx)

        x = self.conv3(x)
        if self.dropout4:
            x = self.dropout4(x, idx)

        x = F.max_pool2d(x, 2)
        x = self.res2(x)
        # if self.dropout5:
        #     x = self.dropout5(x, idx)

        x = self.conv4(x)
        if self.dropout6:
            x = self.dropout6(x, idx)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    
    
    def training_step(self, x, y, indices, use_amp=False, return_preds=False):
        with autocast('cuda', enabled=use_amp):
            preds = self(x, indices)
            loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        if return_preds:
            return loss, preds
        else:
            return loss

    def validation_step(self, x, y, use_amp=False, return_preds=False):
        with torch.no_grad():
            with autocast('cuda', enabled=use_amp):
                preds = self(x)
                loss = self.loss_fn(preds, y)
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.update(preds, y)
        if return_preds:
            return loss, preds
        else:
            return loss

    def predict(self, x):
        with torch.no_grad():
            preds = self(x)
        return preds

    def compute_metrics(self):
        results = {}
        if self.metrics:
            for name, metric in self.metrics.items():
                results[name] = metric.compute().cpu().item()
        return results

    def reset_metrics(self):
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()

    def get_identifier(self):
        # Using self.k which is set to init_channels
        return f"resnet_v1_k{self.k}"

    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)