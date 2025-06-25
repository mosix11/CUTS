import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        init_channels=64,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss,
        metrics:dict=None,
    ):
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels
        self.k = init_channels
        c = init_channels
        
        

        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * c * block.expansion, num_classes)
        
        if weight_init:
            self.apply(weight_init)
            
        if not loss_fn:
            raise RuntimeError('The loss function must be specified!')
        self.loss_fn = loss_fn
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def training_step(self, x, y, use_amp=False, return_preds=False):
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
                results[name] = metric.compute()
        return results
    
    def reset_metrics(self):
        if self.metrics:
            for name, metric in self.metrics.items():
                metric.reset()

    def get_identifier(self):
        return f"resnet18k|k{self.k}"
    
    

    def _count_trainable_parameters(self):
        """
        Counts and returns the total number of trainable parameters in the model.
        These are the parameters whose gradients are computed and are updated during backpropagation.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def make_resnet18k(
    init_channels=64, num_classes=10, weight_init=None, loss_fn=nn.CrossEntropyLoss, metrics=None
) -> PreActResNet:
    """Returns a ResNet18 with width parameter init_channels. (init_channels=64 is standard ResNet18)"""
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=init_channels,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
    
def make_resnet34k(
    init_channels=64, num_classes=10, weight_init=None, loss_fn=nn.CrossEntropyLoss, metrics=None
) -> PreActResNet:
    """Returns a ResNet18 with width parameter init_channels. (init_channels=64 is standard ResNet34)"""
    return PreActResNet(
        PreActBlock,
        [3,4,6,3],
        num_classes=num_classes,
        init_channels=init_channels,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    

def make_resnet18k(
    init_channels=64, num_classes=10, weight_init=None, loss_fn=nn.CrossEntropyLoss, metrics=None
) -> PreActResNet:
    """Returns a ResNet18 with width parameter init_channels. (init_channels=64 is standard ResNet18)"""
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=init_channels,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
def make_resnet18k(
    init_channels=64, num_classes=10, weight_init=None, loss_fn=nn.CrossEntropyLoss, metrics=None
) -> PreActResNet:
    """Returns a ResNet18 with width parameter init_channels. (init_channels=64 is standard ResNet18)"""
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=init_channels,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    
def make_resnet18k(
    init_channels=64, num_classes=10, weight_init=None, loss_fn=nn.CrossEntropyLoss, metrics=None
) -> PreActResNet:
    """Returns a ResNet18 with width parameter init_channels. (init_channels=64 is standard ResNet18)"""
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        init_channels=init_channels,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics,
    )
