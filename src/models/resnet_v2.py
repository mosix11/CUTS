# Pre Activation ResNet or ResNet V2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    
# Pre-Activation BasicBlock for ResNet-9, ResNet-18 and ResNet-34 (ResNet V2)
class PreActBasicBlock(nn.Module):
    """
    Pre-Activation Basic Block for ResNet (ResNet V2).
    Used in ResNet-18 and ResNet-34.
    It consists of two 3x3 convolutional layers with pre-activation.
    The 'expansion' is 1.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        # If the input dimensions (channels or spatial size) do not match the output,
        # a 1x1 convolution is used in the shortcut path to match them.
        # Note: In Pre-Act ResNet, the shortcut itself does not have a BN/ReLU after it.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Pre-activation: BN -> ReLU applied to the input 'x' before the first convolution
        out = F.relu(self.bn1(x))
        # Determine the shortcut path: if shortcut exists, apply it to the pre-activated 'out', otherwise use 'x'
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x

        # Apply first convolution
        out = self.conv1(out)
        # Pre-activation for the second convolution: BN -> ReLU
        out = F.relu(self.bn2(out))
        # Apply second convolution
        out = self.conv2(out)

        # Add the shortcut connection. No ReLU after the addition in Pre-Act ResNet.
        out += shortcut
        return out
    

# Pre-Activation Bottleneck Block for ResNet-50, ResNet-101, and ResNet-152 (ResNet V2)
class PreActBottleneck(nn.Module):
    """
    Pre-Activation Bottleneck Block for ResNet (ResNet V2).
    Used in ResNet-50, ResNet-101, and ResNet-152.
    It consists of three convolutional layers (1x1, 3x3, 1x1) with pre-activation.
    The 'expansion' is 4.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes) # Note: BN for the last 1x1 conv's input
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        # If dimensions don't match, apply a 1x1 conv to the shortcut path.
        # No BN/ReLU after the shortcut conv in Pre-Act ResNet.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        # Pre-activation for the first conv: BN -> ReLU applied to the input 'x'
        out = F.relu(self.bn1(x))
        # Determine the shortcut path: if shortcut exists, apply it to the pre-activated 'out', otherwise use 'x'
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x

        # Apply first 1x1 conv
        out = self.conv1(out)
        # Pre-activation for the 3x3 conv: BN -> ReLU
        out = F.relu(self.bn2(out))
        # Apply 3x3 conv
        out = self.conv2(out)
        # Pre-activation for the last 1x1 conv: BN -> ReLU
        out = F.relu(self.bn3(out))
        # Apply last 1x1 conv
        out = self.conv3(out)

        # Add the shortcut connection. No ReLU after the addition.
        out += shortcut
        return out


# General Pre-Activation ResNet Class
class PreActResNet(nn.Module):
    """
    General Pre-Activation ResNet model implementation (ResNet V2).
    It takes a 'block' type (PreActBasicBlock or PreActBottleneck) and a list of 'num_blocks'
    to define the architecture.
    """
    def __init__(
        self,
        block,
        num_blocks,
        init_channels=64,
        num_classes=10,
        input_image_size=(32, 32),
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss(),
        metrics:dict=None,
    ): 
        
        super(PreActResNet, self).__init__()
        self.in_planes = init_channels # Initial number of input channels for the first layer

        # Determine initial conv layer parameters based on input_image_size
        if input_image_size == (32, 32) or input_image_size == (64, 64):
            # For smaller images, use a smaller kernel and no aggressive downsampling
            conv1_kernel_size = 3
            conv1_stride = 1
            conv1_padding = 1
            maxpool_kernel_size = 1 # No max pooling, or a 1x1 pool effectively
            maxpool_stride = 1
            maxpool_padding = 0
        elif input_image_size == (128, 128):
            # Medium images, slight downsampling
            conv1_kernel_size = 5
            conv1_stride = 2
            conv1_padding = 2
            maxpool_kernel_size = 3
            maxpool_stride = 2
            maxpool_padding = 1
        elif input_image_size == (224, 224) or input_image_size == (256, 256):
            # Standard ImageNet size, use original ResNet initial layer parameters
            conv1_kernel_size = 7
            conv1_stride = 2
            conv1_padding = 3
            maxpool_kernel_size = 3
            maxpool_stride = 2
            maxpool_padding = 1
        else:
            raise ValueError(f"Unsupported input_image_size: {input_image_size}. "
                             "Supported sizes are (32,32), (64,64), (128,128), (224,224), (256,256).")

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels) # BN after conv1, before first residual block's pre-activation
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding)


        # Building Pre-Activation ResNet layers
        self.layer1 = self._make_layer(block, init_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * init_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * init_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * init_channels, num_blocks[3], stride=2)

        # Final Batch Normalization and ReLU before Global Average Pooling
        self.bn_final = nn.BatchNorm2d(8 * init_channels * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)

        # Final average pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * init_channels * block.expansion, num_classes)

        # Initialize weights
        
                
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
        
        
        self.metrics = nn.ModuleDict()
        if metrics:
            for name, metric_instance in metrics.items():
                self.metrics[name] = metric_instance


    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a sequence of pre-activation residual blocks for a given layer.
        The first block in the layer might have a stride > 1 for downsampling.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolutional layer with BN and ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Final pre-activation before global average pooling
        out = self.relu_final(self.bn_final(out))

        # Global average pooling and final fully connected layer
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
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





def PreActResNet9(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBasicBlock,
        num_blocks=[1, 1, 1, 1],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )


def PreActResNet18(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBasicBlock,
        num_blocks=[2, 2, 2, 2],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )
    
def PreActResNet34(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBasicBlock,
        num_blocks=[3, 4, 6, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )
    
def PreActResNet50(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBottleneck,
        num_blocks=[3, 4, 6, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )
    
def PreActResNet101(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBottleneck,
        num_blocks=[3, 4, 23, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )
    
def PreActResNet152(
    init_channels=64, num_classes=10, img_size=(32, 32), weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PreActResNet:
    return PreActResNet(
        block=PreActBottleneck,
        num_blocks=[3, 8, 36, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )
