# Post Activation ResNet or ResNet V1
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseModel

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# BasicBlock for ResNet-9, ResNet-18 and ResNet-34 (ResNet V1)
class BasicBlock(nn.Module):
    """
    Basic Block for ResNet (ResNet V1).
    Used in ResNet-18 and ResNet-34.
    It consists of two 3x3 convolutional layers with post-activation.
    The 'expansion' is 1.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # If the input dimensions (channels or spatial size) do not match the output,
        # a 1x1 convolution is used in the shortcut path to match them.
        # In ResNet V1, the shortcut also includes a BatchNorm.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Apply first conv, BN, and ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply second conv and BN
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection (shortcut takes 'x' directly)
        out += self.shortcut(x)
        # Apply final ReLU after adding the shortcut
        out = F.relu(out)
        return out

# Bottleneck Block for ResNet-50, ResNet-101, and ResNet-152 (ResNet V1)
class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet (ResNet V1).
    Used in ResNet-50, ResNet-101, and ResNet-152.
    It consists of three convolutional layers (1x1, 3x3, 1x1) with post-activation.
    The 'expansion' is 4.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        # If dimensions don't match, apply a 1x1 conv to the shortcut path.
        # In ResNet V1, the shortcut also includes a BatchNorm.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Apply first 1x1 conv, BN, and ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply 3x3 conv, BN, and ReLU
        out = F.relu(self.bn2(self.conv2(out)))
        # Apply last 1x1 conv and BN
        out = self.bn3(self.conv3(out))
        # Add the shortcut connection (shortcut takes 'x' directly)
        out += self.shortcut(x)
        # Apply final ReLU after adding the shortcut
        out = F.relu(out)
        return out

# General Post-Activation ResNet Class (ResNet V1)
class PostActResNet(BaseModel):
    """
    General Post-Activation ResNet model implementation (ResNet V1).
    It takes a 'block' type (BasicBlock or Bottleneck) and a list of 'num_blocks'
    to define the architecture.
    """
    def __init__(
        self,
        block,
        num_blocks,
        init_channels=64,
        num_classes=10,
        input_image_size=(32, 32),
        grayscale:bool=False,
        weight_init=None,
        loss_fn=nn.CrossEntropyLoss(),
        metrics:dict=None,
    ):
        super().__init__(loss_fn=loss_fn, metrics=metrics)
        self.in_planes = init_channels # Initial number of input channels for the first layer
        self.k = init_channels # Added to match the user's original get_identifier method

        input_image_size = tuple(input_image_size)
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
        self.conv1 = nn.Conv2d(1 if grayscale else 3, init_channels, kernel_size=conv1_kernel_size, stride=conv1_stride, padding=conv1_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding)

        # Building ResNet layers
        self.layer1 = self._make_layer(block, init_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * init_channels, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * init_channels, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * init_channels, num_blocks[3], stride=2)

        # Final average pooling and fully connected layer for classification
        # No final BN/ReLU before avgpool in Post-Activation ResNet
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



    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a sequence of residual blocks for a given layer.
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

        # No final BN/ReLU before global average pooling in Post-Activation ResNet
        # The last ReLU is part of the final block's forward pass.

        # Global average pooling and final fully connected layer
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


    def get_identifier(self):
        # Using self.k which is set to init_channels
        return f"resnet_v1_k{self.k}"


# --- Functions to create specific Post-Activation ResNet models (ResNet V1) ---


def PostActResNet9(
    init_channels=64, num_classes=10, img_size=(32, 32), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-18 model using BasicBlock (Post-Activation)."""
    return PostActResNet(
        block=BasicBlock,
        num_blocks=[1, 1, 1, 1],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )

def PostActResNet18(
    init_channels=64, num_classes=10, img_size=(32, 32), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-18 model using BasicBlock (Post-Activation)."""
    return PostActResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )

def PostActResNet34(
    init_channels=64, num_classes=10, img_size=(32, 32), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-34 model using BasicBlock (Post-Activation)."""
    return PostActResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )

def PostActResNet50(
    init_channels=64, num_classes=1000, img_size=(224, 224), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-50 model using Bottleneck block (Post-Activation)."""
    return PostActResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )

def PostActResNet101(
    init_channels=64, num_classes=1000, img_size=(224, 224), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-101 model using Bottleneck block (Post-Activation)."""
    return PostActResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )

def PostActResNet152(
    init_channels=64, num_classes=1000, img_size=(224, 224), grayscale:bool=False, weight_init=None, loss_fn=nn.CrossEntropyLoss(), metrics=None
) -> PostActResNet:
    """ResNet-152 model using Bottleneck block (Post-Activation)."""
    return PostActResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        init_channels=init_channels,
        num_classes=num_classes,
        input_image_size=img_size,
        grayscale=grayscale,
        weight_init=weight_init,
        loss_fn=loss_fn,
        metrics=metrics
    )