"""
Model definitions for experiments
Includes: NiN, VGG16, ResNet18, AlexNet, Inception V3, Inception V4
"""

from .nin import SGXNiN
from .vgg16 import SGXVGG16
from .resnet18 import SGXResNet18
from .alexnet import SGXAlexNet
from .inception_v3 import SGXInceptionV3
from .inception_v4 import SGXInceptionV4

__all__ = [
    'SGXNiN',
    'SGXVGG16', 
    'SGXResNet18',
    'SGXAlexNet',
    'SGXInceptionV3',
    'SGXInceptionV4',
]

