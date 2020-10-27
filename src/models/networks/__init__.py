from . import fcn8_vgg16

from torchvision import models
import torch, os
import torch.nn as nn


def get_network(network_name, n_classes, exp_dict):
    if network_name == 'fcn8_vgg16':
        model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    return model_base

