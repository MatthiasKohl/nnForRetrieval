# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import numpy as np


# autograd function to normalize an input over the rows
# (each vector of a batch is normalized)
# the backward step follows the implementation of
# torch.legacy.nn.Normalize closely
class Normalize2DL2(Function):

    def __init__(self, eps=1e-10):
        super(Normalize2DL2, self).__init__()
        self.eps = eps

    def forward(self, input):
        self.norm2 = input.pow(2).sum(1).add_(self.eps)
        self.norm = self.norm2.pow(0.5)
        output = input / self.norm.expand_as(input)
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        gradInput = self.norm2.expand_as(input) * grad_output
        cross = (input * grad_output).sum(1)
        buf = input * cross.expand_as(input)
        gradInput.add_(-1, buf)
        cross = self.norm2 * self.norm
        gradInput.div_(cross.expand_as(gradInput))
        return gradInput


class NormalizeL2(nn.Module):

    def __init__(self):
        super(NormalizeL2, self).__init__()

    def forward(self, input):
        return Normalize2DL2()(input)


class TuneClassif(nn.Module):
    """
        Image classification network based on a pretrained network
        which is then finetuned to a different dataset
        It's assumed that the last layer of the given network
        is a fully connected (linear) one
    """

    def __init__(self, net, num_classes, untrained_layers=2):
        super(TuneClassif, self).__init__()
        if untrained_layers < 0:
            untrained_layers = sum(1 for _ in net.features) + sum(1 for _ in net.classifier)
        self.features = net.features
        self.classifier = net.classifier
        # make sure we never retrain the first few layers
        # this is usually not needed
        n_features = 0
        for module in self.features:
            if n_features >= untrained_layers:
                break
            if sum(1 for _ in module.parameters()) > 0:
                n_features += 1
                for p in module.parameters():
                    p.requires_grad = False
        n_class = 0
        for module in self.classifier:
            if n_class + n_features >= untrained_layers:
                break
            if sum(1 for _ in module.parameters()) > 0:
                n_class += 1
                for p in module.parameters():
                    p.requires_grad = False

        for name, module in self.classifier._modules.items():
            if module is net.classifier[len(net.classifier._modules) - 1]:
                self.classifier._modules[name] = nn.Linear(module.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Siamese1(nn.Module):
    """
        Define a siamese network
        Given a module, it will duplicate it with weight sharing, concatenate the output and add a linear classifier
    """
    def __init__(self, net, num_classes=100, feature_dim=100):
        super(Siamese1, self).__init__()
        self.features = net.features
        # TODO output size of last filter of net's features should be 6*6
        in_features_factor = 6 ** 2
        for module in self.features:
            if hasattr(module, 'out_channels'):
                in_features = module.out_channels * in_features_factor
        if feature_dim <= 0:
            for module in net.classifier:
                if isinstance(module, nn.modules.linear.Linear):
                    out_features = module.out_features
        else:
            out_features = feature_dim
        self.feature_reduc1 = nn.Sequential(
            nn.Dropout(0.5),
            NormalizeL2(),
            nn.Linear(in_features, out_features)
        )
        self.feature_reduc2 = NormalizeL2()

    def forward_single(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.feature_reduc1(x)
        x = self.feature_reduc2(x)
        return x

    def forward(self, x1, x2=None):
        if self.training:
            return self.forward_single(x1), self.forward_single(x2)
        else:
            return self.forward_single(x1)


def siamese1():
    return Siamese1(models.alexnet(pretrained=True))


# metric loss according to Chopra et al "Learning a Similarity Metric Discriminatively, with Application to Face Verification"
# since we assume normalized vectors, we use Q=2
class MetricL(Function):

    def __init__(self, size_average=True):
        super(MetricL, self).__init__()
        self.size_average = size_average

    # TODO: everything could be done inplace,
    # more difficult though (for norm see torch Cosine Loss)
    def terms(self, input1, input2, y):
        diff = input1 - input2
        energy = diff.norm(1, 1)
        e = energy * 0 + np.e
        exp_term = torch.pow(e, -2.77 * energy / 2)
        return diff, energy, exp_term

    # target takes values in 1 (good), -1 (bad) so (1-target)/2 is 0 for good pairs and 1 for bad ones, (1+target) / 2 inverse
    def forward(self, input1, input2, y):
        _, energy, exp_term = self.terms(input1, input2, y)
        loss_g = (1 + y) * energy * energy / 2
        loss_i = (1 - y) * 2 * exp_term
        loss = (loss_g + loss_i).sum()
        if self.size_average:
            loss = loss / y.size(0)
        self.save_for_backward(input1, input2, y)
        return torch.Tensor([loss])

    def backward(self, grad_output):
        input1, input2, y = self.saved_tensors
        diff, energy, exp_term = self.terms(input1, input2, y)
        # represents the derivative w.r.t. input1 of energy
        diff[diff.lt(0)] = -1
        diff[diff.ge(0)] = 1
        y_g = (1 + y).view(-1, 1).expand_as(input1)
        y_i = (1 - y).view(-1, 1).expand_as(input1)
        energy = energy.expand_as(input1)
        exp_term = exp_term.expand_as(input1)
        grad1 = y_g * diff * energy - 2.77 * y_i * diff * exp_term
        grad2 = -grad1
        if self.size_average:
            grad1.div_(y.size(0))
            grad2.div_(y.size(0))
        return grad1, grad2, None


class MetricLoss(nn.Module):

    def __init__(self, size_average=True):
        super(MetricLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return MetricL(self.size_average)(input1, input2, target)
