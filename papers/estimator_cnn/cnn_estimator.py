"""
cnn_estimator.py

A learned estimator of a disease field: a fully convolutional inpainting network, which maps the
sparse observations of a field to the probability that each cell is diseased and to its value.
Trained on simulator data by train_cnn.py. Requires PyTorch.
"""

import numpy as np
import torch
from torch import nn

from information_model import AbstractScalarFieldIM, latest_per_cell, probability_uncertainty


class InpaintingNet(nn.Module):
    """Dilated convolutions (receptive field 33 cells); no pooling, so that it applies to any grid size.
    Input planes: the observation mask and the observed values (0 where unobserved). 
    Output planes: the logit of the diseased probability, and the logit of the value."""
    def __init__(self, channels = 32):
        super().__init__()
        layers, inputs = [], 2
        for dilation in [1, 2, 4, 8, 1]:
            layers += [nn.Conv2d(inputs, channels, 3, padding=dilation, dilation=dilation), nn.ReLU()]
            inputs = channels
        layers.append(nn.Conv2d(channels, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def input_planes(observations, width, height):
    """The input of the network for the observations of a field: (2, width, height)"""
    planes = np.zeros((2, width, height), dtype=np.float32)
    cells, values = latest_per_cell(observations)
    planes[0, cells[:, 0], cells[:, 1]] = 1.0
    planes[1, cells[:, 0], cells[:, 1]] = values
    return planes


class CNNScalarFieldIM(AbstractScalarFieldIM):
    """The learned estimator of a disease field. The observed cells keep their observed value."""
    UNCERTAINTY = "probability"

    def __init__(self, width, height, default_value = 1.0, model_path = None):
        super().__init__(width, height, default_value)
        self.model_path = model_path
        self.model = None # loaded when first needed

    def estimate(self, observations, prior_value, prior_uncertainty):
        if self.model is None:
            self.model = InpaintingNet()
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(input_planes(observations, self.width, self.height))[None])[0]
        probability = torch.sigmoid(output[0]).numpy().astype(float)
        value = torch.sigmoid(output[1]).numpy().astype(float)
        cells, values = latest_per_cell(observations)
        probability[cells[:, 0], cells[:, 1]] = values < 0.75
        value[cells[:, 0], cells[:, 1]] = values
        self.probability = probability
        return value, probability_uncertainty(probability)
