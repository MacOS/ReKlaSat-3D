#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Stefan Bachhofner.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software, the trained model weights and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
#


import torch

from examples.minkunet import MinkUNet34C


__all__ = [
    'Coordinates',
    'CoordinatesColors',
    'CoordinatesColorsWeighted'
]

STATE_DICT_PREFIX = 'STATE_DICT_EXPERIMENT_ITERATION_'
STATE_DICT_SUFIX = '.pt'
MODELS_BASE_URL = 'https://github.com/MacOS/ReKlaSat-3D/releases/download'
model_urls_base = {
    'coordinates': f'{MODELS_BASE_URL}/v0.1-alpha',
    'coordinates_colors': f'{MODELS_BASE_URL}/v0.2-alpha',
    'coordinates_colors_weighted': f'{MODELS_BASE_URL}/v0.4-alpha'
}

# Number of in in_channels for coordinates.
C_IN_CHANNELS = 3
# Number of in_channels for coordinates and colors.
CC_IN_CHANNELS = 6
# Dimensionality of space.
SPACE_DIM = 3
# Number of classes.
NUM_CLASSES = 6


def get_coordinates_model():
    '''
        Returns MinkUNet34C with 3 input channels for
        x, y, and z.
    '''
    return MinkUNet34C(in_channels=C_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)


def get_coordinates_colors_model():
    '''
        Returns MinkUNet34C with 6 input channels for
        x, y, z, red, green, and blue.
    '''
    return MinkUNet34C(in_channels=CC_IN_CHANNELS, out_channels=NUM_CLASSES, D=SPACE_DIM)


def coordinates(progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    (in_channels=3) with weights after 50 Epochs.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr

    How to use:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates')
    '''
    EPOCH = 50

    model = get_coordinates_model()

    url = f'{model_urls_base["coordinates"]}/{STATE_DICT_PREFIX}{EPOCH}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def coordinates_epoch(epoch=50, progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    (in_channels=3) with weights after <epoch> Epochs.

    Args:
        epoch (int): Specifies the class weights to be returned.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Example:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates_epoch', epoch=40)
    '''
    model = get_coordinates_model()

    url = f'{model_urls_base["coordinates"]}/{STATE_DICT_PREFIX}{epoch}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def coordinates_colors(progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights after 50 Epochs.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.

    How to use:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates_colors')
    '''
    EPOCH = 50

    model = get_coordinates_colors_model()

    url = f'{model_urls_base["coordinates_colors"]}/{STATE_DICT_PREFIX}{EPOCH}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def coordinates_colors_epoch(epoch=50, progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights after <epoch> Epochs.

    Args:
        epoch (int): Specifies the class weights to be returned.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Example:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates_colors_epoch', epoch=40)
    '''
    model = get_coordinates_colors_model()

    url = f'{model_urls_base["coordinates_colors"]}/{STATE_DICT_PREFIX}{epoch}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def coordinates_colors_weighted(progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights, and median class weights, after 400 Epochs.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.

    Example:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates_colors_weighted')
    '''
    EPOCH = 400

    model = get_coordinates_colors_model()

    url = f'{model_urls_base["coordinates_colors_weighted"]}/{STATE_DICT_PREFIX}{EPOCH}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def coordinates_colors_weighted_epoch(epoch=400, progress=True):
    '''
    Returns the model MinkUNet34C from Choey et al. trained on coordinates
    and colors (in_channels=6) with weights, and median class weights, after <epoch> epoch.

    Args:
        epoch (int): Specifies the class weights to be returned.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Example:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'coordinates_colors_weighted_epoch')
    '''
    model = get_coordinates_colors_model()

    url = f'{model_urls_base["coordinates_colors_weighted"]}/{STATE_DICT_PREFIX}{epoch}{STATE_DICT_SUFIX}'

    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=progress)

    return model.load_state_dict(state_dict)


def get_minkunet34c(progress=True, **kwargs):
    '''
    Returns the model MinkUNet34C from Choey et al.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
        **kwargs (dict): Keyword arguments that are passed on to MinkUNet34C. For details, see
            https://github.com/StanfordVL/MinkowskiEngine/blob/master/examples/minkunet.py and
            https://stanfordvl.github.io/MinkowskiEngine/demo/interop.html

    How to use:
        > import torch
        > model = torch.hub.load('MacOS/ReKlaSat-3D', 'get_minkunet34c')
    '''
    return MinkUNet34C(**kwargs)
