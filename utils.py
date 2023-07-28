"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def inverse_normalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    # if mean.ndim == 1:
    #     mean = mean.view(-1, 1, 1)
    # if std.ndim == 1:
    #     std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def imshow(img):
    inverse_normalize(img)
    # inverse_normalize(img)
    npimg = img.numpy()
    # plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_misclassification(misclassified, plot_sample_count=20):
    shortlisted_misclf_images = list()
    mc_list_index = torch.randint(0, len(misclassified), (1,))[0]
    print(mc_list_index)

    fig = plt.figure(figsize=(12, 9))
    for i in range(plot_sample_count):
        a = fig.add_subplot(math.ceil(plot_sample_count/4.0), 4, i+1)
        # All in a batch
        batch_len = misclassified[mc_list_index][0].shape[0] - 1
        batch_idx = torch.randint(0, batch_len, (1,))[0]
        image = misclassified[mc_list_index][0][batch_idx]  # Image
        actual = misclassified[mc_list_index][1][batch_idx]  # Actual
        predicted = misclassified[mc_list_index][2][batch_idx]  # Predicted
        npimg = image.cpu().numpy()
        nptimg = np.transpose(npimg, (1, 2, 0))
        inverse_normalize(torch.Tensor(nptimg))
        plt.imshow(nptimg)
        shortlisted_misclf_images.append(
            (nptimg, classes[actual], classes[predicted], actual, predicted))
        a.axis("off")
        title = f"Actual: {classes[actual]} | Predicted: {classes[predicted]}"
        a.set_title(title, fontsize=10)
    plt.savefig(str('misclassified.jpg'), bbox_inches='tight')
    return shortlisted_misclf_images


def get_lr(optimizer):
    """"
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
