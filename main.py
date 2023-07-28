
import math
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from torchsummary import summary
import argparse
import os

import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dataset import CustomDataLoader
from models.resnet import ResNet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from utils import get_lr
from utils import plot_misclassification


# Set target platform


cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# Custom DataLoader


SEED = 1

# CUDA?
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = (
    dict(shuffle=True, batch_size=512, num_workers=0, pin_memory=True)
    if cuda
    else dict(shuffle=True, batch_size=64)
)

dataloader = CustomDataLoader()

trainset, testset = dataloader.get_train_test_dataset()

# train dataloader
trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

# test dataloader
testloader = torch.utils.data.DataLoader(testset, **dataloader_args)


# Initialize RESNET18 Model


net = ResNet18()
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Model Summary

summary(net, input_size=(3, 32, 32))


# Initialize Optimizer

optimizer = optim.SGD(net.parameters(), lr=0.03, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Learning Rate Finder

lr_finder = LRFinder(net, optimizer, criterion, device=device)
lr_finder.range_test(trainloader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot()  # to inspect the loss-learning rate graph
lr_finder.reset()  # to reset the model and optimizer to their initial state


# Train and Test Loop

train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []


def train(model, device, train_loader, optimizer, epoch, scheduler, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_losses.append(loss)
        lrs.append(get_lr(optimizer))

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} LR={get_lr(optimizer)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_acc.append(100.0 * correct / len(test_loader.dataset))


# Learning Rate Scheduler - OneCycleLR
EPOCHS = 20

scheduler = OneCycleLR(
    optimizer,
    max_lr=1.06E+00,
    steps_per_epoch=len(trainloader),
    epochs=EPOCHS,
    pct_start=5 / EPOCHS,
    div_factor=100,
    three_phase=False,
    final_div_factor=100,
    anneal_strategy="linear",
)


for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(net, device, trainloader, optimizer, epoch, scheduler, criterion)
    test(net, device, testloader, criterion)


train_losses_cpu = [float(x) for x in train_losses]
test_losses_cpu = [float(x) for x in test_losses]


plt.plot(train_losses_cpu, label='train_loss')
plt.legend()
plt.show


plt.plot(test_losses_cpu, label='test_loss')
plt.legend()
plt.show


# Get Misclassified Image, label and prediction
# (Image, actual/target, predicted/pred)
misclassified = list()


def test_misclassified(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            bool_ls = pred.eq(target.view_as(pred)).view_as(target)
            bl = torch.where(bool_ls == False)[0]
            misclassified.append(
                (torch.index_select(data, 0, bl),
                 torch.index_select(target, 0, bl),
                 torch.index_select(pred.view_as(target), 0, bl))
            )


test_misclassified(net, device, testloader)


# Plot Misclassified Images from Testset


plot_sample = 20
shortlisted_misclf_images = plot_misclassification(
    misclassified, plot_sample_count=plot_sample)


# Grad Cam Visualization


target_layers = [net.module.layer3[-1]]

cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)


fig = plt.figure(figsize=(12, 9))
for i in range(len(shortlisted_misclf_images)):
    a = fig.add_subplot(math.ceil(plot_sample/4.0), 4, i+1)
    # All in a batch
    ip_img = shortlisted_misclf_images[i][0]
    # plt.imshow(ip_img)
    input_tensor = torch.Tensor(np.transpose(
        ip_img, (2, 0, 1))).unsqueeze(dim=0)
    targets = [ClassifierOutputTarget(int(shortlisted_misclf_images[i][3]))]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets, aug_smooth=True)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(ip_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)

    a.axis("off")
    title = f"Actual: {shortlisted_misclf_images[i][1]} | Predicted: {shortlisted_misclf_images[i][2]}"
    a.set_title(title, fontsize=10)
plt.savefig(str('misclassified_grad_cam.jpg'), bbox_inches='tight')
