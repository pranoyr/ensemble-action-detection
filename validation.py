import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils.util import AverageMeter, Metric


def val_epoch(model, data_loader, criterion, device, opt):

    model.eval()

    metric = Metric(opt.num_classes)
    losses = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            # compute output
            data, targets = data.to(device), targets.to(device)
            outputs =  model(data)

            # compute loss
            loss = criterion(outputs, targets)

            losses.update(loss.item(), data.size(0))
            outputs = torch.sigmoid(outputs)
            metric.update(outputs, targets)

    # show information
    mAP = metric.compute_metrics()
    print('Validation set ({:d} samples): Average loss: {:.4f}\tmAP: {:.4f}'.format(losses.count, losses.avg, mAP))
    return losses.avg, mAP

    