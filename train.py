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
from utils.util import AverageMeter, Metric
import numpy as np

def train_epoch(model, data_loader, criterion, optimizer, epoch, device, opt):
   
    model.train()
    
    train_loss = 0.0
    metric = Metric(opt.num_classes)
    losses = AverageMeter()
    # Training
    for i, (data, targets) in enumerate(data_loader):
        # compute outputs
        data, targets = data.to(device), targets.to(device)
        outputs =  model(data)

        # compute loss
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        outputs = torch.sigmoid(outputs)
        metric.update(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show information
        if (i+1) % opt.log_interval == 0:
            avg_loss = train_loss / opt.log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, losses.count, len(data_loader.dataset), 100. * (i + 1) / len(data_loader), avg_loss))
            train_loss = 0.0

    # show information
    mAP = metric.compute_metrics()
    print('Train set ({:d} samples): Average loss: {:.4f}\tmAP: {:.4f}'.format(
        losses.count, losses.avg, mAP))

    return losses.avg, mAP