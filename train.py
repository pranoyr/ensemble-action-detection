import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
import time
import argparse
import tensorboardX
import os
import random
from utils.util import AverageMeter, accuracy, ProgressMeter
import numpy as np

def train_epoch(model, data_loader, criterion, optimizer, epoch, device, opt):
   
	model.train()
	
	losses = AverageMeter('Loss', ':.4e')
	accuracies = AverageMeter('Acc', ':6.2f')
	progress = ProgressMeter(
        len(data_loader),
        [losses, accuracies],
        prefix='Train: ')
	# Training
	for batch_idx, (data, targets) in enumerate(data_loader):
		# compute outputs
		data, targets = data.to(device), targets.to(device)

		outputs =  model(data)
		loss = criterion(outputs, targets)

		acc = accuracy(outputs, targets)
		losses.update(loss.item(), data.size(0))
		accuracies.update(acc[0].item(),  data.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# show information
		if batch_idx % opt.log_interval == 0:
			progress.display(batch_idx)
		
	# show information
	print(f' * Train Loss {losses.avg:.3f}, Train Acc {accuracies.avg:.3f}')
	return losses.avg, accuracies.avg