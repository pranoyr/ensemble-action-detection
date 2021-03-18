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
from utils.util import AverageMeter, accuracy, ProgressMeter
import numpy as np

def train_epoch(model, data_loader, criterion, optimizer, epoch, device, opt):
   
	model.train()
	
	losses = AverageMeter('Loss', ':.4e')
	accuracies = AverageMeter('Acc', ':6.2f')
	batch_time = AverageMeter('Time', ':6.3f')
	progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, accuracies],
        prefix='Train: ')
	# Training
	for batch_idx, (data, targets) in enumerate(data_loader):
		# compute outputs
		data, targets = data.to(device), targets.to(device)
		outputs =  model(data)

		# compute loss
		loss = criterion(outputs, targets)
		acc = accuracy(outputs, targets)

		losses.update(loss.item(), data.size(0))
		accuracies.update(acc[0].item(),  data.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# show information
		if batch_idx % opt.log_interval == 0:
			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(data_loader.dataset),
            #     100. * batch_idx / len(data_loader), losses.avg))
			progress.display(batch_idx)
		
	# show information
	print(f' * Loss {losses.avg:.3f}, Accuracy {accuracies.avg:.3f}')
	return losses.avg, accuracies.avg