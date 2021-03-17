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
from utils.util import AverageMeter, accuracy
import numpy as np

def train_epoch(model, data_loader, criterion, optimizer, epoch, device, opt):
   
	model.train()
	
	losses = AverageMeter('Loss', ':.4e')
	accuracies = AverageMeter('Acc@1', ':6.2f')
	# Training
	for i, (data, targets) in enumerate(data_loader):
		# compute outputs
		data, targets = data.to(device), targets.to(device)
		outputs =  model(data)

		# compute loss
		loss = criterion(outputs, targets)
		acc = accuracy(outputs, targets)

		losses.update(loss.item(), data.size(0))
		accuracies.update(acc[0],  data.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# show information
		if (i+1) % opt.log_interval == 0:
			print(f'Epoch: {epoch} [{(i + 1) * len(data)}/{len(data_loader.dataset)}\
				({100. * (i + 1) / len(data_loader) :.0f}%)]  Loss: {losses.avg :.6f}')
		
	# show information
	print(f' * Loss {losses.avg:.3f}, Accuracy {accuracies.avg:.3f}')
	return losses.avg, accuracies.avg