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
from utils.util import AverageMeter, accuracy



def val_epoch(model, data_loader, criterion, device, opt):

	model.eval()

	losses = AverageMeter('Loss', ':.2f')
	accuracies = AverageMeter('Acc', ':.2f')
	with torch.no_grad():
		for (data, targets) in data_loader:
			# compute output
			data, targets = data.to(device), targets.to(device)

			outputs =  model(data)
			loss = criterion(outputs, targets)

			acc = accuracy(outputs, targets)
			losses.update(loss.item(), data.size(0))
			accuracies.update(acc[0].item(),  data.size(0))

	# show information
	print(f' * Val Loss {losses.avg:.3f}, Val Acc {accuracies.avg:.3f}')
	return losses.avg, accuracies.avg

	