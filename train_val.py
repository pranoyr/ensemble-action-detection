import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from models.model_resnet import ResidualNet
from torch.utils import data
from torch.utils.data import Subset

import argparse
import tensorboardX
import os
import random
import numpy as np
from train import train_epoch
from torch.nn import BCEWithLogitsLoss
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from PIL import Image
from dataset import get_training_set, get_validation_set


class MyLazyDataset(data.Dataset):
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __getitem__(self, index):
		if self.transform:
			x = self.transform(Image.fromarray(self.dataset[index][0]))
		else:
			x = Image.fromarray(self.dataset[index][0])
		y = self.dataset[index][1]
		return x, y
	
	def __len__(self):
		return len(self.dataset)


def main():
	opt = parse_opts()
	print(opt)

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{opt.gpu}" if use_cuda else "cpu")

	train_transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		# transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])
	test_transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	training_data = torchvision.datasets.ImageFolder(
		opt.dataset_path, transform=train_transform)
	
	traindataset = MyLazyDataset(training_data, train_transform)
	valdataset = MyLazyDataset(training_data,test_transform)
	
	# Create the index splits for training, validation and test
	train_size = 0.8
	num_train = len(training_data)
	indices = list(range(num_train))
	split = int(np.floor(train_size * num_train))
	split2 = int(np.floor((train_size+(1-train_size)/2) * num_train))
	np.random.shuffle(indices)
	train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

	traindata = Subset(traindataset, indices=train_idx)
	valdata = Subset(valdataset, indices=valid_idx)
	train_loader = torch.utils.data.DataLoader(traindata,
											   batch_size=opt.batch_size,
											   shuffle=True,
											   num_workers=0)
	val_loader = torch.utils.data.DataLoader(valdata,
											 batch_size=opt.batch_size,
											 shuffle=True,
											 num_workers=0)
	print(f'Number of training examples: {len(train_loader.dataset)}')
	print(f'Number of validation examples: {len(val_loader.dataset)}')

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')
	# define model
	model = ResidualNet("ImageNet", opt.depth, opt.num_classes, "CBAM")
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(epoch))
		opt.start_epoch = epoch + 1
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), weight_decay=opt.wt_decay)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	th = 100000
	# start training
	for epoch in range(opt.start_epoch, opt.epochs+1):
		# train, test model
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, opt)
		val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
		scheduler.step(val_loss)

		lr = optimizer.param_groups[0]['lr']  
		
		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc, global_step=epoch)
			summary_writer.add_scalar(
				'acc/val_acc', val_acc, global_step=epoch)
			summary_writer.add_scalar(
				'lr_rate', lr, global_step=epoch)

			state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict':scheduler.state_dict()}
			if val_loss < th:
				torch.save(state, os.path.join('./snapshots', f'{opt.dataset}_model.pth'))
				print("Epoch {} model saved!\n".format(epoch))
				th = val_loss


if __name__ == "__main__":
	main()
