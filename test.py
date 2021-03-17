import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
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
from dataset import get_training_set, get_validation_set



def main():
	opt = parse_opts()
	print(opt)

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{opt.gpu}" if use_cuda else "cpu")

	train_transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((opt.img_H, opt.img_W)),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])
	test_transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((opt.img_H, opt.img_W)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	training_data = get_training_set(opt, train_transform)
	validation_data = get_validation_set(opt, test_transform)
	
	n_train_examples = int(len(training_data)*0.8)
	n_valid_examples = len(training_data) - n_train_examples
	# split data
	training_data, validation_data = torch.utils.data.random_split(training_data, [n_train_examples, n_valid_examples])

	train_loader = torch.utils.data.DataLoader(training_data,
											   batch_size=opt.batch_size,
											   shuffle=True,
											   num_workers=1)
	val_loader = torch.utils.data.DataLoader(validation_data,
											 batch_size=opt.batch_size,
											 shuffle=True,
											 num_workers=1)
	print(f'Number of training examples: {len(train_loader.dataset)}')
	print(f'Number of validation examples: {len(val_loader.dataset)}')

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')
	# define model
	model = resnet18(num_classes=opt.num_classes)

	# if torch.cuda.device_count() > 1:
	#   	print("Let's use", torch.cuda.device_count(), "GPUs!")
  	# 	model = nn.DataParallel(model)
	model = model.to(device)

	if opt.nesterov:
		dampening = 0
	else:
		dampening = opt.dampening
	#define optimizer and criterion
	# optimizer = optim.Adam(model.parameters())
	# optimizer = optim.SGD(
	# 		model.parameters(),
	# 		lr=opt.learning_rate,
	# 		momentum=opt.momentum,
	# 		dampening=dampening,
	# 		weight_decay=opt.weight_decay,
	# 		nesterov=opt.nesterov)
	# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
	# criterion = nn.CrossEntropyLoss()
	# define optimizer and criterion
	optimizer = optim.Adam(model.parameters())
	# loss function
	criterion = BCEWithLogitsLoss()

	# resume model, optimizer if already exists
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(epoch))
		start_epoch = epoch + 1
	else:
		start_epoch = 1

	# start training
	#th = 10000
	for epoch in range(start_epoch, opt.epochs+1):
		val_loss, val_mAP = val_epoch(model, val_loader, criterion, device, opt)

		


if __name__ == "__main__":
	main()
