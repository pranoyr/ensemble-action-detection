import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transforms import GaussianNoise
from efficientnet_pytorch import EfficientNet
import torchvision.datasets as datasets
from torchvision.models import resnet18
from models.model_resnet import ResidualNet
from vit_pytorch.vit import ViT
from torch.utils import data
from torch.utils.data import Subset




import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2, normalize

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
			x = self.transform(image = self.dataset[index][0])["image"]
		else:
			x = self.dataset[index][0]
		y = self.dataset[index][1]
		return x.type(torch.FloatTensor), y
	
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

	# train_transform = transforms.Compose([
	# 	#transforms.RandomCrop(32, padding=3),
	# 	transforms.Resize((256, 256)),
	# 	transforms.RandomHorizontalFlip(0.5),
	# 	transforms.ColorJitter(brightness=[0.2,1]),
	# 	GaussianNoise(0.5),
	# 	# transforms.RandomRotation(10),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
	# 		0.229, 0.224, 0.225])
	# ])

	train_transform = A.Compose([
	A.Resize(256, 256),
	A.OneOf([
	A.HorizontalFlip(p=0.5),
	# A.VerticalFlip(p=0.5),
	A.ShiftScaleRotate(shift_limit= 0.2, scale_limit= 0.2, border_mode=0,
				rotate_limit= 20, value=0, mask_value=0),
	
	# A.RandomResizedCrop(scale = [0.9, 1.0], p=1, height=512, width=512),
	A.GridDropout( holes_number_x=10, holes_number_y=10, ratio=0.4)
	
	]),
	normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
	ToTensorV2(p=1.0)
	])



	test_transform = A.Compose([
		#transforms.RandomCrop(32, padding=3),
		A.Resize(256, 256),
		Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
		ToTensorV2(p=1.0)
	])

	training_data = torchvision.datasets.ImageFolder(
		opt.dataset_path)
	
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
	# model = ResidualNet("ImageNet", opt.depth, opt.num_classes, "CBAM")
	model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=opt.num_classes)
# 	model = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 2,
#     dim = 1024,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

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
