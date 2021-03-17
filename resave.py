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
import cv2
import os
import random
import numpy as np
from train import train_epoch
from torch.nn import BCEWithLogitsLoss
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from datasets.veri_vehicle_attributes import class_mapping
from PIL import Image



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

	class_to_idx = class_mapping()
	idx_to_class = {}
	for name, label in class_to_idx.items():
		idx_to_class[label] = name
   
	transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((opt.img_H, opt.img_W)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	# define model
	model = resnet18(num_classes=40)
	# load weights
	checkpoint = torch.load('/Users/pranoyr/Downloads/model100.pth', map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	torch.save(model.state_dict(), '/Users/pranoyr/Downloads/vehicle_classifier.pth', _use_new_zipfile_serialization=False)

	

if __name__ == "__main__":
	main()
