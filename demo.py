from torch_mtcnn import detect_faces


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mtcnn.mtcnn import MTCNN
from torchvision.models import resnet18
from models.model_resnet import ResidualNet
import cv2
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
from PIL import Image


def main():
	opt = parse_opts()
	print(opt)

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{opt.gpu}" if use_cuda else "cpu")

	idx_to_class = {0:"mask", 1:"unmask"}
   
	transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	# define model
	model = ResidualNet("ImageNet", opt.depth, opt.num_classes, "CBAM")
	checkpoint = torch.load(opt.resume_path, map_location="cpu")
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()

	# img = cv2.imread(opt.img_path)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# img = Image.fromarray(img)
	# img = transform(img)
	# img = torch.unsqueeze(img, dim=0)

	vid = cv2.VideoCapture(0)

	detector = MTCNN()

	while True:
		try:
			ret, image = vid.read()
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# bboxes, _  = detect_faces(Image.fromarray(image), thresholds=[0.90, 0.91, 0.92])

			faces = detector.detect_faces(image)
			bboxes = []
			for i in range(len(faces)):
				# get coordinates
				x1, y1, width, height = faces[i]['box']
				x2, y2 = x1 + width, y1 + height
				bboxes.append([x1, y1, x2, y2])
				break

			print(bboxes)
			image = image[bboxes[0][1] : bboxes[0][3], bboxes[0][0] : bboxes[0][2]]
			draw = image.copy()
			image = Image.fromarray(image)
			image = transform(image)
			image = torch.unsqueeze(image, dim=0)
			
			with torch.no_grad():
				outputs = model(image)
				outputs = nn.Softmax(dim=1)(outputs)
				scores, indices = torch.max(outputs, 1)
				mask = scores > 0.9
				preds = indices[mask]
				print(scores[mask].item())
				print(preds.item())

				print(idx_to_class[preds.item()])
			cv2.imshow('a', draw)
			cv2.waitKey(1)

				# preds = [idx_to_class[label.item()] for label in preds]
		except:
		    continue

if __name__ == "__main__":
	main()
