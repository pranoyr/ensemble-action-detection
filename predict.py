import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from models.model_resnet import ResidualNet
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

    idx_to_class = {"mask":0, "unmask":1}
   
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

    img = cv2.imread(opt.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        outputs = model(img)
        outputs = torch.sigmoid(outputs)
        scores, indices = torch.topk(outputs, dim=1, k=1)
        mask = scores > 0.5
        preds = indices[mask]
    
        preds = [idx_to_class[label.item()] for label in preds]
        print(preds)

if __name__ == "__main__":
    main()
