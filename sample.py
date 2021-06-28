import torch
from skimage.util import random_noise
import cv2
from PIL import Image
import torchvision.transforms as transforms
from transforms import GaussianNoise
import numpy as np

img = Image.open('/Volumes/Neuroplex/iocl/Phoning-Smoking-Datast/JPEGImages/data00049.png')


# gauss_img = torch.tensor(random_noise(np.array(img), mode='gaussian', mean=0, var=0.05, clip=True))
# print(gauss_img.shape)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomCrop(32, padding=3),
    transforms.ColorJitter(brightness=[0.5,1]),
    # transforms.RandomRotation(180),
    GaussianNoise(0.5)])

# img = train_transform(img)

for i in range(1000):
    draw = train_transform(img)
    cv2.imshow('img' ,np.array(draw))
    cv2.waitKey(0)

# cv2.imshow('img' ,gauss_img.numpy())
# cv2.waitKey(0)

