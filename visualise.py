import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from models.model_resnet import ResidualNet
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorboardX
import cv2
import os
from sklearn.manifold import TSNE
import random
import numpy as np
from train import train_epoch
from torch.nn import BCEWithLogitsLoss
from validation import val_epoch
from opts import parse_opts
from torch.optim import lr_scheduler
from PIL import Image


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


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

    labels = {0:"Mask", 1:"UnMask"}
   
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
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = nn.Sequential(*list(model.children())[:-1])
    # print(model)
    # print(model)
    model = model.to(device)
    model.eval()

    X = []
    labels = []
    for img in os.listdir("/Volumes/Neuroplex/kdisc/mask-demo-data/"):
        img_path = f"/Volumes/Neuroplex/kdisc/mask-demo-data/{img}"
        print(img_path)
        img = cv2.imread(img_path)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = torch.from_numpy(img)
        img = Image.fromarray(img)
        img = transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            outputs = model(img)
            X.append(outputs.view(1,-1))

        if ('un' in img_path):
            labels.append("UnMask")
        else:
            labels.append("Mask")
    return X, labels


x, labels = main()
# X = np.array([[0, 0, 0], [0.01, 0.1, 0.1], [1, 0, 1], [1, 0, 1]])
print(torch.cat(x).numpy().shape)
X_embedded = TSNE(n_components=2, n_iter=1000, verbose=True).fit_transform(torch.cat(x).numpy())
# X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(torch.cat(x).numpy())
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = X_embedded[:, 0]
ty = X_embedded[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
colors_per_class = {'Mask':(255,0,0), 'UnMask':(0,255,0)}
# labels = ["Mask"] * 21

for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
# plt.show()
plt.savefig("output.jpg")
       
