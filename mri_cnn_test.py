import glob
import os
import os.path as osp
import pickle
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt

import datasets.dataset as dataset
from datasets.dataset import load_data, CLASS_MAP
import models.models as models
from utils.data_class import BrainDataset
import torchio as tio
#from models.models import FujiNet1 #, Vgg16,

SEED_VALUE = 2481
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

os.environ["CUDA_VISIBLE_DEVICES"]="3"
#この環境変数は最初に宣言しないと有効にならない
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD"], unique=False)
# dataset = load_data(kinds=["PPMI"], classes=["Control"], unique=True)
# dataset = load_data(kinds=["OASIS"], classes=["Control"], unique=True)

# trainとtest用の画像を、同じ患者が分かれて入らないように分ける。
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# train_datadict, val_datadict = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=SEED_VALUE)
pids = []
for i in range(len(dataset)):
    pids.append(dataset[i]["pid"])
gss = GroupShuffleSplit(test_size=1-0.8, random_state=SEED_VALUE)
train_idx, val_idx = list(gss.split(dataset, groups=pids))[0]
train_datadict = dataset[train_idx]
val_datadict = dataset[val_idx]

#  len(train_datadict)

# TorchIO
class ImageTransformio():
    def __init__(self):
        self.transform = {
            "train": tio.Compose([
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=10, isotropic=True,
                                 center="image", default_pad_value="mean", image_interpolation='linear'),
                # tio.transforms.RandomNoise(),
                # tio.transforms.RandomBiasField(),
                # tio.ZNormalization(),
                #tio.transforms.RescaleIntensity((0, 1))
            ]),
            "val": tio.Compose([
                # tio.ZNormalization(),
                # tio.RescaleIntensity((0, 1))  # , in_min_max=(0.1, 255)),
            ])
        }

    def __call__(self, img, phase="train"):
        img_t = torch.tensor(img)
        return self.transform[phase](img_t)

# train/val dataset を作成
train_dataset = BrainDataset(data_dict=train_datadict, transform=ImageTransformio(), phase="train")
val_dataset = BrainDataset(data_dict=val_datadict, transform=ImageTransformio(), phase="val")

print("size of the training dataset = ", len(train_dataset))
print("size of the validation dataset = ", len(val_dataset))
print(f"training image shape = {train_dataset(0)[0].shape}, training label = {train_dataset(0)[1]}")
print(f"test image shape = {val_dataset(0)[0].shape},     test label = {val_dataset(0)[1]}")

# 画像可視化関数
"""
def show_slice(gazo):

    def _voxel2slice(voxel: np.array, aspect: str, slice_idx: int) -> np.array:
        if aspect == 'sagittal':
            slice_img = np.flip(voxel.transpose((0, 2, 1))[slice_idx], 0)
        elif aspect == 'coronal':
            slice_img = np.flip(voxel.transpose((1, 2, 0))[slice_idx], 0)
        elif aspect == 'transverse':
            slice_img = np.flip(voxel.transpose((2, 1, 0))[slice_idx], 0)
        return slice_img

    fig = plt.figure(figsize=(9,3))
    trans = fig.add_subplot(1, 3, 1)
    trans.set_title("transverse", fontsize=12)
    trans.imshow(_voxel2slice(gazo, 'transverse', 50), cmap='gray')
    coronal = fig.add_subplot(1, 3, 2)
    coronal.set_title("coronal", fontsize=12)
    coronal.imshow(_voxel2slice(gazo, 'coronal', 50), cmap='gray')
    sagittal = fig.add_subplot(1, 3, 3)
    sagittal.set_title("sagittal", fontsize=12)
    sagittal.imshow(_voxel2slice(gazo, 'sagittal', 50), cmap='gray')

    fig.show()
"""

idx = 20
img1, label1 = train_dataset(idx)
img2, label2 = train_dataset(idx+1)

print("mean=", img1.mean())
print(f"max={img1.max()} min={img1.min()}")
print(label1)

#show_slice(img1.numpy().reshape(80, 96, 80))
#show_slice(img2.numpy().reshape(80, 96, 80))

#imge = np.clip(image, 0, None)
# print(img1.numpy().reshape(80, 80, 80).mean())
# plt.imshow(np.flip(img1.numpy().reshape(80, 80, 80).transpose(2,0,1)[50],0), cmap="gray")
# plt.imshow(np.flip(img2.numpy().reshape(80, 80, 80).transpose(2,0,1)[50],0), cmap="gray")

# 画像１枚あたりの輝度値ヒストグラム
#idx = 20
# image, label = train_dataset(idx)
# print("mean=", image.mean())
# print(f"max={image.max()} min={image.min()}")
# print(label)
# imge = np.clip(image, 0, None)
# show_img = image.numpy().reshape(80*96*80)
#plt.hist(show_img[show_img > 0.01], bins=255)
#plt.title("Accuracy")
#plt.xlabel("Epoch")
#plt.ylabel("Accuracy")
##plt.legend()

image_list = []
max = 0.01
for image, label in train_dataset:
    if max < image.max():
        max = image.max()
    image_reshape = image.numpy().reshape(80*96*80)
    image_list.append(image_reshape)

for image, label in val_dataset:
    if max < image.max():
        max = image.max()
    image_reshape = image.numpy().reshape(80*96*80)
    image_list.append(image_reshape)
print(len(image_list))

"""
imagelist = np.concatenate(image_list)
plt.title("Histogram of intensity rescale=(0, 1), in_min_max=(1, 255)")
plt.xlabel("Intensity")
plt.ylabel("Number")
plt.hist(imagelist[imagelist > 0.01], bins=255)

"""

print(max)

# 画像の定量評価
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

idx = 20
img1, label1 = train_dataset(idx)
img2, label2 = train_dataset(idx+1)
img1 = np.flip(img1.numpy().reshape(80, 96, 80).transpose(2,0,1)[50],0)
img2 = np.flip(img2.numpy().reshape(80, 96, 80).transpose(2,0,1)[50],0)

mse_none = mean_squared_error(img1, img2)
ssim_none = ssim(img1, img2)

print(ssim_none)

train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=os.cpu_count(), pin_memory=True, shuffle=False)

for inputs, labels in train_dataloader:
    print(labels)

# net choose 
#
net = models.LuckyNet1()
#print(net)

torch.nn.init.kaiming_normal(net.conv1.weight)
torch.nn.init.kaiming_normal(net.conv2.weight)
torch.nn.init.kaiming_normal(net.conv3.weight)
torch.nn.init.kaiming_normal(net.conv4.weight)
torch.nn.init.kaiming_normal(net.fc1.weight)
torch.nn.init.kaiming_normal(net.fc2.weight)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(params=net.parameters(), lr=0.0005, momentum=0.9)
optimizer = optim.Adam(params=net.parameters(), lr=0.001)

def train_model(net, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("Use divice = ", device)

    for epoch in range(num_epochs):
        # train
        net.train()
        loss_avg = 0.0
        acc_avg = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device=device, dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            acc_avg += torch.mean((preds == labels).float()).item() / len(train_dataloader)
            loss_avg += loss.item() / len(train_dataloader)
        train_losses.append(loss_avg)
        train_accs.append(acc_avg)

        # evaluate
        loss_avg = 0.0 
        acc_avg = 0.0
        net.eval()
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            acc_avg += torch.mean((preds == labels).float()).item() / len(val_dataloader)
            loss_avg += loss.item() / len(val_dataloader)

        test_losses.append(loss_avg)
        test_accs.append(acc_avg)
        print(f"EPOCH {epoch+1}  || train loss : {train_losses[epoch]:.4f}, test loss : {test_losses[epoch]:.4f} \
            || train acc : {train_accs[epoch]:.4f} || test acc : {test_accs[epoch]:.4f}")


num_epochs = 250

train_losses = []
test_losses = []
train_accs = []
test_accs = []

train_model(net, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)

fig = plt.figure()
plt.plot(range(250), train_losses, label="train loss")
plt.plot(range(250), test_losses, label="test loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("/data2/lucky/result_fig/2classification_by_cnn/Luckynet_wobatch_acc34.png")

fig88 = plt.figure()
plt.plot(range(250), train_accs, label="train acc")
plt.plot(range(250), test_accs, label="test acc")
plt.title("Accs")
plt.xlabel("Epoch")
plt.ylabel("Accs")
plt.legend()
plt.savefig("/data2/lucky/result_fig/2classification_by_cnn/Luckynet_wobatch_loss34.png")

