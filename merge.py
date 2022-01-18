from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from utils import load_model, AverageMeter, accuracy


datasets = ['./datasets/train1_wasserstein_large', './datasets/train2_PGD_preactresnet', './datasets/train3_PGD_wideresnet', './datasets/train4_light', './datasets/train5_wasserstein']
# datasets = ['./datasets/cifar_train1', './datasets/train2_PGD-8_densenet', './datasets/train3_PGD-8_resnet', './datasets/train4_PGD-4_densenet', './datasets/train5_PGD-4_resnet']
images = []
labels = []
cnt = 0
for dataset in datasets:
    cur_images = np.load(dataset+'_image.npy')
    cur_labels = np.load(dataset+'_label.npy')
    for i in range(cur_images.shape[0]):
        cnt = cnt + 1
        images.append(cur_images[i])
        labels.append(cur_labels[i])

index = [x for x in range(cnt)]
random.shuffle(index)

random_images = []
random_labels = []
for i in range(cnt):
    random_images.append(images[index[i]])
    random_labels.append(labels[index[i]])

images_merge = np.array(random_images).astype(np.uint8)
labels_merge = np.array(random_labels)

print(images_merge.shape, labels_merge.shape)
np.save('data.npy', images_merge)
np.save('label.npy', labels_merge)