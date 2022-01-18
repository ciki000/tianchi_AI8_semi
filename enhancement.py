from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from utils import load_model, AverageMeter, accuracy

import torchattacks
from torchattacks import CW
import cv2

def lightchange(image):
    
    # base: 0.8-0.9
    # 2: 0.85-0.95
    # 3: 0.75-0.88
    light = random.uniform(0.8, 0.9)
    image  = np.power(image, light)
 
    return image

def addsaltnoise(image, p=0.8):
    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]
    #print(h, w)
    mask = np.random.choice((0,1,2), size=(h, w, 1), p=[(1-p)/2., (1-p)/2., p])
    mask = np.repeat(mask, c, axis=2)
    image[mask == 0] = 0
    image[mask == 1] = 255
    return image

def addgaussiannoise(image, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    image = np.clip(image+noise*255, 0, 255)
    return image

image = np.load('./datasets/cifar_test_image.npy')
label = np.load('./datasets/cifar_test_label.npy')

# 亮度调节
for i in range(image.shape[0]):
    image[i] = lightchange(image[i])

# 模糊
# for i in range(image.shape[0]):
#      image[i] = cv2.GaussianBlur(image[i], (3,3), 0)

# 椒盐
# for i in range(image.shape[0]):
#     image[i] = addsaltnoise(image[i])

  
# 高斯噪声
# for i in range(image.shape[0]):
#     image[i] = addgaussiannoise(image[i])

image = np.round(image).astype(np.uint8)
#print(image.shape)
np.save('./datasets/test_light_image.npy', image)
np.save('./datasets/test_light_label.npy', label)