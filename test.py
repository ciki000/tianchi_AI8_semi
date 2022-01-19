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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./datasets/test_Auto_wideresnet_image.npy')
        labels = np.load('./datasets/test_Auto_wideresnet_label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        #self.images = images
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)


use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Data
transform_test = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = MyDataset(transform=transform_test)
testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)

# Model
preactresnet = load_model('preactresnet18').cuda()
preactresnet.load_state_dict(torch.load('./checkpoints/preactresnet_8.pth')['state_dict'])
preactresnet.eval()
wideresnet = load_model('wideresnet').cuda()
wideresnet.load_state_dict(torch.load('./checkpoints/wideresnet_8.pth')['state_dict'])
wideresnet.eval()

preactresnet_accs = AverageMeter()
wideresnet_accs = AverageMeter()

SoftLabels = []
for (input_, soft_label) in tqdm(testloader):
    input_, soft_label = input_.cuda(), soft_label.cuda()
    target = soft_label.argmax(dim=1)

    preactresnet_output = preactresnet(input_)
    # print(target, F.softmax(preactresnet_output, dim=1))
    # break
    wideresnet_output = wideresnet(input_)
    
    # soft_res = F.softmax(preactresnet_output, dim=1).cpu().detach().numpy()
    #soft_wide = F.softmax(wideresnet_output, dim=1).cpu().detach().numpy()

    # for i in range(soft_wide.shape[0]):
    #     SoftLabels.append(soft_wide[i])

    preactresnet_acc = accuracy(preactresnet_output, target)
    wideresnet_acc = accuracy(wideresnet_output, target)
        
    preactresnet_accs.update(preactresnet_acc[0].item(), input_.size(0))
    wideresnet_accs.update(wideresnet_acc[0].item(), input_.size(0))

# soft_labels = np.array(SoftLabels)
# np.save('soft_label.npy', soft_labels)
print(preactresnet_accs.avg, wideresnet_accs.avg)
