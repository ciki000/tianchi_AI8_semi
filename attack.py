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
from torchattacks import CW, PGD, DIFGSM, AutoAttack, APGD, Jitter

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./datasets/cifar_test_image.npy')
        labels = np.load('./datasets/cifar_test_label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def FGSM(model, x, label, eps=0.001):
    x_new = x 
    x_new = Variable(x_new, requires_grad=True)
    
    y_pred = model(x_new)
    loss = cross_entropy(y_pred, label)

    model.zero_grad()
    loss.backward()
    grad = x_new.grad.cpu().detach().numpy()
    grad = np.sign(grad)
    pertubation = grad * eps
    adv_x = x.cpu().detach().numpy() + pertubation
    #adv_x = np.clip(adv_x, clip_min, clip_max)

    x_adv = torch.from_numpy(adv_x).cuda()
    return x_adv

def attack(models, x, y, iter=10, eps=0.001):
    
    ## My implementation

    # for i in range(iter):
    #     for model in models:
    #         x = FGSM(model, x, label, eps)



    ## Use deeprobust

    # PGD
    # adversary_preactresnet = PGD(models[0])
    # adversary_wideresnet = PGD(models[1])
    # attack_params = {'epsilon': 0.1/iter, 'clip_max': 10000.0, 'clip_min': -10000.0, 'num_steps': 5, 'print_process': False}
    # for i in range(iter):
    #     x = adversary_preactresnet.generate(x, y, **attack_params)
    #     x = adversary_wideresnet.generate(x, y, **attack_params)

    # CW
    # adversary_preactresnet = CarliniWagner(models[0])
    # adversary_wideresnet = CarliniWagner(models[1])
    # attack_params = {'epsilon': 0.1/iter, 'clip_max': 10000.0, 'clip_min': -10000.0, 'num_steps': 5, 'print_process': False}
    # for i in range(iter):
    #     x = adversary_preactresnet.generate(x, y, **attack_params)
    #     x = adversary_wideresnet.generate(x, y, **attack_params)



    ## Use torchattacks

    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    norm_preactresnet = nn.Sequential(
        norm_layer,
        models[0]
    ).cuda()
    norm_preactresnet.eval()

    norm_wideresnet = nn.Sequential(
        norm_layer,
        models[1]
    ).cuda()
    norm_wideresnet.eval()

    labels = torch.topk(y, 1)[1].squeeze(1)
    
    # atk_preactresnet = CW(norm_preactresnet, c=1, kappa=0, steps=1000, lr=0.01)
    # atk_preactresnet = PGD(norm_preactresnet, eps=8/255, alpha=1/255, steps=40, random_start=True)
    # atk_preactresnet = DIFGSM(norm_preactresnet, eps=8/255, alpha=2/255, decay=0.0, steps=20, random_start=True)
    # atk_preactresnet = AutoAttack(norm_preactresnet, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    # atk_preactresnet = APGD(norm_preactresnet, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # atk_preactresnet = Jitter(norm_preactresnet, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)

    # atk_wideresnet = CW(norm_wideresnet, c=1, kappa=0, steps=1000, lr=0.01)
    # atk_wideresnet = PGD(norm_wideresnet, eps=8/255, alpha=1/255, steps=40, random_start=True)
    # atk_wideresnet = DIFGSM(norm_wideresnet, eps=8/255, alpha=2/255, decay=0.0, steps=20, random_start=True)
    atk_wideresnet = AutoAttack(norm_wideresnet, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    # atk_wideresnet = APGD(norm_wideresnet, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # atk_wideresnet = Jitter(norm_wideresnet, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
    
    # adv_images = atk_preactresnet(x, labels)
    adv_images = atk_wideresnet(x, labels)
    return adv_images

use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Data
transform_test = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = MyDataset(transform=transform_test)
testloader = data.DataLoader(testset, batch_size=256, shuffle=False)

# Model
preactresnet = load_model('preactresnet18').cuda()
preactresnet.load_state_dict(torch.load('./checkpoints/preactresnet_train.pth')['state_dict'])
preactresnet.eval()
wideresnet = load_model('wideresnet').cuda()
wideresnet.load_state_dict(torch.load('./checkpoints/wideresnet_train.pth')['state_dict'])
wideresnet.eval()

preactresnet_accs = AverageMeter()
wideresnet_accs = AverageMeter()
inputs_adv = []
labels = []
cnt = 0
for (input_, soft_label) in tqdm(testloader):
    input_, soft_label = input_.cuda(), soft_label.cuda()

    models = [preactresnet, wideresnet]
    x = Variable(input_)
    x = attack(models, x, soft_label)

    inv_normalize = transforms.Normalize((-2.4290657439446366, -2.418254764292879, -2.2213930348258706), (4.9431537320810675, 5.015045135406218, 4.975124378109452))
    for i in range(x.shape[0]):
        #inputs_adv.append(np.clip(inv_normalize(x[i].squeeze()).cpu().detach().numpy().transpose((1,2,0)), 0, 1)*255)
        inputs_adv.append(np.clip(x[i].squeeze().cpu().detach().numpy().transpose((1,2,0)), 0, 1)*255)
        labels.append(soft_label[i].squeeze().cpu().numpy())

    # cnt = cnt + 1
    # if (cnt >= 100):
    #     break

#images_adv = np.array(inputs_adv).astype(np.uint8)
images_adv = np.round(np.array(inputs_adv)).astype(np.uint8)
labels_adv = np.array(labels)

np.save('./datasets/test_Auto_wideresnet_image.npy', images_adv)
np.save('./datasets/test_Auto_wideresnet_label.npy', labels_adv)