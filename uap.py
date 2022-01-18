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

from torchattacks.attack import Attack
from torch.nn.modules.loss import _WeightedLoss, _Loss

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./datasets/cifar_train3_image.npy')
        labels = np.load('./datasets/cifar_train3_label.npy')
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

def one_hot(class_labels, num_classes=None): 
    if num_classes==None: 
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.) 
    else: 
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.) 

class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

class UAP(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=0.3,
                 alpha=1/255, steps=40, random_start=True):
        super().__init__("UAP", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = BoundedLogitLossFixedRef(num_classes=10, confidence=10, use_cuda=True)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            #adv_images = torch.clamp(images + delta, min=0, max=1).detach()-images.detach()

        return delta

def gen_uap(inputs_adv, labels, ori_inputs):
    UAP = torch.zeros(size=(10,3,32,32)).cuda()
    for idx, image in enumerate(inputs_adv):
        #print(image.size())
        UAP[labels[idx]] += image
        UAP[labels[idx]] = torch.sign(UAP[labels[idx]])*torch.clamp(abs(UAP[labels[idx]]), 0, 10/255)#np.minimum(abs(UAP[labels[idx]]), 8) #���8��eps
    for idx, image in enumerate(ori_inputs):
        image = image + UAP[labels[idx]] 
        ori_inputs[idx] = image
    return ori_inputs

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
    # atk_wideresnet = AutoAttack(norm_wideresnet, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
    # atk_wideresnet = APGD(norm_wideresnet, norm='Linf', eps=8/255, steps=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # atk_wideresnet = Jitter(norm_wideresnet, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
    atk_wideresnet = UAP(norm_wideresnet, eps=8/255, alpha=2/255, steps=40, random_start=True)
    
    # adv_images = atk_preactresnet(x, labels)
    delta = atk_wideresnet(x, labels)
    return delta

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
    
    delta = attack(models, x, soft_label)

    uap_labels = torch.topk(soft_label, 1)[1].squeeze(1)
    x = gen_uap(delta, uap_labels, x)


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

np.save('./datasets/train3_uap_wideresnet_image.npy', images_adv)
np.save('./datasets/train3_uap_wideresnet_label.npy', labels_adv)