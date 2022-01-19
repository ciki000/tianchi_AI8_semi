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
from config import args_wideresnet, args_preactresnet18
from utils import load_model, AverageMeter, accuracy

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('data.npy')
        labels = np.load('label.npy')
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

class testDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./datasets/eval_image.npy')
        labels = np.load('./datasets/eval_label.npy')
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

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():

    for arch in ['preactresnet18', 'wideresnet']:
        if arch == 'wideresnet':
            args = args_wideresnet
        else:
            args = args_preactresnet18
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        # Model

        model = load_model(arch)
        best_acc = 0  # best test accuracy
        best_epoch = 0

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(args['epochs'])):

            train_loss, train_acc, test_acc= train(trainloader, model, optimizer)
            # print(args)
            print('learning rate:', scheduler.get_lr()[0])
            print('train_acc: {}'.format(train_acc))
            print('test_acc: {}'.format(test_acc))

            # save model
            if (test_acc > best_acc):
                best_acc = test_acc
                best_epoch = epoch + 1
                save_bestcheckpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()

        print('Best acc:')
        print(best_acc)
        print(best_epoch)

def test(model):
    model.eval()
    
    transform_test = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = testDataset(transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)
    
    test_accs = AverageMeter()

    for (input_, soft_label) in testloader:
        input_, soft_label = input_.cuda(), soft_label.cuda()
        target = soft_label.argmax(dim=1)

        model_output = model(input_)

        test_acc = accuracy(model_output, target)
        test_accs.update(test_acc[0].item(), input_.size(0))

    model.train()
    return test_accs.avg

def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    
    test_acc = test(model)
    return losses.avg, accs.avg, test_acc

def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '.pth.tar')
    torch.save(state, filepath)

def save_bestcheckpoint(state, arch):
    filepath = os.path.join(arch + '_best.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    main()
