import datetime
import time
import sys
import pytz
import random
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from adabelief_pytorch import AdaBelief
import torchvision
from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import(Dataset,DataLoader,TensorDataset)
import tqdm
import gzip
import os
import urllib.request
import sys
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import math
from models import (resnet_cifar100,resnet_cifar10,mnist_mlp)
def get_model(dataset_name):
    if dataset_name=='CIFAR100' or 'CIFAR10':
        print('ResNet34')
        return resnet_cifar100.ResNet34()
    
    elif dataset_name=='MNIST':
        print('mnist_mlp')
        return mnist_mlp.MLP()
    
    elif dataset_name=='imagenet':
        print('resnet50')
        resnet = torchvision.models.resnet34(pretrained=False)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 1000)
        return resnet
    
def eval_net(net,dataset,device):
    loader = DataLoader(dataset, drop_last=False, batch_size=1024)
    net.eval()
    ys=[]
    ypreds=[]
    for x,y in (loader):
        x=x.to(device)
        y=y.to(device)
        with torch.no_grad():
            _,y_pred=net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
        ys=torch.cat(ys)
        ypreds=torch.cat(ypreds)
        acc=(ys==ypreds).float().sum()/len(ys)
        return acc.item()

def compute_loss(net,dataset,device):
    loader = DataLoader(dataset, drop_last=False, batch_size=128)
    net.eval()
    score_sum=0.
    for images,labels in (loader):
        images,labels=images.to(device),labels.to(device)
        score_sum+=nn.CrossEntropyLoss()(net(images),labels.view(-1)).item()
    score=float(score_sum/len(loader))
    return score

def get_dataset(dataset_name):
    if dataset_name=='CIFAR100':
        print(dataset_name)
        return (CIFAR100('./data',
        train=True,download=True,transform=transforms.ToTensor()),
        CIFAR100('./data',
        train=False,download=True,transform=transforms.ToTensor()))

    elif dataset_name=='MNIST':
        return (MNIST('./data',
        train=True,download=True,transform=transforms.ToTensor()),
        MNIST('./data',
        train=False,download=True,transform=transforms.ToTensor()))
 
    elif dataset_name=='CIFAR10':
        print(dataset_name)
        return (CIFAR10('./data',
        train=True,download=True,transform=transforms.ToTensor()),
        CIFAR10('./data',
        train=False,download=True,transform=transforms.ToTensor()))
 
    elif dataset_name=='imagenet':
        transform = transforms.Compose(
        [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        trainset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_train', transform=transform)

        testset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_val_for_ImageFolder', transform=transform)


        return trainset,testset
    
    else:
        print(f'{dataset_name} is not exist')
        sys.exit()

def get_dir_name():
    dir_name='step_size_list'
    dir_count=0
    while(os.path.isdir('result/'+dir_name+'_'+str(dir_count))):
        if (sum(os.path.isfile(os.path.join('result/'+dir_name+'_'+str(dir_count),name)) for name in os.listdir('result/'+dir_name+'_'+str(dir_count))))==0:
            dir_name=dir_name+'_'+str(dir_count)
            return dir_name
        dir_count+=1
    dir_name=dir_name+'_'+str(dir_count)
    os.mkdir('result/'+dir_name)
    return dir_name

def mkdir(dir_name):
    print('mkdir '+dir_name)
    os.mkdir('result/'+dir_name)

def get_file_name(dataset_name,batch_size,alg_name):
    file_name=dataset_name+'_'+alg_name+'_'+str(batch_size)+'.bin'
    return file_name



def eval_5_net(net,dataset,device):
    loader = DataLoader(dataset, drop_last=False, batch_size=1000)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.topk(outputs, 5, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1, 1)).sum().item()
    

    return correct/total

def train_net(dataset_name,net,train_set,test_set,optimizer,n_iter,device,alg_name,batch_size,):
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=2)
    train_losses=[]
    train_acc=[]
    val_acc=[]
    val_acc_5=[]
    timesCPU = []
    epochs=[]
    iterations=[]
    test_acces=[]
    grad_norms=[]
    step_size_list=[]
    if_iter_zero=0
    timesCPU_i=0
    iteration=0
    for epoch in range(n_iter):
        if if_iter_zero==0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            timesCPU.append(0)
            if_iter_zero=1
        net.train()
        not_found_inepoch=0
        
        for i,(xx,yy)in (enumerate(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            h=net(xx)
            optimizer.zero_grad()
        
            closure=lambda:nn.CrossEntropyLoss(reduction='mean')(net(xx),yy)
            step_size,loss,grad_norm,not_found=optimizer.step(closure)
            step_size_list.append(step_size)
            grad_norm=grad_norm.to('cpu').detach().numpy().copy()
            grad_norms.append(grad_norm)
            not_found_inepoch+=not_found


        train_loss=compute_loss(net,train_set,device)
        epochs.append(epoch)
        timesCPU_i = time.process_time() - start_time_cpu
        train_losses.append(train_loss)
        timesCPU.append(timesCPU_i)
        train_acc.append(eval_net(net,train_set,device))
        val_acc.append(eval_net(net,test_set,device))
        val_acc_5.append(eval_5_net(net,test_set,device))

        ptrain_losses='{:.5f}'.format(train_losses[-1])
        ptrain_acc='{:.5f}'.format(train_acc[-1])
        pval_acc='{:.5f}'.format(val_acc[-1])
        pval_acc_5='{:.5f}'.format(val_acc_5[-1])
        pgrad_norm='{:.5f}'.format(grad_norm)
        print(f'e:{epoch},l:{ptrain_losses},t_acc{ptrain_acc},v_acc:{pval_acc},val_acc_5:{pval_acc_5},α:{step_size_list[-1]},g:{pgrad_norm},t:{timesCPU[-1]},n_f:{not_found_inepoch}')


    timesCPU=np.asarray(timesCPU)
    step_size_list=np.asarray(step_size_list)
    train_losses=np.asarray(train_losses)
    train_acces=np.asarray(train_acc)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    test_acces=np.asarray(val_acc)
    epochs=np.asarray(epochs)
    grad_norms=np.asarray(grad_norms)

    dict_result = {'algorithm' : alg_name,
                'step_size':step_size_list,
                'dataset' : dataset_name,
                'train_losses': train_losses,
                'train_acces': train_acces,
                'test_acces':test_acces,
                'timesCPU': timesCPU,
                'epochs': epochs,
                'grad_norms':grad_norms}

    return dict_result

   


def train_sfo(dataset_name,net,train_set,test_set,optimizer,n_iter,device,alg_name,batch_size,):
    print('train_sfo')
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    loss_fn=nn.CrossEntropyLoss()
    train_losses=[]
    train_acc=[]
    val_acc=[]
    val_acc_5=[]
    timesCPU = []
    epochs=[]
    test_acces=[]
    grad_norms=[]
    step_size_list=[]
    if_iter_zero=0
    timesCPU_i=0
    iteration=0

    for epoch in range(n_iter):
        if if_iter_zero==0:
            start_time_wall_clock = time.time()
            start_time_cpu = time.process_time()
            timesCPU.append(0)
            if_iter_zero=1
        net.train()
        not_found_inepoch=0
        
        for i,(xx,yy)in (enumerate(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            h=net(xx)
            optimizer.zero_grad()

            closure=lambda:nn.CrossEntropyLoss(reduction='mean')(net(xx),yy)
            step_size,loss,grad_norm,not_found,=optimizer.step(closure)
            step_size_list.append(step_size)
            grad_norm=grad_norm.to('cpu').detach().numpy().copy()
            grad_norms.append(grad_norm)
            not_found_inepoch+=not_found


            if grad_norm<=0.01:
                print(f'e:{epoch},l:{train_losses},t_acc{train_acc},v_acc:{val_acc},val_acc_5:{val_acc_5},α:{step_size_list[-1]},g:{grad_norm},t:{timesCPU[-1]},n_f:{not_found_inepoch}')
                timesCPU=np.asarray(timesCPU)
                step_size_list=np.asarray(step_size_list)
                train_losses=np.asarray(train_losses)
                train_acces=np.asarray(train_acc)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                test_acces=np.asarray(val_acc)
                epochs=np.asarray(epochs)
                grad_norms=np.asarray(grad_norms)

                dict_result = {'algorithm' : alg_name,
                            'step_size':step_size_list,
                            'dataset' : dataset_name,
                            'train_losses': train_losses,
                            'train_acces': train_acces,
                            'test_acces':test_acces,
                            'timesCPU': timesCPU,
                            'epochs': epochs,
                            'grad_norms':grad_norms}

                return dict_result

        train_loss=compute_loss(net,train_set,device)
        epochs.append(epoch)
        timesCPU_i = time.process_time() - start_time_cpu
        train_losses.append(train_loss)
        timesCPU.append(timesCPU_i)
        train_acc.append(eval_net(net,train_set,device))
        val_acc.append(eval_net(net,test_set,device))
        val_acc_5.append(eval_5_net(net,test_set,device))

        ptrain_losses='{:.5f}'.format(train_losses[-1])
        ptrain_acc='{:.5f}'.format(train_acc[-1])
        pval_acc='{:.5f}'.format(val_acc[-1])
        pval_acc_5='{:.5f}'.format(val_acc_5[-1])
        pgrad_norm='{:.5f}'.format(grad_norm)
        print(f'e:{epoch},l:{ptrain_losses},t_acc{ptrain_acc},v_acc:{pval_acc},val_acc_5:{pval_acc_5},α:{step_size_list[-1]},g:{pgrad_norm},t:{timesCPU[-1]},n_f:{not_found_inepoch}')
    timesCPU=np.asarray(timesCPU)
    step_size_list=np.asarray(step_size_list)
    train_losses=np.asarray(train_losses)
    train_acces=np.asarray(train_acc)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    test_acces=np.asarray(val_acc)
    epochs=np.asarray(epochs)
    grad_norms=np.asarray(grad_norms)

    dict_result = {'algorithm' : alg_name,
                'step_size':step_size_list,
                'dataset' : dataset_name,
                'train_losses': train_losses,
                'train_acces': train_acces,
                'test_acces':test_acces,
                'timesCPU': timesCPU,
                'epochs': epochs,
                'grad_norms':grad_norms}

    return dict_result

