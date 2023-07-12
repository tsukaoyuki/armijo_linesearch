import os
from torch import optim
import torch
import math
import train_utils as tu
import pickle
import argparse  
from train_utils import (train_net,train_sfo)
from optimizers import sls,optim_list


parser = argparse.ArgumentParser()  

parser.add_argument('--dataset',type=str,help='dataset',default='CIFAR100')
parser.add_argument('--dir', type=str,help='directly',default='test') 
parser.add_argument('--cuda',type=str,help='gpu',default='0')
parser.add_argument('--epoch',type=int,help='epoch',default=200)
parser.add_argument('--batch',type=int,help='batch size',default=128)
parser.add_argument('--c',type=float,help='c',default=0.1)
parser.add_argument('--algorithm',type=str,help='algorithm',default='SGD+Armijo')

args = parser.parse_args()     

dir_name='result/'+args.dataset+'/'+args.dir
print(f'dir_name: {dir_name}')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

device = 'cuda:'+args.cuda
epoch=args.epoch
c=args.c
dataset_name=args.dataset
alg_name=args.algorithm
batch_size=args.batch

train_set,test_set=tu.get_dataset(dataset_name)
n_batches_per_epoch=math.ceil(len(train_set)/batch_size)

params_dict={'device':device,'n_batches_per_epoch':n_batches_per_epoch,'c':0.1}

net=tu.get_model(dataset_name).to(device)
params_dict['net']=net.parameters()
if alg_name=='SGD+Armijo':
    file_name=tu.get_file_name(dataset_name,batch_size,alg_name+'_'+str(c))
else:
    file_name=file_name=tu.get_file_name(dataset_name,batch_size,alg_name)
print(f'file_name: {file_name}')
print('\n'+alg_name)
l=train_net(dataset_name,
            net,train_set,
            test_set,
            optimizer=optim_list.get_algorithm(alg_name,params_dict),
            n_iter=epoch,
            device=device,
            alg_name=alg_name,
            batch_size=batch_size,)
with open(dir_name+'/'+file_name,'wb')as p:
    pickle.dump(l,p)



