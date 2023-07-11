import os
from torch import optim
import torch
import math
import train_utils as tu
import pickle
import argparse  
from train_utils import (train_net,train_sfo)
from optimizers import sls

parser = argparse.ArgumentParser()  

parser.add_argument('--dataset',type=str,help='dataset',default='CIFAR100')
parser.add_argument('--dir', type=str,help='directly',default='test') 
parser.add_argument('--cuda',type=str,help='gpu',default='0')

args = parser.parse_args()     

dir_name='result/'+args.dataset+'/'+args.dir
print(f'dir_name: {dir_name}')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

device = 'cuda:'+args.cuda
dataset_name=args.dataset
epoch=1
max_batch_size=1024
min_batch_size=32
train_set,test_set=tu.get_dataset(dataset_name)
alg_name='SGD+Armijo'
c_list=[0.05,0.10,0.15]
batch_size=max_batch_size
while (batch_size>=min_batch_size):
    n_batches_per_epoch=math.ceil(len(train_set)/batch_size)
    print(f'batch_size: {batch_size}')
    for c in c_list:
        net=tu.get_model(dataset_name).to(device)
        file_name=tu.get_file_name(dataset_name,batch_size,alg_name+'_'+str(c))
        print(f'file_name: {file_name}')
        print('\n'+alg_name)
        l=train_sfo(dataset_name,net,train_set,test_set,optimizer=sls.SGD(device,net.parameters(),n_batches_per_epoch=n_batches_per_epoch,c=c),n_iter=epoch,device=device,alg_name=alg_name,batch_size=batch_size,)
        with open(dir_name+'/'+file_name,'wb')as p:
            pickle.dump(l,p)
    batch_size=int(batch_size/2)



