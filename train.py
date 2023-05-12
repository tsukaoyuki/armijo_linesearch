from adabelief_pytorch import AdaBelief
import os
from torch import optim
import torch
import math
import train_utils as tu
import pickle
import sys
import argparse  
from train_utils import train_net
import copy
from optimizers import (mySGD,mls,sps,mls_wolf)
import train_utils as ut
#from train_decrease_step_size import train_net
#from train_increase_batch_size import train_net
#from train_sfo import train_net
#import wandb
'''wandb.init(project = 'project',
            name='test')'''
parser = argparse.ArgumentParser()  


parser.add_argument('--dataset',type=str,help='データセット',default='CIFAR100')
parser.add_argument('--dir', type=str,help='保存ディレクトリ指定',default='test') 
parser.add_argument('--lr',type=float,help='ステップサイズ',default=0.1)
parser.add_argument('--cuda',type=str,help='GPU番号',default='6')
parser.add_argument('--inistep',type=float,help='初めのステップサイズ',default=0.01)
parser.add_argument('--iniepoch',type=int,help='どれくらい定数でやるか',default=0)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)


#parser.add_argument('--name',type=str,help='wandb保存名前',default='test')
args = parser.parse_args()     

dir_name='result/'+args.dataset+'/'+args.dir
print(f'dir_name: {dir_name}')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)


'''wandb.init(project = 'project',
            name=args.name)'''

#--------------------------------
device = 'cuda:'+args.cuda
dataset_name=args.dataset
count=50
lr=args.lr
max_batch_size=128
min_batch_size=128
weight_decay=0
resnet_num=34
train_set,test_set=tu.get_dataset(dataset_name)


'''test_set_list=[]
for i in range(len(test_set)):
    test_set_list.append(list(test_set[i]))
    print(tuple(test_set_list))
    print(test_set[i])
    print(type(tuple(test_set_list))
    )
    break
print(test_set)'''



inistep=args.inistep
iniepoch=int(args.iniepoch)


#SGD+Armijo_c_x.xx
#momentum+Armijo_m_x.xx_c_x.xx
#momentum_m_x.xx
#alg_list=['SGD','Adam','momentum_m_0.90','AdamW','RMSprop']
"""alg_list=['SGD+Armijo_c_1e-4','SGD+Armijo_c_1e-3',
'SGD+Armijo_c_1e-2',
'SGD+Armijo_c_1e-1',]"""
#alg_list=['SPS_0.50']
#alg_list=['SGD']
#alg_list=['Adam','SGD','SGD+Armijo_c_1e-1',]
alg_list=['SGD+Armijo_c_1e-1','momentum+Armijo_m_0.90_c_1e-1','SGD','AdamW','RMSprop','Adam','momentum_m_0.90']
#alg_list=['SGD+Armijo_c_1e-1']
#alg_list=['momentum_m_0.90']

'''wandb.config = {
  "learning_rate": lr,
  "epochs": count,
  "batch_size": max_batch_size
}'''

batch_size=max_batch_size
while (batch_size>=min_batch_size):
    n_batches_per_epoch=math.ceil(len(train_set)/batch_size)
    print(f'batch_size: {batch_size}')
    for alg_name in alg_list:
        net=tu.get_model(dataset_name,resnet_num).to(device)

        file_name=tu.get_file_name(dataset_name,batch_size,alg_name)
        print(f'file_name: {file_name}')
        print('\n'+alg_name)
        if 'Armijo' in alg_name:
            if 'momentum' in alg_name:
                l=train_net(dataset_name,net,train_set,test_set,optimizer=mls.SGD(device,net.parameters(),n_batches_per_epoch=n_batches_per_epoch,weight_decay=weight_decay,momentum=float(alg_name[-11:-7]),c=float(alg_name[-4:])),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            else:
                print('c=',float(alg_name[-4:]))
                l=train_net(dataset_name,net,train_set,test_set,optimizer=mls.SGD(device,net.parameters(),weight_decay=weight_decay,n_batches_per_epoch=n_batches_per_epoch,c=float(alg_name[-4:])),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size,)
        else:
            
            print(f'lr={lr},batch size={batch_size}.weight decay={weight_decay}')
            if 'momentum' in alg_name:
                print('m=',float(alg_name[-4:]))
                l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.SGD(net.parameters(),lr=0.1,momentum=float(alg_name[-4:]),weight_decay=weight_decay),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif 'KFAC' in alg_name:
                print('-------')
                l=train_net(dataset_name,net,train_set,test_set,optimizer = KFACOptimizer(net,
                              lr=0.01,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=weight_decay,
                              TCov=args.TCov,
                              TInv=args.TInv),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif 'SPS' in alg_name:
                l=train_net(dataset_name,net,train_set,test_set,optimizer=sps.Sps(net.parameters(),n_batches_per_epoch=n_batches_per_epoch,c=float(alg_name[-4:]),device=device),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif alg_name=='SGD':
                l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.SGD(net.parameters(),lr=0.1,weight_decay=weight_decay),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif alg_name=='Adam':
                l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.Adam(net.parameters(),lr=1e-3,weight_decay=weight_decay),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif alg_name=='RMSprop':
                l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.RMSprop(net.parameters(),lr=1e-2,weight_decay=weight_decay),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            elif alg_name=='AdamW':
                l=train_net(dataset_name,net,train_set,test_set,optimizer=optim.AdamW(net.parameters(),lr=1e-3,weight_decay=weight_decay),n_iter=count,device=device,alg_name=alg_name,batch_size=batch_size)
            else:
                print(alg_name+'is not registered')
        with open(dir_name+'/'+file_name,'wb')as p:
            pickle.dump(l,p)
    batch_size=int(batch_size/2)



