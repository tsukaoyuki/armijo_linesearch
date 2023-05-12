from plot_utils import *
import pickle
import matplotlib.pyplot as plt
import math
import os
l_all=[]
step_list=[]
batch_size_list=[]
batch_size=128
#dir_name='SGD+armijo_b_200'
dir_name='step_size_list_10'
count=6
#dir_name='step_size_list_1'
if os.path.isfile(dir_name+'/CIFAR10_Resnet34_'+str(batch_size)+'.bin'):
    with open(dir_name+'/CIFAR10_Resnet34_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
        print('a')
print(l['step_size'])
print(l['algorithm'])