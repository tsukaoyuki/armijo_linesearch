from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt
import os
import glob
l_all=[]
step_list=[]
batch_size_list=[]
batch_size=128
list_all=[]
dataset_name='r_MNIST'
d='r_MNIST_n'
dir_name='./result/'+dataset_name+'/'+d+'/'

files=glob.glob(dir_name+'*')
for file in files:
    print(file)
    with open(file,'rb') as p:
        l=pickle.load(p)
    #batch_size_list.append(batch_size
    l_all+=[l]
plot_results(l_all,dir_name,batch_size,dataset_name)
    



