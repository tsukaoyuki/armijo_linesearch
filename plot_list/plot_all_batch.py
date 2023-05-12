from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]

list_all=[]
dataset_name='CIFAR100'
algo_name='SGD+Armijo_c_1e-4'
batch_size=1024


dir_name='result/'+dataset_name+'/'+'all/1e-4/'
#dir_name='step_size_list_1'
while (batch_size>=8):
    print(batch_size)
    with open(dir_name+dataset_name+'_'+algo_name+'_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    l['algorithm']=str(batch_size)
    l_all+=[l]
    batch_size=int(batch_size/2)
plot_results(l_all,dir_name,1,dataset_name)
    
