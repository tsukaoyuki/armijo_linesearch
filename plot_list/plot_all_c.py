from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]

list_all=[]
dataset_name='CIFAR100'
algo_name='SGD+Armijo_c_'
batch_size=128
#c_list=['1e-5','1e-4','1e-3']
c_list=['1e-4','1e-3','1e-2','1e-1',]

dir_name='result/'+dataset_name+'/'+dataset_name+'_'+algo_name
#dir_name='step_size_list_1'
for b in batch_size_list:
    for c in c_list:
        print(c)
        with open(dir_name+dataset_name+'_'+algo_name+c+'_'+b+'.bin','rb') as p:
            l=pickle.load(p)
        l['algorithm']=c
        l_all+=[l]
plot_results(l_all,dir_name,1,dataset_name)
    

