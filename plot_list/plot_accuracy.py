from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]

list_all=[]
dataset_name='CIFAR100'

dir_name='result/'+dataset_name+'//'
algo_list=['SGD+Armijo_c_0.10']
for c in c_list:
    algo_list.append(dataset_name+'_'+'SGD+Armijo_c_'+c+'_1024.bin')
for algo,legend,c in zip(algo_list,legend_list,c_list):
    file_name=dir_name+algo
    print(file_name)
    with open(file_name,'rb') as p:
        l=pickle.load(p)
    l_all+=[l]
    print(l.keys())
    print(max(l['test_acces']))
    l['algorithm']=c

plot_results(l_all,dir_name,1,dataset_name)
    

