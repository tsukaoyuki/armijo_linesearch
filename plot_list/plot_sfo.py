from plot_utils_per_step import *
import pickle
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MaxNLocator

l_all=[]
step_list=[]
batch_size_list=[]
batch_size=16
list_all=[]
steps_list=[]


dataset_name='CIFAR100'
dir_name='result/CIFAR100/SGD_001/'


#dir_name='step_size_list_1'

max_step_list=[]
batch_size_list=[]
sfo_list=[]
algo_name='CIFAR100_SGD'
while(batch_size<=1024):
    with open(dir_name+algo_name+'_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    break_epoch=0
    l['algorithm']=l['algorithm']+'_'+str(batch_size)
    for i in range(len(l['grad_norms'])):
        if l['grad_norms'][i]<=0.01:
            break
    max_step_list.append(i)
    batch_size_list.append(batch_size)
    sfo_list.append(batch_size*i)
    batch_size=int(batch_size*2)

for n,b,nb in zip(max_step_list,batch_size_list,sfo_list):
    print(n,b,nb)


fig, ax = plt.subplots(figsize=(10,5))
ax.plot(batch_size_list,max_step_list,marker="o",label='sgd+armijo',color='red')
ax.legend()
ax.grid(True)
plt.tight_layout()
ax.set_xlabel('batch_size')
ax.set_ylabel('steps')
ax.set_xscale('log',base=2)
ax.set_yscale('log',base=10)
#ax.yaxis.set_major_locator(MaxNLocator(9)) 
plt.show()
plt.savefig(dir_name+'/batch_step_'+algo_name+'.png')
plt.close()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(batch_size_list,sfo_list,marker="o",label='sgd+armijo',color='red')
ax.legend()
ax.grid(True)
ax.set_xlabel('batch_size')
ax.set_ylabel('sfo')
ax.set_xscale('log',base=2)
ax.set_yscale('log',base=10)
#ax.yaxis.set_major_locator(MaxNLocator(9)) 
plt.show()
plt.savefig(dir_name+'/sfo_'+algo_name+'.png')
plt.close()


    



