from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

l_all=[]
step_list=[]
batch_size_list=[]
batch_size=128
list_all=[]
dataset_name='CIFAR100'
c_list=[0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
dir_name='result/CIFAR10_each_batches_wd'
y_list=[]
x_list=[]
#dir_name='step_size_list_1'
for c in c_list:
    with open(dir_name+'/SGD+Armijo_+'+str()Resnet34_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    l['algorithm']=str(batch_size)
    top_acc=list(np.sort(l['test_acces'])[::-1])
    print(type(top_acc[0]))
    sum: np.float64=0
    for i in range(5):
        sum=top_acc[i]+sum
    avg=sum/5
    print(avg)
    y_list.append(float(avg))
    x_list.append(batch_size)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x_list,y_list,marker="o",label='sgd+armijo',color='red')
ax.legend()
ax.grid(True)
plt.tight_layout()
ax.set_xlabel('batch_size')
ax.set_ylabel('acc')
ax.set_xscale('log',base=2)
ax.set_yscale('linear')
plt.show()
plt.savefig(dir_name+'/acc_each_batches.png')
plt.close()