from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

batch_size=2048
list_all=[]


dataset_name='CIFAR10'
dir_name='result/'+dataset_name+'_batch/'
y_list=[]
x_list=[]
while(batch_size>=16):
    with open(dir_name+dataset_name+'_SGD+Armijo_c_0.10_'+str(batch_size)+'.bin','rb') as p:
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
    batch_size=int(batch_size/2)

batch_size=2048
dataset_name='CIFAR10'
dir_name='result/'+dataset_name+'_batch_wd/'
y1_list=[]
x1_list=[]
while(batch_size>=16):
    with open(dir_name+'SGD+Armijo_c_0.10_Resnet34_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    l['algorithm']=str(batch_size)
    top_acc=list(np.sort(l['test_acces'])[::-1])
    print(type(top_acc[0]))
    sum: np.float64=0
    for i in range(5):
        sum=top_acc[i]+sum
    avg=sum/5
    print(avg)
    y1_list.append(float(avg))
    x1_list.append(batch_size)
    batch_size=int(batch_size/2)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x_list,y_list,marker="o",label='without weight decay',color='red')
ax.plot(x1_list,y1_list,marker="o",label='with weight decay',color='blue')
ax.legend()
ax.set_title('SGD+Armijo')
ax.grid(True)
plt.tight_layout()
ax.set_xlabel('batch_size')
ax.set_ylabel('acc')
ax.set_xscale('log',base=2)
ax.set_yscale('linear')
plt.show()
plt.savefig('result/hikaku/acc_each_batches.png')
plt.close()

    



