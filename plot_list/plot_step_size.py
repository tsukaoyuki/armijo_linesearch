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
dir_name='hikaku_lr_0001_b_200'
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
dataset_name='CIFAR100'
d='SGD+momentum'
dir_name='./result/'+dataset_name+'/'+d+'/'

files=glob.glob(dir_name+'*')
for file in files:
    print(file)
    with open(file,'rb') as p:
        l=pickle.load(p)
    #batch_size_list.append(batch_size)
    l_all+=[l]
plot_results(l_all,dir_name,batch_size,dataset_name)
    





    plt.plot(range(math.ceil(50000/batch_size)*200),l['step_size'])
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('step_size')
    plt.yscale("linear")
    plt.tight_layout()
    plt.xscale('linear')
    plt.show()
    plt.savefig(dir_name+'/step_size_per_step'+str(batch_size)+'_'+str(i)+'.png')
    plt.close()


