from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt
import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


step_size_per_epoch_list=[]
sum_step_size=0.
batch_size=2048
remain_num=50000


ColorsList = ['b', 'r', 'c', 'g', 'y', 'k', 'm', 'brown','purple']


def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return: 
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]



l_all=[]
step_list=[]
batch_size_list=[]
batch_size=128
list_all=[]
dataset_name='CIFAR10'
d=dataset_name+'_c'
dir_name='./result/'+dataset_name+'/'+d+'/'
alg_name='SGD+Armijo'
c_list=['1e-5','1e-4','1e-3','1e-2','0.10','0.15','0.20','0.25',]
files=glob.glob(dir_name+'*')

fig, ax = plt.subplots(figsize = (10,10))
fig.patch.set_facecolor('white')
for c,color in zip(c_list,ColorsList):
    step_size_per_epoch_list=[]

    print(dir_name+alg_name+c+'.bin')
    with open(dir_name+dataset_name+'_'+alg_name+'_c_'+c+'_128'+'.bin','rb') as p:
        l=pickle.load(p)
    step_num=math.ceil(50000/128)
    result=list(split_list(l['step_size'],step_num))
    for i in result:
        for ii in i:
            sum_step_size+=ii
        step_size_per_epoch_list.append(sum_step_size/step_num)
        sum_step_size=0.
    ax.plot(range(200), step_size_per_epoch_list, label= c, color = color, linewidth=3)

lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
lgd.get_frame().set_linewidth(1.0)
for line in lgd.get_lines():
    line.set_linewidth(3.0)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels))
    
c = patches.Circle( (50,0),3 , facecolor="pink", edgecolor="red", label="vertex")
ax.add_patch(c)
plt.title(dataset_name, fontsize = 50, fontweight='normal')
plt.plot(range(200),step_size_per_epoch_list)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('step_size')
plt.yscale("linear")
plt.tight_layout()
plt.xscale('linear')
plt.show()
if not os.path.isdir(dir_name+'fig'):
    os.mkdir(dir_name+'fig')
print(dir_name+'fig/step_size_list'+'all'+".png")
plt.savefig(dir_name+'fig/step_size_list'+'all'+".png")
plt.savefig("test.png")
plt.close()

