from plot_utils import *
import pickle
import matplotlib.pyplot as plt
import math
import os
l_all=[]
step_list=[]
batch_size_list=[]
batch_size=2048
#dir_name='SGD+armijo_b_200'
dir_name='step_size_list_2'
count=7
#dir_name='step_size_list_1'
while (batch_size>=64):
    for i in range(count):
        if os.path.isfile(dir_name+'/CIFAR10_Resnet34_'+str(batch_size)+'.bin'):
            with open(dir_name+'/CIFAR10_Resnet34_'+str(batch_size)+'.bin','rb') as p:
                l=pickle.load(p)
            count+=1

            print(math.ceil(50000/batch_size)*200)
        else:
            count+=1
            continue


    plt.plot(range(math.ceil(50000/batch_size)*200),l['step_size'])
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('step_size')
    plt.yscale("linear")
    plt.tight_layout()
    plt.xscale('linear')
    plt.title('batch size:'+str(batch_size),fontsize = 15,)
    plt.show()
    plt.savefig(dir_name+'/step_size_per_step'+str(batch_size)+'.png')
    plt.close()
    batch_size=int(batch_size/2)



