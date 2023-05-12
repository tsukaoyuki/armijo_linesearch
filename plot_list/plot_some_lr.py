from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt
import os

l_all=[]
batch_size=128

lr_list=['{:.01f}'.format(0.1),
'{:.02f}'.format(0.01),
'{:.03f}'.format(0.001),
'{:.04f}'.format(0.0001),
'{:.05f}'.format(0.00001),
' '
]
dataset_name='MNIST'
dir_num_list=['01','001','0001','00001','000001','decrease']
alg_list=['SGD','Adam','AdamW','momentum_m_0.90','RMSprop']

for alg_name in alg_list:
    l_all=[]
    for lr,dir_num in zip(lr_list,dir_num_list):
        dir_name='result/'+dataset_name+'/'+dataset_name+'_hikaku_lr_'+dir_num+'/'
        with open(dir_name+'/'+dataset_name+'_'+alg_name+'_128'+'.bin','rb') as p:
            l=pickle.load(p)
        if dir_num=='decrease':
            l['algorithm']='decrease step size'
        else:
            l['algorithm']=lr
        l_all+=[l]
    print(dir_name+'/hikaku_'+dataset_name+'_'+alg_name)  
    if not os.path.isdir(dir_name+'/hikaku_'+dataset_name+'_'+alg_name):
        os.mkdir(dir_name+'/hikaku_'+dataset_name+'_'+alg_name)
        print((dir_name+'/hikaku_'+dataset_name+'_'+alg_name))
    plot_results(l_all,dir_name+'/hikaku_'+dataset_name+'_'+alg_name,1,dataset_name)
    