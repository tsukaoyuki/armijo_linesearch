from plot_utils1 import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]
batch_size=32
list_all=[]
dir_name='SGD_lr_1_b_200'
while(batch_size<=2048):
    with open('./'+dir_name+'/mnist_on_Resnet34_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    l['algorithm']=l['algorithm']+str(batch_size)
    #batch_size_list.append(batch_size)

    plot_results([l],dir_name,batch_size)
    batch_size=int(batch_size*2)
    



