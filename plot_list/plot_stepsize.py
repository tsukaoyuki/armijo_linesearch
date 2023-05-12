from plot_utils import *
import pickle
import matplotlib.pyplot as plt

l_all=[]
step_list=[]
batch_size_list=[]
batch_size=2048
while(batch_size>=16):
    with open('./step_size_all/mnist_on_Resnet18_'+str(batch_size)+'.bin','rb') as p:
        l=pickle.load(p)
    batch_size_list.append(batch_size)
    batch_size=int(batch_size/2)
    
    step_list.append(l['epochs'][-1])
    print(l['step_size'])


"""
sfo_list=[]
for b,s in zip(batch_size_list,step_list):
    sfo_list.append(b*s)
    
print(len(sfo_list))
print(len(batch_size_list))
plt.plot(batch_size_list,sfo_list)
plt.grid()
plt.xlabel('batch size')
plt.ylabel('SFO')
plt.yscale("linear")
plt.tight_layout()
plt.xscale('log',basex=2)
plt.savefig("mnist.png")

plt.show()
#plot_results(l_all)
"""
