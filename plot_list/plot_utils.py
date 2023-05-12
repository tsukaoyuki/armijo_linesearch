import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({'font.size': 18})
plt.rc('font', family='serif')
plt.style.use('seaborn-muted')

cd=os.path.abspath(".")

def plot_results(list_results):

    Name_tilte = list_results[0]['dataset'] 

    ColorsList = ['b', 'r', 'c', 'g', 'y', 'k', 'm', 'brown']
    for i in range(len(list_results)):
        list_results[i]['Color'] = ColorsList[i]

    fig, ax = plt.subplots(figsize = (10,10))
    fig.patch.set_facecolor('white')

    batch_size=512
    for dict_save in list_results:
        ax.plot(dict_save['epochs'], dict_save['train_acces'], label= dict_save['algorithm']+'('+str(batch_size)+')', color = dict_save['Color'], linewidth=3)
        batch_size=int(batch_size/2)

    # ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    ax.set_ylabel('train_accuracy')
    ax.set_xlabel('epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)

    plt.title(Name_tilte, fontsize = 25, fontweight='normal')
    plt.yscale("linear")
    plt.savefig("cifiar100train.png")
    plt.show()


    fig, ax = plt.subplots(figsize = (10,10))
    fig.patch.set_facecolor('white')
    batch_size=512
    for dict_save in list_results:
        ax.plot(dict_save['epochs'], dict_save['train_losses'], label= dict_save['algorithm']+'('+str(batch_size)+')', color = dict_save['Color'], linewidth=3)
        batch_size=int(batch_size/2)

    # ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    ax.set_ylabel('train_loss')
    ax.set_xlabel('Epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)

    plt.title(Name_tilte, fontsize = 25, fontweight='normal')
    plt.grid()
    plt.yscale("log")
    plt.savefig("cifar100train_loss.png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    fig.patch.set_facecolor('white')
    batch_size=512
    for dict_save in list_results:
        ax.plot(dict_save['epochs'], dict_save['test_acces'], label= dict_save['algorithm']+'('+str(batch_size)+')', color = dict_save['Color'], linewidth=3)
        batch_size=int(batch_size/2)
    # ax.grid(True)
    lgd = plt.legend(frameon=True, loc = 'upper right', framealpha = 1, edgecolor = 'black', fancybox = False)
    lgd.get_frame().set_linewidth(1.0)
    for line in lgd.get_lines():
        line.set_linewidth(3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels))

    ax.set_ylabel('test_accuracy')
    ax.set_xlabel('Epochs')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    ax.xaxis.set_tick_params(top='on', direction='in', width=2)
    ax.yaxis.set_tick_params(right='on', direction='in', width=2)

    plt.title(Name_tilte, fontsize = 25, fontweight='normal')
    plt.yscale("linear")
    plt.savefig("cifar100test.png")
    plt.show()

