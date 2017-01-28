###
import pandas as pd
import numpy as np 
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys
import pdb
import math

#s = 100
#sigma = 1
#k = 20
#sample_amount = 250
#learning_parameter = 10
#gens = 10
def deflation_over_generations(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/deflation-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))
    
    X = np.arange(0,s)
    markers = ['x','v','o','*']
    size = [7,7,7,12]
    for idx,row in df.iterrows():
        
        if idx in [0,4,9,29]: #%2 == 0 and idx <= 10:
            theta_prop = [row['t'+str(x)] for x in xrange(s)]
            m1 = np.zeros(s)
            for i in xrange(s):
                for j in xrange(len(theta_prop)):
                    if i >= j:
                        m1[i] += theta_prop[j]
            plt.plot(X,m1, linestyle='dashed', marker=markers[0], markevery=8,markersize=size[0])
            markers.remove(markers[0])
            size.remove(size[0])



    plt.ylim(0-0.01,1+0.01) 
    plt.xlim(0,99) 

    plt.legend(["Initial population", "Generation 5", "Generation 10", "Generation 30"],loc='best',fontsize=12)
#    plt.title('%.2f sigma, %d k, %d samples, %d l' % (sigma,k,sample_amount,learning_parameter))
    plt.ylabel('Proportion of message use',fontsize=19)
    plt.xlabel("State",fontsize=19)

    plt.show()

#deflation_over_generations(100,0.4,30,300,1,50)


def vagueness_single_gen(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/vagueness-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))
    
    X = np.arange(0,s)


    
    for idx,row in df.iterrows():
        if idx == 0 or idx == 1:
            theta_prop = [row['t'+str(x)] for x in xrange(s)]
            m1 = np.zeros(s)
            m2 = np.zeros(s)
            for i in xrange(s):
                for j in xrange(len(theta_prop)):
                    if i >= j:
                        m1[i] += theta_prop[j]
                    else:
                        m2[i] += theta_prop[j]
            plt.plot(X,m1)
            plt.plot(X,m2, linestyle='dashed')
            plt.ylim(0-0.01,1+0.01)
            plt.ylabel('Proportion of use', fontsize=19)
            plt.xlabel('State', fontsize=19)
            plt.legend([r'$m_1$',r'$m_2$'], loc='best',fontsize=19)
            plt.show()

#vagueness_single_gen(100,0.4,20,100,1,50)


def vagueness_one_plot(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/vagueness-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))
    
    X = np.arange(0,s)
    markers = ['x','o']
    style = ['dashed','solid']

    for idx,row in df.iterrows():
        if idx == 0 or idx == 1:
            theta_prop = [row['t'+str(x)] for x in xrange(s)]
            m1 = np.zeros(s)
            for i in xrange(s):
                for j in xrange(len(theta_prop)):
                    if i >= j:
                        m1[i] += theta_prop[j]
            plt.plot(X,m1,linestyle='dashed', marker=markers[0],markevery=10)
            markers.remove(markers[0])
    plt.ylim(0-0.01,1+0.01)
    plt.ylabel('Proportion of m1 use', fontsize=19)
    plt.xlabel('State', fontsize=19)
    plt.legend(['Initial population','First generation'], loc='best',fontsize=19)
    plt.show()


vagueness_one_plot(100,0.4,20,100,1,50)


def heatmap_e_to_delta():
    print 'Loading data'
    df = pd.read_csv('./results/quantifiers_mean_results.csv')
    
    k = 5
    l = 1
    
    final_group = df.loc[:,('k','learning_parameter','epsilon','delta','t11_final')]
    
    group = final_group.loc[final_group['k'] == k]
    group = group.loc[group['learning_parameter'] == l]
    
    t11_rel = group.loc[:,('epsilon', 'delta','t11_final')] #ignore other columns, given that they are fixed
    t11_rec = t11_rel.pivot('epsilon','delta','t11_final') #reshape to have a prior_cost_c by learning_parameter table
    
    sns.set(font_scale=1.2)
    
    vacio = ["" for _ in xrange(9)]
    yticks = [0] + vacio + [0.1] + vacio + [0.2] + vacio + [0.3] + vacio + [0.4] + vacio + [0.5] + vacio + [0.6] + vacio + [0.7] + vacio + [0.8] + vacio + [0.9] + ["" for _ in xrange(8)] + [0.99]
    sns.set(font_scale=1.4)
    ax = sns.heatmap(t11_rec,yticklabels=yticks,xticklabels=yticks)#,yticklabels=yticks)#, yticklabels=yticks) 
#    ax.set(ylabel=r'$\epsilon$',xlabel=r'$\delta$') 
    ax.set_ylabel(r'$\epsilon$', fontsize=35)
    ax.set_xlabel(r'$\delta$', fontsize=35)
#    ax.set_ylabel(r'$\epsilon (P(s_{\forall} \mid s_{\exists\neg\forall}))$', fontsize=15)
#    ax.set_xlabel(r'$\delta (P(s_{\exists\neg\forall} \mid s_{\forall}))$', fontsize=15)

    plt.yticks(rotation=0)
   # ax.tick_params(labelsize=15)
    ax.invert_yaxis()

    plt.show()

#heatmap_e_to_delta()
