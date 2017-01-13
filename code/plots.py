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
    
    for idx,row in df.iterrows():
        if idx in [0,4,9,29]: #%2 == 0 and idx <= 10:
            theta_prop = [row['t'+str(x)] for x in xrange(s)]
            m1 = np.zeros(s)
            for i in xrange(s):
                for j in xrange(len(theta_prop)):
                    if i >= j:
                        m1[i] += theta_prop[j]
            plt.plot(X,m1, linestyle='dashed')

    plt.ylim(0-0.01,1+0.01) 
    plt.legend(["Initial population", "Generation 5", "Generation 10", "Generation 30"],loc='best',fontsize=12)
#    plt.title('%.2f sigma, %d k, %d samples, %d l' % (sigma,k,sample_amount,learning_parameter))
    plt.ylabel('Proportion of use',fontsize=19)
    plt.xlabel("State",fontsize=19)

    plt.show()

#deflation_over_generations(100,0.4,30,300,1,50)
deflation_over_generations(100,2,30,300,1,50)


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

##vagueness_single_gen(100,0.4,20,100,1,50)



#def heatmap_quantifiers():
#    print 'Loading data'
#    df = pd.read_csv('./mean-1000games-c-to-l.csv')
#    
#    a = 1
#    k = 5
#    lam = 30
#    sample = 10
#    
#    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t11_final')]
#    
#    group = final_group.loc[final_group['alpha'] == a]
#    group = group.loc[group['k'] == k]
#    group = group.loc[group['lambda'] == lam]
#    group = group.loc[group['sample_amount'] == sample]
#    
#    t11_rel = group.loc[:,('prior_cost_c', 'learning_parameter','t11_final')] #ignore other columns, given that they are fixed
#    t11_rec = t11_rel.pivot('prior_cost_c','learning_parameter','t11_final') #reshape to have a prior_cost_c by learning_parameter table
#    
#    sns.set(font_scale=1.2)
#    
#    vacio = ["" for _ in xrange(9)]
#    yticks = [0] + vacio + [0.1] + vacio + [0.2] + vacio + [0.3] + vacio + [0.4] + vacio + [0.5] + vacio + [0.6] + vacio + [0.7] + vacio + [0.8] + vacio + [0.9] + ["" for _ in xrange(8)] + [0.99]
#    ax = sns.heatmap(t11_rec,yticklabels=yticks)#, yticklabels=yticks) 
#    ax.set(ylabel='Learning bias c',xlabel='Sampling to MAP parameter l', title=r'Pragmatic L-lack ($\alpha = %d, \lambda = %d$, samples = %d, k = %d)' %(a,lam,sample,k))
#    plt.yticks(rotation=0)
#    
#    ax.invert_yaxis()
#    plt.show()
#
#def dev_over_gens_quantifiers():
#    df = pd.read_csv('./mean-1000games-c-to-l.csv')
#    a = 1
#    k = 5
#    lam = 30
#    sample = 10
#    learn = 3
#    
#    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t9_final','t10_final','t11_final')]
#    group = final_group.loc[final_group['alpha'] == a]
#    group = group.loc[group['k'] == k]
#    group = group.loc[group['lambda'] == lam]
#    group = group.loc[group['sample_amount'] == sample]
#    group = group.loc[group['learning_parameter'] == learn]
#    
#    t_final = group.groupby(['prior_cost_c'])
#    t_final = t_final[['t9_final','t10_final','t11_final']].agg(np.average) 
#    
#    ax = t_final.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
#    ax.set(ylabel="Proportion in population",xlabel='Learning bias c')
#    plt.legend(["prag. L-taut","prag. L-bound","prag. L-lack"], loc='best')
#    
#    plt.show() 
#
#
