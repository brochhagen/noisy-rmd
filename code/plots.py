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
###
###Column headers:###
#####################

s = 100
sigma = 1
k = 10
sample_amount = 100
learning_parameter = 10
gens = 10
def deflation_over_generations(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/deflation-states%d-sigma%d-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))
    
    X = np.arange(0,100)
    
    for idx,row in df.iterrows():
        if idx%2 == 0 and idx <= 10:
            Y = [row['t'+str(x)] for x in xrange(s)]
    #    Y = [eval(x) for x in vals]
            plt.plot(X,Y)
    
    #    plt.plot(X,Y1, marker='*', markersize=12,markevery=4)
    #    plt.plot(X,Y2, marker='D', markersize=7,markevery=5)
    ##    plt.plot(X,Y3,linestyle='dashed')
    #
    #    plt.xlim(min(X),max(X))
    #    plt.ylim(0,max(Y1+Y2)+0.025)
    #    plt.ylabel("Probability of state 1",fontsize=15)
    #    plt.xlabel('Iterations',fontsize=15)
    plt.legend(["Generation 0", "Generation 2", "Generation 4", "Generation 6", "Generation 8", "Generation 10"],loc='best',fontsize=12)
    plt.show()

#deflation_over_generations(100,1,10,100,10,100)
#deflation_over_generations(100,1,10,100,10,10)
deflation_over_generations(100,1,2,10,10,10)
