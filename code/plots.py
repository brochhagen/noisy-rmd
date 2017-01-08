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

def deflation_over_generations(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/deflation-states%d-sigma%d-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))

    X = np.arange(0,101)

    for y in df:
        Y = [eval(df['t'+str(s)] for x in xrange(s)]
        plt.plot(X,Y)

#    plt.plot(X,Y1, marker='*', markersize=12,markevery=4)
#    plt.plot(X,Y2, marker='D', markersize=7,markevery=5)
##    plt.plot(X,Y3,linestyle='dashed')
#
#    plt.xlim(min(X),max(X))
#    plt.ylim(0,max(Y1+Y2)+0.025)
#    plt.ylabel("Probability of state 1",fontsize=15)
#    plt.xlabel('Iterations',fontsize=15)
#    plt.legend(["Sender prior", "Receiver prior"],loc='best',fontsize=12)
    plt.show()

deflation_over_generations(100,1,10,100,10,100)
