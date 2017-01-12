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

s = 100
sigma = 1
k = 10
sample_amount = 100
learning_parameter = 10
gens = 10
def deflation_over_generations(s,sigma,k,sample_amount,learning_parameter,gens):
    print 'Loading data'
    df = pd.read_csv(r"./results/deflation-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv" %(s,sigma,k,sample_amount,learning_parameter,gens))
    
    X = np.arange(0,10)
    
    for idx,row in df.iterrows():
        if idx in [0,4,9,14,19,29]: #%2 == 0 and idx <= 10:
            Y = [row['t'+str(x)] for x in xrange(s)]
            plt.plot(X,Y, linestyle='dashed')
    
    plt.legend(["Generation 0", "Generation 5", "Generation 10", "Generation 15", "Generation 20", "Generation 30"],loc='best',fontsize=12)
    plt.title('%.2f sigma, %d k, %d samples, %d l' % (sigma,k,sample_amount,learning_parameter))
    plt.ylabel('Proportion in population')
    plt.xlabel("theta-threshold (= 1 type)")



    plt.show()

deflation_over_generations(10,0.4,10,300,1,30)
deflation_over_generations(10,1,10,300,1,30)
deflation_over_generations(10,3,10,300,1,30)
