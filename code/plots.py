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
    
    X = np.arange(0,100)
    
    for idx,row in df.iterrows():
        if idx in [0,1,2,5,7,9]: #%2 == 0 and idx <= 10:
            Y = [row['t'+str(x)] for x in xrange(s)]
            plt.plot(X,Y, linestyle='dashed')
    
    plt.legend(["Generation 0", "Generation 1", "Generation 2", "Generation 5", "Generation 7", "Generation 9"],loc='best',fontsize=12)
    plt.title('%.2f sigma, %d k, %d samples, %d l' % (sigma,k,sample_amount,learning_parameter))
    plt.ylabel('Proportion in population')
    plt.xlabel("theta-threshold (= 1 type)")



    plt.show()

deflation_over_generations(100,0.4,50,1000,10,10)
deflation_over_generations(100,1,50,1000,10,10)
deflation_over_generations(100,2,50,1000,10,10)
