import numpy as np

def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    return m


#def normalize(m):
#    for row in xrange(len(m)):
#        if np.sum(m[row]) > 0:
#            m[row] = m[row] / np.sum(m[row])
#        else:
#            m[row] = np.ones(np.shape(m)[1]) / np.shape(m)[1]
#    return m


class Player:
    def __init__(self,lexicon):
        self.lexicon = lexicon
        self.sender_matrix = normalize(lexicon)
#        self.receiver_matrix =  self.receiver_selection_matrix()
   
#    def sender_selection_matrix(self,l):
#        m = np.zeros(np.shape(l))
#        for i in range(np.shape(l)[0]):
#            for j in range(np.shape(l)[1]):
#                m[i,j] = np.exp(self.lam * l[i,j])
#        return normalize(m)
#
#    def receiver_selection_matrix(self):
#        """Take transposed lexicon and normalize row-wise (prior over states plays no role as it's currently uniform)"""
#        m = normalize(np.transpose(self.lexicon))
#        for r in range(np.shape(m)[0]):
#            if sum(m[r]) == 0:
#                for c in range(np.shape(m)[1]):
#                    m[r,c] = 1. / np.shape(m)[0]
#
#        return m
