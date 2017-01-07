import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product, combinations_with_replacement
from scipy import stats
import sys 
import datetime
import csv

##### variables ##########
s = 100 #amount of states
sigma = 1
k = 1  # length of observation sequences
sample_amount = 1000 #amount of k-length samples for each production type ##THERE'S A PROBLEM HERE AND AT GET_MUT_MA


epsilon  = .5 # probability of perceiving S-all, when true state is S-sbna
delta = .1 # probability of perceiving S-sbna, when true state is S-all
learning_parameter = 10 #prob-matching = 1, increments approach MAP
gens = 1000 #number of generations per simulation run
runs = 50 #number of independent simulation runs
state_freq = np.ones(s) / float(s) #frequency of states s_1,...,s_n 
##########################


def get_types(s_amount): #a type is simply a theta-value
    return [x for x in xrange(s_amount)]



print '#Starting, ', datetime.datetime.now()
typeList = get_types(s)

print '#Computing likelihood, ', datetime.datetime.now()

#def get_obs(types,states,state_freqs,sample_amount,k,sigma): #returns sample_amount tuples of k-utterances per type
#    out = []
#    for i in xrange(len(types)): #a parent
#        out_parent = []
#        for j in xrange(sample_amount): #amount of k-length samples
#            parent_production_vector = np.zeros(states)
#            actual_states = [np.random.choice(xrange(states),p=state_freqs) for _ in xrange(k)] #actual state
#            print '###', i
#            print actual_states
#            parent_states = [int(round(np.random.normal(x,sigma))) for x in actual_states] #state as perceived by parent
#            parent_states = [x for x in parent_states if x in xrange(states)] #filter observed states within bounds of state_space
#            print parent_states
#            produced_states = [x for x in parent_states if x >= types[i]]
#            for x in produced_states:
#                parent_production_vector[x] += parent_production_vector[x] + 1
#            print parent_production_vector
#            out_parent.append(parent_production_vector)
#        out.append(out_parent)
#    return out
#
#obs = get_obs(typeList,s,state_freq,sample_amount,k,sigma)

def get_confusability_matrix(states,sigma):
    out = np.zeros([states,states])
    for i in xrange(states):
        for j in xrange(states):
            out[i,j] = stats.norm(i, sigma).pdf(j)
    return out


def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    return m

def m_max(m): #aux function for convenience
    return np.unravel_index(m.argmax(), m.shape)


def get_lh(states):
    return [np.concatenate((np.zeros(x),np.ones(states-x)/(states-x))) for x in xrange(states)]

def get_lh_perturbed(states,sigma):
    likelihoods = get_lh(states) #plain production
    lh_perturbed = get_lh(states) #plain production copy to modify
    state_confusion_matrix = get_confusability_matrix(states,sigma)
    PosteriorState = normalize(np.array([[state_freq[sActual] * state_confusion_matrix[sActual, sPerceived] for sActual in xrange(states)] \
                     for sPerceived in xrange(states)])) # probability of actual state given a perceived state
    DoublePerception = np.array([[np.sum([ state_confusion_matrix[sActual, sTeacher] * PosteriorState[sLearner,sActual] \
                     for sActual in xrange(states)]) for sTeacher in xrange(states) ] for sLearner in xrange(states)])# probability of teacher observing column, given that learner observes row
    for t in xrange(len(likelihoods)):
       for sLearner in xrange(len(likelihoods[t])):
           lh_perturbed[t][sLearner] = np.sum([ DoublePerception[sLearner,sTeacher] * likelihoods[t][sTeacher] for sTeacher in xrange(len(likelihoods[t]))])
    return lh_perturbed


def get_likelihood(states, obs, sigma, kind='plain'):
    if kind == 'perturbed':
        lh = get_lh_perturbed(states,sigma)
    elif kind == 'plain':
        lh = get_lh(states)
        
    out = np.zeros([len(lh), len(obs)])
    for lhi in xrange(len(lh)):
        for o in xrange(len(obs)):
            out[lhi,o] = np.prod([lh[lhi][obs[o][x]] for x in xrange(len(obs[o]))])
    return out

def get_mutation_matrix(states,k,state_freqs,sample_amount,learning_parameter, sigma):
    obs = [[np.random.choice(xrange(states),p=state_freqs) for _ in xrange(k)] for _ in xrange(sample_amount)] #sample of possible observations
    lhs = get_likelihood(states,obs, sigma, kind='plain')
    lhs_perturbed = get_likelihood(states,obs, sigma, kind='perturbed')
    print np.max(lhs[99]), np.max(lhs[98])
    print np.max(lhs_perturbed[99]), np.max(lhs_perturbed[98])

    parametrized_post = normalize(normalize(np.transpose(lhs))**learning_parameter)
    return normalize(np.dot(np.transpose(lhs_perturbed),parametrized_post))


q = get_mutation_matrix(s,k,state_freq,sample_amount,learning_parameter, sigma)

sys.exit()
def get_utils():
    out = np.zeros([len(typeList), len(typeList)])
    for i in range(len(typeList)):
        for j in range(len(typeList)):
            ## this is only correct for "flat state priors"!
            out[i,j] = (np.sum(np.dot(state_confusion_matrix, typeList[i].sender_matrix) * np.transpose(typeList[j].receiver_matrix)) + \
                     np.sum( np.dot(state_confusion_matrix, typeList[j].sender_matrix) * np.transpose(typeList[i].receiver_matrix))) / 4
    return out

print '#Computing utilities, ', datetime.datetime.now()
u = get_utils()
print [np.sum(u[i,:]) for i in xrange(u.shape[1])]

print '#Computing Q, ', datetime.datetime.now()

q = get_mutation_matrix(k,states,messages,likelihoods,state_freq,sample_amount,lexica_prior,learning_parameter,lh_perturbed)



### single run

p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
#p = np.array([1,1,1,1,1,1,1,1,1,1,1,1.0]) / 12
#p_initial = p

for r in range(gens):
    pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
    pPrime = pPrime / np.sum(pPrime)
    p = np.dot(pPrime, q)


print '###Overview of results###', datetime.datetime.now()
print 'Parameters: alpha = %d, c = %.2f, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, gen = %d' % (alpha, cost, lam, k, sample_amount, learning_parameter, gens)
print 'end state:' 
print p

#print 'Q'
#print q[10,10], q[9,10], q[11,10], q[4,10]
#print q[9,9], q[11,11], q[4,4]
#print u[10,10], u[9,9], u[11,11], u[4,4]

#print 'U'
#for i in u: print np.sum(i)
