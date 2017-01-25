import numpy as np
from random import sample
from itertools import product, combinations_with_replacement
from scipy import stats
import sys 
import datetime
import csv

##### variables ##########
s = 100 #amount of states
sigma = 0.4
k = 30  # length of observation sequences
sample_amount = 300 #amount of k-length samples for each production type 
learning_parameter = 1 #prob-matching = 1, increments approach MAP
gens = 50 #number of generations per simulation run
#runs = 50 #number of independent simulation runs
state_freq = np.ones(s) / float(s) #frequency of states s_0,...,s_n-1
##########################

f = csv.writer(open('./results/deflation-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv' %(s,sigma,k,sample_amount,learning_parameter,gens),'wb')) #file to store mean results
f.writerow(["generation","sigma","k","samples","learning"]+['t'+str(x) for x in xrange(s)])

def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    return m

def m_max(m): #auxiliary function for convenience
    return np.unravel_index(m.argmax(), m.shape)

def get_obs(states,state_freqs,sample_amount,k): #returns (non-noisy) sample_amount tuples of k-utterances per type
    out = []
    for i in xrange(s): #a parent/threshold value
        out_parent = []
        for j in xrange(sample_amount): #amount of k-length samples
            actual_states = [np.random.choice(xrange(states),p=state_freqs) for _ in xrange(k)] #actual state
            produced_states = [x for x in actual_states if x >= i] #filter observations by whether parent said something (state >= threshold)
            out_parent.append(produced_states)
        out.append(out_parent)
    return out

def get_confusability_matrix(states,sigma):
    out = np.zeros([states,states])
    myclip_a = 0 #first boundary for distribution
    myclip_b = 99 #second boundary for distribution

    for i in xrange(states):
        for j in xrange(states):
            current_mean = i #actual state
            a, b = (myclip_a - current_mean) / sigma, (myclip_b - current_mean) / sigma
            out[i,j] = stats.truncnorm.pdf(j,a,b,loc=i,scale=sigma)
    return out

def get_lh(states): #send message if state > threshold
    return [np.concatenate((np.zeros(x),np.ones(states-x))) for x in xrange(states)]

def get_lh_perturbed(states,sigma):
    likelihoods = get_lh(states) #plain production
    lh_perturbed = get_lh(states) #plain production copy to modify
    state_confusion_matrix = normalize(get_confusability_matrix(states,sigma))

    PosteriorState = normalize(np.array([[state_freq[sActual] * state_confusion_matrix[sActual, sPerceived] for sActual in xrange(states)] \
                     for sPerceived in xrange(states)])) # probability of actual state given a perceived state
    DoublePerception = np.array([[np.sum([ state_confusion_matrix[sActual, sTeacher] * PosteriorState[sLearner,sActual] \
                     for sActual in xrange(states)]) for sTeacher in xrange(states) ] for sLearner in xrange(states)])# probability of teacher observing column, given that learner observes row

    for t in xrange(len(likelihoods)):
       for sLearner in xrange(len(likelihoods[t])):
           lh_perturbed[t][sLearner] = np.sum([ DoublePerception[sLearner,sTeacher] * likelihoods[t][sTeacher] for sTeacher in xrange(len(likelihoods[t]))])
    return lh_perturbed

def get_likelihood(states, obs, sigma, state_freqs,k, kind='plain'):
    if kind == 'perturbed':
        lh = get_lh_perturbed(states,sigma)
    elif kind == 'plain':
        lh = get_lh(states)
        
    out = np.zeros([len(lh), len(obs)])
    for lhi in xrange(len(lh)):
        type_not_reporting_event = sum([state_freqs[x] for x in xrange(lhi)])
        for o in xrange(len(obs)):
            prob_not_reporting_ob = stats.binom.pmf(k-len(obs[o]),k,type_not_reporting_event)
            out[lhi,o] = np.prod([lh[lhi][obs[o][x]]  for x in xrange(len(obs[o])) ]) * prob_not_reporting_ob
    return out

print '#Starting, ', datetime.datetime.now()
print '#Computing likelihood, ', datetime.datetime.now()

def get_mutation_matrix(states,k,state_freq,sample_amount,learning_parameter, sigma):
    obs = get_obs(states,state_freq,sample_amount,k) #get production data from all types
    out = np.zeros([states,states]) #matrix to store Q

    for parent_type in xrange(states):
        type_obs = obs[parent_type] #Parent production data
        lhs_perturbed = get_likelihood(states, type_obs, sigma, state_freq, k, kind = "perturbed") #P(learner observes data|t_i) for all types;
        lhs = get_likelihood(states, type_obs, sigma, state_freq,k, kind = "plain") #P(parent data|t_i) for all types; without all noise
        parametrized_post = normalize(normalize(np.transpose(lhs))**learning_parameter) #P(t_j|parent data) for all types; P(d|t_j)P(t_j)
        out[parent_type] = np.dot(lhs_perturbed[parent_type],parametrized_post)

    return normalize(out)

print '#Computing Q, ', datetime.datetime.now()

q = get_mutation_matrix(s,k,state_freq,sample_amount,learning_parameter, sigma)

####### single run ######
#p = np.random.dirichlet(np.ones(s)) # unbiased random starting state
p = np.zeros(s)
starting_threshold = 80
p[starting_threshold] = 1 

for r in range(gens):
    f.writerow([str(r),str(sigma),str(k),str(sample_amount),str(learning_parameter)]+[str(x) for x in p])
    print '### Generation %d ###' %r
    print 'Proportion of %.2f players uses threshold %d' % (p[np.argmax(p)], np.argmax(p))
    print 'Proportion of %.2f players uses threshold %d' % (p[starting_threshold], starting_threshold)
    p = np.dot(p, q)

print '###Overview of results###', datetime.datetime.now()
print 'Parameters: sigma = %.2f, k = %d, sample_amount = %d, learning parameter = %d, gens = %d' % (sigma, k, sample_amount, learning_parameter, gens)
print 'incumbent: ', np.argmax(p), 'proportion: ', p[np.argmax(p)]
