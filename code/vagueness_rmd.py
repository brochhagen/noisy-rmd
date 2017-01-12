import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product, combinations_with_replacement
from scipy import stats
import sys 
import datetime
import csv
import pdb 

# observations are currently generated without noise

##### variables ##########
s = 10 #amount of states
sigma = 0.4
k = 10  # length of observation sequences
sample_amount = 1000 #amount of k-length samples for each production type 


learning_parameter = 1 #prob-matching = 1, increments approach MAP
gens = 10 #number of generations per simulation run
#runs = 50 #number of independent simulation runs
state_freq = np.ones(s) / float(s) #frequency of states s_1,...,s_n 
##########################

#f = csv.writer(open('./results/vagueness-states%d-sigma%.2f-k%d-samples%d-l%d-g%d.csv' %(s,sigma,k,sample_amount,learning_parameter,gens),'wb')) #file to store mean results
#f.writerow(["generation","sigma","k","samples","learning"]+['t'+str(x) for x in xrange(s)])

def normalize(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    return m

def m_max(m): #aux function for convenience
    return np.unravel_index(m.argmax(), m.shape)

def get_confusability_matrix(states,sigma):
    out = np.zeros([states,states])
    for i in xrange(states):
        for j in xrange(states):
            out[i,j] = stats.norm(i, sigma).pdf(j)
    return out

def get_lh(states): #send m_1 if state => theta, send m_2 if state < theta
    out = []
    lh_m1 = [np.concatenate((np.zeros(x),np.ones(states-x))) for x in xrange(states)]
    lh_m2 = [np.concatenate((np.ones(x),np.zeros(states-x))) for x in xrange(states)]
    for i in xrange(len(lh_m1)):
        out.append(np.hstack((lh_m1[i][:,np.newaxis],lh_m2[i][:,np.newaxis])))
    return out
   
def get_obs(states,state_freqs,sample_amount,k): #returns (non-noisy) sample_amount tuples of k-utterances per type
    likelihoods = get_lh(states)
    out = []
    for i in xrange(len(likelihoods)): #a parent/threshold value
        out_parent = []
        sts,msgs = np.shape(likelihoods[i])
        doubled_state_freq = np.column_stack((state_freqs,state_freqs)).flatten() #P(s)
        production_vector = likelihoods[i].flatten() * doubled_state_freq #P(s) * P(m|s,t_i)

        for j in xrange(sample_amount): #amount of k-length samples
            parent_production = np.zeros(sts * msgs) #vector to store what parent produced in state
            sampled_obs = [np.random.choice(xrange(len(production_vector)),p=production_vector) for _ in xrange(k)] #idx of 0 is s_0,m_0, idx of 1 is s_0,m_1, ...
            

            for n in xrange(len(sampled_obs)):
                parent_production[sampled_obs[n]] += 1 
            out_parent.append(parent_production)
        out.append(out_parent)
    return out



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
           for m in xrange(len(likelihoods[t][sLearner])):
               lh_perturbed[t][sLearner,m] = np.sum([ DoublePerception[sLearner,sTeacher] * likelihoods[t][sTeacher,m]\
                for sTeacher in xrange(np.shape(likelihoods[t])[0])])
    
    return lh_perturbed


def get_likelihood(states, obs, sigma, kind='plain'):
    if kind == 'perturbed':
        lh = get_lh_perturbed(states,sigma)
    elif kind == 'plain':
        lh = get_lh(states)
        
    out = np.zeros([len(lh), len(obs)])
    for lhi in xrange(len(lh)):
        for o in xrange(len(obs)):
            flat_lhi = lh[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))])
    return out

print '#Starting, ', datetime.datetime.now()
print '#Computing likelihood, ', datetime.datetime.now()

def get_mutation_matrix(states,k,state_freq,sample_amount,learning_parameter, sigma):
    obs = get_obs(states,state_freq,sample_amount,k) #get production data from all types
    out = np.zeros([states,states]) #matrix to store Q

    for parent_type in xrange(states):
        print parent_type
        type_obs = obs[parent_type] #Parent production data
        lhs_perturbed = get_likelihood(states, type_obs, sigma, kind = "perturbed") #P(learner observes data|t_i) for all types;
        lhs = get_likelihood(states, type_obs, sigma, kind = "plain") #P(parent data|t_i) for all types; without all noise
        parametrized_post = normalize(normalize(np.transpose(lhs))**learning_parameter) #P(t_j|parent data) for all types; P(d|t_j)P(t_j)
        out[parent_type] = np.dot(lhs_perturbed[parent_type],parametrized_post)

    return normalize(out)

print '#Computing Q, ', datetime.datetime.now()

q = get_mutation_matrix(s,k,state_freq,sample_amount,learning_parameter, sigma)


### single run

#p = np.random.dirichlet(np.ones(s)) # unbiased random starting state
p = np.zeros(s)
starting_threshold = 6
p[starting_threshold] = 1 

for r in range(gens):
#    pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
#    pPrime = pPrime / np.sum(pPrime)
#    f.writerow([str(r),str(sigma),str(k),str(sample_amount),str(learning_parameter)]+[str(x) for x in p])
    print '### Generation %d ###' %r
    print 'Proportion of %.2f players uses threshold %d' % (p[np.argmax(p)], np.argmax(p))
    print 'Proportion of %.2f players uses threshold %d' % (p[starting_threshold], starting_threshold)
    p = np.dot(p, q)



print '###Overview of results###', datetime.datetime.now()
print 'Parameters: sigma = %.2f, k = %d, sample_amount = %d, learning parameter = %d, gens = %d' % (sigma, k, sample_amount, learning_parameter, gens)
print 'incumbent: ', np.argmax(p), 'proportion: ', p[np.argmax(p)]
sys.exit()

######### snippets for data generated for all types, not by parent_type
#def get_likelihood(states, obs, sigma, kind='plain'):
#    if kind == 'perturbed':
#        lh = get_lh_perturbed(states,sigma)
#    elif kind == 'plain':
#        lh = get_lh(states)
#        
#    out = np.zeros([len(lh), len(obs)])
#    for lhi in xrange(len(lh)):
#        for o in xrange(len(obs)):
#            out[lhi,o] = np.prod([lh[lhi][obs[o][x]] for x in xrange(len(obs[o]))])
#    return out

#def get_mutation_matrix(states,k,state_freqs,sample_amount,learning_parameter, sigma):
#    obs = [[np.random.choice(xrange(states),p=state_freqs) for _ in xrange(k)] for _ in xrange(sample_amount)] #sample of possible observations
#
#    lhs = get_likelihood(states,obs, sigma, kind='plain')
#    
#    lhs_perturbed = get_likelihood(states,obs, sigma, kind='perturbed')
#    
#    parametrized_post = normalize(normalize(lhs)**learning_parameter)
#    out = np.dot(np.transpose(lhs_perturbed),parametrized_post)
#
#    return normalize(np.dot(np.transpose(lhs_perturbed),parametrized_post))#normalize(out)

