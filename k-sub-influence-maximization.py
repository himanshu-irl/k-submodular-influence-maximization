# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Module: 7CCSMPRJ
# Project: Social Network Optimization: k-submodular influence maximization
# Author: Verma, Himanshu
# KID: k20083811
# --------------------------------------------------------

import pandas as pd
import os
import random
import datetime
from operator import add
import numpy as np
import json

random.seed(20083811)

dat_dir = 'dat'
output_dir = 'output'

edge = os.path.join(dat_dir, 'digg_graph_100.tsv')
prob = os.path.join(dat_dir, 'digg_prob_100_1000_pow10.tsv')
output_file = os.path.join(output_dir, '::file_name::')

edge = pd.read_csv(edge, sep='\t', header=None)
prob = pd.read_csv(prob, sep='\t', header=None)


# code is required for encoding while saving 
# output dict as a json file
# provided by user "Jie Yang" on stackoverflow
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def load_graph(edge
               ,prob
               ,k):
    """load_graph reads edge and probability datasets and modifies data 
    structure for simulation and k-greedy consumption
    
    Parameters
    ----------
    edge : pandas dataframe
        user-to-user mapping data
    prob : pandas dataframe
        user-to-user influence probabilities data corresponding to k-topics
    k : int
        number of products or topics
    
    Returns
    -------
    list
        a nested list containing list elements with each list element 
        comprised of user connection tuples corresponding to i-th (i=index) 
        user in the edge dataset at 0th index of tuple and k-influence 
        probabilities of connection as a list at 1st index of tuple.
        
        e.g. [(2061,[0.00580478,0.000499888,0.00359417
                     ,0.00571525,0.000239974,0.00442118
                     ,0.00766344,0.00043991300000000006
                     ,0.0047996,0.00671959])]
        
    """
    
    # ps
    n=0
    m=0
    ps = []
    for index, row in edge.iterrows():
        u = row[0]
        v = row[1]
        vec = []
        ps.append(((u,v),vec))
        # for getting distinct users
        # + 1 as user id starts from 0
        n = max(n, max(u,v)+1) 
        m = m + 1
    
    for index, row in prob.iterrows():
        for j in range(k):
            # print(index, j)
            p = row[j]        
            ps[index][1].append(p)
    
    es = [None]*n # of length n
    for i in range(m):
        u = ps[i][0][0]
        v = ps[i][0][1]
        if es[u] == None:
            es[u] = []

        es[u].append((v, ps[i][1]))
    
    return es

def topic_graph_list(n, k, es0):
  # creating empty list of k-dimension
  es = [None]*k

  # nested list level 1: k elements
  for z in range(k):
      es[z] = [None]*n
      # nested list level 2
      # k'th element (a list) of level 1 consists
      # of n elements (3523)
      for u in range(n):
          # nested list level 3
          # no. of elements = no. of neighbors
          # element value: (neighbor_id, prob. w.r.t k'th topic)
          for i in range(len(es0[u]) if es0[u] != None else 0):
              v = es0[u][i][0]
              p = es0[u][i][1][z]
              if es[z][u] == None:
                  es[z][u] = []
                  
              es[z][u].append((v, p))
    
  return es

def simulate(n
             ,es
             ,k
             ,S
             ,R
             ,mode):
    """
    """
    
    topic_spread_sum = [0]*k
    sum_spread = 0
    # loop for running R MC simulations
    for t in range(R):
        # X prevent activation of users provided in the S set
        X = [False]*n 
        active = [False]*n
        topic_spread = []
        for z in range(k):
            tmp = []
            Q = []
            for i in range(len(S)):
                X[S[i][0]] = True
                if S[i][1] == z:
                    Q.append(S[i][0])
                    active[S[i][0]] = True
            
            # runs until no more further activation is possible
            while len(Q) > 0:
                u = Q[0]
                Q.pop(0)
                active[u] = True
                tmp.append(u)
                # check if element u exits in es at z-topic
                if es[z][u] is not None:
                    for i in range(len(es[z][u])):
                        v = es[z][u][i][0]
                        p = es[z][u][i][1]
                        rand = random.random()
                        # if not X[v] and rand < p:
                        if (not active[v]) and (not X[v]) and rand < p:
                            X[v] = True
                            Q.append(v)
            
            topic_spread.append(sum(active))
                
        n1 = 0
        # calculating total active users at the end of a simulation cycle
        # deactivating activated users in active for the next simulation
        for v in range(n):
            if active[v]:
                n1 = n1+1
                active[v] = False
        
        incr_spread = [0]
        incr_spread.extend(topic_spread[:-1])
        topic_spread = [i-j for i, j in zip(topic_spread, incr_spread)]
        
        topic_spread_sum = list(map(add, topic_spread_sum, topic_spread))
        sum_spread = sum_spread + n1

    if mode == 'all':
        return (1.0 * sum_spread)/R
    elif mode == 'topical':
        return ((1.0 * sum_spread)/R, [t/R for t in topic_spread_sum])
    
    
# k-greed
def k_greed(n
            ,es
            ,k
            ,beta
            ,budget
            ,budget_step_mode=False):  
    
    # queue of n*k (no of users * no of topics)
    que = []
    for v in range(n):
        for z in range(k):
            que.append((1e12, ((v, z),-1)))
            
    # reversed in descending order to implement a cpp priority queue
    que.sort(reverse=True)

    max_spread=0.0
    size_of_current_best=0
    global_num=0
    used = [False]*n
    S = []
    spread_curr = 0

    # dict for saving spread and seed sets at each incremental budget step of 10
    spread_step_dict = {}
    
    # budget is total size constraint (B), |supp(x)| ≤ B
    for j in range(budget):
        next = ()
        num = 0
        while True:
            pp = que[0] # (1000000000000.0, ((3522 (user), 9 (k-topic)), -1))
            que.pop(0) # removes highest priority element from the queue
            # getting user and k'th topic pair
            s = pp[1][0] # (3522, 8)
            last = pp[1][1] # -1
            
            if used[s[0]]:
                num = num + 1
                if len(que)> 0:
                    continue
                else:
                    break
            
            if last == j:
                next = s
                gain = pp[0]                
                num = num + 1
                break
            
            SS = S.copy()
            SS.append(s)
            sigma = simulate(n, es, k, SS, beta, mode='all')
            # to avoid computation wastage when S=[]
            if len(S) > 0:
                psigma = spread_curr
            else:
                psigma = 0
            
            # selected_z and selected_node are initialized 
            # for cost computations
            selected_z = pp[1][0][1]
            selected_node = pp[1][0][0]
            que.append((sigma-psigma, (s, j))) 
            # sigma-psigma: incremental spread
            que.sort(reverse=True)
            num = num + 1
            
        # add a new node
        # filter to avoid adding empty next to S
        # after breaking out of loop with len(que) = 0 and used[s[0]] = True
        if len(next) > 0:
            S.append(next)
            used[next[0]] = True

        # spread of S (current solution)
        spread_curr = simulate(n, es, k, S, beta, mode='topical')
        topical_spread_curr = spread_curr[1]
        spread_curr = spread_curr[0]

        if spread_curr > max_spread:
            max_spread = spread_curr
            max_topical_spread = topical_spread_curr
            size_of_current_best=j+1
            
        global_num = global_num + num

        if budget_step_mode == True and (j+1)%10==0:
            spread_step_dict[j+1] = {'seed_set': S.copy()
                                    ,'solution_size': j+1
                                    ,'spread': spread_curr
                                    ,'topical_spread': topical_spread_curr
                                    ,'evaluations': global_num}

    print(f'seed set: {S}')
    print(f'Solution size = {size_of_current_best}, spread = {max_spread}')
    print(f'topical spread = {max_topical_spread}')
    print(f'Number of evaluations: {global_num}')

    if budget_step_mode == True:
      return spread_step_dict
    else:
      return {'seed_set': S
              ,'solution_size': size_of_current_best
              ,'spread': max_spread
              ,'topical_spread': max_topical_spread
              ,'evaluations': global_num}
  
# single(i) algorithm
def single(n
            ,es
            ,k
            ,beta
            ,topic
            ,budget
            ,budget_step_mode=False):  
    
    # queue of n (no of users)
    que = []
    for v in range(n):
      que.append((1e12, (v, -1)))
            
    # reversed in descending order to implement a cpp priority queue
    que.sort(reverse=True)

    max_spread=0.0
    size_of_current_best=0
    global_num=0
    used = [False]*n
    S = []
    spread_curr = 0

    # dict for saving spread and seed sets at each incremental budget step of 10
    spread_step_dict = {}
    
    # budget is total size constraint (B), |supp(x)| ≤ B
    for j in range(budget):
        next = ()
        num = 0
        while True:
            pp = que[0] # (1000000000000.0, (3522 (user), -1))
            que.pop(0) # removes highest priority element from the queue
            v = pp[1][0] # 3522
            last = pp[1][1] # -1
            s = (v, topic)

            if used[v]:
                num = num + 1
                if len(que)> 0:
                    continue
                else:
                    break
            
            if last == j:
                next = s
                gain = pp[0]
                num = num + 1
                break
                
            SS = S.copy()
            SS.append(s)
            sigma = simulate(n, es, k, SS, beta, mode='all')
            # to avoid computation wastage when S=[]
            if len(S) > 0:
                psigma = spread_curr
            else:
                psigma = 0

            que.append((sigma-psigma, (v, j))) 
            que.sort(reverse=True)
            
            num = num + 1
            
        # add a new node
        # filter to avoid adding empty next to S
        # after breaking out of loop with len(que) = 0 and used[s[0]] = True
        if len(next) > 0:
            S.append(next)
            used[next[0]] = True

        # spread of S (current solution)
        spread_curr = simulate(n, es, k, S, beta, mode='topical')
        topical_spread_curr = spread_curr[1]
        spread_curr = spread_curr[0]

        if spread_curr > max_spread:
            max_spread = spread_curr
            max_topical_spread = topical_spread_curr
            size_of_current_best=j+1
            
        global_num = global_num + num

        if budget_step_mode == True and (j+1)%10==0:
            spread_step_dict[j+1] = {'seed_set': S.copy()
                                    ,'solution_size': j+1
                                    ,'spread': spread_curr
                                    ,'topical_spread': topical_spread_curr
                                    ,'evaluations': global_num}

    print(f'seed set: {S}')
    print(f'Solution size = {size_of_current_best}, spread = {max_spread}')
    print(f'topical spread = {max_topical_spread}')
    print(f'Number of evaluations: {global_num}')

    if budget_step_mode == True:
      return spread_step_dict
    else:
      return {'seed_set': S
              ,'solution_size': size_of_current_best
              ,'spread': max_spread
              ,'topical_spread': max_topical_spread
              ,'evaluations': global_num}

    
def random_spread(n
                  ,k
                  ,es
                  ,beta
                  ,budget
                  ,budget_step_mode=False):
  
  # dict for saving spread and seed sets at each incremental budget step of 10
  spread_step_dict = {}
  S = []
  SS = []
  for j in range(budget):
    while True:
      v = random.randint(0,n)
      if v not in SS:
        SS.append(v)
        break
    z = random.randint(0,k-1)
    S.append((v, z))

    if budget_step_mode == True and (j+1)%10==0:
      spread_curr = simulate(n, es, k, S, beta, mode='topical')
      topical_spread_curr = spread_curr[1]
      spread_curr = spread_curr[0]
      spread_step_dict[j+1] = {'seed_set': S.copy()
                              ,'solution_size': j+1
                              ,'spread': spread_curr
                              ,'topical_spread': topical_spread_curr}
  
  # spread of S (current solution)
  spread_curr = simulate(n, es, k, S, beta, mode='topical')
  topical_spread_curr = spread_curr[1]
  spread_curr = spread_curr[0]

  if budget_step_mode == True:
    return spread_step_dict
  else:
    return {'seed_set': S
            ,'solution_size': j+1
            ,'spread': spread_curr
            ,'topical_spread': topical_spread_curr}


def degree_spread(n
                  ,k
                  ,es
                  ,beta
                  ,budget
                  ,budget_step_mode=False):
    
    # dict for saving spread and seed sets at each incremental budget step of 10
    spread_step_dict = {}
    odeg = []
    S = []
    for v in range(n):
        if es[0][v] == None:
            o = 0
        else:
            o = len(es[0][v])
        odeg.append((o,v))

    odeg.sort(reverse=True)

    for j in range(budget):
        v = odeg[j][1]
        z = random.randint(0,k-1)
        S.append((v,z))

        if budget_step_mode == True and (j+1)%10==0:
          spread_curr = simulate(n, es, k, S, beta, mode='topical')
          topical_spread_curr = spread_curr[1]
          spread_curr = spread_curr[0]
          spread_step_dict[j+1] = {'seed_set': S.copy()
                                  ,'solution_size': j+1
                                  ,'spread': spread_curr
                                  ,'topical_spread': topical_spread_curr}
    
    # spread of S (current solution)
    spread_curr = simulate(n, es, k, S, beta, mode='topical')
    topical_spread_curr = spread_curr[1]
    spread_curr = spread_curr[0]

    if budget_step_mode == True:
      return spread_step_dict
    else:
      return {'seed_set': S
              ,'solution_size': j+1
              ,'spread': spread_curr
              ,'topical_spread': topical_spread_curr}


def topic_marginal_gain(n
                        ,es
                        ,seed_topic
                        ,S
                        ,R):
    """
    """
    delta_set = []
    # loop for running R MC simulations
    for i in range(len(S)):
      user = S[i][0]
      z = seed_topic
      marg_gain = 0
      for t in range(R):
          # X prevent activation of users provided in the S set
          X = [False]*n 
          active = [False]*n
          tmp = []
          Q = [user]
          for i in range(len(S)):
            X[S[i][0]] = True

          # runs until no more further activation is possible
          while len(Q) > 0:
              u = Q[0]
              Q.pop(0)
              active[u] = True
              tmp.append(u)
              # check if element u exits in es at z-topic
              if es[z][u] is not None:
                  for i in range(len(es[z][u])):
                      v = es[z][u][i][0]
                      p = es[z][u][i][1]
                      rand = random.random()
                      if (not active[v]) and (not X[v]) and rand < p:
                          X[v] = True
                          Q.append(v)
                  
          n1 = 0
          # calculating total active users at the end of a simulation cycle
          # deactivating activated users in active for the next simulation
          for v in range(n):
              if active[v]:
                  n1 = n1+1
                  active[v] = False

          marg_gain = marg_gain + n1

      marg_gain = (1.0 * marg_gain)/R
      delta_set.append((marg_gain, user))
    
    delta_set.sort(reverse=True)

    return delta_set


def topic_fsa_spread(n
                    ,es
                    ,k
                    ,beta
                    ,budget
                    ,seed_select_topic
                    ,budget_spread='eq'
                    ,budget_step_mode=False):
  
  # first find all seeds S using the greedy approach
  # the IC spread simulation is done for one generic item i.e. topic with 
  # median spread in single(i) algorithm

  if budget_spread == 'eq':
    budget_ls = [int(budget/k)]*k
  elif budget_spread == 'neq':
    budget_ls = [int(budget/k)]*k
    for b in range(len(budget_ls)):
      if b%2==0 and ((b+1) != k):
        budget_ls[b] = budget_ls[b]+1
      elif b%2==1 and ((b+1) != k):
        budget_ls[b] = budget_ls[b]-1
  
  budget = budget_ls
  single_spread = single(n
                         ,es
                         ,seed_select_topic+1 # to have enough topics for single run for seed_select_topic
                         ,beta
                         ,seed_select_topic
                         ,int(sum(budget))
                         ,budget_step_mode=False)

  # seed set of the given budget is obtained from single(i)
  seed_set = single_spread['seed_set']
  # sorting seed set on the basis of adjusted marginal gain in decreasing order
  seed_set = topic_marginal_gain(n, es, seed_select_topic, seed_set, beta)
  
  S = dict.fromkeys([k for k in range(len(budget))])

  # initializing empty lists (sets) at all keys (topics)
  for key in S.keys():
    S[key] = []

  # selecting topics for user seed set obtained 
  # by maximum spread by IC model
 
  for seed in seed_set:
    u = seed[1]
    # list of topics/companies for which the budget
    # has not been exhausted
    T = []
    for key in list(S.keys()):
      if len(S[key]) < budget[key]:
        T.append(key)
    aplha_min = 1e12

    # getting topic with minimum amp ratio
    for t in T:
      SS_sigma = S.copy()
      SS_sigma = [item for elem in list(SS_sigma.values()) for item in elem]
      if len(SS_sigma) > 0:
        sigma = simulate(n, es, k, SS_sigma, beta, mode='topical')
        topic_spread = sigma[1][t]
        alpha = topic_spread/budget[t]
      else:
        sigma = 0
        alpha = 0
      # minimum amplification factor
      if alpha < aplha_min:
        aplha_min = alpha
        # topic selected
        j = t
    
    # updating parititoned seed set
    S[j].append((u, j))
    print(S)

  # converting seed set dict to list
  S = [item for elem in list(S.values()) for item in elem]
  spread_curr = simulate(n, es, k, S, beta, mode='topical')
  topical_spread_curr = spread_curr[1]
  spread_curr = spread_curr[0]

  return {'seed_set': S
          ,'budget': budget
          ,'budget_spread': budget_spread
          ,'solution_size': sum(budget)
          ,'spread': spread_curr
          ,'topical_spread': topical_spread_curr
          ,'evaluations': 1}


def infl_max(k
             ,edge
             ,prob
             ,beta
             ,budget
             ,budget_step_mode = False
             ,budget_spread = 'na'
             ,algo = 'k_greed'
             ,topic = 1):
    """
    beta: number of simulations (MC simulations)
    """
    global output_file

    # loading loading for topics=4 for fsa as topic(3) is used for phase 1
    # seed identification
    if algo == 'fsa' and k <= 3:
      es0 = load_graph(edge, prob, 4) 
      # number of distinct users in the network
      n = len(es0)
      es = topic_graph_list(n, 4, es0)
    else:
      es0 = load_graph(edge, prob, k) 
      # number of distinct users in the network
      n = len(es0)
      es = topic_graph_list(n, k, es0)
    
    if algo == 'k_greed':
        im_spread = k_greed(n
                            ,es
                            ,k
                            ,beta
                            ,budget
                            ,budget_step_mode=budget_step_mode)
    
    elif algo == 'single':
        im_spread = single(n
                           ,es
                           ,k
                           ,beta
                           ,topic
                           ,budget
                           ,budget_step_mode=budget_step_mode)
    
    elif algo == 'random':
        im_spread = random_spread(n
                                  ,k
                                  ,es
                                  ,beta
                                  ,budget)
    
    elif algo == 'degree':
        im_spread = degree_spread(n
                                  ,k
                                  ,es
                                  ,beta
                                  ,budget)
    elif algo == 'fsa':
        im_spread = topic_fsa_spread(n
                                    ,es
                                    ,k
                                    ,beta
                                    ,budget
                                    ,seed_select_topic=topic
                                    ,budget_spread=budget_spread
                                    ,budget_step_mode=budget_step_mode)
    
    im_spread = {'param': {'k': k
                           ,'n': n
                           ,'beta': beta
                           ,'budget': budget
                           ,'budget_spread': budget_spread
                           ,'algo': algo
                           ,'topic': topic}
                 ,'mode': budget_step_mode
                 ,'result': im_spread}

    # saving im_spread dict to a file
    dt_txt = datetime.datetime.now().strftime('%d%m%Y_%H%M')
    file_name = f'{dt_txt}_{k}topics_{beta}sim_{budget}budget_{budget_step_mode}step_{algo}_{topic}topic.json'

    json_op_file = output_file.replace('::file_name::', file_name)
    json_file = open(json_op_file, "w")
    json.dump(im_spread, json_file, cls=NpEncoder)
    json_file.close()
    
    return im_spread

if __name__ == '__main__':
    
    k=5
    budget = 100
    beta = 100

    k_greed_output = infl_max(k
                              ,edge
                              ,prob
                              ,beta
                              ,budget
                              ,budget_step_mode=False
                              ,algo='k_greed')
    
    topic_fsa_output = infl_max(k
                                ,edge
                                ,prob
                                ,beta
                                ,budget
                                ,budget_step_mode = False
                                ,budget_spread = 'eq'
                                ,algo = 'fsa'
                                ,topic = 3)