
import pandas as pd
import os
from queue import PriorityQueue
import random
import logging as logger
import datetime

random.seed(20083811)

dat_dir = 'dat'

log_file = f"{datetime.datetime.now().strftime('%d%m%Y_%H%M')}_im.log"

edge = os.path.join(dat_dir, 'digg_graph_100.tsv')
prob = os.path.join(dat_dir, 'digg_prob_100_1000_pow10.tsv')
log_file = os.path.join('log', log_file)

edge = pd.read_csv(edge, sep='\t', header=None)
prob = pd.read_csv(prob, sep='\t', header=None)

k=10


def load_graph(edge
               ,prob
               ,k):
    
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


def simulate(n
             ,es
             ,k
             ,S
             ,R
             ,logger):
    
    # int n
    # vector<vector<vector<pair<int, double> > > > &es
    # int k
    # vector<pair<int, int> > &S # user and k'th topic pair
    # int R # no of random graphs (simulations)
    
    logger.info('simulation starts...')

    sum = 0
    # logger.info(f'S=SS: {S}')
    # logger.info(f'R=beta: {R}')
    for t in range(R):
        logger.info(f'{t} MC simulation')
        for z in range(k):
            logger.info(f'{z} topic')
            tmp = [] # vector<int> tmp
            # queue<int> Q;
            Q = []
            
            logger.info(f'length of S: {len(S)}')
            for i in range(len(S)):
                logger.info(f'{i} index of S')
                if S[i][1] == z:
                    logger.info(f'S[i][1] == z | {S[i][1]} == {z}')
                    Q.append(S[i][0])
                    X[S[i][0]] = True
                    logger.info(f'X[S[i][0]] = True | X[{S[i][0]}] = True')
                    
            while len(Q) > 0:
                logger.info(f'u = Q[0] | u = {Q[0]}')
                u = Q[0]
                Q.pop(0)
                active[u] = True
                tmp.append(u) # tmp.push_back(u);
                logger.info(f'len(es[z][u]) | {len(es[z][u])}')
                for i in range(len(es[z][u])):
                    v = es[z][u][i][0]
                    p = es[z][u][i][1]
                    logger.info(f'v = {es[z][u][i][0]}, p = {es[z][u][i][1]}')
                    rand = random.random()
                    logger.info(f'X[v] = {X[v]} and {rand} < {p}')
                    if not X[v] and rand < p:
                        X[v] = True
                        logger.info(f'appending {v} to Q')
                        Q.append(v)
            
            logger.info(f'length of tmp: {len(tmp)}')
            for i in range(len(tmp)):
                X[tmp[i]] = False
                logger.info(f'setting index {tmp[i]} of X to False')
                
        n1 = 0
        logger.info(f'n = {n}')
        for v in range(n):
            logger.info(f'iterating over range(n): iteration {v}')
            logger.info(f'active[v]: {active[v]}')
            logger.info(f'value of n1: {n1}')
            if active[v]:
                n1 = n1+1
                logger.info(f'new value of n1: {n1}')
                active[v] = False
                logger.info(f'setting index {v} of active to False')
                
        sum = sum + n1
        logger.info(f'sum: {sum}')
        
    logger.info(f'expected spread (sum/R): {(1.0 * sum)/R}')
    return (1.0 * sum)/R


# k-greed-ratio starts here
def k_greed(n
            ,es
            ,k
            ,budget
            ,beta):  
    
    # queue of n*k (no of users * no of topics)
    que = []
    for v in range(n):
        for z in range(k):
            que.append((1e12, ((v, z),-1)))
            
    # reversed in descending order to implement a cpp priority queue
    que.sort(reverse=True)

    # len(que) #35230 n * k
    # que[0] (1000000000000.0, ((3522, 9), -1))
    # que[1] (1000000000000.0, ((3522, 8), -1))

    # cost not to be considered
    # total_cost_so_far=0.0
    # max_ratio=0.0
    # max_cost=0.0
    max_spread=0.0
    size_of_current_best=0
    global_num=0

    # vector<bool> used(n);
    used = [False]*n

    # vector<pair<int, int> > S;
    S = []
    budget = 3523

    for j in range(budget):
        logger.info(f'loop {j} (j) of budget')
        next = ()
        num = 0
        print(f'j = {j}')
        while True:
            
            # pair<double, pair<pair<int, int>, int> > pp = que.top();
            pp = que[0] # (1000000000000.0, ((3522 (user), 9 (k-topic)), -1))
            que.pop(0) # removes highest priority element from the queue
            logger.info(f'pp: {pp}')
            print(f'pp = {pp}')
            # getting user and k'th topic pair
            s = pp[1][0] # (3522, 8)
            print(f's = {s}')
            logger.info(f's: {s}')
            last = pp[1][1] # -1
            logger.info(f'last: {last}')
            
            if used[s[0]]:
                print('continue')
                logger.info(f'continue: used[s[0]] = {used[s[0]]}')
                num = num + 1
                continue
            
            if last == j:
                next = s
                gain = pp[0]
                
                print('break')
                print(f'last = {last}')
                print(f'next = {next}')
                print(f'gain = {gain}')
                logger.info('break')
                logger.info(f'last = {last}')
                logger.info(f'next = {next}')
                logger.info(f'gain = {gain}')
                
                num = num + 1
                break
                
            
            SS = []
            SS.append(s)
            logger.info(f'appending: s = {s}')
            print(SS)
            logger.info(SS)
            # double sigma = simulate(n, es, k, SS, beta)
            sigma = simulate(n, es, k, SS, beta, logger)
            psigma = simulate(n, es, k, S, beta, logger)
            # why do we need sigma and psigma? Their difference is not used 
            # post computation
            # answer: As que is a priority queue, it's elements will get
            # sorted in reverse order and sigma-psigma is critical for 
            # getting priority order
            
            # selected_z and selected_node are initialized 
            # for cost computations
            selected_z = pp[1][0][1]
            print(f'selected_z = {selected_z}')
            logger.info(f'selected_z = {selected_z}')
            selected_node = pp[1][0][0]
            # cost_of_selected_node_at_z = costs[]
            print(f'selected_node = {selected_node}')
            logger.info(f'selected_node = {selected_node}')
            que.append((sigma-psigma, (s, j)))
            print(f'append: (s, j) = ({s}, {j})')
            logger.info(f'appending to que: (sigma-psigma, (s, j)) = {(sigma-psigma, (s, j))}')
            #que.append((500, (s, j)))
            que.sort(reverse=True)
            
            num = num + 1
            
        # add a new node
        print(f'next = {next}')
        logger.info(f'next = {next}')
        S.append(next)
        
        print(f'set used to True for next[0] = {next[0]}')
        logger.info(f'set used to True for next[0] = {next[0]}')
        used[next[0]] = True

        # spread of S (current solution)
        spread_curr = simulate(n, es, k, S, beta, logger)

        if spread_curr > max_spread:
            max_spread = spread_curr
            size_of_current_best=j+1

        global_num = global_num + num

    print(f'Solution size = {size_of_current_best}, spread = {max_spread}')
    print(f'Number of evaluations: {global_num}')



if __name__ == '__main':
    
    logger.basicConfig(filename=log_file
                        ,level=logger.DEBUG
                        ,format='%(asctime)s %(message)s')
    
    logger.info('creating graph data structure of the network data...')
    es0 = load_graph(edge, prob, k)
    
    # es0[1] # index position of es0 is node u = 1
    # and elements of es0[1] are list containing
    # connections of u = 1 and corresponding IP's of k-topics
     #    [(2291,
     #  [0.00119935,
     #   0.00307573,
     #   0.0041323,
     #   0.00153893,
     #   0.00518785,
     #   0.00896364,
     #   0.00384334,
     #   0.00849733,
     #   0.00135917,
     #   0.00584457]),
     # (2311,
     #  [0.00603356,
     #   0.00150897,
     #   0.00443114,
     #   0.00695812,
     #   0.00696805,
     #   0.00955863,
     #   0.00378354,
     #   0.00740521,
     #   0.00821944,
     #   0.0023575]),
     # (2348, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
     # (2492,
     #  [0.00302587,
     #   0.00233754,
     #   0.0073158,
     #   0.00673947,
     #   0.0059142,
     #   0.00934051,
     #   0.00926118,
     #   0.00201816,
     #   0.000269967,
     #   0.00626228]),
     # (2719,
     #  [0.00820951,
     #   0.00606339,
     #   0.00780247,
     #   0.00255705,
     #   0.00575504,
     #   0.00414226,
     #   0.00250717,
     #   0.00450086,
     #   0.00040992400000000004,
     #   0.00277652]),
     # (3152, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
     
    # number of distinct users in the network
    n = len(es0)
    
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
    
    # k'th topic list
    # es[k]
    # es[0] # for getting k=0 topic list
    # len(es[0]) 3523
    
    # neighbors of u=1 user and IP's for k=0 topic
    # es[k][u]
    # es[0][1]
    
    # neighbor at v=0 index of u=1 user and IP's for k=0 topic
    # es[k][u][v]
    # es[0][1][0]
    
    # k'th topic, u'th user, and v'th neighbor's prob
    # es[k][u][v][1]
    # es[0][1][0][1]
    
    # neighbors of u=1 and prob. of each neighbor w.r.t to k=0
    # es[0][1]
    # [(2291, 0.00119935),
    #  (2311, 0.00603356),
    #  (2348, 0.0),
    #  (2492, 0.00302587),
    #  (2719, 0.00820951),
    #  (3152, 0.0)]
    
    X = [False]*n
    active = [False]*n
    beta = 100
    # working without costs in contrast to C++ code
    budget = 3523
    
    k_greed(n ,es, k, budget, beta)
    
    simulate(n,es,k,S=[(3522, 7)],R=100,logger=logger)
