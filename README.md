# 7CCSMPRJ - Social Network Optimization: k-submodular Influence Maximization

## Getting started
The README file contains instructions to understand, configure and execute all the algorithms simulated in the experimental evaluation. 

### Prerequisites
* Python >= 3.8.5
* Python libraries: os, pandas, random, datetime, operator, numpy, json

### Statement for work certification
```
I verify that I am the sole author of the programmes contained in this archive, except where explicitly stated to the contrary.

Himanshu Verma
2nd Sep 2021
```

### Author
```
Verma, Himanshu
k20083811
himanshu.verma@kcl.ac.uk
```
***

## 1. Supplemental folder structure

The supplemental folder for `Social Network Optimization: k-submodular Influence Maximization` by Himanshu Verma (k20083811) is composed of the 
fundamental components employed in the experimental evaluation:
* Source code of the dissertation project: `k-sub-influence-maximization.py`
* `dat` folder containing publicly available dataset called `Digg 2009 data set` (https://www.isi.edu/~lerman/downloads/digg2009.html) bifurcated into two .tsv files:
	* `digg_graph_100.tsv`: Contains edge links among the users of Digg website
	* `digg_prob_100_1000_pow10.tsv`: Includes edge probability of the edges given in `digg_graph_100.tsv` for distinct topics
* `output` folder is responsible for saving output files (.JSON files) of the source code

***

## 2. Source Code

The source code of this project, `k-sub-influence-maximization.py`, is developed in Python codebase by leveraging the C++ based source code of "Algorithms for Optimizing the Ratio of Monotonek-Submodular Functions" (by Hau Chan, Grigorios Loukides, Zhenghui Shu, ECML/PKDD 2020) https://doi.org/10.1007/978-3-030-67664-3_1

### User-defined functions:

Multiple functions are defined in the source code, performing assorted tasks such as performing ETL (Extract, Transform, Load) operations of social network data and simulating the functioning of primary algorithms and baselines.

The following functions are used for loading and creating (ETL operations) graphical data structures:
* `load_graph`: Reads edge and probability datasets and modifies data structure for simulation and k-greedy consumption
* `topic_graph_list`: Further modifies the data structure of the output delivered by `load_graph` for spread simulation

The following functions are essential for simulating the spread of the algorithms discussed in the thesis:
* `simulate`: Performs Monte Carlo (MC) simulations for algorithms
* `k_greed`: Simulates k-Greedy algorithm
* `single`: Simulates Single(i) baseline algorithm
* `random_spread`: Simulates Random baseline algorithm
* `degree_spread`: Simulates Degree baseline algorithm
* `topic_marginal_gain`: Simulates marginal gain and used in Fair Seed Allocation (FSA) approach based Needy Greedy-IC algorithm
* `topic_fsa_spread`: Simulates Needy Greedy-IC algorithm

### Parameters of master function:

`infl_max` function collates all the above-mentioned functions under a single master function for simulating expected influence spread under varying configurations. Different configurations are arranged by controlling the following parameters:
* `k`: Number of topics/items/products/dimensions
* `edge`: Pandas dataframe containing graph edges
* `prob`: Pandas dataframe containing edge probabilities
* `beta`: Number of Monte Carlo (MC) simulations
* `budget`: Number of desired users in the seed set
* `budget_step_mode`: If set to `True`, `infl_max` delivers the expected spread at even budget steps of 10. For instance, `infl_max` outputs influence spread for budget values ranging from 10 to 100 in steps of 10 if the `budget` is equal to 100
* `budget_spread`: Controls split of the budget among the topics (either equal or non-equal split) (only for FSA based Needy Greedy-IC)
* `algo`: Parameter for setting algorithm, with default value set to `k_greed`
	* `k_greed`: k-Greedy
	* `fsa`: Needy Greedy-IC
	* `single`: Single(i)
	* `random`: Random
	* `degree`: Degree 
* `topic`: Selects topic for Single(i) algorithm, with default value set to `1` (Only for Single(i))


```
Note: In addition to algorithmic and data ETL functions, the source code defines a class called NpEncoder 
(provided by user "Jie Yang" on stackoverflow.com at https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741) 
for overcoming issues encountered while saving output dictionary (dict) as JSON file.  
```

### Example of running algorithms using `infl_max` function:
```
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
```