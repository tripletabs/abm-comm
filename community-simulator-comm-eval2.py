#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Examining the influence of user's "context mobility" on the formation of online communities

Created on Sun Apr 8 20:48 2020
"""


import pycxsimulator
from pylab import *
import networkx as nx
import random as rd
import numpy as np
import numpy.matlib as npm
from pyDOE import lhs
from collections import Counter



### defualt Model Parameters ---------------------------------------------------

# Networks
# 1. Barabasi Albert graph: nx.barabasi_albert_graph(n, m)
# avergae degree = 2m
N = 1000                     # the number of nodes
m = 1                       # number of edges to attach from a new node
# 2. Watts Strogatz graph: nx.watts_strogatz_graph(n, k, p)
# average degree = k
# N = 200                   # same number of nodes
k = 2*m                     # each node is connected to k nearest neighbors
p = 0.05                    # the probability of rewiring each edge

# Communities
NUM_topics = 5               # number of sub-topics under an overarching theme
TOPICS = range(0, NUM_topics)

# agents
pf_mean = 90
pf_sd = 10
pf = np.random.normal(pf_mean, pf_sd, N) # 0 - 100

# pre-penalized probabilities
pc_mean = 0.5
pc_sd = 0.1
pc = np.random.normal(pc_mean, pc_sd, N) # 0 - 100

pb_mean = 0.5
pb_sd = 0.1
pb = np.random.normal(pb_mean, pb_sd, N) # 0 - 100

ps_mean = 0.2
ps_sd = 0.1
ps = np.random.normal(ps_mean, ps_sd, N) # 0 - 100

### ----------------------------------------------------------------------------



### other utils ----------------------------------------------------------------

def like_minded(n1, n2, g):
    if g.node[n1]["topic"] == g.node[n2]["topic"]:
        return True
    else:
        return False

def sort_nodes(graph):
    nodes_topics = [(n, graph.node[n]["topic"]) for n in graph.nodes]
    sorted_nodes_topics = sorted(nodes_topics, key=lambda x: x[1])
    sorted_nodes = [pair[0] for pair in sorted_nodes_topics]
    return sorted_nodes

### ----------------------------------------------------------------------------



### model body -----------------------------------------------------------------

def initialize():
    global g, nextg, coverage_array, perfmce_array

    # g = nx.watts_strogatz_graph(N, k, p)
    g = nx.barabasi_albert_graph(N, m)
    g.pos = nx.spring_layout(g)

    # set up initial attributes
    node_attr_dict = {}
    # topic_ls = []
    for i in range(N):
        random_topic = rd.choice(TOPICS)
        # topic_ls.append(random_topic)
        node_attr = {"topic": random_topic,
                    "pc": pc[i],
                    "pb": pb[i],
                    "ps": ps[i],
                    "pf": pf[i]}
        node_attr_dict[i] = node_attr
    nx.set_node_attributes(g, node_attr_dict)

    # records community member to evaluate coverage/performance
    topic_member_dict = {}
    for k in range(NUM_topics):
        topic_member_dict[k] = [n for n in g.nodes if g.node[n]["topic"] == k]
    partition = [set(ls) for k,ls in topic_member_dict.items()]

    # initialize arrays
    coverage_array = [nx.community.quality.coverage(g, partition)]
    perfmce_array = [nx.community.quality.performance(g, partition)]


    nextg = g.copy()
    nextg.pos = g.pos


def update():
    global g, nextg, coverage_array, perfmce_array

    # update graph
    nextg = g.copy()
    nextg.pos = g.pos

    for a in g.nodes:
        my_topic = g.node[a]["topic"]
        other_topics = list(set(TOPICS) - {my_topic})
        neighbors = [*g.neighbors(a)]
        strangers = list(set(g.nodes) - set(neighbors) - {a})
        pc_a = g.node[a]["pc"]
        pb_a = g.node[a]["pb"]
        ps_a = g.node[a]["ps"]
        pf_a = g.node[a]["pf"]
        ppc_a = ((100-pf_a)/100)*pc_a
        ppb_a = ((100+pf_a)/100)*pb_a
        pps_a = ((100-pf_a)/100)*ps_a

        # break up with current neighbors
        for nb in neighbors:
            if like_minded(a, nb, g): # if they have common interests
                if random() < pb_a and (nextg.has_edge(a, nb)):
                    nextg.remove_edge(a, nb) # going to break at a pre-penalized probability
                    # print(f"removed {a}-{nb}")
            else: # if they have no common interests
                if random() < ppb_a and (nextg.has_edge(a, nb)):
                    nextg.remove_edge(a, nb) # going to break at a higher probability
                    # print(f"removed {a}-{nb}")

        # connect with distant strangers
        for stg in strangers:
            if like_minded(a, stg, g): # if they have common interests
                if random() < pc_a and (nextg.has_edge(a, stg) == False):
                    nextg.add_edge(a, stg) # going to connect at a pre-penalized probability
            else: # if they have no common interests
                if random() < ppc_a and (nextg.has_edge(a, stg) == False):
                    nextg.add_edge(a, stg) # going to connect at a lower probability

        # switch to a new topic
        if random() < pps_a:
            g.node[a]["topic"] = rd.choice(other_topics)


    topic_member_dict = {}
    for k in range(NUM_topics):
        topic_member_dict[k] = [n for n in g.nodes if g.node[n]["topic"] == k]
    partition = [set(ls) for k,ls in topic_member_dict.items()]

    coverage_array.append(nx.community.quality.coverage(g, partition))
    perfmce_array.append(nx.community.quality.performance(g, partition))

    g = nextg.copy()
    g.pos = nextg.pos




# def observe():
#     global g
#     cla()
#     nodelist = sort_nodes(g)
#     adj_matrix = nx.adjacency_matrix(g, nodelist=nodelist).todense()
#     imshow(adj_matrix)
#     show()
#     pass

### ----------------------------------------------------------------------------
# pycxsimulator.GUI().start(func=[initialize, observe, update])



timesteps = 100  # model converged pretty early

pf_mean_ls = np.arange(50, 95, 5)
cmap1 = ["peachpuff","lightsalmon","salmon","tomato","orangered","red","firebrick","darkred","maroon"]
cmap2 = ["lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "royalblue", "blue", "mediumblue", "darkblue", "midnightblue"]

fig,ax = plt.subplots()
for i,pf_mean in enumerate(pf_mean_ls):
    print("run:", i)
    pf = np.random.normal(pf_mean, pf_sd, N) # 0 - 100

    initialize()
    for j in range(0, timesteps):
        print("timestep:", j)
        update()

    curve = plt.plot(coverage_array, label="coverage", color=cmap1[i])
    curve = plt.plot(perfmce_array, label="performace", color=cmap2[i])

ax.set_xlabel("timestep")
ax.set_ylabel("score")
ax.set_title("Sensitivity Test for Penalizing Factor")
show(curve)
savefig("pf-sensitivity.png")
