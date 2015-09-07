__author__ = 'yuxinsun'
import networkx as nx
import numpy as np
import itertools
from itertools import chain
import random
from compiler.ast import flatten
from sklearn.preprocessing import normalize
from collections import OrderedDict

# This file contains functions for:
# Compute the p-spectrum string kernel for the base feature space
# Build the DAG for the FSM model using both weighting schemes


# Create a word list from alphabet
# This word list contains all possible combinations of length p strings
# Input:
# alphabet: amino acid alphabet
# p: spectrum
# Output:
# word_list: a list of all combinations of length p strings
def create_word_list(alphabet, p):

    word_list = list(itertools.product(alphabet, repeat=p))
    for i in range(0,len(word_list)):
        word_list[i] = ''.join((itertools.chain(word_list[i])))

    return word_list


# Process data
# Split the CDR3 sequences in each file with a space
# Input:
# file_list: a list of file names
# path: path of data files
# l: number of CDR3 sequences to be randomly selected. If l = "all", all sequences will be kept
# Output:
# data: list of processed CDR3s
def process_data(file_list, path, l):
    from itertools import chain
    import random

    data = []
    for file in file_list:
        print('Processing data: ', file)
        file_object = open(path+file+'.txt')
        file_name = file_object.readlines()
        file_name = [s.strip('\r\n') for s in file_name]

        data_temp = []
        for file_line in file_name:
            s = file_line.split()
            data_temp.append(s[0])

        # Randomly select l CDR3s
        if l != 'all':
            random.shuffle(data_temp, random.random)
            data_temp = data_temp[:l]

        # Split the CDR3s with spaces
        data.append(' '.join(chain(data_temp)))

    return data


# Process data by splitting the CDR3 sequences with a space
# This function is particularly designed for P277 mice files, which have different formats from other files
# Input:
# file_list: a list of file names
# path: path of data files
# l: number of CDR3 sequences to be randomly selected. If l = "all", all sequences will be kept
# Output:
# data: list of processed CDR3s
def process_data_count(file_list, path, l):

    data = []
    for file in file_list:
        print('Processing data: ', file)
        data_temp = []
        file_object = open(path+file+'.txt')
        file_name = file_object.readlines()
        file_name = [s.strip('\r\n') for s in file_name]

        del file_name[0]

        # Split the CDR3s with spaces
        for fileline in file_name:
            s = fileline.split(',')
            s = (s[0]+' ')*int(s[1])
            s = s.split(' ')
            data_temp.append(s[:-1])

        data_temp = flatten(data_temp)

        # Randomly select l CDR3s
        if l != 'all':
            random.shuffle(data_temp, random.random)
            data_temp = data_temp[:l]

        data.append(' '.join(chain(data_temp)))

    return data


# Build a DAG for training data
# Input:
# data: training data
# pmin: minimum state length
# pmax: maximum state length
# thre: threshold t
# alphabet: amino acid alphabet
# kernel_type: string or fisher - when set to string, this function returns a DAG equivalent to the p-spectrum kernel
# Output:
# DAG G
def KernelDAG(data, pmin, pmax, thre, alphabet, kernel_type):
    word_list = set(create_word_list(alphabet, pmin))
    M = len(data)
    edge_dic = {}  # Python dictionary of edges
    kern_dic = {}  # Python dictionary of string kernel or Fisher features

    for m in range(0, M):
        print('Data item: %d' % m)
        for i in range(0, len(data[m])-pmin+1+1):
           # Compute term frequencies of substrings of length pmin-1 to pmax
           for j in range(pmin-1, pmax+1):
               if i+j <= len(data[m]):
                sub = data[m][i:i+j]
                if sub in kern_dic:
                    kern_dic[sub][m] += 1
                elif ' ' not in sub:
                    kern_dic[sub] = np.zeros(M)
                    kern_dic[sub] = kern_dic[sub].astype(float)
                    kern_dic[sub][m] = 1

    base_dic = dict((k, v) for k, v in kern_dic.items() if len(k) <= pmax)  # a Python dictionary of all possible states
    base_dic = dict((k, v) for k, v in base_dic.items() if np.sum(v) >= thre)  # a Python dictionary of base features
    kern_dic = dict((k, v) for k, v in kern_dic.items() if len(k) >= pmin)  # a Python dictionary of all non zero substrings

    word_list = word_list - word_list.intersection(set(base_dic.keys()))  # all length pmin-1 transitions that are not in base_dic
    for word in word_list:
        edge_dic[(word[:-1], word[1:])] = np.zeros(M)

    for k, v in kern_dic.items():
        if np.sum(v) >= thre:
            if kernel_type == 'fisher':
                edge_dic[(k[:-1], k)] = v
            elif kernel_type == 'string':
                edge_dic[(k[:-1], k[1:])] = v
        else:
            for i in range(len(k), pmin-1, -1):
                if k[-i:] in base_dic and k[:-1] in base_dic:
                    edge_dic[(k[:-1], k[-i:])] = v
                    continue

    kern_dic.clear()
    base_dic.clear()

    # Add edges to an empty DAG G
    G = nx.DiGraph()
    G.add_edges_from(edge_dic.keys())
    nx.set_edge_attributes(G, 'kern_unnorm', edge_dic)

    alphabet_set = set(alphabet)

    # Add edges that represents non-existing transitions (states of length pmin-1 only
    for node in G.nodes():
        if 'X' in node:  # Remove symbol "X" in CDR3 sequences as it does not belong to 20 amino acids
            G.remove_node(node)
            continue

        suc = G.successors(node)
        suc = set([temp[-1] for temp in suc])
        inter = alphabet_set-alphabet_set.intersection(suc)

        for letter in inter:
            G.add_edge(node, node[-1]+letter, kern_unnorm=np.zeros(M))

    return G


# Build the DAG for test data
# Input:
# data: test data
# G_train: DAG for training data
# pmin: minimum state length
# pmax: maximum state length
# Output:
# G_test: DAG for test data
def kernelDAGTest(data, G_train, pmin, pmax):
    print('Computing Test DAG')
    edges = nx.edges(G_train)
    kern = dict.fromkeys(edges, 0)
    kern_temp = {}

    for i in range(0, len(data)):
        for j in range(pmin-1, pmax+1):
            if data[i:i+j] in kern_temp:
                kern_temp[data[i:i+j]] += 1
            else:
                kern_temp[data[i:i+j]] = 1

    for edge in edges:
        key = edge[0]+edge[1][-1]
        if key in kern_temp:
            kern[edge] = kern_temp[key]

    G_test = nx.DiGraph()
    G_test.add_edges_from(edges)
    nx.set_edge_attributes(G_test, 'kern_unnorm', kern)

    return G_test


# Compute transition probabilities for training data
# Input:
# dag: DAG for training data
# thre: threshold t
# Output:
# dag: DAG for training data with transition probabilities
def tranDAG(dag, thre):

    for node in dag.nodes():
        s = (np.sum(dag[node][x]['kern_unnorm']) for x in dag.successors(node))
        s = sum(s)
        for x in dag.successors(node):
            if s == 0:
                dag[node][x]['tran'] = 0
            else:
                dag[node][x]['tran'] = np.sum(dag[node][x]['kern_unnorm'])/s
            if dag[node][x]['tran'] < thre:
                dag.remove_edge(node, x)

    iso = nx.isolates(dag)
    dag.remove_nodes_from(iso)
    return dag


# Compute transition probabilities for test data
# Input:
# dag: DAG for test data
# G: DAG for training data
# Output:
# dag: DAG for test data with transition probabilities
def tranDAGTest(dag, G):
    tran = nx.get_edge_attributes(G, 'tran')
    nx.set_edge_attributes(dag, 'tran', tran)

    return dag


# Weight the features using log(1/p) or (log(1/p))^2
# Normalise the weighted features
# Input:
# G: DAG for training data
# power: power = 2: (log(1/p))^2; power = 1: log(1/p)
# Output:
# G: normalised DAG for training data
def normDAGl2(G, power):
    kern = nx.get_edge_attributes(G, 'kern_unnorm')
    tran = nx.get_edge_attributes(G, 'tran')

    kern = OrderedDict(sorted(kern.items(), key=lambda t: t[0]))
    val = kern.values()
    key = kern.keys()

    tran = OrderedDict(sorted(tran.items(), key=lambda t: t[0]))
    tran = tran.values()

    val = np.asarray(val, dtype=float)
    tran = np.asarray(tran, dtype=float)
    tran = np.log(1/tran)  # logarithm weighting: log(1/p)
    tran[tran == np.inf] = 0
    tran[np.isnan(tran)] = 0

    if power == 2:  # logarithm weighting: (log(1/p))^2
        tran = np.square(tran)

    # Normalisation
    if len(val.shape) == 2:
        kern = val*tran[:, None]  # avoid numeric problems when using logarithm weighting
        kern = normalize(kern, norm='l2', axis=0)
    else:
        kern = val*tran
        kern = kern/np.linalg.norm(kern)  # avoid numeric problems when using logarithm weighting

    kern = dict(zip(key, kern))
    nx.set_edge_attributes(G, 'kern', kern)

    # Delete edges with zero kernels - better keep zero kernels as practically including zero kernels performs better
    # for edge in G.edges():
    #     if float(np.sum(G[edge[0]][edge[1]]['kern'])) == 0.:
    #         G.remove_edge(edge[0], edge[1])

    # Remove isolated nodes
    iso = nx.isolates(G)
    G.remove_nodes_from(iso)

    return G


# Weight the features and normalise the DAG for test data
# Input:
# G_test: DAG for test data
# power: power = 2: (log(1/p))^2; power = 1: log(1/p)
# Output:
# G_test: normalised DAG for test data
def normDAGl2Test(G_test, power):
    kern = nx.get_edge_attributes(G_test, 'kern_unnorm')
    tran = nx.get_edge_attributes(G_test, 'tran')

    kern = OrderedDict(sorted(kern.items(), key=lambda t: t[0]))
    val = kern.values()
    key = kern.keys()

    tran = OrderedDict(sorted(tran.items(), key=lambda t: t[0]))
    tran = tran.values()

    val = np.asarray(val, dtype=float)
    tran = np.asarray(tran, dtype=float)
    tran = np.log(1/tran)  # logarithm weighting
    tran[tran == np.inf] = 0
    tran[np.isnan(tran)] = 0

    if power == 2:
      tran = np.square(tran)

    if len(val.shape) == 2:
        # kern = val/tran[:, None]
        kern = val*tran[:, None]  # avoid numeric problems when using logarithm weighting
        kern = normalize(kern, norm='l2', axis=0)
    else:
        kern = val*tran
        kern = kern/np.linalg.norm(kern)

    kern = dict(zip(key, kern))
    nx.set_edge_attributes(G_test, 'kern', kern)

    return G_test


# String kernel: return a base feature matrix of the p-spectrum kernel
# Input:
# data: input data (both training and test data)
# alphabet: alphabet of amino acids
# p: spectrum
# Output:
# kern: a m*20^p matrix of base features, m being the sample size
def strKern(data, alphabet, p):
    word_list = create_word_list(alphabet, p)

    kern = []
    count = 1
    for ele in data:
        print('Current data: %d' % count)
        dic = dict.fromkeys(word_list, 0)
        dic = OrderedDict(sorted(dic.items(), key=lambda t: t[0]))
        for j in range(0, len(ele)-p+1):
            if ele[j:j+p] in dic:
                dic[ele[j:j+p]] += 1
        kern.append(dic.values())
        count += 1

    return kern


# String kernel: return Python dictionary of the base feature space of the p-spectrum kernel
# String kernel: return a base feature matrix of the p-spectrum kernel
# Input:
# data: input data (both training and test data)
# alphabet: alphabet of amino acids
# p: spectrum
# Output:
# kern: a Python dictionary of base features
def strKernDic(data, alphabet, p):
    word_list = create_word_list(alphabet, p)

    count = 1
    kern_dic = dict.fromkeys(word_list)
    kern_dic = OrderedDict(sorted(kern_dic.items(), key=lambda t: t[0]))

    for k in kern_dic.keys():
        kern_dic[k] = np.zeros(len(data))

    for i in range(0, len(data)):
        print('Current data: %d' % count)
        ele = data[i]

        for j in range(0, len(ele)-p+1):
            if ele[j:j+p] in kern_dic:
                kern_dic[ele[j:j+p]][i] += 1

        count += 1

    return kern_dic


# Normalise the string kernel
# Input:
# kern_dic: Python dictionary of the spectrum kernel
# Output:
# kern_dic: normalised Python dictionary of the spectrum kernel
def normStr(kern_dic):
    kern_dic = OrderedDict(sorted(kern_dic.items(), key=lambda t: t[0]))
    val = kern_dic.values()
    key = kern_dic.keys()

    val = np.asarray(val, dtype=float)
    val = normalize(val, norm='l2', axis=0)

    kern_dic = dict(zip(key, val))

    return kern_dic


# Process the data on an existing DAG
# Input:
# G_train: DAG to be copied
# data_total: new data
# pmin: minimum state length
# pmax: maximum state length
# Output:
# G: DAG for new data
def CopyDAG(G_train, data_total, pmin, pmax):
    edges = nx.edges(G_train)
    kern = dict.fromkeys(edges)
    M = len(data_total)
    kern_temp = {}

    for edge in edges:
        kern[edge] = np.zeros(M)

    for m in range(M):
        print('Data item: %d' % m)
        data = data_total[m]
        for i in range(0, len(data)-pmin+1+1):
            for j in range(pmin-1, pmax+1):
                if data[i:i+j] in kern_temp:
                    kern_temp[data[i:i+j]][m] += 1
                else:
                    kern_temp[data[i:i+j]] = np.zeros(M)
                    kern_temp[data[i:i+j]][m] = 1

    for edge in edges:
        key = edge[0]+edge[1][-1]
        if key in kern_temp:
            kern[edge] = kern_temp[key]

    G = nx.DiGraph()
    G.add_edges_from(edges)
    nx.set_edge_attributes(G, 'kern_unnorm', kern)

    return G