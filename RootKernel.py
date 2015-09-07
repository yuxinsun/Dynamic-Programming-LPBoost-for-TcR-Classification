__author__ = 'yuxinsun'
import networkx as nx
import numpy as np
import itertools
from itertools import chain
import random
from compiler.ast import flatten
from sklearn.preprocessing import normalize
from collections import OrderedDict


# Compute the String Kernel and Returns Root Kernel
# Input:
# file_list: list of file names
# p: spectrum
# alphabet: alphabet that documents are generated from

# Create a word list from alphabet
def create_word_list(alphabet, p):

    word_list = list(itertools.product(alphabet, repeat=p))
    for i in range(0,len(word_list)):
        word_list[i] = ''.join((itertools.chain(word_list[i])))

    return word_list


# Process data
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

        if l != 'all':
            random.shuffle(data_temp, random.random)
            data_temp = data_temp[:l]

        data.append(' '.join(chain(data_temp)))
        # data.append(data_temp)
    return data


# Process data with sequences and count
def process_data_count(file_list, path, l):

    data = []
    for file in file_list:
        print('Processing data: ', file)
        data_temp = []
        file_object = open(path+file+'.txt')
        file_name = file_object.readlines()
        file_name = [s.strip('\r\n') for s in file_name]

        del file_name[0]

        for fileline in file_name:
            s = fileline.split(',')
            s = (s[0]+' ')*int(s[1])
            s = s.split(' ')
            data_temp.append(s[:-1])

        data_temp = flatten(data_temp)
        if l != 'all':
            random.shuffle(data_temp, random.random)
            data_temp = data_temp[:l]

        data.append(' '.join(chain(data_temp)))
        # data.append(data_temp)

    return data


# Compute root kernel - inefficient, use root_kernel instead
def root_kernel_in(data, alphabet, p):
    from re import findall

    # Create word list
    word_list = create_word_list(alphabet, p)

    # Create empty kernel matrix
    kern = []
    kern_dic_whole = dict.fromkeys(word_list, 0)

    for data_item in data:
        # Create an empty dictionary
        kern_dic = dict.fromkeys(word_list, 0)

        for words in kern_dic.keys():
            kern_dic[words] = len(findall('(?=%s)' % words, data_item))
            kern_dic_whole[words] += kern_dic[words]

        kern.append(kern_dic.values())

    return kern, kern_dic_whole


# Compute root kernel
def root_kernel(data, alphabet, p):
    from collections import OrderedDict

    # Create word list
    word_list = create_word_list(alphabet, p)

    # Create kernel matrix
    kern = []
    kern_dic_whole = dict.fromkeys(word_list, 0)
    kern_dic_whole = OrderedDict(sorted(kern_dic_whole.items(), key=lambda t: t[0]))

    for data_item in data:
        # Create an empty dictionary
        kern_dic = dict.fromkeys(word_list, 0)
        kern_dic = OrderedDict(sorted(kern_dic.items(), key=lambda t: t[0]))

        for i in range(0, len(data_item)-p+1):
            if data_item[i:i+p] in kern_dic:
                kern_dic[data_item[i:i+p]] += 1
                kern_dic_whole[data_item[i:i+p]] += 1

        kern.append(list(kern_dic.values()))

    return kern, kern_dic_whole


# Compute transition probabilities from dictionary of root kernel
def trans_prob(kernel_dic):
    # Create an empty dictionary of total transitions
    # such that trans_dic[ab] = sum_x kernel_dic[abx]
    trans_dic = {}
    for words in kernel_dic.keys():
        if words[:-1] in trans_dic:
            trans_dic[words[:-1]] += kernel_dic[words]
        else:
            trans_dic[words[:-1]] = kernel_dic[words]

    # Compute transition probabilities
    for words in kernel_dic.keys():
        if trans_dic[words[:-1]] == 0:
            kernel_dic[words] = 0
        else:
            kernel_dic[words] = kernel_dic[words]/trans_dic[words[:-1]]

    # trans = kernel_dic.values()
    # trans = list(trans)

    return kernel_dic


# def kernelDAG(data, pmin, pmax, thre):
#
#     kern_dic = {}
#     base_dic = {}
#
#     for m in range(0, len(data)):
#         print('Data item: %d' % m)
#         for i in range(0, len(data[m])-pmin+1+1):
#             # Create a dictionary where each entry stores the frequency of a pmin-1 substring
#             base = data[m][i:i+pmin-1]
#             if base in base_dic:
#                 base_dic[base][m] += 1
#             elif ' ' not in base:
#                 base_dic[base] = np.zeros(len(data))
#                 base_dic[base][m] = 1
#
#             # Create a dictionary where each entry stores the frequency of transition of length pmin to pmax
#             # Key for the dictionary is a tuple of substring pairs
#             # e.g.: transition XYZ: dictionary[(XY, XYZ)] = tf(XYZ), with tf being the term frequency
#             for j in range(pmin, pmax+1):
#                 if i+j <= len(data[m]):
#                     sub = data[m][i:i+j]
#                     tup = (sub[:-1], sub)
#                     if tup in kern_dic:
#                         kern_dic[tup][m] += 1
#                     elif ' ' not in sub:
#                         kern_dic[tup] = np.zeros(len(data))
#                         kern_dic[tup] = kern_dic[tup].astype(float)
#                         kern_dic[tup][m] = 1
#
#     # Exclude entries of base_dic where mean value is less than threshold
#     base_dic = dict((k, v) for k, v in base_dic.items() if np.mean(v) >= thre)
#
#     # Include entries of kern_dic where mean value is greater than or equal to threshold
#     # If mean value is less than threshold, base_dic is checked
#     # e.g.: if kern_dic[(XY, XYZ)] < threshold, but both XY and YZ exist in base_dic
#     #       we delete kern_dic[(XY, XYZ)] but add kern_dic[(XY, YZ)] = tf(XYZ)
#     for k, v in kern_dic.items():
#         if np.mean(v) < thre:
#
#             if len(k[1]) == pmin and k[0] in base_dic and k[1][1:] in base_dic:
#                 kern_dic[(k[0], k[1][1:])] = v
#                 del kern_dic[k]
#             else:
#                 del kern_dic[k]
#
#     # Build DAG from dictionary kern_dic
#     # Each key represents an edge on the DAG while each value represents an edge attribute 'kern'
#     G = nx.DiGraph()
#     G.add_edges_from(kern_dic.keys())
#     nx.set_edge_attributes(G, 'kern', kern_dic)
#
#     kern_dic.clear()
#
#     return G


def KernelDAG(data, pmin, pmax, thre, alphabet, kernel_type):
    word_list = set(create_word_list(alphabet, pmin))
    M = len(data)
    edge_dic = {}
    kern_dic = {}

    for m in range(0, M):
        print('Data item: %d' % m)
        for i in range(0, len(data[m])-pmin+1+1):
           for j in range(pmin-1, pmax+1):  # computes term frequency of substrings of length pmin-1 to pmax
               if i+j <= len(data[m]):
                sub = data[m][i:i+j]
                if sub in kern_dic:
                    kern_dic[sub][m] += 1
                elif ' ' not in sub:
                    kern_dic[sub] = np.zeros(M)
                    kern_dic[sub] = kern_dic[sub].astype(float)
                    kern_dic[sub][m] = 1

    base_dic = dict((k, v) for k, v in kern_dic.items() if len(k) <= pmax)  # a dictionary of all possible states
    base_dic = dict((k, v) for k, v in base_dic.items() if np.sum(v) >= thre)
    kern_dic = dict((k, v) for k, v in kern_dic.items() if len(k) >= pmin)  # a dictionary of all non zero substrings

    word_list = word_list - word_list.intersection(set(base_dic.keys()))
    for word in word_list:
        edge_dic[(word[:-1], word[1:])] = np.zeros(M)  # a dictionary of all length pmin-1 transitions that are not in base_dic

    # edge_dic = {}
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

    G = nx.DiGraph()
    G.add_edges_from(edge_dic.keys())
    nx.set_edge_attributes(G, 'kern_unnorm', edge_dic)

    alphabet_set = set(alphabet)
    # add edges that represents non-existing transitions (states of length pmin-1 only
    for node in G.nodes():
        if 'X' in node:
            G.remove_node(node)
            continue

        suc = G.successors(node)
        suc = set([temp[-1] for temp in suc])
        inter = alphabet_set-alphabet_set.intersection(suc)

        for letter in inter:
            G.add_edge(node, node[-1]+letter, kern_unnorm=np.zeros(M))

    return G




# def kernelDAGTest(data, G, pmin, pmax, thre):
#
#     edge_key = G.edges()
#
#     G_Test = kernelDAG(data, pmin, pmax, thre)
#
#     dic = nx.get_edge_attributes(G_Test, 'kern')
#     for e in edge_key:
#         if e not in dic:
#             dic[e] = 0
#
#     for e in dic.keys():
#         if e not in edge_key:
#             del dic[e]
#
#     # Build DAG from dictionary kern_dic
#     # Each key represents an edge on the DAG while each value represents an edge attribute 'kern'
#     G = nx.DiGraph()
#     G.add_edges_from(dic.keys())
#     nx.set_edge_attributes(G, 'kern', dic)
#
#
#     dic.clear()
#
#     return G


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


# Compute transition probabilities with DAG
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


# Compute transition probabilities with DAG
def tranDAGTest(dag, G):
    tran = nx.get_edge_attributes(G, 'tran')
    nx.set_edge_attributes(dag, 'tran', tran)

    # edge = G.edges()
    #
    # for e in edge:
    #     node = e[1]
    #     node_pre = e[0]
    #     dag[node_pre][node]['tran'] = G[node_pre][node]['tran']

    return dag


# Normalisation
def normDAG(dag):

    # s = (np.square(dag[e[0]][e[1]]['kern']/np.square(dag[e[0]][e[1]]['tran'])) for e in dag.edges())
    s = (np.square(dag[e[0]][e[1]]['kern']/dag[e[0]][e[1]]['tran']) for e in dag.edges())
    # We use K(d, s) = tf(.)/tran(.) here, assuming a uniform probability for substring for convenience in DP
    s = np.sum(s)
    s = np.asarray(s)
    s = np.sqrt(s)

    for edge in dag.edges():
        nd = edge[1]
        nd_pre = edge[0]

        # dag[nd_pre][nd]['kern'] /= (s*np.square(dag[nd_pre][nd]['tran']))
        dag[nd_pre][nd]['kern'] /= (s*dag[nd_pre][nd]['tran'])

    return dag

def normDAGTest(G_test, G_train):
    # s = (np.square(G_test[e[0]][e[1]]['kern']/np.square(G_train[e[0]][e[1]]['tran'])) for e in G_train.edges())
    s = (np.square(G_test[e[0]][e[1]]['kern']/G_train[e[0]][e[1]]['tran']) for e in G_train.edges())
    s = np.sum(s)
    s = np.asarray(s)
    s = np.sqrt(s)

    for edge in G_test.edges():
        nd = edge[1]
        nd_pre = edge[0]

        # G_test[nd_pre][nd]['kern'] /= (s*np.square(G_train[nd_pre][nd]['tran']))
        G_test[nd_pre][nd]['kern'] /= (s*G_train[nd_pre][nd]['tran'])
        # G_test[nd_pre][nd]['tran'] = G_train[nd_pre][nd]['tran']

    return G_test


# l2 normalisation
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
    nx.set_edge_attributes(G, 'kern', kern)

    # delete edges with zero kernels
    # for edge in G.edges():
    #     if float(np.sum(G[edge[0]][edge[1]]['kern'])) == 0.:
    #         G.remove_edge(edge[0], edge[1])

    # remove isolated nodes
    iso = nx.isolates(G)
    G.remove_nodes_from(iso)

    return G


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


# String Kernel
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


def normStr(kern_dic):
    kern_dic = OrderedDict(sorted(kern_dic.items(), key=lambda t: t[0]))
    val = kern_dic.values()
    key = kern_dic.keys()

    val = np.asarray(val, dtype=float)
    val = normalize(val, norm='l2', axis=0)

    kern_dic = dict(zip(key, val))

    return kern_dic


def testtest(data, pmin, pmax, thre):

    kern_dic = {}
    base_dic = {}

    for m in range(0, len(data)):
        print('Data item: %d' % m)
        for i in range(0, len(data[m])-pmin+1+1):
            # Create a dictionary where each entry stores the frequency of a pmin-1 substring
            base = data[m][i:i+pmin-1]
            if base in base_dic:
                base_dic[base][m] += 1
            elif ' ' not in base:
                base_dic[base] = np.zeros(len(data))
                base_dic[base][m] = 1

            # Create a dictionary where each entry stores the frequency of transition of length pmin to pmax
            # Key for the dictionary is a tuple of substring pairs
            # e.g.: transition XYZ: dictionary[(XY, XYZ)] = tf(XYZ), with tf being the term frequency
            for j in range(pmin, pmax+1):
                if i+j <= len(data[m]):
                    sub = data[m][i:i+j]
                    tup = (sub[:-1], sub[1:])
                    if tup in kern_dic:
                        kern_dic[tup][m] += 1
                    elif ' ' not in sub:
                        kern_dic[tup] = np.zeros(len(data))
                        kern_dic[tup] = kern_dic[tup].astype(float)
                        kern_dic[tup][m] = 1


    # Build DAG from dictionary kern_dic
    # Each key represents an edge on the DAG while each value represents an edge attribute 'kern'
    G = nx.DiGraph()
    G.add_edges_from(kern_dic.keys())
    nx.set_edge_attributes(G, 'kern', kern_dic)

    kern_dic.clear()

    return G


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