__author__ = 'yuxinsun'

import networkx as nx
import numpy as np
from RootKernel import create_word_list
import operator


# Dynamic programming for FSM with DAG
# Input:
# t: u*y, where u is the misclassification cost, y is the desired labels
# dag: DAG
# l: length of transitions
# Output:
# opt: stop criterion in LPBoost
# kern: a column of features for the selected substring
# sub: selected substring
# att: 'Positive' or 'Negative'
def DPNormPN(t, dag, l):

    M = len(t)
    t = np.asarray(t)
    nml = np.sqrt(l)

    for node in dag.nodes():
        dag.node[node]['opt_pos'] = np.zeros(l+1, float)  # l+1 vector of max values can be achieved at node "node"/ pos
        dag.node[node]['pos_pos'] = ['N/A']*(l+1)  # l+1 vector of pointers to previous node, which can give the max/ pos
        dag.node[node]['opt_neg'] = np.zeros(l+1, float)  # l+1 vector of max values can be achieved at node "node"/ neg
        dag.node[node]['pos_neg'] = ['N/A']*(l+1)  # l+1 vector of pointers to previous node, which can give the max/ neg
        dag.node[node]['att'] = []  # a scalar that indicates whether positive or negative kernels can give the max
        dag.node[node]['opt'] = 0  # a scalar that stores the max between positive and negative values


    counter = 1
    while counter <= l:
        # Loop over all nodes
        for nd in dag.nodes():
            # Loop over all edges that points to node nd
            # for ed in dag.in_edges(nd):  # inefficient, use DiGraph.predecessors() instead
            for nd_pre in dag.predecessors(nd):
                # nd_pre = ed[0]  # previous node under current iteration
                temp = np.dot(dag[nd_pre][nd]['kern'], t)

                # Positive kernels
                key_pos = temp/nml + dag.node[nd_pre]['opt_pos'][counter-1]

                if key_pos > dag.node[nd]['opt_pos'][counter]:
                    dag.node[nd]['opt_pos'][counter] = key_pos
                    dag.node[nd]['pos_pos'][counter] = nd_pre

                # Negative kernels
                key_neg = -temp/nml + dag.node[nd_pre]['opt_neg'][counter-1]

                if key_neg > dag.node[nd]['opt_neg'][counter]:
                    dag.node[nd]['opt_neg'][counter] = key_neg
                    dag.node[nd]['pos_neg'][counter] = nd_pre

            if counter == l:
                if dag.node[nd]['opt_pos'][counter] >= dag.node[nd]['opt_neg'][counter]:
                    dag.node[nd]['att'] = 'Positive'
                    dag.node[nd]['opt'] = dag.node[nd]['opt_pos'][counter]
                else:
                    dag.node[nd]['att'] = 'Negative'
                    dag.node[nd]['opt'] = dag.node[nd]['opt_neg'][counter]

        counter += 1

    # Track substring
    # Positive
    opt = nx.get_node_attributes(dag, 'opt')  # get 'opt' attributes from all nodes
    opt_key = opt.keys()
    opt_val = opt.values()
    opt_val = np.asarray(opt_val)
    ind = np.argmax(opt_val)  # get index for the node that gives the maximum at l transitions
    nd = opt_key[ind]  # get the state at the end of best path
    opt = np.max(opt_val)


    # Track best path and compute corresponding
    sub = []
    sub.append(nd)
    kern = np.zeros([M, 1])
    att = dag.node[nd]['att']
    for i in range(l, 0, -1):
        if att == 'Positive':
            nd_pre = dag.node[nd]['pos_pos'][i]
            if nd_pre == 'N/A':
                break
            kern += dag[nd_pre][nd]['kern']
            sub.append(nd_pre)
            nd = nd_pre
        else:
            nd_pre = dag.node[nd]['pos_neg'][i]
            if nd_pre == 'N/A':
                break
            kern -= dag[nd_pre][nd]['kern']
            sub.append(nd_pre)
            nd = nd_pre
    kern /= nml

    return opt, kern, sub, att


# Feature selection
# Input:
# dag: DAG where features are to be selected
# sub: a list of nodes/ states
# att: 'Positive' or 'Negative'
# Output:
# kern: a column of features for the selected substring

def featNorm(dag, subs, att):

    kern = []

    for j in range(0, len(subs)):
        sub = subs[j]
        sub.reverse()
        key = 0
        for i in range(0, len(sub)-1):
            key += np.asarray(dag[sub[i]][sub[i+1]]['kern'])

        if att[j] == 'Negative':
            key = -key

        key /= np.sqrt(len(sub)-1)
        kern.append(key)

    kern = np.asarray(kern)
    kern = np.transpose(kern)


    return kern


# Sometimes the returned substrings do not need to be reversed
def featNormNew(dag, subs, att):

    kern = []

    for j in range(0, len(subs)):
        sub = subs[j]
        key = 0
        for i in range(0, len(sub)-1):
            key += np.asarray(dag[sub[i]][sub[i+1]]['kern'])

        if att[j] == 'Negative':
            key = -key
        key /= np.sqrt(len(sub)-1)
        kern.append(key)

    kern = np.asarray(kern)
    kern = np.transpose(kern)

    return kern


##########################################################
#                                                        #
#         Dynamic Programming for String Kernels         #
#                                                        #
##########################################################

# Dynamic programming for spectrum kernels
# Input:
# x: input data
# t: u*y, where u is the misclassification cost, y is the desired labels
# L: length of substrings (not transitions/paths)
# alphabet: amino acid alphabet
# Output:
# val: stop criterion for LPBoost
# kern: a column of features for selected substrings
# sub: selected substring
# att: 'Positive' or 'Negative'
def DPString(x, t, L, alphabet):

    word = create_word_list(alphabet,3)
    M = len(t)
    N = len(alphabet)
    dic = dict(zip(word, x.transpose()))

    DP_pos = np.zeros([N, N])
    DP_neg = np.zeros([N, N])
    SUB_pos = []
    SUB_neg = []

    nml = np.sqrt(L-3+1)

    for l in range(0, L-3+1):
        DPre_pos = np.copy(DP_pos)
        DPre_neg = np.copy(DP_neg)

        SUB_pos.append({})
        SUB_neg.append({})

        for l2 in range(0, len(alphabet)):
            ltr2 = alphabet[l2]

            for l1 in range(0, len(alphabet)):
                ltr1 = alphabet[l1]
                temp_pos = {}
                temp_neg = {}

                for l3 in range(0, len(alphabet)):
                    ltr3 = alphabet[l3]
                    temp_pos[ltr3] = DPre_pos[l3, l2] + np.dot(np.transpose(t), dic[ltr1+ltr2+ltr3])/nml
                    temp_neg[ltr3] = DPre_neg[l3, l2] - np.dot(np.transpose(t), dic[ltr1+ltr2+ltr3])/nml


                temp = dict(map(lambda item: (item[1],item[0]),temp_pos.items()))
                pos_key = temp[max(temp.keys())]
                pos_val = temp_pos[pos_key]

                temp = dict(map(lambda item: (item[1],item[0]),temp_neg.items()))
                neg_key = temp[max(temp.keys())]
                neg_val = temp_neg[neg_key]

                DP_pos[l2, l1] = pos_val
                DP_neg[l2, l1] = neg_val

                SUB_pos[l][ltr2+ltr1] = pos_key
                SUB_neg[l][ltr2+ltr1] = neg_key

    # Track subscripts
    pos_val = np.max(DP_pos)
    neg_val = np.max(DP_neg)

    if pos_val >= neg_val:
        val = pos_val
        att = 'Positive'
        temp = np.unravel_index(DP_pos.argmax(), DP_pos.shape)
        l2 = temp[0]
        l1 = temp[1]

        ltr2 = alphabet[l2]
        ltr1 = alphabet[l1]
        sub = ltr1+ltr2

        kern = np.zeros([M, 1])
        for i in range(L-3,-1,-1):
            temp = SUB_pos[i][ltr2+ltr1]
            sub += temp
            ltr2 = sub[-1]
            ltr1 = sub[-2]


            kern += np.transpose(np.matrix(dic[sub[-3:]]))

        kern = kern/nml
    else:
        val = neg_val
        att = 'Negative'
        temp = np.unravel_index(DP_neg.argmax(), DP_neg.shape)
        l2 = temp[0]
        l1 = temp[1]

        ltr2 = alphabet[l2]
        ltr1 = alphabet[l1]
        sub = ltr1+ltr2

        kern = np.zeros([M, 1])
        for i in range(L-3,-1,-1):
            temp = SUB_neg[i][ltr2+ltr1]
            sub += temp
            ltr2 = sub[-1]
            ltr1 = sub[-2]

            kern -= np.transpose(np.matrix(dic[sub[-3:]]))

        kern = kern/nml


    return val, kern, sub, att


# Return features according to selected substrings
# Input:
# x: input data
# subs: selected substrings
# atts: 'Positive' or 'Negative'
# word: word list of all possible combinations of p amino acid, p being the spectra
# Output:
# kern: columns of features for selected substrings

def featString(x, subs, atts, word):
    dic = dict(zip(word, x.transpose()))
    kern = []

    for k in range(0, len(subs)):
        sub = subs[k]
        att = atts[k]
        temp = 0
        for i in range(0, len(sub)-3+1):
            temp += dic[sub[i:i+3]]
        temp = temp/np.sqrt(len(sub)-3+1)

        if att == 'Negative':
            temp = -temp

        kern.append(temp)

    return kern