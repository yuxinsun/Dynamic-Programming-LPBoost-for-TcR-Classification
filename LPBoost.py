__author__ = 'yuxinsun'

import numpy as np
import DynamicProgramming as dp
# Linear Programming Boosting
# Required Library:
# numpy
# cvxopt

# Convex optimisation with cvxopt
# Input:
# x: hypothesis space - m*n matrix
# y: desired labels - m*1 vector
# D: D in LPBoost, usually D = 1/(m*nu) - scalar
# Output:
# Optimisation results:
# u: primal classification cost - m*1 vector
# beta: primal variable - scalar
# c4.multiplier: dual weights - m*1 vector

def LPcvx(z, y, D):
    from cvxopt import *

    m = y.size

    # Initialise
    u = modeling.variable(m, 'u')
    u.value = matrix(np.ones(m)/m)
    beta = modeling.variable(1, 'beta')
    beta.value = 0

    # Constraints
    c1 = (modeling.sum(u) == 1)
    c2 = (u <= D)
    c3 = (u >= 0)
    c4 = (modeling.dot(matrix(z), u) <= beta)

    # Solve problems
    lp = modeling.op(beta, [c1, c2, c3, c4])
    solvers.options['show_progress']=False
    sol = lp.solve()

    return u.value, beta.value, c4.multiplier.value


# LPBoost over DAG
# Input:
# y: desired labels
# D: 1/(m*nu)
# dag: DAG
# l: length of path
# maxIter: maximum iteration
# Output:
# u: misclassification cost
# beta: beta in LP optimisation
# a: weight vector
# subs: selected transitions
# att: positive or negative
# F: selected subset of the feature space
def LPNorm(y, D, l, dag, maxIter):

    m = y.size

    # Initialise
    u = np.ones(m)/m
    t = np.multiply(u, y)
    val, kern, sub, att = dp.DPNormPN(t, dag, l)

    counter = 1
    crit = val
    beta = 0
    F = []
    subs = []
    atts = []

    # LPBoost optimisation
    while crit >= beta+10**(-6) and counter <= maxIter:  # for a higher convergence rate

        subs.append(sub)
        atts.append(att)

        F.append(kern*y)
        opt = np.asarray(F)
        opt = np.transpose(opt)

        [u, beta, a] = LPcvx(opt, y, D)
        u = np.squeeze(np.asarray(u))

        t = np.multiply(u, y)
        val, kern, sub, att = dp.DPNormPN(t, dag, l)  # dynamic programming over DAG
        crit = val

        beta = np.asarray(beta)
        beta = beta[0]
        print('Iteration: %d, beta: %f, criterion: %f' % (counter, beta, crit))

        counter += 1

    u = np.asarray(u)
    a = np.asarray(a)

    a[np.less(a, 10**-5)] = 0

    return u, beta, a, subs, atts, F


# LPBoost for string kernels
# Input:
# x: input data matrix
# y: desired labels
# D = 1/(m*nu)
# maxiter: maximum iterations
# Output:
# u: misclassification cost
# beta: beta in LP optimisation
# a: weight vector
# idx: indices of selected substrings
def LPBoostAg(x, y, D, maxIter):

    m = y.size
    u = np.ones(m)/m

    temp = np.multiply(u, y)
    h = np.dot(temp, x)


    ind = np.argmax(h)
    crit = np.max(h)
    beta = 0
    F = []
    idx = []

    counter = 1
    while crit >= beta+10**(-6) and counter <= maxIter:

        F.append(x[:, ind]*y)
        opt = np.asarray(F)
        opt = np.transpose(opt)

        idx.append(ind)

        [u, beta, a] = LPcvx(opt, y, D)
        u = np.squeeze(np.asarray(u))

        temp = np.multiply(u, y)
        h = np.dot(temp, x)

        ind = np.argmax(h)
        crit = np.max(h)

        beta = np.asarray(beta)
        beta = beta[0]

        print('Iteration: %d, beta: %f, criterion: %f' % (counter, beta, crit))

        counter += 1

    return u, beta, a, idx


# Dynamic programming for the spectrum kernels
# Input:
# x: input data matrix
# y: desired labels
# D = 1/(m*nu)
# l: length of substrings
# alphabet: amino acid alphabet
# maxiter: maximum iterations
# Output:
# u: misclassification cost
# beta: beta in LP optimisation
# a: weight vector
# subs: selected substrings
# attributes of selected substrings, positive or negative
def LPDynamic(x, y, D, l, alphabet, maxIter):
    m = y.size

    # Initialise
    u = np.ones(m)/m
    t = np.multiply(u, y)
    val, kern, sub, att = dp.DPString(x, t, l, alphabet)

    counter = 1
    crit = val
    beta = 0
    F = []
    subs = []
    atts = []

    while crit >= beta+10**(-6) and counter <= maxIter:  # for a higher convergence rate

        subs.append(sub)
        atts.append(att)

        # temp = np.multiply(kern,np.transpose(np.matrix(y)))
        kern = np.squeeze(np.asarray(kern))
        F.append(kern*y)
        opt = np.asarray(F)
        opt = np.transpose(opt)

        [u, beta, a] = LPcvx(opt, y, D)
        u = np.squeeze(np.asarray(u))

        t = np.multiply(u, y)
        val, kern, sub, att = dp.DPString(x, t, l, alphabet)

        crit = val

        beta = np.asarray(beta)
        beta = beta[0]
        print('Iteration: %d, beta: %f, criterion: %f' % (counter, beta, crit))

        counter += 1

    u = np.asarray(u)
    a = np.asarray(a)

    a[np.less(a, 10**-5)] = 0

    return u, beta, a, subs, atts, F


