#! /usr/bin/python

from __future__ import print_function

import theano
import theano.tensor as T

from utils import floatX


"""
General Optimizer Structure: (adadelta, adam, rmsprop, sgd)
Parameters
----------
    learning_rate: theano shared variable
        learning rate, currently only necessary for sgd
    parameters: OrderedDict()
        dict of shared variables {name: variable}
    grads:
        list of gradients
    inputs :
        inputs required to compute gradients
    cost :
        objective of optimization

Returns
-------
    f_grad_shared : compute cost, update optimizer shared variables
    f_update : update parameters
"""


# See "ADADELTA: An adaptive learning rate method", Matt Zeiler (2012) arXiv
# preprint http://arxiv.org/abs/1212.5701
def adadelta(learning_rate, parameters, grads, inputs, cost):
    zipped_grads = [theano.shared(p.get_value() * floatX(0.), name='%s_grad' % k)
                    for k, p in parameters.iteritems()]
    running_up2 = [theano.shared(p.get_value() * floatX(0.), name='%s_rup2' % k)
                   for k, p in parameters.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * floatX(0.), name='%s_rgrad2' % k)
                      for k, p in parameters.iteritems()]

    zg_up = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2_up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=zg_up + rg2_up, profile=False)

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in
             zip(zipped_grads, running_up2, running_grads2)]
    ru2_up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters.itervalues(), updir)]

    f_update = theano.function([learning_rate], [], updates=ru2_up + param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update


#    See Lecture 6.5, Coursera: Neural Networks for Machine Learning (2012),
#    Tieleman, T. and Hinton. G. for original methods
#
#    This implementation (with Nesterov Momentum) is described well in:
#    "Generating Sequences with Recurrent Neural Networks", Alex Graves, arxiv preprint
#    http://arxiv.org/abs/1308.0850
def rmsprop(learning_rate, parameters, grads, inputs, cost):
    zipped_grads = [theano.shared(p.get_value() * floatX(0.), name='%s_grad' % k)
                    for k, p in parameters.iteritems()]
    running_grads = [theano.shared(p.get_value() * floatX(0.), name='%s_rgrad' % k)
                     for k, p in parameters.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * floatX(0.), name='%s_rgrad2' % k)
                      for k, p in parameters.iteritems()]

    zg_up = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg_up = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2_up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=zg_up + rg_up + rg2_up, profile=False)

    updir = [theano.shared(p.get_value() * floatX(0.), name='%s_updir' % k) for k, p in parameters.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in
                 zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(parameters.itervalues(), updir_new)]
    f_update = theano.function([learning_rate], [], updates=updir_new + param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update


# See "Adam: A Method for Stochastic Optimization" Kingma et al. (ICLR 2015)
# Theano implementation adapted from Soren Kaae Sonderby (https://github.com/skaae)
# preprint: http://arxiv.org/abs/1412.6980
def adam(learning_rate, parameters, grads, inputs, cost):
    g_shared = [theano.shared(p.get_value() * floatX(0.), name='%s_grad' % k) for k, p in parameters.iteritems()]
    gs_up = [(gs, g) for gs, g in zip(g_shared, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=gs_up)
    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8
    updates = []
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - b1 ** i_t
    fix2 = 1. - b2 ** i_t
    lr_t = lr0 * (T.sqrt(fix2) / fix1)

    for p, g in zip(parameters.values(), g_shared):
        m = theano.shared(p.get_value() * floatX(0.))
        v = theano.shared(p.get_value() * floatX(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([learning_rate], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


# Vanilla SGD
def sgd(learning_rate, parameters, grads, inputs, cost):
    g_shared = [theano.shared(p.get_value() * floatX(0.), name='%s_grad' % k) for k, p in parameters.iteritems()]
    gs_up = [(gs, g) for gs, g in zip(g_shared, grads)]

    f_grad_shared = theano.function(inputs, cost, updates=gs_up, profile=False)

    p_up = [(p, p - learning_rate * g) for p, g in zip(parameters.itervalues(), g_shared)]
    f_update = theano.function([learning_rate], [], updates=p_up, profile=False)

    return f_grad_shared, f_update


def get_optimizer(name, learning_rate, parameters, grads, inputs, cost):
    return eval(name)(learning_rate, parameters, grads, inputs, cost)


__all__ = [
    'sgd',
    'adam',
    'adadelta',
    'rmsprop',
    'get_optimizer',
]
