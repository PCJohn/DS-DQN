import numpy as np
import random
from random import random as rnd

import qlearn
import knobs

IN_SYM = ['a','b']

def samples(sample_count):
    smp = []
    N = len(IN_SYM)
    sizes = map(int,knobs.MAX_LEN*np.random.random(sample_count))
    for i in xrange(sample_count):
        in_str = ''.join([random.choice(IN_SYM) for _ in xrange(sizes[i])])
        smp.append((in_str,in_str[::-1]))
    return smp

ql = qlearn.QLearner('stack','nn_q')
ql.train('rev')
print ql.play('abbab')
