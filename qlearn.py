import tfac
import numpy as np
import importlib
import random
from random import random as rnd
import tensorflow as tf

import knobs

STOP_EXEC = 'stop'
NO_IN = '_'

class QLearner:
    
    def __init__(self, bb_file, nn_file):
        self.bb_mod = importlib.import_module(bb_file)
        self.bbox = self.bb_mod.BlackBox()
        self.ops = self.bb_mod.OPS
        self.ops.append(STOP_EXEC)
        self.n_ops = len(self.ops)
        self.sym = self.bb_mod.SYM
        self.sym.append(NO_IN)
        self.n_sym = len(self.sym)
        self.op_mat = np.diag(np.ones(len(self.ops)))
        self.sym_mat = np.diag(np.ones(len(self.sym)))
        
        self.nn_mod = importlib.import_module(nn_file)
        self.sess = tfac.start_sess()
        self.state_len = len(self.bbox.state)*self.n_sym
        self.q,self.x_hold,self.y_hold = self.nn_mod.build_net(self.sess,self.state_len,self.n_ops)


    def enc_op(self,op):
        return self.op_mat[self.ops.index(op)]

    def enc_sym(self,sym):
        return self.sym_mat[self.sym.index(sym)]

    def enc_state(self,sym_list):
        return np.concatenate([self.enc_sym(sym) for sym in sym_list])

    def dec_op(self,op_v):
        return self.ops[max((self.op_mat*op_v).argmax(1))]

    def dec_sym(self,sym_v):
        return self.sym[max((self.sym_mat*sym_v).argmax(1))]

    def replay_mem(self,samples,max_steps):
        replay_mem = []
        out_str = ''
        for in_str,out_str in samples:
            for i in xrange(max_steps):
                if i < len(in_str):
                    in_c = in_str[i]
                else:
                    in_c = NO_IN
                self.bbox.read(in_c)
                s = np.array([self.enc_state(self.bbox.state)])
                q_t = self.q.eval(session=self.sess,feed_dict={self.x_hold:s})[0]
                if random.random() < knobs.EXPLORE:
                    a_ind = random.choice(range(q_t.shape[0]))
                else:
                    a_ind = np.argmax(q_t)
                a = self.ops[a_ind]
                r = 0
                #print a,'==',self.bbox.state,'==',
                out_c = self.bbox.act(a,in_c)
                if i+1 < len(out_str):
                    exp_c = out_str[i+1]
                else:
                    exp_c = NO_IN
                if out_c == exp_c:
                    r = knobs.REWARD
                q = np.copy(q_t)
                if (a == STOP_EXEC) | (i == max_steps):
                    q[a_ind] = r               
                    replay_mem.append((s,q))
                    break
                else:
                    s = np.array([self.enc_state(self.bbox.state)])
                    #print self.bbox.state
                    q_t1 = self.q.eval(session=self.sess,feed_dict={self.x_hold:s})[0]
                    q[a_ind] = r+knobs.GAMMA*np.max(q_t1)
                    #print q-q_t,'--',np.max(q_t1),'--',r,'--',out_c,'--',exp_c,'--',a_ind
                    #print q,'--',q_t,'--',r,'--',out_c,'--',exp_c
                    replay_mem.append((s,q))
        random.shuffle(replay_mem)
        X,Y = map(np.array,zip(*replay_mem))
        return (X,Y)

    def play(self,in_str):
        out_str = ''
        a = None
        i = 0
        for i in xrange(knobs.MAX_STEPS):
            if i < len(in_str):
                in_c = in_str[i]
            else:
                in_c = NO_IN
            self.bbox.read(in_c)
            s = np.array([self.enc_state(self.bbox.state)])
            a = self.ops[np.argmax(self.q.eval(session=self.sess,feed_dict={self.x_hold:s}))]
            #print s,'--',a
            if a == STOP_EXEC:
                break
            out_c = self.bbox.act(a,in_c)
            if out_c != None:
                out_str = out_str+out_c
        return out_str

    def batch_test_seq(self,seq_list):
        return np.mean([knobs.seq_sim(out_str,self.play(in_str)) for in_str,out_str in seq_list])

    def train(self, trainer_file):
        train_mod = importlib.import_module(trainer_file)
        max_steps = knobs.MAX_STEPS
        for _ in xrange(2):
            samples = train_mod.samples(knobs.SAMPLE_BATCH)
            random.shuffle(samples)
            N = len(samples)
            train_set = samples[:N/4]
            test_set = samples[3*N/4:]
            print train_set[:3]
            X,Y = self.replay_mem(samples,max_steps)
            tfac.train(self.sess,self.q,self.x_hold,self.y_hold,
                       X,Y,
                       lrate=knobs.LRATE,
                       epsilon=knobs.ADAM_EPS,
                       n_epoch=knobs.N_EPOCH,
                       batch_size=knobs.MINIBATCH,
                       print_epoch=knobs.PRINT_EPOCH)
            acc = self.batch_test_seq(test_set)
            print acc
        #tfac.save(self.sess,knobs.SAVE_PATH)
