import tfac
import tensorflow as tf
import numpy as np
import random

A = [1,0,0]
B = [0,1,0]
EMPTY = [0,0,1]

class Stack:
    
    def __init__(self):
        self.v = []
    
    def push(self,x):
        self.v.append(x)
    
    def pop(self):
        if len(self.v) == 0:
            return
        x = self.v[-1]
        self.v = self.v[:-1]
        return x
    
    def top(self):
        if len(self.v) == 0:
            return '$'
        return self.v[-1]

def encode(ch):
    if ch == 'a':
        return A
    elif ch == 'b':
        return B
    elif ch == '$':
        return EMPTY

def decode()

def read(sess, y, seq):
    s = Stack()
    for ch in seq:
        in_v = encode(ch)+encode(s.top())
        print in_v
        y.eval(session=sess,feed_dict={x_hold:,keeP_prob:})
        


#Build the net in the session
def build_net(sess):
    in_len = 6
    in_dep = 1

    x_hold = tf.placeholder(tf.float32,shape=[None,in_len])
    y_hold = tf.placeholder(tf.float32,shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold,[-1,in_len,in_len,in_dep])

    dim = 32 * 4*4
        
    #Fully connected layer - 600 units
    of = tf.reshape(o3,[-1,dim])
    w4 = tfac.weight([dim,100])
    b4 = tfac.bias([100])
    o4 = nn.relu(tf.matmul(of,w4)+b4)

    o4 = nn.dropout(o4, keep_prob)

    #Output softmax layer - 2 units
    w5 = tfac.weight([100,2])
    b5 = tfac.bias([2])
    y = nn.softmax(tf.matmul(o4,w5)+b5)

    sess.run(tf.initialize_all_variables())

    return y,x_hold,y_hold,keep_prob

#Method to run the training
def train_net():
    train,val,test = face_ds.load_find_ds()
    sess = tfac.start_sess()
    y,x_hold,y_hold,keep_prob = build_net(sess)
    out = y.eval(session=sess,feed_dict={x_hold: ,keep_prob: })
    '''acc = tfac.train(sess,
                    y,
                    x_hold,
                    y_hold,
                    keep_prob,
                    train[0],train[1],
                    test[0],test[1],
                    lrate=1e-4,
                    epsilon=1e-16,
                    n_epoch=8,
                    batch_size=100,
                    print_epoch=1,
                    save_path=model_path)
    print "Accuracy:",acc'''
    
    sess.close()

read_seq('abba')
