import knobs
import tfac
import tensorflow as tf
from tensorflow import nn as nn

#Build the net in the session
def build_net(sess,n_in,n_out):
    x_hold = tf.placeholder(tf.float32,shape=[None,n_in])
    q_hold = tf.placeholder(tf.float32,shape=[None,knobs.FC_OUT])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold,[-1,n_in,1,1])

    #Fully connected layer - 100 units
    in_v = tf.reshape(x_hold,[-1,n_in])
    hid = tfac.linear(in_v,n_in,units=knobs.FC1,activation='relu')
    q = tfac.linear(hid,knobs.FC1,units=knobs.FC_OUT,activation='relu')
    sess.run(tf.initialize_all_variables())

    return q,x_hold,q_hold
