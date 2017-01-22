"""
Accessory methods for using TensorFlow -- Mostly taken out from the TensorFlow tutorials!

Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)
"""

import tensorflow as tf
from tensorflow import nn as nn
import numpy as np
import random

#Make weight and bias variables -- From the TensorFlow tutorial
def weight(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)

def bias(shape):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(intial)

def linear(v_in,n_in,units=100,activation='relu'):
    w = weight([n_in,units])
    b = bias([units])
    out = tf.matmul(v_in,w)+b
    if activation == 'relu':
        out = nn.relu(out)
    elif activation == 'sigmoid':
        out = nn.sigmoid(out)
    elif activation == 'softmax':
        out = nn.softmax(out)
    elif activation == 'softplus':
        out = nn.softplus(out)
    return out

#Finds the product of a dimension tuple to find the total length
def dim_prod(dim_arr):
    return np.prod([d for d in dim_arr if d != None])

#Start a TensorFlow session
def start_sess():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    return sess

#Train the model
def train(sess, y, x_hold, y_hold, X, Y, valX=None, valY=None, lrate=1e-4, epsilon=1e-8, n_epoch=5, batch_size=50, print_epoch=1):
    init_var_list = set(tf.all_variables())
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_hold*tf.log(y+1e-10), reduction_indices=[1]))
    mean_squared =  tf.reduce_mean(0.5*tf.square(tf.sub(y_hold,y)))
    #train_step = tf.train.AdamOptimizer(learning_rate=lrate,epsilon=epsilon).minimize(mean_squared)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(mean_squared)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_hold,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #Flatten the input images for the placeholder
    flat_len = dim_prod(x_hold._shape_as_list())
    X = X.reshape((X.shape[0],flat_len))

    print 'Starting training session...'

    sess.run(tf.initialize_variables(set(tf.all_variables())-init_var_list))

    batch_num = 0
    batches = batchify(X,Y,batch_size)
    print 'Number of batches:',len(batches)
    train_accuracy = None
    val_accuracy = None
    for i in range(n_epoch):
        avg_acc = 0
        random.shuffle(batches)
        for batchX,batchY in batches:
            avg_acc = avg_acc + accuracy.eval(session=sess, feed_dict={x_hold:batchX, y_hold:batchY})
            train_step.run(session=sess,feed_dict={x_hold:batchX, y_hold:batchY})
        if i%print_epoch == 0:
            train_accuracy = avg_acc/len(batches)
            print 'Epoch '+str(i)+': '+str(train_accuracy)
    if (not valX is None) & (not valY is None):
        #Validation
        valX = valX.reshape((valX.shape[0],flat_len))
        val_accuracy = accuracy.eval(session=sess,feed_dict={x_hold:valX, y_hold:valY})
        print 'Val acc:',val_accuracy

    if val_accuracy != None:
        return val_accuracy
    return train_accuracy

def save(sess,path):
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess,path)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(path+'_graph',sess.graph)
    writer.flush()
    writer.close()
    print 'Model saved'

#Test a model
def test(sess, X, Y, model_path):
    correct_prediction = tf.equal(tf.argmax(self.net,1), tf.argmax(self.y_hold,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,model_path)
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    test_accuracy = accuracy.eval(session=sess,feed_dict={x_hold:X,y_hold:Y})
    return test_accuracy

#Split to mini batches
def batchify(X, Y, batch_size):
    batches = [(X[i:i+batch_size],Y[i:i+batch_size]) for i in xrange(0,X.shape[0],batch_size)]
    random.shuffle(batches)
    return batches
