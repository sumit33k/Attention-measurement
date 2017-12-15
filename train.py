# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:28:03 2017

@author: sumit
"""

import numpy as np
import tensorflow as tf


Dim = 300*2
numFilter = 20
batch_size = 1

def filter(sent):
    '''
    if a character is ',' or '.', etc., it should be separated 
    '''
    s = ''
    for i in sent:        
        if not (i.isdigit() or i.isalpha()):
            i = ' '+i+' '
        s = s + i
    return s

def weight_variable(shape):
    w = tf.truncated_normal(shape = shape, stddev = 0.1)
    return tf.Variable(w)

def bias_variable(shape):
    b = tf.constant(0.1, shape = shape)
    return b

def holistic_Filter(x, ws, W, b, pooling='max'):
    '''
    
    '''
    y = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding="SAME")
    #y = tf.reshape(tf.reduce_mean(y, axis=2), [batch_size, -1, 1, numFilter])
    o = tf.nn.relu(y + b) #shape = [batch_size, len+1-ws, height=1, numFilter]
    # pooling ---groupA--- 
    if pooling == 'max':
        o_pooling = tf.reduce_max(o, axis = 1) # shape = [batch_size, height=1, numFilter], the rank is reduced by 1. 
    elif pooling =='min':
        o_pooling = tf.reduce_min(o, axis = 1)
    else:
        o_pooling = tf.reduce_mean(o, axis = 1)
    return o_pooling

def per_dimension_Filter(x, ws, W, b, pooling='max'):
    '''
    
    '''  
    x_shape = x.get_shape()
    n = x_shape[2]
    x_unpack = tf.unpack(x, axis = 2)
    W_unpack = tf.unpack(W, axis = 1)
    b_unpack = tf.unpack(b, axis = 1)
    y = []
    for i in range(n):
        xi = x_unpack[i]
        wi = W_unpack[i]
        bi = b_unpack[i]
        conv_y = tf.nn.relu(tf.nn.conv1d(xi, wi, stride=1, padding="SAME") + bi) 
        y.append(conv_y)
    
    o = tf.pack(y, axis = 2) # [batch=1, len+1-ws, height=Dim , numFilter]
    # pooling --groupB---
    if pooling == 'max':
        o_pooling = tf.reduce_max(o, axis = 1) #[batch=1, height=Dim, numFilter]
    else:
        o_pooling = tf.reduce_max(o, axis = 1)
    return o_pooling




def groupA_block(x, w, b, wss=[1, 2, 100], poolings = ['max', 'min', 'mean']):
    y = []
    for j in range(len(poolings)):    
        pooling = poolings[j]
        tmp_y = []
        for i in range(len(wss)):
            ws = wss[i]
            o = holistic_Filter(x, ws, w[i], b[i], pooling)
            tmp_y.append(o)
            
        y.append(tmp_y)
    return y
    
def groupB_block(x, w, b, wss=[1, 2], poolings = ['max','min']):
    y = []
    for j in range(len(poolings)):
        pooling = poolings[j]
        tmp_y = []
        for i in range(len(wss)):
            ws = wss[i]
            o = per_dimension_Filter(x, ws, w[i], b[i], pooling)
            tmp_y.append(o)
        y.append(tmp_y)
    return y

def cosine_distance(a, b):
	inner_product = tf.reduce_sum(tf.multiply(a, b), axis=1)
	cos_angle = tf.divide(inner_product, tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(a), axis=1)), tf.sqrt(tf.reduce_sum(tf.square(b), axis=1))))
	return cos_angle
                  
def comU(a, b, tag = 2):

    fea = []
    fea.append(cosine_distance(a, b))
    #fea.append(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(a,b)), axis=1)))
    fea.append(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(a,b)), axis=1)))
    if tag == 2:
        fea.append(tf.reduce_max(tf.abs(tf.sub(a, b)), axis=1))
    #print 'fea=', fea
    return tf.pack(fea, axis=1)


def similarity_measurement_layer(BlockA1, BlockA2, BlockB1, BlockB2, ws1=3, ws2=2):
    '''
    similarity measurement
    '''
    #----- algorithm 1
   
    featureH = []
    for i in range(3):
        regM1 = tf.concat(1, BlockA1[i])
        regM2 = tf.concat(1, BlockA2[i])
        for k in range(numFilter):
            featureH.append(comU(regM1[:,:,k], regM2[:,:,k], 1))
            

    #----- algorithm 2
    featureA = []
    featureB = []
    for i in range(3):
        for j in range(ws1):
            for k in range(ws1):
                featureA.append(comU(BlockA1[i][j][:,0,:], BlockA2[i][k][:,0,:]))
             
                    
    for i in range(2):
        for j in range(ws2):
            for k in range(numFilter):
                featureB.append(comU(BlockB1[i][j][:,:,k], BlockB2[i][j][:,:,k]))
    return tf.concat(1, featureH+featureB)


def linear_layer(x, w, b):
    '''
    linear layer
    '''
    return tf.matmul(x, w) + b
    
def activation_layer(x):
    '''
    activation layer
    '''
    return tf.nn.tanh(x)

def log_softmax_layer(x):
    '''
    log-softmax layer
    '''
    return tf.nn.softmax(x)



## ---------------------------  build network ---------------------------------

hindn = 150
outputn = 6
lr = 0.1

xs1 = tf.placeholder(tf.float32, [None, Dim], name='input_sentenc')
xs2 = tf.placeholder(tf.float32, [None, Dim], name='input_sentenc')


ys = tf.placeholder(tf.float32, [outputn], name='input_score')
xs1_4d = tf.reshape(xs1, [batch_size, -1, Dim, 1])
xs2_4d = tf.reshape(xs2, [batch_size, -1, Dim, 1])
ys_2d = tf.reshape(ys, [batch_size, outputn])

# conv and pooling layer
wsA = [1,2,100]
wsB = [1,2]
w1_conv = [weight_variable([wsA[0], Dim, 1, numFilter]),\
           weight_variable([wsA[1], Dim, 1, numFilter]),\
           weight_variable([wsA[2], Dim, 1, numFilter])]

b1_conv = [bias_variable([numFilter]),\
           bias_variable([numFilter]),\
           bias_variable([numFilter])]

#print 'w1_conv', w1_conv[0].get_shape()

y1_A = groupA_block(xs1_4d, w1_conv, b1_conv, wsA)
y2_A  = groupA_block(xs2_4d, w1_conv, b1_conv, wsA)


w2_conv = [weight_variable([wsB[0], Dim, 1, numFilter]),
           weight_variable([wsB[1], Dim, 1, numFilter])]
b2_conv = [bias_variable([numFilter, Dim]),
           bias_variable([numFilter, Dim])]
y1_B = groupB_block(xs1_4d, w2_conv, b2_conv, wsB)
y2_B = groupB_block(xs2_4d, w2_conv, b2_conv, wsB)
#print 'y1_A', y1_A
#print 'y1_B', y1_B
#similarity_measurement_layer 
feature = similarity_measurement_layer(y1_A, y2_A, y1_B, y2_B)
#feature = tf.reduce_max(tf.concat(1, y1_B[0]+y1_B[1]+y2_B[0]+y2_B[1]), axis=2)



# linear layer 1
w1_linear = weight_variable([numFilter*2*3*2+numFilter*3*2, hindn]) 
#w1_linear = weight_variable([81, hindn]) 
b1_linear = bias_variable([hindn])
y_linear1 = linear_layer(feature, w1_linear, b1_linear)

# activstion layer
y_activation = activation_layer(y_linear1)

# linear layer 2
w2_linear = weight_variable([hindn, outputn]) 
b2_linear = bias_variable([outputn])
y_linear2 = linear_layer(y_activation, w2_linear, b2_linear)

# log-softmax layer
y_softmax = log_softmax_layer(y_linear2)
#print 'y_softmax', y_softmax.get_shape()
#print 'ys', ys_2d.get_shape()


cross_entropy = -tf.reduce_sum(ys_2d*tf.log(y_softmax))
train = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_2d, 1), tf.argmax(y_softmax, 1)), tf.float32))
#------------------------------------------------------------------------------
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


#-------------------------- bulid dict of word-embaddings 
f = open('./paragram-phrase-XXL.txt','r')
dict = {}
while 1:
    line = f.readline()
    if not line:
        break
    items = line.strip().split()
    if len(items) != 301:
        continue
    value = []
    for i in range(1,301):
        value.append(float(items[i]))
    dict[items[0]] = value

f.close()
print "Build dict finished"

def word2vec(sent):
    '''
    word to vector
    '''
    sent = sent.lower()
    sent_vec = []
    for i in range(len(sent)):
        try:
            tmp = dict[sent[i]]
            sent_vec.append(tmp)
        except:
            pass
    return sent_vec
    
'''attention-based word embadding '''
def attention_based(sent1, sent2):
    def softmax(x):
    	return np.exp(x) / np.sum(np.exp(x), axis=0)

    sent1_vec = np.array(sent1, dtype=np.float32)
    sent2_vec = np.array(sent2, dtype=np.float32)    
    D = np.dot(sent1_vec, np.transpose(sent2_vec))
    for i in range(len(sent1_vec)):
        for j in range(len(sent2_vec)):
            D[i][j] = D[i][j]/(np.linalg.norm(sent1_vec[i])*np.linalg.norm(sent2_vec[j]))

    E1 = np.sum(D, axis = 1)
    E2 = np.sum(D, axis = 0)
    A1 = softmax(E1)
    A1 = A1.reshape([len(A1), 1])
    A2 = softmax(E2)
    A2 = A2.reshape([len(A2), 1])
    sent1_attention = np.hstack((sent1_vec, np.multiply(sent1_vec, A1)))
    sent2_attention = np.hstack((sent2_vec, np.multiply(sent2_vec, A2)))
    return sent1_attention,sent2_attention

# train data
train_data = open('./train.txt','r')
train_sentence1 = []
train_sentence2 = []
train_y = []
f1 = 0
while f1<1100:
    f1 += 1
    line = train_data.readline()
    if not line:
        #print line
        break
    items = line.strip().split('\t')
    if len(items) != 3:
        continue
    vector1,vector2 = attention_based(word2vec(items[0]), word2vec(items[1]))
    #train_sentence1.append(tf.Variable(vector1, tf.float32))
    train_sentence1.append(vector1)
    #train_sentence2.append(tf.Variable(vector2, tf.float32))
    train_sentence2.append(vector2)
    score_int = int(round(float(items[2])))
    y = [0]*6
    y[score_int] = 1
    train_y.append(np.array(y, dtype=np.float32))
train_data.close()
print "import train data finished"

test_data = open('./test.txt','r')
test_sentence1 = []
test_sentence2 = []
test_y = []
f2=0
while f2<110:
    f2 += 1
    line = test_data.readline()
    if not line:
        #print line
        break
    items = line.strip().split('\t')
    if len(items) != 3:
        continue
    vector1, vector2 = attention_based(word2vec(items[0]), word2vec(items[1]))
    test_sentence1.append(vector1)
    test_sentence2.append(vector2)
    score_int = int(round(float(items[2])))
    y = [0]*6
    y[score_int] = 1
    test_y.append(np.array(y, dtype=np.float32))
test_data.close()
print "import test data finished"

#------------------------------------------------------------------------


for i in range(1000):
    x1 = train_sentence1[i]
    x2 = train_sentence2[i]
    y = train_y[i]
    sess.run(train, feed_dict={xs1:x1, xs2:x2, ys:y})

    if i%10 == 0:
        score = []
        for j in range(100):
            x1_test = test_sentence1[j]
            x2_test = test_sentence2[j]
            y_test = test_y[j]
            accarcy = sess.run([acc], feed_dict={xs1: x1_test, xs2: x2_test, ys:y_test})
            score.append(accarcy*1.0)
        print i, np.mean(score)

