# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

save_path = 'output/checkpoint.ckpt'
ReadAtten = True
WriteAtten = True
## MODEL PARAMETERS ## 

A,B = 28,28 # image width,height
img_size = B*A # the canvas size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 5 # read glimpse grid width/height
write_n = 5 # write glimpse grid width/height
read_size = 2*read_n*read_n if ReadAtten else 2*img_size
write_size = write_n*write_n if WriteAtten else img_size
z_size = 10 # QSampler output size
T = 10 # MNIST generation sequence length
batch_size = 128
train_iters = 10000
learning_rate = 1e-3 # learning rate for optimizer
eps = 1e-8 # epsilon for numerical stability

## OPTIMIZER ## 

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads = optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
	if g is not None:
		grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)


# 训练

data_directory = os.path.join(data_dir, "mnist")
if not os.path.exists(data_directory):
	os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

sess=tf.InteractiveSession(config=config)
tf.initialize_all_variables().run()
saver = tf.train.Saver()


if os.path.exists(save_path):
	saver.restore(sess, save_path)


for i in range(train_iters):
	xtrain,_=train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
	feed_dict={x: xtrain}
	Lx_, Lz_, _ = sess.run([Lx, Lz, train_op], feed_dict)
	if i%100==0:
		print "iter=%d : Lx: %f Lz: %f" % (i, Lx_, Lz_)

## TRAINING FINISHED ## 

canvases = sess.run(cs, feed_dict) # generate some examples
canvases = np.array(canvases) # T x batch x img_size

out_file = os.path.join(data_dir,"draw_data.npy")
np.save(out_file, [canvases, Lxs, Lzs])
print("Outputs saved in file: %s" % out_file)

ckpt_file = os.path.join(data_dir,"drawmodel.ckpt")
print("Model saved in file: %s" % saver.save(sess, ckpt_file))

sess.close()

print('Done drawing! Have a nice day! :)')
