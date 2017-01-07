# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import draw
import cmtf.data.data_mnist as mnist


save_path = 'output/checkpoint.ckpt'

# 检测目录是否存在
save_dir = os.path.dirname(save_path)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


# 生成图
hp = draw.default_hp()
graph = tf.Graph()
model = draw.DRAW(graph, hp)


# 训练
with graph.as_default():
	# 优化方法
	optimizer = tf.train.AdamOptimizer(hp.learning_rate, beta1=0.5)

	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * 0.1
	loss = model.loss + l2_loss

	grads = optimizer.compute_gradients(loss)
	for i,(g,v) in enumerate(grads):
		if g is not None:
			grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
	train_op = optimizer.apply_gradients(grads)

	# GPU
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	config.gpu_options.allow_growth = True

	# sess、saver
	sess=tf.InteractiveSession(config=config)
	saver = tf.train.Saver()
	tf.initialize_all_variables().run()

	# # restore
	# if os.path.exists(save_path):
	#  	saver.restore(sess, save_path)

	data = mnist.read_data_sets()

	# train
	for i in range(hp.epochs):
		x_, _ = data.train.next_batch(hp.batch_size)
		x_ = (x_ > 0.5).astype(np.float32)		#二值化

		Lx_, Lz_, _ = sess.run([model.Lx, model.Lz, train_op], {model.x: x_})
		if (i+1)%10==0:
			str_output = "epoch: %d   Lx: %.2f   Lz: %.2f" % (i, Lx_, Lz_)
			if (i+1)%500 == 0:
				str_output += '   save'
				saver.save(sess, save_path)
			print str_output

	sess.close()
