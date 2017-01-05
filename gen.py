# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import draw
import cmtf.data.data_mnist as mnist


save_path = 'output/checkpoint.ckpt'


# 生成图
hp = draw.default_hp()
graph = tf.Graph()
model = draw.DRAW(graph, hp)


with graph.as_default():
	# GPU
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	config.gpu_options.allow_growth = True

	# sess、saver
	sess=tf.InteractiveSession(config=config)
	tf.initialize_all_variables().run()

	# restore
	model.restore(sess, save_path)

	images = sess.run(model.generated_images_sequences)
	print images

	sess.close()
