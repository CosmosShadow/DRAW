# coding: utf-8
import os
import imageio
import numpy as np
import tensorflow as tf

import cmtf.data.data_mnist as mnist
from ImageOperation.images2one import *
from scipy import misc

import draw


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

	unit_images = []
	images = sess.run(model.sampled_tensors)
	for T, image in enumerate(images):
		imgs = image.reshape(-1, hp.A, hp.B)[:100]
		img = images2one(imgs)
		img = (np.clip(img, 0.0, 1.0) *  255).astype(np.uint8)
		unit_images.append(img)
		imageio.imwrite('images/' +str(T) + '.png', img)

	imageio.mimsave('images/gen.gif', unit_images, duration=1.0)

	sess.close()
