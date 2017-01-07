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
	# GPU
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	config.gpu_options.allow_growth = True

	# sess
	sess=tf.InteractiveSession(config=config)
	tf.initialize_all_variables().run()

	model.restore(sess, save_path)

	data = mnist.read_data_sets()

	x_, _ = data.train.next_batch(hp.batch_size)
	x_ = (x_ > 0.5).astype(np.float32)		#二值化

	images = sess.run(model.output_tensors, {model.x: x_})
	
	for T, image in enumerate(images):
		imgs = image.reshape(-1, hp.A, hp.B)[:100]
		img = images2one(imgs)
		img = (np.clip(img, 0.0, 1.0) *  255).astype(np.uint8)
		unit_images.append(img)
		imageio.imwrite('images/' +str(T) + '.png', img)

	imageio.mimsave('images/gen.gif', unit_images, duration=1.0)

	sess.close()
