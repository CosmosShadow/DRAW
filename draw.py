# coding: utf-8
import tensorflow as tf
from cmtf.func import lib as tflib
from cmtf.model.tf_object import TFObject

# 超参
def default_hp():
	return tflib.HParams(
	                   # 结构参数
	                   ReadAtten = True,	#读是否采用Attention机制
	                   WriteAtten = True,
	                   A = 28,	#图片大小 A * B
	                   B = 28,
	                   enc_size = 256,	#LSTM encode大小
	                   dec_size = 256,	#LSTM decode大小
	                   read_n = 5,		#读格子大小
	                   write_n = 5,		#写格子大小
	                   z_size = 10,		#采样大小
	                   T = 10,			#步长
	                   # 训练参数
	                   batch_size = 128,
	                   epochs = 10000,
	                   learning_rate = 1e-2,
	                   # 生成参数
	                   gen_batch_size_sqrt = 11,
	                   )

# 模型
class DRAW(TFObject):
	def __init__(self, graph, hp, scope='draw'):
		self.graph = graph
		self.hp = hp
		self.scope = scope
		self.build_graph()

	def restore(self, session, checkpoint_file, checkpoint_scope='draw'):
		super(DRAW, self).restore(session, checkpoint_file, checkpoint_scope)

	def build_graph(self):
		with self.graph.as_default():
			with tf.device(lambda op: ''):
				with tf.variable_scope(self.scope):
					A = self.hp.A
					B = self.hp.B
					T = self.hp.T

					self.x = tf.placeholder(tf.float32,shape=(self.hp.batch_size, A * B)) # input (self.hp.batch_size * hp.A * B)
					epsilon = tf.random_normal((self.hp.batch_size, self.hp.z_size), mean=0, stddev=1) # Qsampler noise
					lstm_enc = tf.nn.rnn_cell.LSTMCell(self.hp.enc_size, state_is_tuple=True) # encoder Op
					lstm_dec = tf.nn.rnn_cell.LSTMCell(self.hp.dec_size, state_is_tuple=True) # decoder Op
					read = read_attn if self.hp.ReadAtten else read_no_attn
					write = write_atten if self.hp.WriteAtten else write_no_attn

					cs = [0] * T #生成序列
					# LSTM状态
					h_dec_prev = tf.zeros((self.hp.batch_size, self.hp.dec_size))
					enc_state = lstm_enc.zero_state(self.hp.batch_size, tf.float32)
					dec_state = lstm_dec.zero_state(self.hp.batch_size, tf.float32)

					self.Lz = 0.0
					# 构建模型
					DO_SHARE = None
					output_tensor = tf.zeros((self.hp.batch_size, A * B))
					self.output_tensors = [output_tensor]
					for t in range(self.hp.T):
						# read
						x_hat = self.x - tf.sigmoid(output_tensor) # error image
						r = read(self.x, x_hat, h_dec_prev, self.hp.read_n, A, B, DO_SHARE)
						# encode
						h_enc, enc_state = encode(lstm_enc, enc_state, tf.concat(1, [r, h_dec_prev]), DO_SHARE)
						# Q
						z, mean, stddev = sampleQ(h_enc, self.hp.z_size, epsilon, DO_SHARE)
						# vae loss
						vae_loss = get_vae_cost(mean, stddev)
						self.Lz = self.Lz + vae_loss
						# decode
						h_dec, dec_state = decode(lstm_dec, dec_state, z, DO_SHARE)
						# write
						output_tensor = output_tensor + write(h_dec, self.hp.write_n, A, B, DO_SHARE)
						self.output_tensors.append(output_tensor)
						h_dec_prev = h_dec

						# LSTM参数重用
						DO_SHARE = True

					# 代价函数
					self.Lx = tf.reduce_sum(binary_crossentropy(self.x, tf.nn.sigmoid(output_tensor)))
					self.loss = self.Lx + self.Lz

					# 生成过程
					DO_SHARE = True
					gen_batch_size = self.hp.gen_batch_size_sqrt * self.hp.gen_batch_size_sqrt
					self.gen_z = tf.random_normal((gen_batch_size, self.hp.z_size), mean=0, stddev=1) # Qsampler noise
					# self.gen_z = tf.placeholder(tf.float32, shape=(gen_batch_size, self.hp.z_size))
					gen_dec_state = lstm_dec.zero_state(gen_batch_size, tf.float32)
					sampled_tensor = tf.zeros((gen_batch_size, A * B))
					self.sampled_tensors = []
					for t in range(T):
						# decode
						gen_h_dec, gen_dec_state = decode(lstm_dec, gen_dec_state, self.gen_z, DO_SHARE)
						# write
						sampled_tensor = sampled_tensor + write(gen_h_dec, self.hp.write_n, A, B, DO_SHARE)
						# store to sequences
						self.sampled_tensors.append(tf.nn.sigmoid(sampled_tensor))


def linear(x, output_dim):
	w = tf.get_variable("w", [x.get_shape()[1], output_dim]) 
	b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
	return tf.matmul(x,w) + b

def filterbank(gx, gy, sigma2, delta, N, A, B):
	grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
	mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
	mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
	a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
	b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
	mu_x = tf.reshape(mu_x, [-1, N, 1])
	mu_y = tf.reshape(mu_y, [-1, N, 1])
	sigma2 = tf.reshape(sigma2, [-1, 1, 1])
	Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
	Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
	# normalize, sum over A and B dims
	eps=1e-8
	Fx = Fx/tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
	Fy = Fy/tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
	return Fx, Fy

def attn_window(scope, h_dec, N, A, B, DO_SHARE):
	with tf.variable_scope(scope, reuse=DO_SHARE):
		params = linear(h_dec, 5)
	gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(1, 5, params)
	gx = (A+1)/2*(gx_+1)
	gy = (B+1)/2*(gy_+1)
	sigma2 = tf.exp(log_sigma2)
	delta = (max(A, B)-1)/(N-1)*tf.exp(log_delta) # batch x N
	return filterbank(gx, gy, sigma2, delta, N, A, B) + (tf.exp(log_gamma), )

## READ ## 
def read_no_attn(x, x_hat, h_dec_prev, N, A, B, DO_SHARE):
	return tf.concat(1, [x, x_hat])

def filter_img(img, Fx, Fy, gamma, N, A, B):
	Fxt = tf.transpose(Fx, perm = [0, 2, 1])
	img = tf.reshape(img, [-1, B, A])
	glimpse = tf.batch_matmul(Fy, tf.batch_matmul(img, Fxt))
	glimpse = tf.reshape(glimpse, [-1, N*N])
	return glimpse * tf.reshape(gamma, [-1,1])

def read_attn(x, x_hat, h_dec_prev, N, A, B, DO_SHARE):
	Fx, Fy, gamma = attn_window("read", h_dec_prev, N, A, B, DO_SHARE)
	x = filter_img(x, Fx, Fy, gamma, N, A, B)
	x_hat = filter_img(x_hat, Fx, Fy, gamma, N, A, B)
	return tf.concat(1, [x, x_hat]) # concat along feature axis


def encode(lstm_enc, state, input, DO_SHARE):
	with tf.variable_scope("encoder", reuse=DO_SHARE):
		return lstm_enc(input, state)


def sampleQ(h_enc, z_size, epsilon, DO_SHARE):
	eps = 1e-8
	with tf.variable_scope("mean", reuse=DO_SHARE):
		mean = linear(h_enc, z_size)
	with tf.variable_scope("stddev", reuse=DO_SHARE):
		stddev = tf.sqrt(tf.exp(linear(h_enc, z_size)))
	return (mean + stddev*epsilon, mean, stddev)

def get_vae_cost(mean, stddev, epsilon=1e-8):
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) - 2.0 * tf.log(stddev + epsilon) - 1.0))

def decode(lstm_dec, state, input, DO_SHARE):
	with tf.variable_scope("decoder",reuse=DO_SHARE):
		return lstm_dec(input, state)


def write_no_attn(h_dec, N, A, B, DO_SHARE):
	with tf.variable_scope("write", reuse=DO_SHARE):
		return linear(h_dec, A*B)


def write_atten(h_dec, N, A, B, DO_SHARE):
	with tf.variable_scope("writeW",reuse=DO_SHARE):
		w = linear(h_dec, N*N)
	w = tf.reshape(w, [-1, N, N])
	Fx, Fy, gamma = attn_window("write", h_dec, N, A, B, DO_SHARE)
	Fyt = tf.transpose(Fy, perm = [0, 2, 1])
	wr = tf.batch_matmul(Fyt, tf.batch_matmul(w, Fx))
	wr = tf.reshape(wr, [-1, A*B])
	return wr * tf.reshape(1.0/gamma, [-1, 1])


def binary_crossentropy(t, o):
	eps=1e-8
	return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



