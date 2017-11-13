import tensorflow as tf
import numpy as np
import os, json
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def matrix_factorization(lamda, beta, rm, d):
	R = tf.placeholder(tf.float32, shape=(None, ), name ='R')
	x_index = tf.placeholder(tf.int32, shape=(None, 1), name ='x_index')
	y_index = tf.placeholder(tf.int32, shape=(None, 1), name ='y_index')
	
	P = tf.Variable(tf.random_uniform([i0, d], -1.0, 1.0))
	Q = tf.Variable(tf.random_uniform([u0, d], -1.0, 1.0))
	
	arr1 = tf.gather_nd(P, y_index)
	arr2 = tf.gather_nd(Q, x_index)
	R_pred = tf.reduce_sum(tf.multiply(arr1, arr2), 1)

	first_term = tf.reduce_sum(tf.square(R_pred - R))
	second_term = (lamda/2.0)*(tf.square(tf.norm(P))+ tf.square(tf.norm(Q)))
	loss = second_term + first_term

	return x_index, y_index, R, loss

def run():
	global i0, u0
	inter = './inter_files/'
	with open(inter+'indices','r') as f:
		index = json.load(f)

	with open(inter+'values','r') as f:
		value = json.load(f)

	u0,i0 = index[0][-1], index[1][-1]
	del index[0][-1]
	del index[1][-1]
	index = np.array(index)

	x_index, y_index, R, loss = matrix_factorization(0.1, 0.1, 0, 100)
	optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		x_feed = np.array([[x] for x in index[0]])
		y_feed = np.array([[y] for y in index[1]])

		feed_dict = {x_index:x_feed, y_index:y_feed, R:value}
		
		loss_val, _ = session.run([loss, optimizer], feed_dict)
		print(loss_val)
		while True:
			loss_val2, _ = session.run([loss, optimizer], feed_dict)
			delta = loss_val-loss_val2
			if abs(delta) < 0.01:
				break
			loss_val = loss_val2
			print(loss_val)

if __name__ == '__main__':
	run()