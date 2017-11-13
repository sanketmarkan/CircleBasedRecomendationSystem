import tensorflow as tf
import numpy as np
import os, json
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def social_matrix_factorization(lamda, beta, rm, d):
	R = tf.placeholder(tf.float32, shape=(None, ), name ='R')
	x_index = tf.placeholder(tf.int32, shape=(None, 1), name ='x_index')
	y_index = tf.placeholder(tf.int32, shape=(None, 1), name ='y_index')
	
	social_x_index = tf.placeholder(tf.int32, shape=(None, 1), name ='social_x_index')
	social_y_index = tf.placeholder(tf.int32, shape=(None, 1), name ='social_y_index')
	social_x_value = tf.placeholder(tf.float32, shape=(None, 1), name ='social_x_value')
	social_y_value = tf.placeholder(tf.float32, shape=(None, 1), name ='social_y_value')
	
	P = tf.Variable(tf.random_uniform([i0, d], -1.0, 1.0))
	Q = tf.Variable(tf.random_uniform([u0, d], -1.0, 1.0))
	
	arr1 = tf.gather_nd(P, y_index)
	arr2 = tf.gather_nd(Q, x_index)
	R_pred = tf.reduce_sum(tf.multiply(arr1, arr2), 1)

	arr3 = tf.gather_nd(Q, social_x_index)
	arr4 = tf.gather_nd(Q, social_y_index)
	arr5 = social_x_value * arr3 - social_y_value * arr4
	iQ = tf.reduce_sum(arr5, 0)

	first_term = tf.reduce_sum(tf.square(R_pred - R))
	second_term = (lamda/2.0)*(tf.square(tf.norm(P))+ tf.square(tf.norm(Q)))
	third_term = (beta/2.0)*(tf.reduce_sum(iQ*iQ))
	loss = second_term + first_term + third_term

	return x_index, y_index, R, loss, social_x_index, social_y_index, social_x_value, social_y_value

def equalTrust(s):
	social_index = [0]*2
	social_value = [0]*2
	for i in [0,1]:
		social_index[i] = []
		social_value[i] = []
	
	for i in range(len(s)):
		k = len(s[i])
		for j in s[i]:
			social_index[0].append(i)
			social_index[1].append(j)
			social_value[0].append(1.0/k)
			social_value[1].append(1.0/k)
	
	return social_index, social_value

def run():
	global i0, u0
	inter = './inter_files/'
	with open(inter+'indices','r') as f:
		index = json.load(f)

	with open(inter+'values','r') as f:
		value = json.load(f)

	with open(inter+'s_list', 'r') as f:
		s = json.load(f)

	social_index, social_value = equalTrust(s)

	u0, i0 = index[0][-1], index[1][-1]
	del index[0][-1]
	del index[1][-1]
	index = np.array(index)

	x_index, y_index, R, loss, A, B, C, D = social_matrix_factorization(0.1, 0.1, 0, 100)
	optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)

		x_feed = np.array([[x] for x in index[0]])
		y_feed = np.array([[y] for y in index[1]])

		social_x_feed = np.array([[x] for x in social_index[0]])
		social_y_feed = np.array([[y] for y in social_index[1]])

		social_x_value_feed = np.array([[x] for x in social_value[0]])
		social_y_value_feed = np.array([[x] for x in social_value[1]])

		feed_dict = {x_index:x_feed, y_index:y_feed, R:value}
		feed_dict[A] = social_x_feed
		feed_dict[B] = social_y_feed
		feed_dict[C] = social_x_value_feed
		feed_dict[D] = social_y_value_feed
		
		loss_val, _ = session.run([loss, optimizer], feed_dict)
		print(loss_val)
		while True:
			loss_val2, _ = session.run([loss, optimizer], feed_dict)
			delta = loss_val-loss_val2
			if abs(delta) < 0.1:
				break
			loss_val = loss_val2
			print(loss_val)

if __name__ == '__main__':
	run()