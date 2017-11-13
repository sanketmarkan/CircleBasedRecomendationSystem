import tensorflow as tf
import numpy as np
import os, json
import random, math
from random import shuffle

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

	first_term = tf.reduce_mean(tf.square(R_pred - R))
	second_term = (lamda/2.0)*(tf.square(tf.norm(P))+ tf.square(tf.norm(Q)))
	third_term = (beta/2.0)*(tf.reduce_mean(iQ*iQ))
	loss = second_term + first_term + third_term
	return x_index, y_index, R, loss, social_x_index, \
			social_y_index, social_x_value, social_y_value, \
			first_term

def equalTrust(s, cat, A):
	ma = {}
	for x in A:
		ma[x] = 1

	social_index = [0]*2
	social_value = [0]*2
	for i in [0,1]:
		social_index[i] = []
		social_value[i] = []

	arr = [0]*len(s)
	for i in range(len(s)):
		for j in s[i]:
			try:
				b = ma[i]
				b = ma[j] 
				arr[i] += 1
			except:
				continue

	for i in range(len(s)):
		k = arr[i]
		for j in s[i]:
			try:
				b = ma[i]
				b = ma[j]
				social_index[0].append(i)
				social_index[1].append(j)
				social_value[0].append(1.0/k)
				social_value[1].append(1.0/k)
			except:
				continue

	return social_index, social_value

def init():
	global i0, u0, noCat
	inter = './inter_files/'
	with open(inter+'indices_map','r') as f:
		ma = json.load(f)

	with open(inter+'s_list', 'r') as f:
		s = json.load(f)

	noCat, u0, i0 = ma["nothappening"][0][0], ma["nothappening"][1][0], ma["nothappening"][2][0]
	del ma["nothappening"]

	return ma, s

def run(index, value, social_index, social_value, test_index, test_value, rm, noe):
	x_index, y_index, R, loss, A, B, C, D, L2 = social_matrix_factorization(0.1, 30, rm, 100)
	optimizer = tf.train.GradientDescentOptimizer(5e-6).minimize(loss)
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

		noIt = 0
		while True:
			loss_val2, _ = session.run([loss, optimizer], feed_dict)
			delta = loss_val-loss_val2
			if noIt >= 30:
				break
			loss_val = loss_val2
			noIt += 1

			if noIt%10==0:
				print "****", delta, noIt, loss_val
			# 	x_feed = np.array([[x] for x in test_index[0]])
			# 	y_feed = np.array([[y] for y in test_index[1]])
			# 	feed_dict2 = {x_index:x_feed, y_index:y_feed, R:test_value}
			# 	Lo2 = session.run(L2, feed_dict2)
			# 	print math.sqrt(Lo2), noIt
		x_feed = np.array([[x] for x in test_index[0]])
		y_feed = np.array([[y] for y in test_index[1]])
		feed_dict2 = {x_index:x_feed, y_index:y_feed, R:test_value}
		Lo2 = math.sqrt(session.run(L2, feed_dict2))/noe
	return Lo2

def feedRun(cat, A, B, s):
	
	Q = np.array(B)
	rm = np.mean(Q)
	C, D = equalTrust(s, cat, A[0])

	error = 0
	k = 5
	le = len(B)

	x = range(le)
	shuffle(x)

	A = np.array(A)
	B  = np.array(B)

	A[0] = A[0][x]
	A[1] = A[1][x]
	B = B[x]

	A = A.tolist()
	B = B.tolist()
	
	for i in range(k):
		A1 = [A[0][(i*le)/k:((i+1)*le)/k], A[1][(i*le)/k:((i+1)*le)/k]]
		B1 = B[(i*le)/k:((i+1)*le)/k]

		A2 = [A[0][:(i*le)/k]+ A[0][((i+1)*le)/k:], A[1][:(i*le)/k] + A[1][((i+1)*le)/k:]]
		B2 = B[0:(i*le)/k]+ B[((i+1)*le)/k:]

		error += run(A1, B1, C, D, A2, B2, rm, k)

		print i, " Done ", cat, "with", error

	return error/k

if __name__ == '__main__':
	ma, s = init()
	cat_list = ["Software", "Books", "Music"]
	cat_list = ["Software", "Books", "Music", "Toys", "Videos & DVDs", "Destinations", "Cars", \
					"Kids' TV Shows", "Video Games", "Chain Restaurants"]

	ma2 = {}
	for cat in ma.keys():
		if cat not in cat_list:
			continue
		A = [ma[cat][0], ma[cat][1]]
		B = ma[cat][2]

		rmse = feedRun(cat, A, B, s)
		print "RMSE error for", cat, rmse

		ma2[cat] = rmse

	print ma2
