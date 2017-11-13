import os
import json

user_dir = '../epinions/wot/'
inter_dir = './inter_files/'

noUsers = 75888
noE = 0
adj = [0]*noUsers

for i in range(noUsers):
	filename = str(i)+"-trusts"
	adj[i] = []
	with open(user_dir+filename, "r") as f:
		st = f.read().split('\n')
		for x in st:
			if x:
				x = int(x.split('\t')[0])
				if x < noUsers:
					adj[i].append(x)
					noE += 1

print noUsers, noE

with open(inter_dir+"s_list", "w") as f:
	json.dump(adj, f)
