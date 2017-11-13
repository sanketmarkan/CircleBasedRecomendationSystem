import json

user_dir = '../epinions/reviews/'
inter_dir = './inter_files/'

#75888
noUsers = 75888
ma = {}
noProducts = 0

for i in range(noUsers):
	with open(user_dir+str(i)) as f:
		a = f.read().split('\n')
		for x in a:
			y =  x.split('\t')
			if len(y)>1:
				category = y[-5]
				rating = y[-3]
				product = y[3]
				if rating == "na":
					continue

				try:
					b = ma[product]
				except:
					ma[product] = noProducts
					noProducts += 1
x_val = []
y_val = []
val = []
print noUsers, noProducts

for i in range(noUsers):
	with open(user_dir+str(i)) as f:
		a = f.read().split('\n')
		for x in a:
			y =  x.split('\t')
			if len(y)>1:
				category = y[-5]
				rating = y[-3]
				product = y[3]
				if rating == "na":
					continue
				
				x_val.append(i)
				y_val.append(ma[product])
				val.append(min(float(rating), 5.0))

with open(inter_dir+"indices", "w") as f:
	x_val.append(noUsers)
	y_val.append(noProducts)
	indices = [x_val, y_val]
	json.dump(indices, f)

with open(inter_dir+"values", "w") as f:
	json.dump(val, f)
