import json

user_dir = '../epinions/reviews/'
inter_dir = './inter_files/'

ma = {}
ma2 = {}

noUsers = 75888
noProducts = 0
noCat = 0

# cat_list = ["Software", "Books", "Music", "Toys", "Videos & DVDs", "Destinations", "Cars", "Kids' TV Shows", \
#  "Video Games", "Chain Restaurants"]

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

				try:
					b = ma2[category]
				except:
					ma2[category] = noCat
					noCat += 1

ma2["nothappening"] = 4
for key in ma2.keys():
	ma2[key] = [0]*3
	for i in range(3):
		ma2[key][i] = []

print noUsers, noProducts, noCat
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
				
				ma2[category][0].append(i)
				ma2[category][1].append(ma[product])
				ma2[category][2].append(min(float(rating), 5.0))

with open(inter_dir+"indices_map", "w") as f:
	ma2["nothappening"][0].append(noCat)
	ma2["nothappening"][1].append(noUsers)
	ma2["nothappening"][2].append(noProducts)
	json.dump(ma2, f)
