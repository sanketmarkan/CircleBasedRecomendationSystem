def expertTrust(s, cat, A):
	ma,total = {}, 0
	for x in A:
		try:
			ma[x] += 1
		except:
			ma[x] = 1

	social_index = [0]*2
	social_value = [0]*2
	for i in [0,1]:
		social_index[i] = []
		social_value[i] = []

	arr = [0]*len(s)
	total = [0]*len(s)
	for i in range(len(s)):
		for j in s[i]:
			try:
				b = ma[i]
				b = ma[j] 
				arr[i] += 1
				total[i] += ma[j]
			except:
				continue

	for i in range(len(s)):
		k = arr[i]
		for j in s[i]:
			try:
				b = ma[i]
				b = ma[j]
				xd = total[i] + ma[i];
				social_index[0].append(i)
				social_index[1].append(j)
				social_value[0].append(1.0/k)
				social_value[1].append(ma[j]/float(xd))
			except:
				continue

	return social_index, social_value

