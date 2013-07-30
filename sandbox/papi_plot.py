import sys

input_files = sorted(sys.argv[1:])

np = len(input_files)
events = None 
labels = []
max_iter = 0
data = []

# pass 1
p = 0
for filename in input_files:
	with open(filename, "r") as f:
		for line in f:
			if line.find("iter") != -1: # header line
				if not events:
					h = line.strip().split("\t")
					events = h[2:]
			else:
				d = line.strip().split("\t")
				iter = d[0]
				label = d[1]
				
				if label not in labels:
					labels.append(label)
				max_iter = max(iter, max_iter)
				dp = {'p':p, 'iter':iter, 'label':label}
				
				if len(events) != len(d[2:]):
					raise ValueError, "length mismatch!"
					
				for x in zip(events, d[2:]):
					dp[x[0]] = x[1]
				data.append(dp)
	p += 1				
					
print events
print labels
print max_iter
print len(data)
#for d in data:
#	print d