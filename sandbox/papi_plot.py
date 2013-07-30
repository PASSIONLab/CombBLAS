import sys

input_files = sorted(sys.argv[1:])

np = len(input_files)
events = None 
labels = []
max_iter = 0
data = []

# pass 1 - load
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
				iter = int(d[0])
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

def scatter_dump():
	global data, events, labels, max_iter, np
	
	from matplotlib import pyplot as plt
	import numpy as np
	import matplotlib

	colors = ['r', 'g', 'b', 'y']

	f, axarr = plt.subplots(len(events), sharex=True)
	#axarr[0].plot(x, y)
	#axarr[0].set_title('Sharing X axis')
	#axarr[1].scatter(x, y)

	for event in events:
		plot_id = events.index(event)
		
		sx = []
		sy = []
		sc = []
		for dp in data:
			x = dp['iter']*len(labels) + labels.index(dp['label'])
			sx.append(x)
			sy.append(dp[event])
			sc.append(colors[dp['p']]) # color
	
		axarr[plot_id].scatter(sx, sy, c=sc, marker='x', label=None)
		axarr[plot_id].set_ylabel(event)
		#plt.legend(loc=2)

	# set up the ticks
	plt.xlabel("counters")
	ax = plt.gca()
	num_ticks = len(labels)*(max_iter+1)
	ax.set_xticks(range(num_ticks))
	ticklabels = []
	for i in range(num_ticks):
		l = labels[i%len(labels)] + " %d"%(int(i/(len(labels))))
		ticklabels.append(l)
	ax.set_xticklabels(ticklabels)
	plt.setp(ax.get_xticklabels(), rotation='vertical')
	#ax = plt.subplot(111)
	#ax.set_xticklabels(["one", "two", "three"])

	plt.xlim([-1,num_ticks+1])
	
	f.set_tight_layout(True)
	#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	plt.show()

scatter_dump()