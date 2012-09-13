import math

# makes numbers pretty
def splitthousands(s, sep=','):
	s = str(int(s))
	if (len(s) <= 3): return s  
	return splitthousands(s[:-3], sep) + sep + s[-3:]

def compute_stats(data, israte=False):
	n = len(data)
	data.sort()
	
	#min
	min = data[0]
	
	if n >= 4:
		#first quartile
		t = (n+1) / 4.0 - 1
		k = int(t)
		if (t == k):
			q1 = data[k]
		else:
			q1 = 3*(data[k]/4.0) + data[k+1]/4.0;
		
	if n >= 2:
		# median
		t = (n+1) / 2.0 - 1
		k = int(t)
		if (t == k):
			median = data[k]
		else:
			median = data[k]/2.0 + data[k+1]/2.0;
	
	if n >= 4:
		# third quartile
		t = 3*((n+1) / 4.0) - 1
		k = int(t)
		if (t == k):
			q3 = data[k]
		else:
			q3 = data[k]/4.0 + 3*(data[k+1]/4.0);

	#max
	max = data[n-1];
	
	#mean
	sum = 0.0
	for i in range(n-1, -1, -1):
		sum = sum + data[i]
	mean = sum/n;
	
	#standard deviation
	s = 0.0
	for k in range(n-1, -1, -1):
		tmp = data[k] - mean
		s = s + tmp*tmp
	sampleStdDev = math.sqrt(s/(n-1))

	#harmonic mean
	s = 0.0
	for k in range(0,n):
		if (data[k]):
			s = s + 1.0/data[k]
	if (s == 0):
		harmonicMean = 0
	else:
		harmonicMean = n/s
	m = s/n
		
	#harmonic sample standard deviation
	s = 0.0
	for k in range(0, n):
		if (data[k]):
			tmp = 1.0/data[k] - m;
		else:
			tmp = -m
		s = tmp*tmp
	harmonicSampleStdDev = (math.sqrt (s)/(n-1)) * harmonicMean * harmonicMean
	
	# total
	total = 0
	for d in data:
		total += d

	ret = {}
	ret["min"] = min
	ret["q1"] = q1
	ret["median"] = median
	ret["q3"] = q3
	ret["max"] = max
	ret["total"] = total
	if israte:
		ret["harmonic_mean"] = harmonicMean
		ret["harmonic_stddev"] = harmonicSampleStdDev
	else:
		ret["mean"] = mean
		ret["stddev"] = sampleStdDev
	return ret

# prints statistics about an array
# the algorithms here are based on the graph500 reference implementation,
# graph500-1.2/graph500.c:218
def printstats(data, label, israte, printSomethingAnyway=False, printTotal=False):
	n = len(data)
	data.sort()
	
	if printSomethingAnyway:
		if n == 0:
			data = [0, 0, 0, 0]
		elif n == 1:
			val = data[0]
			data = [val, val]
		n = len(data)
	else:
		if n == 0:
			return
		if n == 1:
			print "            %s: \t%20.17e"%(label, data[0])
			return
	
	stats = compute_stats(data, israte=israte)
	
	if n >= 2:
		print "            min_%s: \t%20.17e"%(label, stats["min"])
	if n >= 4:
		print "  firstquartile_%s: \t%20.17e"%(label, stats["q1"])
	if n >= 2:
		print "         median_%s: \t%20.17e"%(label, stats["median"])
	if n >= 4:
		print "  thirdquartile_%s: \t%20.17e"%(label, stats["q3"])
	if n >= 1:
		print "            max_%s: \t%20.17e"%(label, stats["max"])
	if (israte):
		print "  harmonic_mean_%s: \t%20.17e"%(label, stats["harmonic_mean"])
		print "harmonic_stddev_%s: \t%20.17e"%(label, stats["harmonic_stddev"])
	else:
		print "           mean_%s: \t%20.17e"%(label, stats["mean"])
		print "         stddev_%s: \t%20.17e"%(label, stats["stddev"])
	if printTotal:
		print "          total_%s: \t%e"%(label, stats["total"])

