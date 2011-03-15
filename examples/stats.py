import math

# makes numbers pretty
def splitthousands(s, sep=','):
	s = str(int(s))
	if (len(s) <= 3): return s  
	return splitthousands(s[:-3], sep) + sep + s[-3:]

# prints statistics about an array
# the algorithms here are based on the graph500 reference implementation,
# graph500-1.2/graph500.c:218
def printstats(data, label, israte):
	n = len(data)
	data.sort()
	
	if n == 0:
		return
	if n == 1:
		print "            %s: %20.17e"%(label, data[0])
		return
	
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
	
	if n >= 2:
		print "            min_%s: %20.17e"%(label, min)
	if n >= 4:
		print "  firstquartile_%s: %20.17e"%(label, q1)
	if n >= 2:
		print "         median_%s: %20.17e"%(label, median)
	if n >= 4:
		print "  thirdquartile_%s: %20.17e"%(label, q3)
	if n >= 1:
		print "            max_%s: %20.17e"%(label, max)
	if (israte):
		print "  harmonic_mean_%s: %20.17e"%(label, harmonicMean)
		print "harmonic_stddev_%s: %20.17e"%(label, harmonicSampleStdDev)
	else:
		print "           mean_%s: %20.17e"%(label, mean)
		print "         stddev_%s: %20.17e"%(label, sampleStdDev)

