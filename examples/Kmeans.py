import kdt
import time
import getopt
import sys
import random
import math
from stats import splitthousands

# Usage for a file
def usage():
	print "Kmeans.py [-nDATA_SIZE] [-mATTRIBUTES] [-kCLUSTERS] [-fFILE]"
	print "Kmeans.py creates a random matrix using the input dimensions, and calls the kmeans function on it."
	print "to cluster the data set. The DATA_SIZE represents the number of data elements to be generated."
	print "Since we are using the matrix representation, the number of data points is equal to the rows of the matrix."
	print "The -m argument represents the attributes of the element. These are the columns in the matrix representation."
	print "-k is the number of clusters. The matrix generated is effectively a set of n elements in an m-dimensional"
	print "vector space"
	print "The -f FILE argument is for storing the matrix and the resulting clusters into files."
	print "The matrix generated is stored in FILE.mat and the cluster matrix is stored in FILE.clust. Results are stored by"
	print "default in default.mat and default.clust"
	print "Example:"
	print "Generating a matrix of 100*10 and clustering it into 5 clusters. Results stored in demo.mat and demo.clust."
	print "python Kmeans.py -n 100 -m 10 -k 5 -f demo"


# Setup up the command line arguments.
try:
	opts, args = getopt.getopt(sys.argv[1:], "hn:m:k:f:", ["help", "rows=", "cols=", "clusters=", "file="])
except getopt.GetoptError, err:
	# print help information and exit:
	if kdt.master():
		print str(err) # will print something like "option -a not recognized"
		usage()
	sys.exit(2)

# Default values for rows, columns and number of clusters.

n = 1000
m = 19
k = 10
plotfile = "default"

for o,a in opts:
	if o in ("-h","--help"):
		usage()
		sys.exit(0)	
	if o in ("-n","--rows"):
		n = int(a)
	if o in ("-m","--cols"):
		m = int(a)
	if o in ("-k","--clusters"):
		k = int(a)
	if o in ("-f","--file"):
		if(m <= 3):
			plotfile = a 
		else:
			print "The dimensions are such that result cannot be plotted through gnuplot.\
			       Please enter dimensions (m) <= 3"
			sys.exit(2)

# Some corner cases which are logically not possible. e.g. Sorting 3 data points into 5 clusters.
if k > n:
	k = n

# A single data point has one cluster
if n == 1:
	print "Chose n > 1. n = 1 is a trivial case."
	sys.exit(1)

# setup the matrix
src1 = kdt.Vec.range(n)
dest1 = kdt.Vec(n,sparse = False)
values1 = kdt.Vec(n,sparse = False) 	

src1.randPerm()
dest1.apply(lambda x: random.randint(0,m-1))
values1.apply(lambda x: random.randint(1,5))

E = kdt.Mat(src1,dest1,values1,m,n)

# Create a random matrix as a sum of many random matrices.
for i in range(1,10):

	src1.randPerm()
	dest1.apply(lambda x: random.randint(0,m-1))
	values1.apply(lambda x: random.randint(1,5))
	
	E1 = kdt.Mat(src1,dest1,values1,m,n)
	E = E.__add__(E1)

# Time the algorithm
before = time.time()
B = kdt.Mat.kmeans(E,m,n,k)
time = time.time() - before

print ""
print "Dividing an "+str(n)+"-by-"+str(m)+" matrix into "+str(k)+" clusters"
print "Time required = "+str(time)
print ""

# Print the original matrix and the resultant cluster representation.
plotmat = open(plotfile+'.mat','w')
plotclust = open(plotfile+'.clust','w')
plotmat.write(str(E))
plotclust.write(str(B))
plotmat.close()
plotclust.close()
	
