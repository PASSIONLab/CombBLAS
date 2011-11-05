from DiGraph import DiGraph
from Vec import Vec
from Mat import Mat
from Util import *

import random
import math
from Mat import Mat
from Vec import Vec

def add(x,y):
	#print "Add",x,y
	return x+y

def mul(x,y):
	#print "Mul",x,y
	return (x-y)*(x-y)

def _kmeans(self):
	
	#######################################################################	
	# Start by creating an n*k eigen-vector matrix E.		
	
	# The dimension of the matrix = the number of vertices	
	n = 1000
	# The number of eigen vectors
	k = 10

	src1 = Vec.range(n)
	dest1 = Vec(n,sparse = False)
	values1 = Vec(n,sparse = False) 	
	
	src1.randPerm()
	dest1.apply(lambda x: random.randint(0,k-1))
	values1.apply(lambda x: random.randint(1,5))
	
	#print src1,dest1,values1
	#print src1,dest1,values1
		
	E = Mat(src1,dest1,values1,k,n)
	#print E

	for i in range(1,10):

		src1.randPerm()
		dest1.apply(lambda x: random.randint(0,k-1))
		values1.apply(lambda x: random.randint(1,5))
		
		E1 = Mat(src1,dest1,values1,k,n)
		E = E.__add__(E1)
	
	#######################################################################	
	
	# Obtain the C matrix as a submatrix of E. Transpose operation for making 
	# distance calculation easier.

	vec1 = Vec.range(0,k)
	vec2 = Vec.range(0,k)
	
	C = E.copy()
	C = Mat._toMat(E._m_.SubsRef(vec1._v_,vec2._v_))
	#print Mat._toMat(E._m_.SubsRef(vec1._v_,vec2._v_))
	C.transpose()
	
	# This is the distance matrix to obtain the shift between centroids in two 
	# successive iterations.	
	vec2.randPerm()
	zeros = Vec.zeros(k)
	#Cold = Mat(vec1,vec2,zeros,k,k)	
	Cold = C.copy()

	print "E:",E

	# Clustering algorithm begins here. i represent the number of iterations executed.
	while 1:	
		
		# Creating the semiring for matrix multiplication to calculate the distance.
		addFn = lambda x,y: ((x)+(y))
		mulFn = lambda x,y: (x-y)*(x-y)
		sR    	   = sr(add,mul)			

		# Obtain the distance matrix.	
		D          = E.SpGEMM(C,semiring=sR)
		#print "D:",D
		
		# Obtain the minimum for every row.
		minvec     = D.reduce(Mat.Row,min,init=1e308)		
	
		#print "minvec:", minvec
		# Calculate a boolean matrix, where a(i,j) = 1 if the shortest centroid to ith
		# row is the jth centroid.

		B = D.copy()
		B.scale(minvec,lambda x,y : int(x==y),dir = Mat.Row)		
		
		#print "B:",B
		
		# Total number of vectors in each cluster. This is useful for calculating the new
		# centroid matrix.
		total = B.reduce(Mat.Column,lambda x,y: x + y)
		
		#print "total:", total
		# Do ET * B and scale it using the total array to get the new C.
		E.transpose()
		C = E.SpGEMM(B,semiring=sr_plustimes)
		#print "Cnew:", C
		E.transpose()
		#print 'C',C
		C.scale(total,lambda x,y: y != 0 and x/y or 0,dir = Mat.Column)
		#print "Cnew:", C
	
		S = C.eWiseApply(Cold, lambda x,y: (x-y)*(x-y))
		S._prune(lambda x : x==0)
		
		if S.getnnn() == 0:
			break

		#Dist = Cold.SpGEMM(C,semiring=sR)
		Cold = C.copy()
		
		#print 'B'
		#print B
		#print Dist
		
	print B

	return None

def _cluster_spectral(self):
	"""
	Performs Spectral Clustering.
	"""

	# self is a DiGraph
	#This code was directly copied from bfs -- Don't know what this exactly means.
	
	if not self.isObj() and self._hasFilter(self):
		raise NotImplementedError, 'DiGraph(element=default) with filters not supported'
	if self.isObj():
		#tmpG = self.copy()._toDiGraph()
		matrix = self.copy(element=0).e
	else:
		matrix = self.e

	self._kmeans(matrix)
	
	return None

DiGraph._cluster_spectral = _cluster_spectral
DiGraph._kmeans = _kmeans
