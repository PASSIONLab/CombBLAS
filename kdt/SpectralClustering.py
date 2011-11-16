from DiGraph import DiGraph
from Vec import Vec
from Mat import Mat
from Util import *

import random
import math

def kmeans(E,k):
	
	'''
	kmeans(E,k) : Performs K-means clustering on data stored in matrix format.
	The function takes on two arguments - E, the matrix representation of data to be
	clustered and k, the number of clusters to be formed. The E matrix is an n*m matrix
	with n rows representing n data points, and m columns being the m attributes of
	each point. kmeans returns a vector b where, each vertex belongs to cluster b[i]
	'''	

	# Get the matrix dimensions
	n = E.nrow()
	m = E.ncol()

	# Obtain the Centroid matrix (C) as a submatrix of E. Transpose operation for making 
	# distance calculation easier.

	vec1 = Vec.range(0,n)
	vec1.randPerm()
	vec1 = vec1[vec1.range(k)]
	vec2 = Vec.range(0,m)
	
	C = E[vec1,vec2]
	#C = Mat._toMat(E._m_.SubsRef(vec1._v_,vec2._v_))
	C.transpose()
	
	# Etrans is used later during the algorithm to find the sum of all data points in a centroid.
	Etrans = E.copy()
	Etrans.transpose()

	# This is the distance matrix to obtain the shift between centroids in two 
	# successive iterations.	
	Cold = C.copy()

	# Clustering algorithm begins here. i represent the number of iterations executed.
	while 1:	
		
		# Creating the semiring for matrix multiplication to calculate the distance.
		addFn = lambda x,y: ((x)+(y))
		mulFn = lambda x,y: (x-y)*(x-y)
		sR    	   = sr(addFn,mulFn)			

		# Obtain the distance matrix.	
		D          = E.SpGEMM(C,semiring=sR)
		
		# Obtain the minimum for every row.
		minvec     = D.reduce(Mat.Row,min,init=1e308)		
	
		# Calculate a boolean matrix, where a(i,j) = 1 if the shortest centroid to ith
		# row is the jth centroid.

		B = D.copy()
		B.scale(minvec,lambda x,y : int(x==y),dir = Mat.Row)		
		
		# Total number of vectors in each cluster. This is useful for calculating the new
		# centroid matrix.
		total = B.reduce(Mat.Column,lambda x,y: x + y)
		
		# Do ET * B and scale it using the total array to get the new C.
		C = Etrans.SpGEMM(B,semiring=sr_plustimes)

		C.scale(total,lambda x,y: y != 0 and x/y or 0,dir = Mat.Column)
		# The shift matrix S calculates the distance between corresponding centroids
		# of the current and previous iteration. If S = 0, terminate the algorithm.	

		S = C.eWiseApply(Cold, lambda x,y: (x-y)*(x-y))
		S._prune(lambda x : x==0)
		
		if S.nnn() == 0:
			break
		
		# Copy the current centroid into Cold for the next iteration.
		Cold = C.copy()
	

	# Convert the B matrix to a b vector. b[i] represents the cluster i belongs to.
	# This is achieved by multiplying B with a vector [0,1,..,k-1]. The semiring 
	# can be the scalar addition and multiplication, but it cannot handle the case
	# when a data point may belong to more than 1 cluster.

	rangevec = Vec.range(0,k,sparse=True)
			
	def addFn(x,y):
		return  x == 0 and y or x
	def mulFn(x,y):
		return x*y
	
	b = B.SpMV(rangevec,semiring = sr(addFn,mulFn))
	return b

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

	
	return None

DiGraph._cluster_spectral = _cluster_spectral
Mat.kmeans = kmeans
