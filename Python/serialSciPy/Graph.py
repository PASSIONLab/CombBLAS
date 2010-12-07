import numpy as np
import scipy as sc
import scipy.sparse as sp

class Graph:
	#ToDo: privatize .spmat name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	print "in Graph"

	def __init__(self, edgev, size):
		#print "in Graph/__init__"
		# include edges in both directions
		self.spmat = sp.csr_matrix((edgev[1],edgev[0]), shape=(size)) + sp.csr_matrix(((edgev[1][1],edgev[1][0]), edgev[0]), shape=(size));

	def __len__(self):
		#print "in Graph/len"
		return int(np.max(self.spmat.get_shape()));

	#FIX:  should only return 1 of the 2 directed edges for simple graphs
	def toEdgeV(self):		
		[ij, v] = self._toArrays(self.spmat);
		return EdgeV(ij, v);

	@staticmethod
	def _toArrays(spmat):		# similar to toEdgeV, except returns arrays
		[i, j] = spmat.nonzero();
		v1 = spmat[i,j].getA1();
		v = np.reshape(v1, (len(v1),));
		i = i.reshape((len(i),));
		j = j.reshape((len(j),));
		return ((i,j), v)

	def nedges(self):
		return self.spmat.getnnz();

	def nverts(self):
		return np.shape(self.spmat)[0];

	def degree(self):
		tmp = self._spones(self.spmat);
		return np.asarray(tmp.sum(0))

	@staticmethod
	def _spones(spmat):		
		[nr, nc] = spmat.get_shape();
		[ij, v] = Graph._toArrays(spmat);
		[i, j] = ij;				# ToDo:  suitable only for simple graphs
		i = i.flatten();
		j = j.flatten();
		return sp.csr_matrix((sc.tile(1, (sc.shape(i)[0],)), (i, j)), shape = (nr,nc))

	@staticmethod
	def _sub2ind(size, row, col):		# ToDo:  extend to >2D
		(nr, nc) = size;
		ndx = row + col*nr;
		return ndx

	@staticmethod
	def _SpMV_times_max(X,Y):		# ToDo:  extend to 2-/3-arg versions 
		[nrX, ncX] = X.get_shape()
		shY = sc.shape(Y)
		nrY = shY[0];
		if ncX != nrY:
			print "Inner dimensions of X and Y do not match"
 			return
		if len(shY) > 1 and shY[1] > 1:
			print "Y must be a column vector"
			return
		#ToDo:  make this work with a truly sparse Z
		Z = sp.csr_matrix(-np.Inf * np.ones((nrX, ),np.dtype(X))).T;
		for i in range(nrX):
 			for k in range(ncX):
 				Z[i,0] = max( (Z[i])[0,0], X[i,k]*Y[k,] );
 				#was:  Z[i,0] = min( Z[i], X[i,k]*Y[k,] );
			foo = Z[i];	#debug

		return Z
		
	@staticmethod
	def _SpMV_sel_max(X,Y):		# ToDo:  extend to 2-/3-arg versions 
		[nrX, ncX] = X.get_shape()
		shY = sc.shape(Y)
		nrY = shY[0];
		if ncX != nrY:
			print "Inner dimensions of X and Y do not match"
 			return
		if len(shY) > 1 and shY[1] > 1:
			print "Y must be a column vector"
			return
		#ToDo:  make this work with a truly sparse Z
		Z = sp.csr_matrix(-np.Inf * np.ones((nrX, ),np.dtype(X))).T;
		for i in range(nrX):
 			for k in range(ncX):
				if X[i,k] <> 0 and Y[k,] <> 0:
 					Z[i,0] = max( (Z[i])[0,0], Y[k,] );
 					#was:  Z[i,0] = min( Z[i], X[i,k]*Y[k,] );
			#foo = Z[i];	#debug

		return Z
		

class EdgeV:
	print "in EdgeV"


	# vertex endpoints are each a tuple of flattened NumPy column vectors
	def __init__(self, verts, values):
		error = False;
		if type(values).__name__ <> 'ndarray':
			error = True;
		if len(np.shape(values)) <> 1:
			error = True;
		for i in range(len(verts)):
			if type(verts[i]).__name__ <> 'ndarray':
				error = True;
			if len(np.shape(verts[i])) <> 1:
				error = True;
		if error:
			raise ValueError('inputs must be NumPy flattened arrays or tuples of flattened arrays')
		if np.shape(verts[0])[0] <> np.shape(values)[0]:
			raise ValueError('length of vertex and values arrays must be the same')
		self.__verts = verts;
		self.__values = values;

	def __len__(self):
		return len(self.__verts[0]);

	def __getitem__(self, i):
		if i == 0:
			return self.__verts;
		if i == 1:
			return self.__values;
		raise ValueError('index out of range');
		return;

	def vertsShape(self):
		return np.shape(self.__verts[0]);

	def valuesShape(self):
		return np.shape(self.__values);

	def verts(self):
		return self.__verts;

	def values(self):
		return self.__values;
		

#	No VertexV class for now

#class VertexV:
#	print "in VertexV"
#
#	def __init__(self, ndces):

