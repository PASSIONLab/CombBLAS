import numpy as np
import scipy as sc
import scipy.sparse as sp
import pyCombBLAS as pcb
import PyCombBLAS as PCB
import feedback

class Graph:
	#ToDo: privatize .spm name (to .__spmat)
	#ToDo: implement bool, not, _sprand

	#print "in Graph"

	def __init__(self, edgev, size):
		# include edges in both directions
		# NEW:  not clear that any of the SpParMat constructors actually do the following, so
		#   may be a new function
		self.spm = spm.SpParMat(edgev, size=size);

	def __len__(self):
		return self.spm.nvert();

	# NOTE:  no shape() for a graph;  use nvert/nedge instead	
	#def shape(self):
	#	return (self.spm.nvert(), self.spm.nvert());

	#FIX:  should only return 1 of the 2 directed edges for simple graphs
	def toEdgeV(self):		
		[ij, v] = self._toVectors(self.spm);
		return EdgeV(ij, v);

	@staticmethod
	def _toVectors(spmat):		# similar to toEdgeV, except returns arrays
		[i, j] = spmat.nonzero();		# NEW:  don't see a way to do this with current SpParMat
		v = spmat[i,j];				# not NEW:  looks like already supported by general indexing
		return ((i,j), v)

	def nedge(self):
		return self.spm.nedge();

	def nvert(self):
		return self.spm.nvert();

	def degree(self):
		tmp = spm.reduce(self._spones(self.spm), 0, '+');	# FIX: syntax
		return tmp;

	@staticmethod
	def _spones(spmat):		
		[nr, nc] = spmat.shape();
		[ij, ign] = Graph._toEdgeV(spmat);
		return Graph.Graph(ij, spv.ones(len(ign)));

	@staticmethod
	def _sub2ind(size, row, col):		# ToDo:  extend to >2D
		(nr, nc) = size;
		ndx = row + col*nr;
		return ndx

	@staticmethod
	def _SpMV_times_max(X,Y):		# ToDo:  extend to 2-/3-arg versions 
		[nrX, ncX] = X.shape()
		[nrY, ncY] = Y.shape()
		if ncX != nrY:
			raise ValueError, "Inner dimensions of X and Y do not match"
		if ncY > 1:
			raise ValueError, "Y must be a column vector"

		Z = spm.SpMV_SelMax(X, Y);	# assuming function creates its own output array

		return Z
		

class EdgeV:
	# NOTE:  vertex numbers go from 1 to N unlike Python's 0 to N-1 


	# For now, vertex endpoints are each a SpParVec, as is values
	# ToDo:  create a way for a client edge-vector to become a distributed edge-vector
	def __init__(self, verts, values):
		error = False;
		if type(values).__name__ <> 'SpParVec':
			error = True;
		for i in range(len(verts)):		# 'len' overloaded for SPVs
			if type(verts[i]).__name__ <> 'SpParVec':
				error = True;
		if error:
			raise ValueError('inputs must be SpParVecs')
		if len(verts[0]) <> len(values):
			raise ValueError('length of vertex and values vectors must be the same')
		self.__verts = verts;
		self.__values = values;


	#FIX: inconsistency between len and getitem;  len returns #verts;  getitem[0] returns verts
	def __len__(self):
		return self.__verts.nvert();

	def __getitem__(self, i):
		if i == 0:
			return self.__verts;
		if i == 1:
			return self.__values;
		raise ValueError('index out of range');
		return;

	def nvert(self):
		return self.__verts.nvert();

	def nvalues(self):
		return self.__values.nvalues();

	def verts(self):
		return self.__verts;

	def values(self):
		return self.__values;
		

#	VertexV class

#class VertexV:
#	print "in VertexV"
#
#	def __init__(self, ndces):
#		print "VertexV __init__ not implemented"
#		pass;


#	def __len__(self):
#		return self.__verts.nvert();


sendFeedback = feedback.sendFeedback;

