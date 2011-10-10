import math
import pyCombBLAS as pcb
import Graph as gr
from Graph import ParVec, SpParVec, master
from DiGraph import DiGraph

import time

class HyGraph(gr.Graph):
	InOut = 1

	# NOTE:  vertices are associated with columns;  edges are associated
	#	with rows
	def __init__(self,*args):
		"""
		creates a new HyGraph instance.  Can be called in one of the 
		following forms:

	HyGraph():  creates a HyGraph instance with no vertices or edges.  Useful
		as input for genGraph500Edges.

	HyGraph(edgeNumV, incidentVertexV, weightV, nvert, [nedge])
		create a HyGraph instance.  Each element of the first (ParVec) 
		argument denotes an edge number;  each corresponding element of
		the second (ParVec) argument denotes the number of a vertex to 
		which the edge is incident.  A single edge number can occur an 
		arbitrary number of times in the first argument; all the vertices 
		denoted for the same edge collectively define the hyperedge.  
		The third argument may be a ParVec instance or a scalar;  if a 
		ParVec, then its corresponding elements denote the weight of the
		edge for its incident vertex;  if a scalar, then it denotes the 
		weight of all edges for all incident vertices.  The nvert argument
		denotes the number of vertices in the resulting HyGraph (not all
		vertices must have incident edges).  The optional nedge argument
		denotes the number of edges in the resulting HyGraph (not all edges
		must be incident vertices).

		Input Arguments:
			edgeNumV:  a ParVec containing integers denoting the 
			    edge number of each edge.
			incidentVertexV:  a ParVec containing integers denoting 
			    incident vertices.
			weightV:  a ParVec containing double-precision floating-
			    point numbers denoting the weight of each edge incident
			    to a vertex, or a double-precision floating-point 
			    scalar denoting the weight of all edges for all
			    incident vertices.
			nvert:  an integer scalar denoting the number of vertices.
			nedge:  an optional integer scalar argument denoting the 
			    number of edges

		Output Argument:  
			ret:  a HyGraph instance

		SEE ALSO:  toParVec
		"""
		if len(args) == 0:
			self._spm = pcb.pySpParMat()
			self._spmT = pcb.pySpParMat()
		elif len(args) == 4:
			[i,j,v,nv] = args
			if len(i) != len(j):
				raise KeyError, 'first two arguments must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v)
			ne = int(i.max()) + 1
			if j.max() > nv-1:
				raise KeyError, 'at least one second index greater than #vertices'
			self._spm = pcb.pySpParMat(ne,nv,i._dpv,j._dpv,v._dpv)
			self._spmT = self._spm.copy()
			self._spmT.Transpose()
		elif len(args) == 5:
			[i,j,v,nv,ne] = args
			if len(i) != len(j):
				raise KeyError, 'first two arguments must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v)
			if i.max() > ne-1:
				raise KeyError, 'at least one first index greater than #edges'
			if j.max() > nv-1:
				raise KeyError, 'at least one second index greater than #vertices'
			self._spm = pcb.pySpParMat(ne,nv,i._dpv,j._dpv,v._dpv)
			self._spmT = self._spm.copy()
			self._spmT.Transpose()
		else:
			raise NotImplementedError, "only 0-, 4- and 5-argument cases supported"

#	def __add__(self, other):
#		"""
#		adds corresponding edges of two DiGraph instances together,
#		resulting in edges in the result only where an edge exists in at
#		least one of the input DiGraph instances.
#		"""
#		if type(other) == int or type(other) == long or type(other) == float:
#			raise NotImplementedError
#		if self.nvert() != other.nvert():
#			raise IndexError, 'Graphs must have equal numbers of vertices'
#		elif isinstance(other, DiGraph):
#			ret = self.copy()
#			ret._spm += other._spm
#			#ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.plus());  # only adds if both mats have nonnull elems!!
#		return ret
#
#	def __div__(self, other):
#		"""
#		divides corresponding edges of two DiGraph instances together,
#		resulting in edges in the result only where edges exist in both
#		input DiGraph instances.
#		"""
#		if type(other) == int or type(other) == long or type(other) == float:
#			ret = self.copy()
#			ret._spm.Apply(pcb.bind2nd(pcb.divides(),other))
#		elif self.nvert() != other.nvert():
#			raise IndexError, 'Graphs must have equal numbers of vertices'
#		elif isinstance(other,DiGraph):
#			ret = self.copy()
#			ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.divides())
#		else:
#			raise NotImplementedError
#		return ret
#
#	def __getitem__(self, key):
#		"""
#		implements indexing on the right-hand side of an assignment.
#		Usually accessed through the "[]" syntax.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			key:  one of the following forms:
#			    - a non-tuple denoting the key for both dimensions
#			    - a tuple of length 2, with the first element denoting
#			        the key for the first dimension and the second 
#			        element denoting for the second dimension.
#			    Each key denotes the out-/in-vertices to be addressed,
#			    and may be one of the following:
#				- an integer scalar
#				- the ":" slice denoting all vertices, represented
#				  as slice(None,None,None)
#				- a ParVec object containing a contiguous range
#				  of monotonically increasing integers 
#		
#		Output Argument:
#			ret:  a DiGraph instance, containing the indicated vertices
#			    and their incident edges from the input DiGraph.
#
#		SEE ALSO:  subgraph
#		"""
#		#ToDo:  accept slices for key0/key1 besides ParVecs
#		if type(key)==tuple:
#			if len(key)==1:
#				[key0] = key; key1 = -1
#			elif len(key)==2:
#				[key0, key1] = key
#			else:
#				raise KeyError, 'Too many indices'
#		else:
#			key0 = key;  key1 = key
#		if type(key0) == int or type(key0) == long or type(key0) == float:
#			tmp = ParVec(1)
#			tmp[0] = key0
#			key0 = tmp
#		if type(key1) == int or type(key0) == long or type(key0) == float:
#			tmp = ParVec(1)
#			tmp[0] = key1
#			key1 = tmp
#		if type(key0)==slice and key0==slice(None,None,None):
#			key0mn = 0; 
#			key0tmp = self.nvert()
#			if type(key0tmp) == tuple:
#				key0mx = key0tmp[0] - 1
#			else:
#				key0mx = key0tmp - 1
#		else:
#			key0mn = int(key0.min()); key0mx = int(key0.max())
#			if len(key0)!=(key0mx-key0mn+1) or not (key0==ParVec.range(key0mn,key0mx+1)).all():
#				raise KeyError, 'Vector first index not a range'
#		if type(key1)==slice and key1==slice(None,None,None):
#			key1mn = 0 
#			key1tmp = self.nvert()
#			if type(key1tmp) == tuple:
#				key1mx = key1tmp[1] - 1
#			else:
#				key1mx = key1tmp - 1
#		else:
#			key1mn = int(key1.min()); key1mx = int(key1.max())
#			if len(key1)!=(key1mx-key1mn+1) or not (key1==ParVec.range(key1mn,key1mx+1)).all():
#				raise KeyError, 'Vector second index not a range'
#		[i, j, v] = self.toParVec()
#		sel = ((i >= key0mn) & (i <= key0mx) & (j >= key1mn) & (j <= key1mx)).findInds()
#		newi = i[sel] - key0mn
#		newj = j[sel] - key1mn
#		newv = v[sel]
#		ret = DiGraph(newi, newj, newv, key0mx-key0mn+1, key1mx-key1mn+1)
#		return ret
#
#	def __iadd__(self, other):
#		if type(other) == int or type(other) == long or type(other) == float:
#			raise NotImplementedError
#		if self.nvert() != other.nvert():
#			raise IndexError, 'Graphs must have equal numbers of vertices'
#		elif isinstance(other, DiGraph):
#			#dead tmp = pcb.EWiseApply(self._spm, other._spm, pcb.plus())
#			self._spm += other._spm
#		return self
#
#	def __imul__(self, other):
#		if type(other) == int or type(other) == long or type(other) == float:
#			self._spm.Apply(pcb.bind2nd(pcb.multiplies(),other))
#		elif isinstance(other,DiGraph):
#			self._spm = pcb.EWiseApply(self._spm,other._spm, pcb.multiplies())
#		else:
#			raise NotImplementedError
#		return self
#
#	def __mul__(self, other):
#		"""
#		multiplies corresponding edges of two DiGraph instances together,
#		resulting in edges in the result only where edges exist in both
#		input DiGraph instances.
#
#		"""
#		if type(other) == int or type(other) == long or type(other) == float:
#			ret = self.copy()
#			ret._spm.Apply(pcb.bind2nd(pcb.multiplies(),other))
#		elif self.nvert() != other.nvert():
#			raise IndexError, 'Graphs must have equal numbers of vertices'
#		elif isinstance(other,DiGraph):
#			ret = self.copy()
#			ret._spm = pcb.EWiseApply(self._spm,other._spm, pcb.multiplies())
#		else:
#			raise NotImplementedError
#		return ret
#
#	def __neg__(self):
#		ret = self.copy()
#		ret._spm.Apply(pcb.negate())
#		return ret
#
	#ToDo:  put in method to modify _REPR_MAX
	_REPR_MAX = 100
	def __repr__(self):
		if self.nvert() == 0:
			return 'Null HyGraph object'
		if self.nvert()==1:
			[i, j, v] = self.toParVec()
			if len(v) > 0:
				print "%d %f" % (v[0], v[0])
			else:
				print "%d %f" % (0, 0.0)
		else:
			[i, j, v] = self.toParVec()
			[iT, jT, vT] = self.toParVec(shadow=True)	#DEBUG
			if len(i) < self._REPR_MAX:
				print i,j,v
				print iT, jT, vT			#DEBUG
		return ' '
#
#	def _SpMM(self, other):
#		"""
#		"multiplies" two DiGraph instances together as though each was
#		represented by a sparse matrix, with rows representing out-edges
#		and columns representing in-edges.
#		"""
#		selfnv = self.nvert()
#		if type(selfnv) == tuple:
#			[selfnv1, selfnv2] = selfnv
#		else:
#			selfnv1 = selfnv; selfnv2 = selfnv
#		othernv = other.nvert()
#		if type(othernv) == tuple:
#			[othernv1, othernv2] = othernv
#		else:
#			othernv1 = othernv; othernv2 = othernv
#		if selfnv2 != othernv1:
#			raise ValueError, '#in-vertices of first graph not equal to #out-vertices of the second graph '
#		ret = DiGraph()
#		ret._spm = self._spm.SpMM(other._spm)
#		return ret
#
	def npin(self):
		"""
		calculates the cardinality of each edge of the passed HyGraph 
		instance.

		Input Arguments:
			self:  a HyGraph instance

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    cardinality of the corresponding edge.

		SEE ALSO:  rank, antirank 
		"""
		if self.nedge() == 0:
			return ParVec.zeros(self.nedge())
		ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
		return ParVec.toParVec(ret)

#	def copy(self):
#		"""
#		creates a deep copy of a DiGraph instance.
#
#		Input Argument:
#			self:  a DiGraph instance.
#
#		Output Argument:
#			ret:  a DiGraph instance containing a copy of the input.
#		"""
#		ret = DiGraph()
#		ret._spm = self._spm.copy()
#		return ret
		
	def degree(self):
		"""
		calculates the degree of each vertex of the passed HyGraph instance.

		Input Arguments:
			self:  a HyGraph instance

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    degree of the corresponding vertex.

		SEE ALSO:  sum 
		"""
		if self.nedge() == 0:
			return ParVec.zeros(self.nvert())
		ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
		return ParVec.toParVec(ret)
#
#	# in-place, so no return value
#	def removeSelfLoops(self):
#		"""
#		removes all edges whose source and destination are the same
#		vertex, in-place in a DiGraph instance.
#
#		Input Argument:
#			self:  a DiGraph instance, modified in-place.
#
#		"""
#		if self.nvert() > 0:
#			self._spm.removeSelfLoops()
#		return
#
#	@staticmethod
#	def fullyConnected(n,m=None):
#		"""
#		creates edges in a DiGraph instance that connects each vertex
#		directly to every other vertex.
#
#		Input Arguments:
#			n:  an integer scalar denoting the number of vertices in
#			    the graph that may potentially have out-edges.
#			m:  an optional argument, which if specified is an integer
#			    scalar denoting the number of vertices in the graph
#			    that may potentially have in-edges.
#
#		Output Argument:
#			ret:  a DiGraph instance with directed edges from each
#			    vertex to every other vertex. 
#		"""
#		if m == None:
#			m = n
#		i = (ParVec.range(n*m) % n).floor()
#		j = (ParVec.range(n*m) / n).floor()
#		v = ParVec.ones(n*m)
#		ret = DiGraph(i,j,v,n,m)
#		return ret
#
#	def genGraph500Edges(self, scale):
#		"""
#		creates edges in a DiGraph instance that meet the Graph500 
#		specification.  The graph is symmetric. (See www.graph500.org 
#		for details.)
#
#		Input Arguments:
#			self:  a DiGraph instance, usually with no edges
#			scale:  an integer scalar representing the logarithm base
#			    2 of the number of vertices in the resulting DiGraph.
#			    
#		Output Argument:
#			ret:  a double-precision floating-point scalar denoting
#			    the amount of time to converted the created edges into
#			    the DiGraph instance.  This equals the value of Kernel 1
#			    of the Graph500 benchmark.
#		"""
#		elapsedTime = self._spm.GenGraph500Edges(scale)
#	 	return elapsedTime
#
	#in-place, so no return value
	def invertEdgesVertices(self):
		"""
		inverts in-place the meaning of vertex and edge in a HyGraph 
		instance.  Each edge incident to a set of vertices becomes a vertex
		incident to a set of edges and vice versa.

		Input Argument:
			self:  a HyGraph instance, modified in-place.
		"""
		tmp = self._spm
		self._spm = self._spmT
		self._spmT = tmp

	@staticmethod
	def load(fname):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a HyGraph instance. The
		lines of the file are interpreted as edge-number, vertex-number,
		and optional weight, just like the HyGraph constructor.

		Input Argument:
			fname:  a filename from which the HyGraph data will be loaded.
		Output Argument:
			ret:  a HyGraph instance containing the graph represented
			    by the file's contents.

		NOTE:  The Matrix Market format numbers edges and vertices from 1 to
		N.  Python and KDT number edges and vertices from 0 to N-1.  The 
		load method makes this conversion while reading the data and creating
		the graph.

		SEE ALSO:  HyGraph, save, UFget
		"""
		# Verify file exists.
		file = open(fname, 'r')
		file.close()
		
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = HyGraph()
		ret._spm = pcb.pySpParMat()
		ret._spm.load(fname)
		ret._spmT = ret._spm.copy()
		ret._spmT.Transpose()
		return ret

#	def max(self, dir=InOut):
#		"""
#		finds the maximum weights of the appropriate edges of each vertex 
#		of the passed DiGraph instance.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			dir:  a direction of edges over which to find the maximum,
#			    with choices being DiGraph.Out (default), DiGraph.In, or 
#			    DiGraph.InOut.
#
#		Output Argument:
#			ret:  a ParVec instance with each element containing the
#			    maximum of the weights of the corresponding vertex.
#
#		SEE ALSO:  degree, min 
#		"""
#		#ToDo:  is default to InOut best?
#		if dir == DiGraph.InOut:
#			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.max())
#			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.max())
#			return ParVec.toParVec(tmp1+tmp2)
#		elif dir == DiGraph.In:
#			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.max())
#			return ParVec.toParVec(ret)
#		elif dir == DiGraph.Out:
#			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.max())
#			return ParVec.toParVec(ret)
#		else:
#			raise KeyError, 'Invalid edge direction'
#
#	def min(self, dir=InOut):
#		"""
#		finds the minimum weights of the appropriate edges of each vertex 
#		of the passed DiGraph instance.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			dir:  a direction of edges over which to find the minimum,
#			    with choices being DiGraph.Out (default), DiGraph.In, or 
#			    DiGraph.InOut.
#
#		Output Argument:
#			ret:  a ParVec instance with each element containing the
#			    minimum of the weights of the corresponding vertex.
#
#		SEE ALSO:  degree, max 
#		"""
#		#ToDo:  is default to InOut best?
#		if dir == DiGraph.InOut:
#			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.min())
#			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.min())
#			return ParVec.toParVec(tmp1+tmp2)
#		elif dir == DiGraph.In:
#			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.min())
#			return ParVec.toParVec(ret)
#		elif dir == DiGraph.Out:
#			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.min())
#			return ParVec.toParVec(ret)
#
#	def mulNot(self, other):
#		"""
#		multiplies corresponding edge weights of two DiGraph instances,
#		taking the logical not of the second argument before doing the 
#		multiplication.  In effect, each nonzero edge of the second
#		argument deletes its corresponding edge of the first argument.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			other:  another DiGraph instance
#
#		Output arguments:
#			ret:  a DiGraph instance 
#		"""
#		if self.nvert() != other.nvert():
#			raise IndexError, 'Graphs must have equal numbers of vertices'
#		else:
#			ret = DiGraph()
#			ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.multiplies(), True)
#		return ret
#

	def _nnn(self):
		"""
		returns the number of nonnulls in the underlying sparse-matrix
		representation, which is typically different from the number of
		hyperedges.

		SEE ALSO:  nedge
		"""
		return self._spm.getnnz()

	def nedge(self):
		"""
		returns the number of hyperedges in the passed HyGraph instance.

		SEE ALSO: nvert, degree  
		"""
		return self._spm.getnrow()


	def nvert(self):
		"""
		returns the number of vertices in the given HyGraph instance.

		Input Argument:
			self:  a HyGraph instance.

		Output Argument:
			ret:  an integer denoting the number of vertices

		SEE ALSO:  nedge, degree
		"""
		return self._spm.getncol()
#
#	##in-place, so no return value
#	#def ones(self):
#	#	"""
#	#	sets every edge in the graph to the value 1.
#
#	#	Input Argument:
#	#		self:  a DiGraph instance, modified in place.
#
#	#	Output Argument:
#	#		None.
#
#	#	SEE ALSO:  set
#	#	"""
#	#	self._spm.Apply(pcb.set(1))
#	#	return

	def antirank(self):
		"""
		calculates the antirank (the minimum cardinality of any edge) of the
		passed HyGraph instance.

		Input Arguments:
			self:  a HyGraph instance

		Output Argument:
			ret:  the antirank of the HyGraph instance

		SEE ALSO:  npin, rank 
		"""
		return self.npin().min()

	def rank(self):
		"""
		calculates the rank (the maximum cardinality of any edge) of the
		passed HyGraph instance.

		Input Arguments:
			self:  a HyGraph instance

		Output Argument:
			ret:  the rank of the HyGraph instance

		SEE ALSO:  npin, antirank 
		"""
		return self.npin().max()

#
#	def save(self, fname):
#		"""
#		saves the contents of the passed DiGraph instance to a file named
#		fname in the Coordinate Format of the Matrix Market Exchange Format.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			fname:  a filename to which the DiGraph data will be saved.
#
#		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
#		N.  Python and KDT number vertex numbers from 0 to N-1.  The save
#		method makes this conversion while writing the data.
#
#		SEE ALSO:  load, UFget
#		"""
#		self._spm.save(fname)
#		return
#
#	#in-place, so no return value
#	def scale(self, other, dir=Out):
#		"""
#		multiplies the weights of the appropriate edges of each vertex of
#		the passed DiGraph instance in-place by a vertex-specific scale 
#		factor.
#
#		Input Arguments:
#			self:  a DiGraph instance, modified in-place
#			dir:  a direction of edges to scale, with choices being
#			    DiGraph.Out (default) or DiGraph.In.
#
#		Output Argument:
#			None.
#
#		SEE ALSO:  * (DiGraph.__mul__), mulNot
#		"""
#		if not isinstance(other,gr.SpParVec):
#			raise KeyError, 'Invalid type for scale vector'
#		selfnv = self.nvert()
#		if type(selfnv) == tuple:
#			[selfnv1, selfnv2] = selfnv
#		else:
#			selfnv1 = selfnv; selfnv2 = selfnv
#		if dir == DiGraph.In:
#			if selfnv2 != len(other):
#				raise IndexError, 'graph.nvert()[1] != len(scale)'
#			self._spm.ColWiseApply(other._spv, pcb.multiplies())
#		elif dir == DiGraph.Out:
#			if selfnv1 != len(other):
#				raise IndexError, 'graph.nvert()[1] != len(scale)'
#			self._T()
#			self._spm.ColWiseApply(other._spv,pcb.multiplies())
#			self._T()
#		else:
#			raise KeyError, 'Invalid edge direction'
#		return
#
	##in-place, so no return value
	def set(self, value):
		"""
		sets the value of every edge incident to every vertex in the 
		graph to the given value.

		Input Arguments:
			self:  a HyGraph instance, modified in place.
			value:  a scalar integer or double-precision floating-
			    point value.

		Output Argument:
			None.

		SEE ALSO:  ones
		"""
		self._spm.Apply(pcb.set(value))
		self._spmT.Apply(pcb.set(value))
		return
#
#	def subgraph(self, ndx1, ndx2=None):
#		"""
#		creates a new DiGraph instance consisting of only designated vertices 
#		of the input graph and their indicent edges.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			ndx1:  an integer scalar or a ParVec of consecutive vertex
#			    numbers to be included in the subgraph along with edges
#			    starting from these vertices.
#			ndx2:  an optional argument; if specified, is an integer
#			    scalar or a ParVec of consecutive vertex numbers to
#			    be included in the subgraph along with any edges ending
#			    at these vertices.
#			 
#		Output Argument:
#			ret:  a DiGraph instance composed of the selected vertices
#			    and their incident edges.
#
#		SEE ALSO:  DiGraph.__getitem__
#		"""
#		if type(ndx2) == type(None) and (ndx2 == None):
#			ndx2 = ndx1
#		ret = self[ndx1, ndx2]
#		return ret
#
#	def sum(self, dir=Out):
#		"""
#		adds the weights of the appropriate edges of each vertex of the
#		passed DiGraph instance.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			dir:  a direction of edges to sum, with choices being
#			    DiGraph.Out (default), DiGraph.In, or DiGraph.InOut.
#
#		Output Argument:
#			ret:  a ParVec instance with each element containing the
#			    sum of the weights of the corresponding vertex.
#
#		SEE ALSO:  degree 
#		"""
#		if dir == DiGraph.InOut:
#			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity())
#			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity())
#			return ParVec.toParVec(tmp1+tmp2)
#		elif dir == DiGraph.In:
#			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity())
#			return ParVec.toParVec(ret)
#		elif dir == DiGraph.Out:
#			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity())
#			return ParVec.toParVec(ret)
#		else:
#			raise KeyError, 'Invalid edge direction'
#
	_T = invertEdgesVertices

	# in-place, so no return value
	def toBool(self):
		"""
		converts the HyGraph instance in-place such that each edge has only
		a Boolean (True) value, thereby consuming less space and making
		some operations faster.

		Input Argument:
			self:  a HyGraph instance that is overwritten by the method

		Output Argument:
			None.
		"""
		if isinstance(self._spm, pcb.pySpParMat):
			self._spm  = pcb.pySpParMatBool(self._spm)
			self._spmT = pcb.pySpParMatBool(self._spmT)

	def toDiGraph(self):
		"""
		converts the HyGraph instance to its analogous DiGraph, where each
		hyperedge is replaced by a pair of directed edges between each pair
		of vertices in the hyperedge.
		"""
		ret = DiGraph();
		ret._spm = self._spmT.SpMM(self._spm)
		return ret

	def toParVec(self,shadow=False):
		"""
		decomposes a DiGraph instance to 3 ParVec instances, with each
		element of the first ParVec denoting the edge number, the 
		corresponding element of the second ParVec denoting the vertex to
		which the edge is incident, and the corresponding element of
		the third ParVec denoting the value or weight of the edge incident
		to that vertex.

		Input Argument:
			self:  a HyGraph instance

		Output Argument:
			ret:  a 3-element tuple with ParVec instances denoting the
			    edge number, vertex number, and weight, respectively.

		SEE ALSO:  HyGraph 
		"""
		ne = self._nnn()
		reti = ParVec(ne)
		retj = ParVec(ne)
		retv = ParVec(ne)
		if not shadow:
			self._spm.Find(reti._dpv, retj._dpv, retv._dpv)
		else:
			self._spmT.Find(reti._dpv, retj._dpv, retv._dpv)
		return (reti, retj, retv)

#	@staticmethod
#	def twoDTorus(n):
#		"""
#		constructs a DiGraph instance with the connectivity pattern of a 2D
#		torus;  i.e., each vertex has edges to its north, west, south, and
#		east neighbors, where the neighbor may be wrapped around to the
#		other side of the torus.  
#
#		Input Parameter:
#			nnodes:  an integer scalar that denotes the number of nodes
#			    on each axis of the 2D torus.  The resulting DiGraph
#			    instance will have nnodes**2 vertices.
#
#		Output Parameter:
#			ret:  a DiGraph instance with nnodes**2 vertices and edges
#			    in the pattern of a 2D torus. 
#		"""
#		N = n*n
#		nvec =   ((ParVec.range(N*4)%N) / n).floor()	 # [0,0,0,...., n-1,n-1,n-1]
#		nvecil = ((ParVec.range(N*4)%N) % n).floor()	 # [0,1,...,n-1,0,1,...,n-2,n-1]
#		north = gr.Graph._sub2ind((n,n),(nvecil-1) % n,nvec)	
#		south = gr.Graph._sub2ind((n,n),(nvecil+1) % n,nvec)
#		west = gr.Graph._sub2ind((n,n),nvecil, (nvec-1) % n)
#		east = gr.Graph._sub2ind((n,n),nvecil, (nvec+1) % n)
#		Ndx = ParVec.range(N*4)
#		northNdx = Ndx < N
#		southNdx = (Ndx >= N) & (Ndx < 2*N)
#		westNdx = (Ndx >= 2*N) & (Ndx < 3*N)
#		eastNdx = Ndx >= 3*N
#		col = ParVec.zeros(N*4)
#		col[northNdx] = north
#		col[southNdx] = south
#		col[westNdx] = west
#		col[eastNdx] = east
#		row = ParVec.range(N*4) % N
#		ret = DiGraph(row, col, 1, N)
#		return ret

#	# ==================================================================
#	#  "complex ops" implemented below here
#	# ==================================================================
#
#
#	#	creates a breadth-first search tree of a Graph from a starting
#	#	set of vertices.  Returns a 1D array with the parent vertex of 
#	#	each vertex in the tree; unreached vertices have parent == -1.
#	#	sym arg denotes whether graph is symmetric; if not, need to transpose
#	#
	def bfsTree(self, root):
		"""
		calculates a breadth-first search tree from the edges in the
		passed HyGraph, starting from the root vertex.  "Breadth-first"
		in the sense that all vertices reachable in step i are added
		to the tree before any of the newly-reachable vertices' reachable
		vertices are explored.  The returned tree (implied in the parent
		relationships) consists of simple edges, not the original 
		hyperedges of the HyGraph instance.

		Input Arguments:
			root:  an integer denoting the root vertex for the tree

		Input Arguments:
			parents:  a ParVec instance of length equal to the number
			    of vertices in the HyGraph, with each element denoting 
			    the vertex number of that vertex's parent in the tree.
			    The root is its own parent.  Unreachable vertices
			    have a parent of -1. 

		SEE ALSO: isBfsTree 
		"""
		parents = pcb.pyDenseParVec(self.nvert(), -1)
		fringeV = pcb.pySpParVec(self.nvert())
		parents[root] = root
		fringeV[root] = root
		while fringeV.getnee() > 0:
			fringeV.setNumToInd()
			fringeE = self._spm.SpMV_SelMax(fringeV)
			fringeV = self._spmT.SpMV_SelMax(fringeE)
			pcb.EWiseMult_inplacefirst(fringeV, parents, True, -1)
			parents[fringeV] = 0
			parents += fringeV
		return ParVec.toParVec(parents)
	

#		# returns tuples with elements
#		# 0:  True/False of whether it is a BFS tree or not
#		# 1:  levels of each vertex in the tree (root is 0, -1 if not reached)
	def isBfsTree(self, root, parents, sym=False):
		"""
		validates that a breadth-first search tree in the style created
		by bfsTree is correct.

		Input Arguments:
			root:  an integer denoting the root vertex for the tree
			parents:  a ParVec instance of length equal to the number
			    vertices in the HyGraph, with each element denoting 
			    the vertex number of that vertex's parent in the tree.
			    The root is its own parent.  Vertices unreachable
			    from the root have a parent of -1. 
		
		Output Arguments:
			ret:  a 2-element tuple.  The first element is an integer
			    whose value is 1 if the graph is a BFS tree and whose
			    value is the negative of the first test below that failed
			    if one of them failed.  If the graph is a BFS tree,
			    the second element of the tuple is a ParVec of length
			    equal to the number of vertices in the HyGraph, with
			    each element denoting the level in the tree at which
			    the vertex resides.  The root resides in level 0, its
			    direct neighbors in level 1, and so forth.  Unreachable 
			    vertices have a level value of -1.  If the graph is not
			    a BFS tree (one of the tests failed), the second
			    element of the tuple is None.
		
		Tests:
			The tests implement some of the Graph500 (www.graph500.org) 
			specification. (Some of the Graph500 tests also depend on 
			input edges.)
			1:  The tree does not contain cycles,  that every vertex
			    with a parent is in the tree, and that the root is
			    not the destination of any tree edge.
			2:  Tree edges are between vertices whose levels differ 
			    by exactly 1.

		SEE ALSO: bfsTree 
		"""
		diG = self.toDiGraph()
		diG.removeSelfLoops()
		diG.toBool()
		ret = diG.isBfsTree(root, parents)
		return ret
#	
#	# returns a Boolean vector of which vertices are neighbors
#	def neighbors(self, source, nhop=1, sym=False):
#		"""
#		calculates, for the given DiGraph instance and starting vertices,
#		the vertices that are neighbors of the starting vertices (i.e.,
#		reachable within nhop hops in the graph).
#
#		Input Arguments:
#			self:  a DiGraph instance
#			source:  a Boolean ParVec with True (1) in the positions
#			    of the starting vertices.  
#			nhop:  a scalar integer denoting the number of hops to 
#			    use in the calculation. The default is 1.
#			sym:  a scalar Boolean denoting whether the DiGraph is 
#			    symmetric (i.e., each edge from vertex i to vertex j
#			    has a companion edge from j to i).  If the DiGraph 
#			    is symmetric, the operation is faster.  The default 
#			    is False.
#
#		Output Arguments:
#			ret:  a ParVec of length equal to the number of vertices
#			    in the DiGraph, with a True (1) in each position for
#			    which the corresponding vertex is a neighbor.
#
#			    Note:  vertices from the start vector may appear in
#			    the return value.
#
#		SEE ALSO:  pathsHop
#		"""
#		if not sym:
#			self._T()
#		dest = ParVec(self.nvert(),0)
#		fringe = SpParVec(self.nvert())
#		fringe[source] = 1
#		for i in range(nhop):
#			self._spm.SpMV_SelMax_inplace(fringe._spv)
#			dest[fringe.toParVec()] = 1
#		if not sym:
#			self._T()
#		return dest
#		
#	# returns:
#	#   - source:  a vector of the source vertex for each new vertex
#	#   - dest:  a Boolean vector of the new vertices
#	#ToDo:  nhop argument?
#	def pathsHop(self, source, sym=False):
#		"""
#		calculates, for the given DiGraph instance and starting vertices,
#		which can be viewed as the fringe of a set of paths, the vertices
#		that are reachable by traversing one graph edge from one of the 
#		starting vertices.  The paths are kept distinct, as only one path
#		will extend to a given vertex.
#
#		Input Arguments:
#			self:  a DiGraph instance
#			source:  a Boolean ParVec with True (1) in the positions
#			    of the starting vertices.  
#			sym:  a scalar Boolean denoting whether the DiGraph is 
#			    symmetric (i.e., each edge from vertex i to vertex j
#			    has a companion edge from j to i).  If the DiGraph 
#			    is symmetric, the operation is faster.  The default 
#			    is False.
#
#		Output Arguments:
#			ret:  a ParVec of length equal to the number of vertices
#			    in the DiGraph.  The value of each element of the ParVec 
#			    with a value other than -1 denotes the starting vertex
#			    whose path extended to the corresponding vertex.  In
#			    the case of multiple paths potentially extending to
#			    a single vertex, the highest-numbered starting vertex
#			    is chosen as the source. 
#
#		SEE ALSO:  neighbors
#		"""
#		if not sym:
#			self._T()
#		#HACK:  SelMax is actually doing a Multiply instead of a Select,
#		#    so it doesn't work "properly" on a general DiGraph, whose
#		#    values can't be counted on to be 1.  So, make a copy of
#		#    the DiGraph and set all the values to 1 as a work-around. 
#		self2 = self.copy()
#		self2.ones()
#		ret = ParVec(self2.nvert(),-1)
#		fringe = source.find()
#		fringe.spRange()
#		self2._spm.SpMV_SelMax_inplace(fringe._spv)
#		ret[fringe] = fringe
#		if not sym:
#			self._T()
#		return ret
#
#
#	def normalizeEdgeWeights(self, dir=Out):
#		"""
#		Normalize the outward edge weights of each vertex such
#		that for Vertex v, each outward edge weight is
#		1/outdegree(v).
#		"""
#		degscale = self.degree(dir)
#		degscale._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0), pcb.identity(), pcb.bind1st(pcb.divides(), 1)))			
#		self.scale(degscale.toSpParVec(), dir)
#		
#	def pageRank(self, epsilon = 0.1, dampingFactor = 0.85):
#		"""
#		Compute the PageRank of vertices in the graph.
#
#		The PageRank algorithm computes the percentage of time
#		that a "random surfer" spends at each vertex in the
#		graph. If the random surfer is at Vertex v, she will
#		take one of two actions:
#		    1) She will hop to another vertex to which Vertex
#                       v has an outward edge. Self loops are ignored.
#		    2) She will become "bored" and randomly hop to any
#                       vertex in the graph. This action is taken with
#                       probability (1 - dampingFactor).
#
#		When the surfer is visiting a vertex that is a sink
#		(i.e., has no outward edges), she hops to any vertex
#		in the graph with probability one.
#
#		Optional argument epsilon controls the stopping
#		condition. Iteration stops when the 1-norm of the
#		difference in two successive result vectors is less
#		than epsilon.
#
#		Optional parameter dampingFactor alters the results
#		and speed of convergence, and in the model described
#		above dampingFactor is the percentage of time that the
#		random surfer hops to an adjacent vertex (rather than
#		hopping to a random vertex in the graph).
#
#		See "The PageRank Citation Ranking: Bringing Order to
#		the Web" by Page, Brin, Motwani, and Winograd, 1998
#		(http://ilpubs.stanford.edu:8090/422/) for more
#		information.
#		"""
#
#		# We don't want to modify the user's graph.
#		G = self.copy()
#		G._T()
#		nvert = G.nvert()
#
#		# Remove self loops.
#		G.removeSelfLoops()
#
#		# Handle sink nodes (nodes with no outgoing edges) by
#		# connecting them to all other nodes.
#
#		sinkV = G.degree(DiGraph.In)
#		sinkV._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(0), pcb.set(1./nvert)))
#
#		# Normalize edge weights such that for each vertex,
#		# each outgoing edge weight is equal to 1/(number of
#		# outgoing edges).
#		G.normalizeEdgeWeights(DiGraph.In)
#
#		# PageRank loop.
#		delta = 1
#		dv1 = ParVec(nvert, 1./nvert)
#		v1 = dv1.toSpParVec()
#		prevV = SpParVec(nvert)
#		onesVec = SpParVec.ones(nvert)
#		dampingVec = onesVec * ((1 - dampingFactor)/nvert)
#		while delta > epsilon:
#			prevV = v1.copy()
#			v2 = G._spm.SpMV_PlusTimes(v1._spv)
#
#			# Compute the inner product of sinkV and v1.
#			sinkContrib = sinkV.copy()
#			sinkContrib._dpv.EWiseApply(v1._spv, pcb.multiplies())
#			sinkContrib = sinkContrib._dpv.Reduce(pcb.plus())
#			
#			v1._spv = v2 + (onesVec*sinkContrib)._spv
#			v1 = v1*dampingFactor + dampingVec
#			delta = (v1 - prevV)._spv.Reduce(pcb.plus(), pcb.abs())
#		return v1
#
#		
#	def centrality(self, alg, **kwargs):
#		"""
#		calculates the centrality of each vertex in the DiGraph instance,
#		where 'alg' can be one of 
#		    'exactBC':  exact betweenness centrality
#		    'approxBC':  approximate betweenness centrality
#
#		Each algorithm may have algorithm-specific arguments as follows:
#		    'exactBC':  
#		        normalize=True:  normalizes the values by dividing by 
#		                (nVert-1)*(nVert-2)
#		    'approxBC':
#			sample=0.05:  the fraction of the vertices to use as sources 
#				and destinations;  sample=1.0 is the same as exactBC
#		        normalize=True:  normalizes the values by multiplying by 
#				nVerts / (nVertsCalculated * (nVerts-1) * (nVerts-2))
#		The return value is a ParVec with length equal to the number of
#		vertices in the DiGraph, with each element of the ParVec containing
#		the centrality value of the vertex.
#		"""
#		if alg=='exactBC':
#			cent = DiGraph._approxBC(self, sample=1.0, **kwargs)
#			#cent = DiGraph._bc(self, 1.0, self.nvert())
#		elif alg=='approxBC':
#			cent = DiGraph._approxBC(self, **kwargs)
#		elif alg=='kBC':
#			raise NotImplementedError, "k-betweenness centrality unimplemented"
#		elif alg=='degree':
#			raise NotImplementedError, "degree centrality unimplemented"
#		else:
#			raise KeyError, "unknown centrality algorithm (%s)" % alg
#	
#		return cent
#	
#	
#	def _approxBC(self, sample=0.05, normalize=True, nProcs=pcb._nprocs(), memFract=0.1, BCdebug=0):
#		"""
#		calculates the approximate or exact (with sample=1.0) betweenness
#		centrality of the input DiGraph instance.  _approxBC is an internal
#		method of the user-visible centrality method, and as such is
#		subject to change without notice.  Currently the following expert
#		argument is supported:
#		    - memFract:  the fraction of node memory that will be considered
#			available for a single strip in the strip-mining
#			algorithm.  Fractions that lead to paging will likely
#			deliver atrocious performance.  The default is 0.1.  
#		"""
#		A = self.copy()
#		Anv = A.nvert()
#		if BCdebug>0:
#			print "in _approxBC, A.nvert=%d, nproc=%d" % (Anv, nProcs)
#		self.ones()
#		#Aint = self.ones()	# not needed;  Gs only int for now
#		N = A.nvert()
#		bc = ParVec(N)
#		#nProcs = pcb._nprocs()
#		nVertToCalc = int(self.nvert() * sample)
#		# batchSize = #rows/cols that will fit in memory simultaneously.
#		# bcu has a value in every element, even though it's literally
#		# a sparse matrix (DiGraph).  So batchsize is calculated as
#		#   nrow = memory size / (memory/row)
#		#   memory size (in edges)
#		#        = 2GB * 0.1 (other vars) / 18 (bytes/edge) * nProcs
#		#   memory/row (in edges)
#		#        = self.nvert()
#		physMemPCore = 2e9; bytesPEdge = 18
#		#memFract = 0.1;
#		batchSize = int(2e9 * memFract / bytesPEdge * nProcs / N)
#		nBatches = int(math.ceil(float(nVertToCalc) / float(batchSize)))
#		nPossBatches = int(math.ceil(float(N) / float(batchSize)))
#		if sample == 1.0:
#			startVs = range(0,nVertToCalc,batchSize)
#			endVs = range(batchSize, nVertToCalc, batchSize)
#			if nVertToCalc % batchSize != 0:
#				endVs.append(nVertToCalc)
#			numVs = [y-x for [x,y] in zip(startVs,endVs)]
#		else:
#			perm = ParVec.range(nPossBatches)
#			perm.randPerm()
#			#   ideally, could use following 2 lines, but may have
#			#   only 1-2 batches, which makes index vector look
#			#   like a Boolean, which doesn't work right
#			#startVs = ParVec.range(nBatches)[perm[ParVec.range(nBatches]]
#			#numVs = [min(x+batchSize,N)-x for x in startVs]
#			tmpRange = ParVec.range(nPossBatches)
#			startVs = ParVec.zeros(nBatches)
#			numVs = ParVec.zeros(nBatches)
#			for i in range(nBatches):
#				startVs[i] = tmpRange[perm[i]]*batchSize
#				numVs[i] = min(startVs[i]+batchSize,N)-startVs[i]
#
#		if BCdebug>0:
#			print "batchSz=%d, nBatches=%d, nPossBatches=%d" % (batchSize, nBatches, nPossBatches)
#		for [startV, numV] in zip(startVs, numVs):
#			startV = int(startV); numV = int(numV)
#			if BCdebug>0:
#				print "startV=%d, numV=%d" % (startV, numV)
#			bfs = []		
#			batch = ParVec.range(startV, startV+numV)
#			curSize = len(batch)
#			nsp = DiGraph(ParVec.range(curSize), batch, 1, curSize, N)
#			fringe = A[batch,ParVec.range(N)]
#			depth = 0
#			while fringe.nedge() > 0:
#				before = time.time()
#				depth = depth+1
#				if BCdebug>1 and depth>1:
#					nspne = tmp.nedge(); tmpne = tmp.nedge(); fringene = fringe.nedge()
#					if master():
#					    print "BC: in while: depth=%d, nsp.nedge()=%d, tmp.nedge()=%d, fringe.nedge()=%d" % (depth, nspne, tmpne, fringene)
#				nsp = nsp+fringe
#				tmp = fringe.copy()
#				tmp.ones()
#				bfs.append(tmp)
#				tmp = fringe._SpMM(A)
#				if BCdebug>1:
#					#nspsum = nsp.sum(DiGraph.Out).sum() 
#					#fringesum = fringe.sum(DiGraph.Out).sum()
#					#tmpsum = tmp.sum(DiGraph.Out).sum()
#					if master():
#						#print depth, nspsum, fringesum, tmpsum
#						pass
#				fringe = tmp.mulNot(nsp)
#				if BCdebug>1 and master():
#					print "    %f seconds" % (time.time()-before)
#	
#			bcu = DiGraph.fullyConnected(curSize,N)
#			# compute the bc update for all vertices except the sources
#			for depth in range(depth-1,0,-1):
#				# compute the weights to be applied based on the child values
#				w = bfs[depth] / nsp * bcu
#				if BCdebug>2:
#					tmptmp = w.sum(DiGraph.Out).sum()
#					if master():
#						print tmptmp
#				# Apply the child value weights and sum them up over the parents
#				# then apply the weights based on parent values
#				w._T()
#				w = A._SpMM(w)
#				w._T()
#				w *= bfs[depth-1]
#				w *= nsp
#				bcu += w
#	
#			# update the bc with the bc update
#			if BCdebug>2:
#				tmptmp = bcu.sum(DiGraph.Out).sum()
#				if master():
#					print tmptmp
#			bc = bc + bcu.sum(DiGraph.In)	# column sums
#	
#		# subtract off the additional values added in by precomputation
#		bc = bc - nVertToCalc
#		if normalize:
#			nVertSampled = sum(numVs)
#			bc = bc * (float(N)/float(nVertSampled*(N-1)*(N-2)))
#		return bc
#	
#	def cluster(self, alg, **kwargs):
#	#		ToDo:  Normalize option?
#		"""
#		Deferred implementation for KDT v0.1
#			
#		"""
#		raise NotImplementedError, "clustering not implemented for v0.1"
#		if alg=='Markov' or alg=='markov':
#			clus = _markov(self, **kwargs)
#	
#		elif alg=='kNN' or alg=='knn':
#			raise NotImplementedError, "k-nearest neighbors clustering not implemented"
#	
#		else:
#			raise KeyError, "unknown clustering algorithm (%s)" % alg
#	
#		return clus
