import math
import pyCombBLAS as pcb
import Graph as gr
from Graph import ParVec, SpParVec, master

import time

class DiGraph(gr.Graph):
	InOut = 1
	In = 2
	Out = 3

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
	def __init__(self,*args):
		"""
		creates a new DiGraph instance.  Can be called in one of the 
		following forms:

	DiGraph():  creates a DiGraph instance with no vertices or edges.  Useful as input for genGraph500Edges.

	DiGraph(sourceV, destV, weightV, n)
	DiGraph(sourceV, destV, weightV, n, m)
		create a DiGraph Instance with edges with source represented by 
		each element of sourceV and destination represented by each 
		element of destV with weight represented by each element of 
		weightV.  In the 4-argument form, the resulting DiGraph will 
		have n out- and in-vertices.  In the 5-argument form, the 
		resulting DiGraph will have n out-vertices and m in-vertices.

		Input Arguments:
			sourceV:  a ParVec containing integers denoting the 
			    source vertex of each edge.
			destV:  a ParVec containing integers denoting the 
			    destination vertex of each edge.
			weightV:  a ParVec containing double-precision floating-
			    point numbers denoting the weight of each edge.
			n:  an integer scalar denoting the number of out-vertices 
			    (and also in-vertices in the 4-argument case).
			m:  an integer scalar denoting the number of in-vertices.

		Output Argument:  
			ret:  a DiGraph instance

		Note:  If two or more edges have the same source and destination
		vertices, their weights are summed in the output DiGraph instance.

		SEE ALSO:  toParVec
		"""
		if len(args) == 0:
			self._spm = pcb.pySpParMat()
		elif len(args) == 1:	# no longer used
			[arg] = args
			if arg < 0:
				self._spm = pcb.pySpParMat()
			else:
				raise NotImplementedError, '1-argument case only accepts negative value'
		elif len(args) == 4:
			[i,j,v,nv] = args
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v)
			if i.max() > nv-1:
				raise KeyError, 'at least one first index greater than #vertices'
			if j.max() > nv-1:
				raise KeyError, 'at least one second index greater than #vertices'
			self._spm = pcb.pySpParMat(nv,nv,i._dpv,j._dpv,v._dpv)
		elif len(args) == 5:
			[i,j,v,nv1,nv2] = args
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = ParVec.broadcast(len(i),v)
			if i.max() > nv1-1:
				raise KeyError, 'at least one first index greater than #vertices'
			if j.max() > nv2-1:
				raise KeyError, 'at least one second index greater than #vertices'
			self._spm = pcb.pySpParMat(nv1,nv2,i._dpv,j._dpv,v._dpv)
		else:
			raise NotImplementedError, "only 1, 4, and 5 argument cases supported"

	def __add__(self, other):
		"""
		adds corresponding edges of two DiGraph instances together,
		resulting in edges in the result only where an edge exists in at
		least one of the input DiGraph instances.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other, DiGraph):
			ret = self.copy()
			ret._spm += other._spm
			#ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.plus());  # only adds if both mats have nonnull elems!!
		return ret

	def __div__(self, other):
		"""
		divides corresponding edges of two DiGraph instances together,
		resulting in edges in the result only where edges exist in both
		input DiGraph instances.
		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret._spm.Apply(pcb.bind2nd(pcb.divides(),other))
		elif self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other,DiGraph):
			ret = self.copy()
			ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.divides())
		else:
			raise NotImplementedError
		return ret

	def __getitem__(self, key):
		"""
		FIX:  fix documentation

		implements indexing on the right-hand side of an assignment.
		Usually accessed through the "[]" syntax.

		Input Arguments:
			self:  a DiGraph instance
			key:  one of the following forms:
			    - a non-tuple denoting the key for both dimensions
			    - a tuple of length 2, with the first element denoting
			        the key for the first dimension and the second 
			        element denoting for the second dimension.
			    Each key denotes the out-/in-vertices to be addressed,
			    and may be one of the following:
				- an integer scalar
				- the ":" slice denoting all vertices, represented
				  as slice(None,None,None)
				- a ParVec object containing a contiguous range
				  of monotonically increasing integers 
		
		Output Argument:
			ret:  a DiGraph instance, containing the indicated vertices
			    and their incident edges from the input DiGraph.

		SEE ALSO:  subgraph
		"""
		#ToDo:  accept slices for key0/key1 besides ParVecs
		if type(key)==tuple:
			if len(key)==1:
				[key0] = key; key1 = -1
			elif len(key)==2:
				[key0, key1] = key
			else:
				raise KeyError, 'Too many indices'
		else:
			key0 = key;  key1 = key
		if type(key0) == int or type(key0) == long or type(key0) == float:
			tmp = ParVec(1)
			tmp[0] = key0
			key0 = tmp
		if type(key1) == int or type(key0) == long or type(key0) == float:
			tmp = ParVec(1)
			tmp[0] = key1
			key1 = tmp
		#if type(key0)==slice and key0==slice(None,None,None):
		#	key0mn = 0; 
		#	key0tmp = self.nvert()
		#	if type(key0tmp) == tuple:
		#		key0mx = key0tmp[0] - 1
		#	else:
		#		key0mx = key0tmp - 1
		#if type(key1)==slice and key1==slice(None,None,None):
		#	key1mn = 0 
		#	key1tmp = self.nvert()
		#	if type(key1tmp) == tuple:
		#		key1mx = key1tmp[1] - 1
		#	else:
		#		key1mx = key1tmp - 1
		
		ret = DiGraph()
		ret._spm = self._spm.SubsRef(key0._dpv, key1._dpv)
		return ret

	def __iadd__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			raise NotImplementedError
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other, DiGraph):
			#dead tmp = pcb.EWiseApply(self._spm, other._spm, pcb.plus())
			self._spm += other._spm
		return self

	def __imul__(self, other):
		if type(other) == int or type(other) == long or type(other) == float:
			self._spm.Apply(pcb.bind2nd(pcb.multiplies(),other))
		elif isinstance(other,DiGraph):
			self._spm = pcb.EWiseApply(self._spm,other._spm, pcb.multiplies())
		else:
			raise NotImplementedError
		return self

	def __mul__(self, other):
		"""
		multiplies corresponding edges of two DiGraph instances together,
		resulting in edges in the result only where edges exist in both
		input DiGraph instances.

		"""
		if type(other) == int or type(other) == long or type(other) == float:
			ret = self.copy()
			ret._spm.Apply(pcb.bind2nd(pcb.multiplies(),other))
		elif self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		elif isinstance(other,DiGraph):
			ret = self.copy()
			ret._spm = pcb.EWiseApply(self._spm,other._spm, pcb.multiplies())
		else:
			raise NotImplementedError
		return ret

	def __neg__(self):
		ret = self.copy()
		ret._spm.Apply(pcb.negate())
		return ret

	#ToDo:  put in method to modify _REPR_MAX
	_REPR_MAX = 100
	def __repr__(self):
		if self.nvert() == 0:
			return 'Null DiGraph object'
		if self.nvert()==1:
			[i, j, v] = self.toParVec()
			if len(v) > 0:
				print "%d %f" % (v[0], v[0])
			else:
				print "%d %f" % (0, 0.0)
		else:
			[i, j, v] = self.toParVec()
			if len(i) < self._REPR_MAX:
				print i,j,v
		return ' '

	def _SpGEMM(self, other):
		"""
		"multiplies" two DiGraph instances together as though each was
		represented by a sparse matrix, with rows representing out-edges
		and columns representing in-edges.
		"""
		selfnv = self.nvert()
		if type(selfnv) == tuple:
			[selfnv1, selfnv2] = selfnv
		else:
			selfnv1 = selfnv; selfnv2 = selfnv
		othernv = other.nvert()
		if type(othernv) == tuple:
			[othernv1, othernv2] = othernv
		else:
			othernv1 = othernv; othernv2 = othernv
		if selfnv2 != othernv1:
			raise ValueError, '#in-vertices of first graph not equal to #out-vertices of the second graph '
		ret = DiGraph()
		ret._spm = self._spm.SpGEMM(other._spm)
		return ret

	def contract(self, groups=None, clusterParents=None):
		"""
		contracts all vertices that are like-numbered in the groups
		argument into single vertices, removing all edges between
		vertices in the same group and retaining edges where any
		vertex in a group has an edge to a vertex in another group.
		The result DiGraph will have as many vertices as the maximum
		value in groups.

		Input Arguments:
			self:  a DiGraph instance.
			
			Specify exactly one of the following:
			groups:  a ParVec denoting into which group (result
				vertex) each input vertex should be placed
			clusterParents: a ParVec denoting which vertex an input vertex
			    should be collapsed into.

		Output Argument:
			ret:  a DiGraph instance
		"""
		#ToDo:  implement weighting properly
		n = self.nvert()

		if type(self.nvert()) == tuple:
			raise NotImplementedError, 'only implemented for square graphs'
		if groups is None and clusterParents is None:
			raise KeyError, 'groups or collapseInto must be specified'

		if groups is not None and len(groups) != n:
			raise KeyError, 'len(groups) does not match self.nvert()'
		if groups is not None and (groups.min < 0 or groups.max() >= len(groups)):
			raise KeyError, 'at least one groups value negative or greater than len(groups)-1'
		if clusterParents is not None and len(clusterParents) != n:
			raise KeyError, 'len(clusterParents) does not match self.nvert()'
		if clusterParents is not None and (clusterParents.min < 0 or clusterParents.max() >= n):
			raise KeyError, 'at least one groups value negative or greater than len(clusterParents)-1'
			
		if clusterParents is not None:
			# convert to groups
			groups = DiGraph.convClusterParentToGroup(clusterParents)
		
		nvRes = int(groups.max()+1)
		origVtx = ParVec.range(n)
		# lhrMat == left-/right-hand-side matrix
		lrhMat = DiGraph(groups, origVtx, ParVec.ones(n), nvRes, n)
		tmpMat = lrhMat._SpGEMM(self)
		lrhMat._T()
		res = tmpMat._SpGEMM(lrhMat)
		return res;
	
	@staticmethod
	def convClusterParentToGroup(clusterParents, retInvPerm = False):
		"""
		converts a component list composed of parent vertices into
		a component list composed of cluster numbers.
		For k clusters, each cluster number will be in the range [0,k).

		Input Arguments:
			clusterParents:  a vector where each element specifies which vertex
			    is its cluster parent. It is assumed that each cluster has only
			    one parent vertex.
			retInvPerm:  if True, says to return a permutation going from group
			    number to original parent vertex.

		Output Argument:
			ret:  a vertex of the same length as clusterParents where the elements
			    are in the range [0,k) and correspond to cluster group numbers.
			    If retInvPerm is True, the return is a tuple which also contains
			    a permutation going from group number to original parent vertex.
		"""
		
		n = len(clusterParents)
		
		# Count the number of elements in each parent's component to identify the parents
		countM = DiGraph(clusterParents, ParVec.range(n), ParVec.ones(n), n)
		counts = countM._spm.Reduce(pcb.pySpParMat.Row(), pcb.plus())
		del countM
		
		# sort to put them all at the front
		sorted = counts.copy()
		sorted.Apply(pcb.negate())
		perm = sorted.Sort()
		sorted.Apply(pcb.negate())
		
		# find inverse of sort permutation so that [1,2,3,4...] can be put back into the
		# original parent locations
		invM = DiGraph(ParVec.toParVec(perm), ParVec.range(n), ParVec.ones(n), n)
		invPerm = invM._spm.SpMV(SpParVec.range(n)._spv, pcb.TimesPlusSemiring()).dense()
		del invM
		
		# Find group number for each parent vertex
		groupNum = invPerm
		
		# Broadcast group number to all vertices in cluster
		broadcastM = DiGraph(ParVec.range(n), clusterParents, ParVec.ones(n), n)
		ret = broadcastM._spm.SpMV(groupNum.sparse(), pcb.TimesPlusSemiring()).dense()
		
		if retInvPerm:
			return ParVec.toParVec(ret), ParVec.toParVec(perm)
		else:
			return ParVec.toParVec(ret)
	
	@staticmethod
	def convMaskToIndices(mask):
		verts = mask._dpv.FindInds(pcb.bind2nd(pcb.equal_to(), 1))
		return ParVec.toParVec(verts)

	def copy(self):
		"""
		creates a deep copy of a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  a DiGraph instance containing a copy of the input.
		"""
		ret = DiGraph()
		ret._spm = self._spm.copy()
		return ret
		
	def degree(self, dir=Out):
		"""
		calculates the degrees of the appropriate edges of each vertex of 
		the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges to count, with choices being
			    DiGraph.Out (default), DiGraph.In, or DiGraph.InOut.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    degree of the weights of the corresponding vertex.

		SEE ALSO:  sum 
		"""
		if self.nedge() == 0:
			nv = self.nvert()
			if type(nv) == tuple:
				nv0 = nv[0]; nv1 = nv[1];
			else:
				nv0 = nv; nv1 = nv;
			if dir == DiGraph.In or dir == DiGraph.InOut:
				return ParVec.zeros(nv0)
			elif dir == DiGraph.Out:
				return ParVec.zeros(nv1)
			else:
				raise KeyError, 'Unknown edge-direction'
		if dir == DiGraph.InOut:
			#ToDo:  can't do InOut if nonsquare graph
			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
			return ParVec.toParVec(tmp1+tmp2)
		elif dir == DiGraph.In:
			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
			return ParVec.toParVec(ret)
		elif dir == DiGraph.Out:
			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(1), pcb.set(0)))
			return ParVec.toParVec(ret)
		else:
			raise KeyError, 'Invalid edge direction'

	# in-place, so no return value
	def removeSelfLoops(self):
		"""
		removes all edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.

		"""
		if self.nvert() > 0:
			self._spm.removeSelfLoops()
		return

	def addSelfLoops(self, selfLoopAttr=1):
		"""
		removes all edges whose source and destination are the same
		vertex, in-place in a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
			selfLoopAttr: the value to put on each self loop

		"""
		if self.nvert() > 0:
			self += DiGraph.eye(self.nvert(), selfLoopAttr=selfLoopAttr)
		return

	@staticmethod
	def fullyConnected(n,m=None):
		"""
		creates edges in a DiGraph instance that connects each vertex
		directly to every other vertex.

		Input Arguments:
			n:  an integer scalar denoting the number of vertices in
			    the graph that may potentially have out-edges.
			m:  an optional argument, which if specified is an integer
			    scalar denoting the number of vertices in the graph
			    that may potentially have in-edges.

		Output Argument:
			ret:  a DiGraph instance with directed edges from each
			    vertex to every other vertex. 
		"""
		if m is None:
			m = n
		i = (ParVec.range(n*m) % n).floor()
		j = (ParVec.range(n*m) / n).floor()
		v = ParVec.ones(n*m)
		ret = DiGraph(i,j,v,n,m)
		return ret
	
	@staticmethod
	def eye(n, selfLoopAttr=1):
		"""
		creates edges in a DiGraph instance where each vertex
		has exactly one edge connecting to itself.

		Input Arguments:
			n:  an integer scalar denoting the number of vertices in
			    the graph.
			selfLoopAttr: the value to put on the self loop.

		Output Argument:
			ret:  a DiGraph instance with directed edges from each
			    vertex to itself. 
		"""
		return DiGraph(ParVec.range(n),ParVec.range(n),ParVec(n, selfLoopAttr),n,n)

	def genGraph500Edges(self, scale):
		"""
		creates edges in a DiGraph instance that meet the Graph500 
		specification.  The graph is symmetric. (See www.graph500.org 
		for details.)

		Input Arguments:
			self:  a DiGraph instance, usually with no edges
			scale:  an integer scalar representing the logarithm base
			    2 of the number of vertices in the resulting DiGraph.
			    
		Output Argument:
			ret:  a double-precision floating-point scalar denoting
			    the amount of time to converted the created edges into
			    the DiGraph instance.  This equals the value of Kernel 1
			    of the Graph500 benchmark. Timing the entire genGraph500Edges
			    call would also time the edge generation, which is not part
			    of Kernel 1.
			    Degrees of all vertices.
		"""
		degrees = pcb.pyDenseParVec(1, 1)
		elapsedTime = self._spm.GenGraph500Edges(scale, degrees)
	 	return (elapsedTime, ParVec.toParVec(degrees))

	@staticmethod
	def load(fname):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a DiGraph instance.

		Input Argument:
			fname:  a filename from which the DiGraph data will be loaded.
		Output Argument:
			ret:  a DiGraph instance containing the graph represented
			    by the file's contents.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The load
		method makes this conversion while reading the data and creating
		the graph.

		SEE ALSO:  save, UFget
		"""
		# Verify file exists.
		file = open(fname, 'r')
		file.close()
		
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = DiGraph()
		ret._spm = pcb.pySpParMat()
		ret._spm.load(fname)
		return ret

	def max(self, dir=InOut):
		"""
		finds the maximum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the maximum,
			    with choices being DiGraph.InOut (default), DiGraph.In, or 
			    DiGraph.Out.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    maximum of the weights of the corresponding vertex.

		SEE ALSO:  degree, min 
		"""
		#ToDo:  is default to InOut best?
		if dir == DiGraph.InOut:
			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.max())
			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.max())
			return ParVec.toParVec(tmp1+tmp2)
		elif dir == DiGraph.In:
			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.max())
			return ParVec.toParVec(ret)
		elif dir == DiGraph.Out:
			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.max())
			return ParVec.toParVec(ret)
		else:
			raise KeyError, 'Invalid edge direction'

	def min(self, dir=InOut):
		"""
		finds the minimum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the minimum,
			    with choices being DiGraph.Out (default), DiGraph.In, or 
			    DiGraph.InOut.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    minimum of the weights of the corresponding vertex.

		SEE ALSO:  degree, max 
		"""
		#ToDo:  is default to InOut best?
		if dir == DiGraph.InOut:
			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.min())
			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.min())
			return ParVec.toParVec(tmp1+tmp2)
		elif dir == DiGraph.In:
			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.min())
			return ParVec.toParVec(ret)
		elif dir == DiGraph.Out:
			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.min())
			return ParVec.toParVec(ret)

	def mulNot(self, other):
		"""
		multiplies corresponding edge weights of two DiGraph instances,
		taking the logical not of the second argument before doing the 
		multiplication.  In effect, each nonzero edge of the second
		argument deletes its corresponding edge of the first argument.

		Input Arguments:
			self:  a DiGraph instance
			other:  another DiGraph instance

		Output arguments:
			ret:  a DiGraph instance 
		"""
		if self.nvert() != other.nvert():
			raise IndexError, 'Graphs must have equal numbers of vertices'
		else:
			ret = DiGraph()
			ret._spm = pcb.EWiseApply(self._spm, other._spm, pcb.multiplies(), True)
		return ret

	def nedge(self, vpart=None):
		"""
		returns the number of edges in (each partition of the vertices
		of) the DiGraph instance, including edges with zero weight.

		Input Arguments:
			self:  a DiGraph instance
			vpart: an optional argument; a ParVec instance denoting a 
			    partition of the vertices of the graph.  The value of
			    each element of vpart denotes the partition in which
			    the corresponding vertex resides.  Note that this is
			    the same format as the return value from cluster().

		Output Arguments:
			ret:  the number of edges in the DiGraph (if vpart not
			    specified) or a ParVec instance with each element
			    denoting the number of edges in the corresponding
			    partition (if vpart specified).

		SEE ALSO:  npart, contract
		"""
		nv = self.nvert()
		if vpart is None:
			return self._spm.getnee()
		else:
			if self.nvert() != len(vpart):
				raise KeyError,'vpart must be same length as number of vertices in DiGraph instance'
			retLen = int(vpart.max())+1
		if self.nvert() == 0:
			if retLen == 1:
				return 0
			else:
				return ParVec.zeros(retLen)
		if retLen == 1:
			ret = self._spm.getnee()
		else:
			selfcopy = self.copy()
			selfcopy.set(1)
			C = DiGraph(vpart,ParVec.range(nv),
				ParVec.ones(nv), retLen, nv)
			tmpMat = C._SpGEMM(selfcopy)
			C._T()
			tmpMat = tmpMat._SpGEMM(C)
			#HACK!!
			#ret = tmpMat[diag]
			ret = ParVec(retLen)
			for i in range(retLen):
				ret[i] = int(tmpMat[i].sum()[0])
		return ret
			

	#FIX:  good idea to have this return an int or a tuple?
	def nvert(self, vpart=None):
		"""
		ToDo:  fix docstring for vertex-partition angle

		returns the number of vertices in the given DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  if the DiGraph was created with the same number of
			    vertices potentially having in- and out-edges, the
			    return value is a scalar integer of that number.  If
			    the DiGraph was created with different numbers of
			    vertices potentially having in- and out-edges, the
			    return value is a tuple of length 2, with the first
			    (second) element being the number of vertices potentially 
			    having out-(in-)edges.

		SEE ALSO:  nedge, degree
		"""
		nrow = self._spm.getnrow()
		ncol = self._spm.getncol()
		if nrow!=ncol:
			return (nrow, ncol)
		if vpart is None:
			return nrow
		if len(vpart) != nrow:
			raise KeyError, 'vertex partition length not equal to number of vertices'
		ret = vpart.hist()
		return ret


	##in-place, so no return value
	#def ones(self):
	#	"""
	#	sets every edge in the graph to the value 1.

	#	Input Argument:
	#		self:  a DiGraph instance, modified in place.

	#	Output Argument:
	#		None.

	#	SEE ALSO:  set
	#	"""
	#	self._spm.Apply(pcb.set(1))
	#	return

	#in-place, so no return value
	def reverseEdges(self):
		"""
		reverses the direction of each edge of a DiGraph instance in-place,
		switching its source and destination vertices.

		Input Argument:
			self:  a DiGraph instance, modified in-place.
		"""
		self._spm.Transpose()

	def save(self, fname):
		"""
		saves the contents of the passed DiGraph instance to a file named
		fname in the Coordinate Format of the Matrix Market Exchange Format.

		Input Arguments:
			self:  a DiGraph instance
			fname:  a filename to which the DiGraph data will be saved.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The save
		method makes this conversion while writing the data.

		SEE ALSO:  load, UFget
		"""
		self._spm.save(fname)
		return

	#in-place, so no return value
	def scale(self, other, dir=Out):
		"""
		multiplies the weights of the appropriate edges of each vertex of
		the passed DiGraph instance in-place by a vertex-specific scale 
		factor.

		Input Arguments:
			self:  a DiGraph instance, modified in-place
			other: a ParVec whose elements are used
			dir:  a direction of edges to scale, with choices being
			    DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			None.

		SEE ALSO:  * (DiGraph.__mul__), mulNot
		"""
		if not isinstance(other,gr.ParVec):
			raise KeyError, 'Invalid type for scale vector'
		selfnv = self.nvert()
		if type(selfnv) == tuple:
			[selfnv1, selfnv2] = selfnv
		else:
			selfnv1 = selfnv; selfnv2 = selfnv
		if dir == DiGraph.In:
			if selfnv2 != len(other):
				raise IndexError, 'graph.nvert()[1] != len(scale)'
			self._spm.DimWiseApply(pcb.pySpParMat.Column(), other._dpv, pcb.multiplies())
		elif dir == DiGraph.Out:
			if selfnv1 != len(other):
				raise IndexError, 'graph.nvert()[1] != len(scale)'
			self._spm.DimWiseApply(pcb.pySpParMat.Row(), other._dpv, pcb.multiplies())
		else:
			raise KeyError, 'Invalid edge direction'
		return
	##in-place, so no return value
	#def set(self, value):
	#	"""
	#	sets every edge in the graph to the given value.

	#	Input Arguments:
	#		self:  a DiGraph instance, modified in place.
	#		value:  a scalar integer or double-precision floating-
	#		    point value.

	#	Output Argument:
	#		None.

	#	SEE ALSO:  ones
	#	"""
	#	self._spm.Apply(pcb.set(value))
	#	return

	def subgraph(self, ndx1=None, ndx2=None, mask=None):
		"""
		creates a new DiGraph instance consisting of only designated vertices 
		of the input graph and their indicent edges.

		Input Arguments:
			self:  a DiGraph instance
			ndx1:  an integer scalar or a ParVec of consecutive vertex
			    numbers to be included in the subgraph along with edges
			    starting from these vertices.
			ndx2:  an optional argument; if specified, is an integer
			    scalar or a ParVec of consecutive vertex numbers to
			    be included in the subgraph along with any edges ending
			    at these vertices.
			mask:  a length nverts ParVec with True elements for vertices
			    that should be kept and False elements for vertices to
			    be discarded.
			 
		Output Argument:
			ret:  a DiGraph instance composed of the selected vertices
			    and their incident edges.

		SEE ALSO:  DiGraph.__getitem__
		"""
		if ndx1 is None and mask is None:
			raise KeyError, 'Either indices or a mask must be provided'
		
		if ndx1 is None:
			# convert mask to indices
			verts = mask._dpv.FindInds(pcb.bind2nd(pcb.equal_to(), 1))
			ndx1 = ParVec.toParVec(verts)
			ndx2 = None
		
		if ndx2 is None:
			ndx2 = ndx1
		ret = self[ndx1, ndx2]
		return ret

	def sum(self, dir=Out):
		"""
		adds the weights of the appropriate edges of each vertex of the
		passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges to sum, with choices being
			    DiGraph.Out (default), DiGraph.In, or DiGraph.InOut.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    sum of the weights of the corresponding vertex.

		SEE ALSO:  degree 
		"""
		if dir == DiGraph.InOut:
			tmp1 = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity())
			tmp2 = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity())
			return ParVec.toParVec(tmp1+tmp2)
		elif dir == DiGraph.In:
			ret = self._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.identity())
			return ParVec.toParVec(ret)
		elif dir == DiGraph.Out:
			ret = self._spm.Reduce(pcb.pySpParMat.Row(),pcb.plus(), pcb.identity())
			return ParVec.toParVec(ret)
		else:
			raise KeyError, 'Invalid edge direction'

	_T = reverseEdges

	# in-place, so no return value
	def toBool(self):
		"""
		converts the DiGraph instance in-place such that each edge has only
		a Boolean (True) value, thereby consuming less space and making
		some operations faster.

		Input Argument:
			self:  a DiGraph instance that is overwritten by the method

		Output Argument:
			None.
		"""
		if isinstance(self._spm, pcb.pySpParMat):
			self._spm = pcb.pySpParMatBool(self._spm)

	def toParVec(self):
		"""
		decomposes a DiGraph instance to 3 ParVec instances, with each
		element of the first ParVec denoting the source vertex of an edge,
		the corresponding element of the second ParVec denoting the 
		destination vertex of the edge, and the corresponding element of
		the third ParVec denoting the value or weight of the edge.

		Input Argument:
			self:  a DiGraph instance

		Output Argument:
			ret:  a 3-element tuple with ParVec instances denoting the
			    source vertex, destination vertex, and weight, respectively.

		SEE ALSO:  DiGraph 
		"""
		ne = self.nedge()
		reti = ParVec(ne)
		retj = ParVec(ne)
		retv = ParVec(ne)
		self._spm.Find(reti._dpv, retj._dpv, retv._dpv)
		#ToDo:  return nvert() of original graph, too
		return (reti, retj, retv)

	@staticmethod
	def twoDTorus(n):
		"""
		constructs a DiGraph instance with the connectivity pattern of a 2D
		torus;  i.e., each vertex has edges to its north, west, south, and
		east neighbors, where the neighbor may be wrapped around to the
		other side of the torus.  

		Input Parameter:
			nnodes:  an integer scalar that denotes the number of nodes
			    on each axis of the 2D torus.  The resulting DiGraph
			    instance will have nnodes**2 vertices.

		Output Parameter:
			ret:  a DiGraph instance with nnodes**2 vertices and edges
			    in the pattern of a 2D torus. 
		"""
		N = n*n
		nvec =   ((ParVec.range(N*4)%N) / n).floor()	 # [0,0,0,...., n-1,n-1,n-1]
		nvecil = ((ParVec.range(N*4)%N) % n).floor()	 # [0,1,...,n-1,0,1,...,n-2,n-1]
		north = gr.Graph._sub2ind((n,n),(nvecil-1) % n,nvec)	
		south = gr.Graph._sub2ind((n,n),(nvecil+1) % n,nvec)
		west = gr.Graph._sub2ind((n,n),nvecil, (nvec-1) % n)
		east = gr.Graph._sub2ind((n,n),nvecil, (nvec+1) % n)
		Ndx = ParVec.range(N*4)
		northNdx = Ndx < N
		southNdx = (Ndx >= N) & (Ndx < 2*N)
		westNdx = (Ndx >= 2*N) & (Ndx < 3*N)
		eastNdx = Ndx >= 3*N
		col = ParVec.zeros(N*4)
		col[northNdx] = north
		col[southNdx] = south
		col[westNdx] = west
		col[eastNdx] = east
		row = ParVec.range(N*4) % N
		ret = DiGraph(row, col, 1, N)
		return ret

	# ==================================================================
	#  "complex ops" implemented below here
	# ==================================================================


	#	creates a breadth-first search tree of a Graph from a starting
	#	set of vertices.  Returns a 1D array with the parent vertex of 
	#	each vertex in the tree; unreached vertices have parent == -1.
	#	sym arg denotes whether graph is symmetric; if not, need to transpose
	#
	def bfsTree(self, root, sym=False):
		"""
		calculates a breadth-first search tree from the edges in the
		passed DiGraph, starting from the root vertex.  "Breadth-first"
		in the sense that all vertices reachable in step i are added
		to the tree before any of the newly-reachable vertices' reachable
		vertices are explored.

		Input Arguments:
			root:  an integer denoting the root vertex for the tree
			sym:  a Boolean denoting whether the DiGraph is symmetric
			    (i.e., each edge from vertex i to vertex j has a
			    companion edge from j to i).  If the DiGraph is 
			    symmetric, the operation is faster.  The default is 
			    False.

		Input Arguments:
			parents:  a ParVec instance of length equal to the number
			    of vertices in the DiGraph, with each element denoting 
			    the vertex number of that vertex's parent in the tree.
			    The root is its own parent.  Unreachable vertices
			    have a parent of -1. 

		SEE ALSO: isBfsTree 
		"""
		if not sym:
			self._T()
		parents = pcb.pyDenseParVec(self.nvert(), -1)
		fringe = pcb.pySpParVec(self.nvert())
		parents[root] = root
		fringe[root] = root
		while fringe.getnee() > 0:
			fringe.setNumToInd()
			self._spm.SpMV_SelMax_inplace(fringe)
			pcb.EWiseMult_inplacefirst(fringe, parents, True, -1)
			parents[fringe] = 0
			parents += fringe
		if not sym:
			self._T()
		return ParVec.toParVec(parents)
	
	
		# returns tuples with elements
		# 0:  True/False of whether it is a BFS tree or not
		# 1:  levels of each vertex in the tree (root is 0, -1 if not reached)
	def isBfsTree(self, root, parents, sym=False):
		"""
		validates that a breadth-first search tree in the style created
		by bfsTree is correct.

		Input Arguments:
			root:  an integer denoting the root vertex for the tree
			parents:  a ParVec instance of length equal to the number
			    vertices in the DiGraph, with each element denoting 
			    the vertex number of that vertex's parent in the tree.
			    The root is its own parent.  Vertices unreachable
			    from the root have a parent of -1. 
			sym:  a scalar Boolean denoting whether the DiGraph is 
			    symmetric (i.e., each edge from vertex i to vertex j
			    has a companion edge from j to i).  If the DiGraph 
			    is symmetric, the operation is faster.  The default 
			    is False.
		
		Output Arguments:
			ret:  a 2-element tuple.  The first element is an integer,
			    whose value is 1 if the graph is a BFS tree and whose
			    value is the negative of the first test below that failed,
			    if one of them failed.  If the graph is a BFS tree,
			    the second element of the tuple is a ParVec of length 
			    equal to the number of vertices in the DiGraph, with 
			    each element denoting the level in the tree at which 
			    the vertex resides.  The root resides in level 0, its
			    direct neighbors in level 1, and so forth.  Unreachable vertices have a
			    level value of -1.  If the graph is not a BFS tree (one
			    of the tests failed), the second element of the
			    tuple is None.
		
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
		ret = 1		# assume valid
		nvertG = self.nvert()
	

		# spec test #1
		# Confirm that the tree is a tree;  i.e., that it does not
		# have any cycles (visited more than once while building
		# the tree) and that every vertex with a parent is
		# in the tree. 

		# build a new graph from just tree edges
		tmp2 = parents != ParVec.range(nvertG)
		treeEdges = (parents != -1) & tmp2
		treeI = parents[treeEdges.findInds()]
		treeJ = ParVec.range(nvertG)[treeEdges.findInds()]
		del tmp2, treeEdges
		# root cannot be destination of any tree edge
		if (treeJ == root).any():
			return (-1, None)
		# note treeJ/TreeI reversed, so builtGT is transpose, as
		#   needed by SpMV
		builtGT = DiGraph(treeJ, treeI, 1, nvertG)
		visited = ParVec.zeros(nvertG)
		visited[root] = 1
		fringe = SpParVec(nvertG)
		fringe[root] = root
		cycle = False
		multiparents = False
		while fringe.nnn() > 0 and not cycle and not multiparents:
			fringe.spOnes()
			newfringe = SpParVec.toSpParVec(builtGT._spm.SpMV_PlusTimes(fringe._spv))
			if visited[newfringe.toParVec().findInds()].any():
				cycle = True
				break
			if (newfringe > 1).any():
				multiparents = True
				break
			fringe = newfringe
			visited[fringe] = 1
		if cycle or multiparents:
			return (-1, None)
		del visited, builtGT
		
		# spec test #2
		#    tree edges should be between verts whose levels differ by 1
		
		# calculate level in the tree for each vertex; root is at level 0
		# about the same calculation as bfsTree, but tracks levels too
		if not sym:
			self.reverseEdges()
		parents2 = ParVec.zeros(nvertG) - 1
		parents2[root] = root
		fringe = SpParVec(nvertG)
		fringe[root] = root
		levels = ParVec.zeros(nvertG) - 1
		levels[root] = 0
	
		level = 1
		while fringe.nnn() > 0:
			fringe.spRange()
			#ToDo:  create PCB graph-level op
			self._spm.SpMV_SelMax_inplace(fringe._spv)
			#ToDo:  create PCB graph-level op
			pcb.EWiseMult_inplacefirst(fringe._spv, parents2._dpv, True, -1)
			parents2[fringe] = fringe
			levels[fringe] = level
			level += 1
		if not sym:
			self.reverseEdges()
		del parents2
		if (levels[treeI]-levels[treeJ] != -1).any():
			return (-2, None)
	
		return (ret, levels)
	
	# returns a Boolean vector of which vertices are neighbors
	def neighbors(self, source, nhop=1, sym=False):
		"""
		calculates, for the given DiGraph instance and starting vertices,
		the vertices that are neighbors of the starting vertices (i.e.,
		reachable within nhop hops in the graph).

		Input Arguments:
			self:  a DiGraph instance
			source:  a Boolean ParVec with True (1) in the positions
			    of the starting vertices.  
			nhop:  a scalar integer denoting the number of hops to 
			    use in the calculation. The default is 1.
			sym:  a scalar Boolean denoting whether the DiGraph is 
			    symmetric (i.e., each edge from vertex i to vertex j
			    has a companion edge from j to i).  If the DiGraph 
			    is symmetric, the operation is faster.  The default 
			    is False.

		Output Arguments:
			ret:  a ParVec of length equal to the number of vertices
			    in the DiGraph, with a True (1) in each position for
			    which the corresponding vertex is a neighbor.

			    Note:  vertices from the start vector may appear in
			    the return value.

		SEE ALSO:  pathsHop
		"""
		if not sym:
			self._T()
		dest = ParVec(self.nvert(),0)
		fringe = SpParVec(self.nvert())
		fringe[source] = 1
		for i in range(nhop):
			self._spm.SpMV_SelMax_inplace(fringe._spv)
			dest[fringe.toParVec()] = 1
		if not sym:
			self._T()
		return dest
		
	# returns:
	#   - source:  a vector of the source vertex for each new vertex
	#   - dest:  a Boolean vector of the new vertices
	#ToDo:  nhop argument?
	def pathsHop(self, source, sym=False):
		"""
		calculates, for the given DiGraph instance and starting vertices,
		which can be viewed as the fringe of a set of paths, the vertices
		that are reachable by traversing one graph edge from one of the 
		starting vertices.  The paths are kept distinct, as only one path
		will extend to a given vertex.

		Input Arguments:
			self:  a DiGraph instance
			source:  a Boolean ParVec with True (1) in the positions
			    of the starting vertices.  
			sym:  a scalar Boolean denoting whether the DiGraph is 
			    symmetric (i.e., each edge from vertex i to vertex j
			    has a companion edge from j to i).  If the DiGraph 
			    is symmetric, the operation is faster.  The default 
			    is False.

		Output Arguments:
			ret:  a ParVec of length equal to the number of vertices
			    in the DiGraph.  The value of each element of the ParVec 
			    with a value other than -1 denotes the starting vertex
			    whose path extended to the corresponding vertex.  In
			    the case of multiple paths potentially extending to
			    a single vertex, the highest-numbered starting vertex
			    is chosen as the source. 

		SEE ALSO:  neighbors
		"""
		if not sym:
			self._T()
		#HACK:  SelMax is actually doing a Multiply instead of a Select,
		#    so it doesn't work "properly" on a general DiGraph, whose
		#    values can't be counted on to be 1.  So, make a copy of
		#    the DiGraph and set all the values to 1 as a work-around. 
		self2 = self.copy()
		self2.ones()
		ret = ParVec(self2.nvert(),-1)
		fringe = source.find()
		fringe.spRange()
		self2._spm.SpMV_SelMax_inplace(fringe._spv)
		ret[fringe] = fringe
		if not sym:
			self._T()
		return ret


	def normalizeEdgeWeights(self, dir=Out):
		"""
		Normalize the outward edge weights of each vertex such
		that for Vertex v, each outward edge weight is
		1/outdegree(v).
		"""
		degscale = self.degree(dir)
		degscale._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0), pcb.identity(), pcb.bind1st(pcb.divides(), 1)))			
		self.scale(degscale, dir)
		
	def pageRank(self, epsilon = 0.1, dampingFactor = 0.85):
		"""
		Compute the PageRank of vertices in the graph.

		The PageRank algorithm computes the percentage of time
		that a "random surfer" spends at each vertex in the
		graph. If the random surfer is at Vertex v, she will
		take one of two actions:
		    1) She will hop to another vertex to which Vertex
                       v has an outward edge. Self loops are ignored.
		    2) She will become "bored" and randomly hop to any
                       vertex in the graph. This action is taken with
                       probability (1 - dampingFactor).

		When the surfer is visiting a vertex that is a sink
		(i.e., has no outward edges), she hops to any vertex
		in the graph with probability one.

		Optional argument epsilon controls the stopping
		condition. Iteration stops when the 1-norm of the
		difference in two successive result vectors is less
		than epsilon.

		Optional parameter dampingFactor alters the results
		and speed of convergence, and in the model described
		above dampingFactor is the percentage of time that the
		random surfer hops to an adjacent vertex (rather than
		hopping to a random vertex in the graph).

		See "The PageRank Citation Ranking: Bringing Order to
		the Web" by Page, Brin, Motwani, and Winograd, 1998
		(http://ilpubs.stanford.edu:8090/422/) for more
		information.
		"""

		# We don't want to modify the user's graph.
		G = self.copy()
		G._T()
		nvert = G.nvert()

		# Remove self loops.
		G.removeSelfLoops()

		# Handle sink nodes (nodes with no outgoing edges) by
		# connecting them to all other nodes.

		sinkV = G.degree(DiGraph.In)
		sinkV._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(), 0), pcb.set(0), pcb.set(1./nvert)))

		# Normalize edge weights such that for each vertex,
		# each outgoing edge weight is equal to 1/(number of
		# outgoing edges).
		G.normalizeEdgeWeights(DiGraph.In)

		# PageRank loop.
		delta = 1
		dv1 = ParVec(nvert, 1./nvert)
		v1 = dv1.toSpParVec()
		prevV = SpParVec(nvert)
		onesVec = SpParVec.ones(nvert)
		dampingVec = onesVec * ((1 - dampingFactor)/nvert)
		while delta > epsilon:
			prevV = v1.copy()
			v2 = G._spm.SpMV_PlusTimes(v1._spv)

			# Compute the inner product of sinkV and v1.
			sinkContrib = sinkV.copy()
			sinkContrib._dpv.EWiseApply(v1._spv, pcb.multiplies())
			sinkContrib = sinkContrib._dpv.Reduce(pcb.plus())
			
			v1._spv = v2 + (onesVec*sinkContrib)._spv
			v1 = v1*dampingFactor + dampingVec
			delta = (v1 - prevV)._spv.Reduce(pcb.plus(), pcb.abs())
		return v1

		
	def centrality(self, alg, **kwargs):
		"""
		calculates the centrality of each vertex in the DiGraph instance,
		where 'alg' can be one of 
		    'exactBC':  exact betweenness centrality
		    'approxBC':  approximate betweenness centrality

		Each algorithm may have algorithm-specific arguments as follows:
		    'exactBC':  
		        normalize=True:  normalizes the values by dividing by 
		                (nVert-1)*(nVert-2)
		    'approxBC':
			sample=0.05:  the fraction of the vertices to use as sources 
				and destinations;  sample=1.0 is the same as exactBC
		        normalize=True:  normalizes the values by multiplying by 
				nVerts / (nVertsCalculated * (nVerts-1) * (nVerts-2))
		The return value is a ParVec with length equal to the number of
		vertices in the DiGraph, with each element of the ParVec containing
		the centrality value of the vertex.
		"""
		if alg=='exactBC':
			cent = DiGraph._approxBC(self, sample=1.0, **kwargs)
			#cent = DiGraph._bc(self, 1.0, self.nvert())
		elif alg=='approxBC':
			cent = DiGraph._approxBC(self, **kwargs)
		elif alg=='kBC':
			raise NotImplementedError, "k-betweenness centrality unimplemented"
		elif alg=='degree':
			raise NotImplementedError, "degree centrality unimplemented"
		else:
			raise KeyError, "unknown centrality algorithm (%s)" % alg
	
		return cent
	
	
	def _approxBC(self, sample=0.05, normalize=True, nProcs=pcb._nprocs(), memFract=0.1, BCdebug=0, batchSize=-1, retNVerts=False):
		"""
		calculates the approximate or exact (with sample=1.0) betweenness
		centrality of the input DiGraph instance.  _approxBC is an internal
		method of the user-visible centrality method, and as such is
		subject to change without notice.  Currently the following expert
		argument is supported:
		    - memFract:  the fraction of node memory that will be considered
			available for a single strip in the strip-mining
			algorithm.  Fractions that lead to paging will likely
			deliver atrocious performance.  The default is 0.1.  
		"""
		A = self.copy()
		Anv = A.nvert()
		if BCdebug>0 and master():
			print "in _approxBC, A.nvert=%d, nproc=%d" % (Anv, nProcs)

		if BCdebug>1 and master():
			print "Apply(set(1))"
		self.ones()
		#Aint = self.ones()	# not needed;  Gs only int for now
		if BCdebug>1 and master():
			print "spm.getnrow and col()"
		N = A.nvert()
		if BCdebug>1 and master():
			print "densevec(%d, 0)"%N
		bc = ParVec(N)
		if BCdebug>1 and master():
			print "getnrow()"
		nVertToCalc = int(math.ceil(self.nvert() * sample))
		nVertToCalc = min(nVertToCalc, self.nvert())
		
		# batchSize = #rows/cols that will fit in memory simultaneously.
		# bcu has a value in every element, even though it's literally
		# a sparse matrix (DiGraph).  So batchsize is calculated as
		#   nrow = memory size / (memory/row)
		#   memory size (in edges)
		#        = 2GB * 0.1 (other vars) / 18 (bytes/edge) * nProcs
		#   memory/row (in edges)
		#        = self.nvert()
		physMemPCore = 2e9; bytesPEdge = 18
		if (batchSize < 0):
			batchSize = int(2e9 * memFract / bytesPEdge * nProcs / N)
		batchSize = min(nVertToCalc, batchSize)
		
		nBatches = int(math.ceil(float(nVertToCalc) / float(batchSize)))
		nPossBatches = int(math.ceil(float(N) / float(batchSize)))
		
		# sources for the batches
		# the i-th batch is defined as randVerts[ startVs[i] to (startVs[i]+numV[i]) ]
		randVerts = ParVec.range(Anv)
		
		if master():
			print "NOTE! SKIPPING RANDPERM()! starting vertices will be sequential."
		#randVerts.randPerm()
		
		if (batchSize >= nVertToCalc):
			startVs = [0]
			endVs = [nVertToCalc]
			numVs = [nVertToCalc]
		else: #elif sample == 1.0:
			startVs = range(0,nVertToCalc,batchSize)
			endVs = range(batchSize, nVertToCalc, batchSize)
			endVs.append(nVertToCalc)
			numVs = [y-x for [x,y] in zip(startVs,endVs)]
		if False: #else:
			if BCdebug>1 and master():
				print "densevec iota(0, %d) (i think in that order)"%nPossBatches
			perm = ParVec.range(nPossBatches)
			if BCdebug>1 and master():
				print "densevec randperm()"
			perm.randPerm()
			#   ideally, could use following 2 lines, but may have
			#   only 1-2 batches, which makes index vector look
			#   like a Boolean, which doesn't work right
			#startVs = ParVec.range(nBatches)[perm[ParVec.range(nBatches]]
			#numVs = [min(x+batchSize,N)-x for x in startVs]
			if BCdebug>1 and master():
				print "densevec iota(0, %d) (i think in that order)"%nPossBatches
			tmpRange = ParVec.range(nPossBatches)
			if BCdebug>1 and master():
				print "densevec(%d, 0)"%nBatches
			startVs = ParVec.zeros(nBatches)
			if BCdebug>1 and master():
				print "densevec(%d, 0)"%nBatches
			numVs = ParVec.zeros(nBatches)
			for i in range(nBatches):
				if BCdebug>1 and master():
					print "dense vec SubsRef, GetElement and SetElement"
				startVs[i] = tmpRange[perm[i]]*batchSize
				if BCdebug>1 and master():
					print "dense vec GetElement and SetElement"
				numVs[i] = min(startVs[i]+batchSize,N)-startVs[i]

		if BCdebug>0 and master():
			print "batchSz=%d, nBatches=%d, nPossBatches=%d" % (batchSize, nBatches, nPossBatches)
		if BCdebug>1 and master():
			print "summary of batches:"
			print "startVs:",startVs
			print "  numVs:", numVs
		for [startV, numV] in zip(startVs, numVs):
			startV = int(startV); numV = int(numV)
			if BCdebug>0 and master():
				print "startV=%d, numV=%d" % (startV, numV)
			bfs = []		
			batchRange = ParVec.range(startV, startV+numV)
			batch = randVerts[batchRange]
			curSize = len(batch)
			#next:  nsp is really a SpParMat
			nsp = DiGraph(ParVec.range(curSize), batch, 1, curSize, N)
			#next:  fringe should be Vs; indexing must be impl to support that; seems should be a collxn of spVs, hence a SpParMat
			fringe = A[batch,ParVec.range(N)]
			depth = 0
			while fringe.nedge() > 0:
				before = time.time()
				depth = depth+1
				if BCdebug>1 and depth>1:
					nspne = tmp.nedge(); tmpne = tmp.nedge(); fringene = fringe.nedge()
					if master():
					    print "BC: in while: depth=%d, nsp.nedge()=%d, tmp.nedge()=%d, fringe.nedge()=%d" % (depth, nspne, tmpne, fringene)
				nsp += fringe
				tmp = fringe.copy()
				tmp.ones()
				bfs.append(tmp)
				#next:  changes how???
				tmp = fringe._SpGEMM(A)
				if BCdebug>1:
					#nspsum = nsp.sum(DiGraph.Out).sum() 
					#fringesum = fringe.sum(DiGraph.Out).sum()
					#tmpsum = tmp.sum(DiGraph.Out).sum()
					if master():
						#print depth, nspsum, fringesum, tmpsum
						pass
				# prune new-fringe to new verts
				fringe = tmp.mulNot(nsp)
				if BCdebug>1 and master():
					print "    %f seconds" % (time.time()-before)
	
			bcu = DiGraph.fullyConnected(curSize,N)
			# compute the bc update for all vertices except the sources
			for depth in range(depth-1,0,-1):
				# compute the weights to be applied based on the child values
				w = bfs[depth] / nsp 
				w *= bcu
				if BCdebug>2:
					tmptmp = w.sum(DiGraph.Out).sum()
					if master():
						print tmptmp
				# Apply the child value weights and sum them up over the parents
				# then apply the weights based on parent values
				w._T()
				w = A._SpGEMM(w)
				w._T()
				w *= bfs[depth-1]
				w *= nsp
				bcu += w
	
			# update the bc with the bc update
			if BCdebug>2:
				tmptmp = bcu.sum(DiGraph.Out).sum()
				if master():
					print tmptmp
			bc = bc + bcu.sum(DiGraph.In)	# column sums
	
		# subtract off the additional values added in by precomputation
		bc = bc - nVertToCalc
		if normalize:
			nVertSampled = sum(numVs)
			bc = bc * (float(N)/float(nVertSampled*(N-1)*(N-2)))
		
		if retNVerts:
			return bc,nVertSampled
		else:
			return bc
	
	def cluster(self, alg, **kwargs):
	#		ToDo:  Normalize option?
		"""
		Deferred implementation for KDT v0.1
			
		"""
		if alg=='Markov' or alg=='markov':
			G = DiGraph._markov(self, **kwargs)
			clus = G.connComp()
			return clus, G
	
		elif alg=='kNN' or alg=='knn':
			raise NotImplementedError, "k-nearest neighbors clustering not implemented"
	
		else:
			raise KeyError, "unknown clustering algorithm (%s)" % alg
	
		return clus

	def connComp(self, sym=False):
		"""
		Finds the connected components of the graph by BFS.
		Output Arguments:
			ret:  a ParVec of length equal to the number of vertices
			    in the DiGraph.  The value of each element of the ParVec 
			    denotes a cluster root for that vertex.
		"""
		
		G = self.copy()
		n = G.nvert()
		
		if not sym:
			temp = self.copy()
			temp._T()
			G += temp
			
		G += DiGraph(ParVec.range(n), ParVec.range(n), ParVec.ones(n), n)
		G._spm.Apply(pcb.set(1))
		
		# Future: use a dense accumulator and a sparse frontier to take advantage
		# of vertices that are found in the correct component and will not be
		# reshuffled.
		#component = ParVec.range(G.nvert())
		#frontier = component.toSpParVec()._spv
		frontier = SpParVec.range(n)._spv
		
		def iterop(vals):
			vals[1] = int(vals[0] != vals[1])
		
		delta = 1
		while delta > 0:
			last_frontier = frontier
			frontier = G._spm.SpMV(frontier, pcb.SecondMaxSemiring())

			pcb.EWise(iterop, [pcb.EWise_OnlyNZ(frontier), last_frontier])
			delta = last_frontier.Reduce(pcb.plus())
		
		return ParVec.toParVec(frontier.dense())
	
	def __findLargestComponent(self, sym=False): # deprecated
		"""
		Returns a subgraph that consists of the largest component of self. 
		Components are found on an undirected version of self.
		Output Arguments:
			ret:  a DiGraph consisting of the largest connected component
			    in this graph.
		"""
		
		if sym:
			G = self
		else:
			G = self.copy()
			G._T()
			G += self
		components = G.connComp(sym=True)
		n = self.nvert()
		
		# Find the largest component
		
		# Count the number of elements in each component
		countM = DiGraph(components, ParVec.range(n), ParVec.ones(n), n)
		counts = countM._spm.Reduce(pcb.pySpParMat.Row(), pcb.plus())
		
		# Find the element with the largest count
		maxCount = counts.Reduce(pcb.max())
		maxV = counts.FindInds(pcb.bind2nd(pcb.equal_to(), maxCount))[0]
		
		# Create a list of vertices in this component
		verts = components._dpv.FindInds(pcb.bind2nd(pcb.equal_to(), maxV))
		verts = ParVec.toParVec(verts)
		
		return verts

	def _markov(self, expansion=2, inflation=2, addSelfLoops=False, selfLoopWeight=1, prunelimit=0.00001, sym=False, retNEdges=False):
		"""
		Performs Markov Clustering (MCL) on self and returns a graph representing the clusters.
		"""
		
		EPS = 0.001
		#EPS = 10**(-100)
		chaos = 1000
		
		#Check parameters
		if expansion <= 1:
			raise KeyError, 'expansion parameter must be greater than 1'
		if inflation <= 1:
			raise KeyError, 'inflation parameter must be greater than 1'
		
		A = self.copy()
		#if not sym:
			#A = A + A.Transpose() at the points where A is 0 or null
		
		#Add self loops
		N = A.nvert()
		if addSelfLoops:
			A.addSelfLoops(selfLoopWeight)
		
		#Create stochastic matrix
	
		#Avoid divide-by-zero error
		sums = A.sum(DiGraph.In)
		sums._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0),
			pcb.set(1),
			pcb.identity()))
		
		A.scale( ParVec.ones(A.nvert()) / sums , dir=DiGraph.In )
		
		if retNEdges:
			nedges = 0
		
		#Iterations tally
		iterNum = 0
		
		#MCL Loop
		while chaos > EPS and iterNum < 300:
			iterNum += 1;
		
			#Expansion - A^(expansion)
			if retNEdges:
				AA = A.copy()
			for i in range(1, expansion):
				if retNEdges:
					AA._spm.Apply(pcb.set(1))
					AA = AA._SpGEMM(AA)
					nedges += AA.sum(DiGraph.In)._dpv.Reduce(pcb.plus())
				A = A._SpGEMM(A)
		
			#Inflation - Hadamard power - greater inflation parameter -> more granular results
			A._spm.Apply(pcb.bind2nd(pcb.pow(), inflation))
			
			#Re-normalize
			sums = A.sum(DiGraph.In)
			sums._dpv.Apply(pcb.ifthenelse(pcb.bind2nd(pcb.equal_to(), 0),
				pcb.set(1),
				pcb.identity()))

			A.scale( ParVec.ones(A.nvert()) / sums, dir=DiGraph.In)
		
			#Looping Condition:
			colssqs = A._spm.Reduce(pcb.pySpParMat.Column(),pcb.plus(), pcb.bind2nd(pcb.pow(), 2))
			colmaxs = A._spm.Reduce(pcb.pySpParMat.Column(), pcb.max(), 0.0)
			chaos = ParVec.toParVec(colmaxs - colssqs).max()
			#print "chaos=",chaos

			# Pruning implementation - switch out with TopK / give option
			A._spm.Prune(pcb.bind2nd(pcb.less(), prunelimit))
			#print "number of edges remaining =", A._spm.getnee()
		
		#print "Iterations = %d" % iterNum
		
		if retNEdges:
			return A,nedges

		return A
