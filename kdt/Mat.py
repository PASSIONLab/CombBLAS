import math
#from Graph import master
from Vec import Vec
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binaryObj
from Util import _op_make_binary_pred
from Util import _sr_get_python_mul
from Util import _sr_get_python_add
from Util import _makePythonOp

import kdt.pyCombBLAS as pcb

import time

class Mat:
	Column  = pcb.pySpParMat.Column()
	Row = pcb.pySpParMat.Row()
	All = Row + Column + 1

	# NOTE:  for any vertex, out-edges are in the column and in-edges
	#	are in the row
	def __init__(self, i=None, j=None, v=None, n=None, m=None, element=0):
		"""
		FIX:  doc
		creates a new Mat instance.  Can be called in one of the 
		following forms:

	Mat():  creates an empty Mat instance with no elements.

	Mat(i, j, v, n)
	Mat(i, j, v, n, m)
		create a Mat instance with n columns, m rows, and initialized
		by the tuples i,j,v. 
		edges with source represented by 
		each element of sourceV and destination represented by each 
		element of destV with weight represented by each element of 
		weightV.  In the 4-argument form, the resulting Mat will 
		have n out- and in-vertices.  In the 5-argument form, the 
		resulting Mat will have n out-vertices and m in-vertices.

		Input Arguments:
			i:  a Vec containing integers denoting the 
			    (0-based) row of each element.
			i:  a Vec containing integers denoting the 
			    (0-based) column of each element.
			v:  a Vec containing the value of each element.
			n:  an integer scalar denoting the number of columns
			    (out-vertices for an adjacency matrix) 
			    (and also rows in the 4-argument case).
			m:  an integer scalar denoting the number of rows
			    (in-vertices for an adjacency matrix)

		Output Argument:  
			ret:  a Mat instance

		Note:  If two or more elements (tuples) have the same i and j
		values then their values are summed.

		SEE ALSO:  toVec
		"""
		if m is None and n is not None:
			m = n
		if i is None:
			if n is not None: #create a Mat with an underlying pySpParMat* of the right size with no nonnulls
				nullVec = pcb.pyDenseParVec(0,0)
			if isinstance(element, (float, int, long)):
				if n is None:
					self._m_ = pcb.pySpParMat()
				else:
					self._m_ = pcb.pySpParMat(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, bool):
				if n is None:
					self._m_ = pcb.pySpParMatBool()
				else:
					self._m_ = pcb.pySpParMatBool(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj1):
				if n is None:
					self._m_ = pcb.pySpParMatObj1()
				else:
					self._m_ = pcb.pySpParMatObj1(m,n,nullVec,nullVec, nullVec)
			elif isinstance(element, pcb.Obj2):
				if n is None:
					self._m_ = pcb.pySpParMatObj2()
				else:
					self._m_ = pcb.pySpParMatObj2(m,n,nullVec,nullVec, nullVec)
			self._identity_ = element
		elif i is not None and j is not None and n is not None:
			#j = sourceV
			#i = destV
			#v = valueV
			if type(v) == tuple and isinstance(element,(float,int,long)):
				raise NotImplementedError, 'tuple valueV only valid for Obj element'
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = Vec(len(i), v, sparse=False)
#			if i.max() > n-1:
#				raise KeyError, 'at least one first index greater than #vertices'
#			if j.max() > n-1:
#				raise KeyError, 'at least one second index greater than #vertices'
			
			if i.isObj() or j.isObj():
				raise ValueError, "sourceV and destV cannot be objects!"
			if i.isSparse() or j.isSparse():
				raise ValueError, "sourceV and destV cannot be sparse!"
			
			if isinstance(v, Vec):
				element = v._identity_
				if v.isSparse():
					raise ValueError, "valueV cannot be sparse!"
			else:
				element = v
			
			if isinstance(element, (float, int, long)):
				self._m_ = pcb.pySpParMat(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, bool):
				self._m_ = pcb.pySpParMatBool(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj1):
				self._m_ = pcb.pySpParMatObj1(m,n,i._v_,j._v_,v._v_)
			elif isinstance(element, pcb.Obj2):
				self._m_ = pcb.pySpParMatObj2(m,n,i._v_,j._v_,v._v_)
			self._identity_ = Mat._getExampleElement(self._m_)
		else:
			raise ValueError, "Incomplete arguments to Mat()"
	
	@staticmethod
	def _getExampleElement(pcbMat):
		if isinstance(pcbMat, pcb.pySpParMat):
			return 0.0
		if isinstance(pcbMat, pcb.pySpParMatBool):
			return True
		if isinstance(pcbMat, pcb.pySpParMatObj1):
			return pcb.Obj1()
		if isinstance(pcbMat, pcb.pySpParMatObj2):
			return pcb.Obj2()
		raise NotImplementedError, 'Unknown vector type!'

	@staticmethod
	def _toMat(pcbMat):
		ret = Mat()
		ret._m_ = pcbMat
		ret._identity_ = Mat._getExampleElement(pcbMat)
		return ret

	def copy(self, element=None):
		"""
		creates a deep copy of a DiGraph instance.

		Input Argument:
			self:  a DiGraph instance.

		Output Argument:
			ret:  a DiGraph instance containing a copy of the input.
		"""

		if hasattr(self,'_filter_'):
#			if type(self.nvert()) is tuple:
#				raise NotImplementedError, 'only square Mats for now'
			class tmpU:
				_filter_ = self._filter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._filter_)):
						if not tmpU._filter_[i](x):
							return True
					return False
			tmpInstance = tmpU()
			#ret = Mat._toMat(self._m_.copy())
			#ret._m_.Prune(pcb.unaryObjPred(tmpInstance.fn))
			ret = self._prune(pcb.unaryObjPred(tmpInstance.fn), False, ignoreFilter=True)
		else:
			ret = Mat._toMat(self._m_.copy())
		
		# TODO: integrate filter/copy and element conversion
		# so they are not a separate steps that make two copies of the matrix.
		if element is not None and type(self._identity_) is not type(element):
			#if not isinstance(element, (bool, float, int, long)):
				# because EWiseApply(pySpParMat,pySpParMatObj)
				#   applies only where the first argument has
				#   non-nulls;  the only way I know to avoid
				#   is to use the result of 
				#   pySpParMat(pySpParMatObj), which
				#   only works for converting to doubleints
			#	raise NotImplementedError, 'can only convert to long for now'
			
			if isinstance(element, bool):
				ret = Mat._toMat(pcb.pySpParMatBool(ret._m_))
			elif isinstance(element, (float, int, long)):		
				# FIX: remove following 2 lines when EWiseApply works 
				#   as noted above 
				ret = Mat._toMat(pcb.pySpParMat(ret._m_))
				
				if self.isObj():
					def func(x, y): 
						#ToDo:  assumes that at least x or y is an ObjX
						if isinstance(x,(float,int,long)):
							ret = y.coerce(x, False)
						else:
							ret = x.coerce(y, True)
						return ret
					ret = ret.eWiseApply(ret, func, True, True)
			else: # object matrix
				raise NotImplementedError, 'can only convert to long for now'
				
		return ret

	def isObj(self):
		return not isinstance(self._identity_, (float, int, long, bool))
		#try:
		#	ret = hasattr(self,'_elementIsObject') and self._elementIsObject
		#except AttributeError:
		#	ret = False
		#return ret

	def isBool(self):
		return isinstance(self._m_, (pcb.pySpParMatBool))

	def isScalar(self):
		return isinstance(self._m_, (pcb.pySpParMat))

	def nrow(self):
		return self._m_.getnrow()
		
	def ncol(self):
		return self._m_.getncol()
	
	def nnn(self):
		"""
		returns the number of existing elements in this matrix.
		"""
		if self._hasFilter():
			if self._hasMaterializedFilter():
				return self._materialized.nnn()
			return int(self.reduce(Mat.All, (lambda x,y: x+y), uniOp=(lambda x: 1), init=0))
			
		return self._m_.getnee()		
	
	def toBool(self, inPlace=True):
		"""
		converts the Mat instance in-place such that each element only has
		a Boolean (True) value, thereby consuming less space and making
		some operations faster.

		Input Argument:
			self:  a Mat instance that is overwritten by the method

		Output Argument:
			None.
		"""
		
		if inPlace:
			if not self.isBool():
				self._m_ = pcb.pySpParMatBool(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMatBool(self._m_))

	def toScalar(self, inPlace=True):
		"""
		converts the Mat instance in-place such that each element only has
		a scalar (64-bit) value.

		Input Argument:
			self:  a Mat instance that is overwritten by the method

		Output Argument:
			None.
		"""
		
		if inPlace:
			if not self.isScalar():
				self._m_ = pcb.pySpParMat(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMat(self._m_))

	def toVec(self):
		"""
		decomposes a DiGraph instance to 3 Vec instances, with each
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
		if self._hasFilter():
			mat = self.copy()
		else:
			mat = self
			
		ne = mat.nnn()
		if ne != 0:
			reti = Vec(ne, element=0, sparse=False)
			retj = Vec(ne, element=0, sparse=False)
			retv = Vec(ne, element=self._identity_, sparse=False)
			mat._m_.Find(reti._v_, retj._v_, retv._v_)
		else:
			reti = Vec(0, sparse=False)
			retj = Vec(0, sparse=False)
			retv = Vec(0, element=self._identity_, sparse=False)
		#ToDo:  return nvert() of original graph, too
		return (reti, retj, retv)

	# NEEDED: tests
	#ToDo:  put in method to modify _REPR_MAX

	def _reprHeader(self):
		nnn = self.nnn()
		ret = "" + str(self.nrow()) + "-by-" + str(self.ncol()) + " (row-by-col) Mat with " + str(nnn) + " elements.\n"
		if self._hasFilter():
			nee = self._m_.getnee()
			ret += "%d filter(s) remove %d additional elements (%d total elements stored).\n"%(len(self._filter_), (nee-nnn), (nee))
		return ret
	
	def _reprTuples(self):
		[i, j, v] = self.toVec()
		ret = self._reprHeader()
		print ret
		print "i (row index): ", repr(i)
		print "j (col index): ", repr(j)
		print "v (value)    : ", repr(v)
		#return ret + repr(i) + repr(j) + repr(v)
		return ""

	def _reprGrid(self):
		[i, j, v] = self.toVec()
		ret = self._reprHeader()

		# make empty 2D array, I'm sure there's a more proper way to initialize it
		mat = []
		widths = []
		for rowc in range(self.nrow()):
			r = []
			for colc in range(self.ncol()):
				r.append("-")
				if rowc == 0:
					widths.append(1)
			mat.append(r)
		
		# manually fill the array with matrix values
		for count in range(len(i)):
			mat[int(i[count])][int(j[count])] = str(v[count])
			widths[int(j[count])] = max(widths[int(j[count])], len(mat[int(i[count])][int(j[count])]))
		
		# print row by row
		for rowc in range(self.nrow()):
			for colc in range(self.ncol()):
				ret += (mat[rowc][colc]).center(widths[colc]) + "  "
			ret += "\n"
		
		return ret
				
	_REPR_MAX = 400
	def __repr__(self):

		if self.ncol() < 20:
			# pretty print a nice matrix
			return self._reprGrid()
		elif self.nnn() < self._REPR_MAX:
			return self._reprTuples()
		else:
			ret = self._reprHeader()
			return ret + "Too many elements to print."

	# NEEDED: support for filters
	@staticmethod
	def load(fname, element=0.0):
		"""
		loads the contents of the file named fname (in the Coordinate Format 
		of the Matrix Market Exchange Format) into a Mat instance.

		Input Argument:
			fname:  a filename from which the matrix data will be loaded.
		Output Argument:
			ret:  a Mat instance containing the matrix represented
			    by the file's contents.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The load
		method makes this conversion while reading the data and creating
		the graph.

		SEE ALSO:  save, UFget
		"""
		# Verify file exists.
		# TODO: make pcb load throw an error if the file cannot be opened.
		file = open(fname, 'r')
		file.close()
		
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = Mat(element=element)
		ret._m_.load(fname)
		return ret

	# NEEDED:  support for filters
	def save(self, fname):
		"""
		saves the contents of the passed Mat instance to a file named
		fname in the Coordinate Format of the Matrix Market Exchange Format.

		Input Arguments:
			self:  a DiGraph instance
			fname:  a filename to which the matrix data will be saved.

		NOTE:  The Matrix Market format numbers vertex numbers from 1 to
		N.  Python and KDT number vertex numbers from 0 to N-1.  The save
		method makes this conversion while writing the data.

		SEE ALSO:  load, UFget
		"""
		if self._hasMaterializedFilter():
			self._materialized.save(fname)
			return

		if self._hasFilter():
			mat = self.copy()
			mat._m_.save(fname)
		else:
			self._m_.save(fname)

###########################
### Generators
###########################

	@staticmethod
	def generateRMAT(scale, fillFactor=16, initiator=[.57, .19, .19, .05], delIsolated=True, element=True):
		"""
		generates a Kroenecker product matrix using the Graph500 RMAT graph generator.
		
		Output Argument:
			ret: a tuple: Mat, Vec, time.
				Mat is the matrix itself,
				Vec is a vector containing the degrees of each vertex in the original graph
				time is the time for the Graph500 Kernel 1.
		"""
		degrees = Vec(0, element=1.0, sparse=False)
		matrix = Mat(element=element)
		if matrix.isObj():
			raise NotImplementedError,"RMAT generation to supported on object matrices yet."
		kernel1Time = matrix._m_.GenGraph500Edges(scale, degrees._v_, fillFactor, delIsolated, initiator[0], initiator[1], initiator[2], initiator[3])
		return matrix, degrees, kernel1Time

	@staticmethod
	def eye(n, m=None, element=1.0):
		"""
		creates an identity matrix. The resulting matrix is n-by-n
		with `element` on the main diagonal.
		In other words, ret[i,i] = element for 0 < i < n.

		Input Arguments:
			n:  number of columns of the matrix
			m:  number of rows of the matrix
			element: the value to put on the main diagonal elements.

		Output Argument:
			ret:  an identity matrix. 
		"""
		if m is None:
			m = n
		nnz = min(n, m)
		if nnz <= 0:
			raise KeyError,"need n > 0 and m > 0"
		return Mat(Vec.range(nnz),Vec.range(nnz),Vec(nnz, element, sparse=False),n, m)

	@staticmethod
	def ones(n,m=None, element=1.0):
		"""
		creates an `m`-by-`n` matrix with all elements set to `element`.

		Input Arguments:
			n:  an integer specifying the number of rows in the matrix.
			m:  an integer specifying the number of columns in the matrix.
			    If omitted, it defaults to the same value as `n`.
		element: the value to set every Mat element to. Default is 1.0.

		Output Argument:
			ret:  an m-by-n full Mat instance. 
		"""
		
		if m is None:
			m = n
		i = (Vec.range(n*m) / n).floor()
		j = (Vec.range(n*m) % n).floor()
		v = Vec.ones(n*m, element=element)
		ret = Mat(i,j,v,n,m)
		return ret

##########################
### Filtering Methods
##########################

	# in-place, so no return value
	def addFilter(self, filter):
		"""
		adds a vertex filter to the Mat instance.  

		A vertex filter is a Python function that is applied elementally
		to each vertex in the Mat, with a Boolean True return value
		causing the vertex to be considered and a False return value
		causing it not to be considered.

		Vertex filters are additive, in that each vertex must pass all
		filters to be considered.  All vertex filters are executed before
		a vertex is considered in a computation.
#FIX:  how is an argument passed to the function?

		Input Arguments:
			self:  a Mat instance
			filter:  a Python function

		SEE ALSO:
			delEFilter  
		"""
		if hasattr(self, '_filter_'):
			self._filter_.append(filter)
		else:
			self._filter_ = [filter]
		
		self._dirty()
		return
	
	def _addSpMVFilteredVec(self, vec):
		"""
		Adds support for matching a vector's filtering during SpMV. I.e. so that if vec[i] is
		filtered out then row and column i of `self` will also be filtered out. Necessary to
		properly filter out graph vertices if this Mat represents an adjacency matrix.
		"""
		
		self._dirty()
		return
		
	# in-place, so no return value
	def delFilter(self, filter=None):
		"""
		deletes a filter from the Mat instance.  

		Input Arguments:
			filter:  either a Python predicate function which has
			    been previoiusly added to this instance by a call to
			    addFilter or None, which signals the deletion of all
			    filters.

		SEE ALSO:
			addFilter  
		"""
		if not hasattr(self, '_filter_'):
			raise KeyError, "no filters previously created"
		if filter is None:
			del self._filter_	# remove all filters
			self.deMaterializeFilter()
		else:
			self._filter_.remove(filter)
			if len(self._filter_) == 0:
				del self._filter_
				self.deMaterializeFilter()
		
		self._dirty()
		return

	def materializeFilter(self):
		# a materialized filter is being used if the self._materialized element exists.
		self._materialized = 1
		# do the materialization
		self._dirty()
	
	def deMaterializeFilter(self):
		if self._hasMaterializedFilter():
			del self._materialized

	def _hasFilter(self):
		try:
			ret = (hasattr(self,'_filter_') and len(self._filter_)>0)
		except AttributeError:
			ret = False
		return ret
	
	def _hasMaterializedFilter(self):
		try:
			ret = hasattr(self,'_materialized')
		except AttributeError:
			ret = False
		return ret
	
	def _updateMaterializedFilter(self):
		self._materialized = self.copy()
	
	def _dirty(self):
		if self._hasMaterializedFilter():
			del self._materialized
			self._updateMaterializedFilter()
	
	_eWiseIgnoreMaterializedFilter = False
	def _copyBackMaterialized(self):
		if self._hasMaterializedFilter():
			self._eWiseIgnoreMaterializedFilter = True
			self.eWiseApply(self._materialized, (lambda s, m: m), allowANulls=False, allowBNulls=False, inPlace=True)
			self._eWiseIgnoreMaterializedFilter = False
		else:
			raise ValueError, "Internal Error: copy back a materialized filter when no materialized filter exists!"

##########################
### Basic Methods
##########################

	# NEEDED: tests
	#in-place, so no return value
	def apply(self, op):#, other=None, notB=False):
		"""
		applies the given operator to every edge in the Mat

		Input Argument:
			self:  a Mat instance, modified in place.
			op:  a Python or pyCombBLAS function

		Output Argument:  
			None.

		"""
		if self._hasMaterializedFilter():
			ret = self._materialized.apply(op)
			self._copyBackMaterialized()
			return ret
		
		if self._hasFilter():
			op = _makePythonOp(op)
			op = FilterHelper.getFilteredUniOpOrSelf(self, op)
			#if self.isSparse():
			#	op = FilterHelper.getFilteredUniOpOrSelf(self, op)
			#else:
			#	op = FilterHelper.getFilteredUniOpOrOpVal(self, op, self._identity_)
		
		self._m_.Apply(_op_make_unary(op))
		return

		if self._hasFilter():
			if self._hasMaterializedFilter():
				raise NotImplementedError, "this operation does not support materialized filters"
			raise NotImplementedError, "this operation does not support filters yet."

		if other is None:
			if not isinstance(op, pcb.UnaryFunction):
				self._m_.Apply(pcb.unaryObj(op))
			else:
				self._m_.Apply(op)
		else:
			if not isinstance(op, pcb.BinaryFunction):
				self._m_ = pcb.EWiseApply(self._m_, other._m_, pcb.binaryObj(op), notB)
			else:
				self._m_ = pcb.EWiseApply(self._m_, other._m_, op, notB)

		self._dirty()

	def count(self, dir, pred=None):
		"""
		returns the number of elements for which `pred` is true.
		"""
		if pred is None:
			pred = lambda x: bool(x)
		
		return self.reduce(dir, (lambda x,y: x+y), uniOp=pred, init=0.0)

	# NEEDED: tests
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, doOp=None, inPlace=False, allowIntersect=True, predicate=False):
		"""
		ToDo:  write doc
		
		See Also: Vec.eWiseApply
		"""
		
		if predicate:
			raise NotImplementedError, "predicate output not yet supported for Mat.eWiseApply"
		
		if not isinstance(other, Mat):
			raise KeyError, "eWiseApply works on two Mat objects."
			# use apply()?
			return

		if not self._eWiseIgnoreMaterializedFilter:
			if self._hasMaterializedFilter() and inPlace:
				raise ValueError, "Materialized filters are read-only."

			if self._hasMaterializedFilter() and not other._hasMaterializedFilter():
				return self._materialized.eWiseApply(other, op, allowANulls, allowBNulls, doOp, inPlace)
			if not self._hasMaterializedFilter() and other._hasMaterializedFilter():
				return self.eWiseApply(other._materialized, op, allowANulls, allowBNulls, doOp, inPlace)
			if self._hasMaterializedFilter() and other._hasMaterializedFilter():
				return self._materialized.eWiseApply(other._materialized, op, allowANulls, allowBNulls, doOp, inPlace)
		# else:
		#   ignoring materialized filters is used for copying data back from the materialized filter to the main Mat data structure

		ANull = self._identity_
		BNull = other._identity_
		#superOp, doOp = FilterHelper.getEWiseFilteredOps(self, other, op, doOp, allowANulls, allowBNulls, ANull, BNull, allowIntersect)
		
		##if doOp is not None:
		# new version
		if inPlace:
			self._m_ = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(op), _op_make_binary_pred(doOp), allowANulls, allowBNulls, ANull, BNull, allowIntersect, _op_make_unary_pred(FilterHelper.getFilterPred(self)), _op_make_unary_pred(FilterHelper.getFilterPred(other)))
			self._dirty()
			return
		else:
			m = pcb.EWiseApply(self._m_, other._m_, _op_make_binary(op), _op_make_binary_pred(doOp), allowANulls, allowBNulls, ANull, BNull, allowIntersect, _op_make_unary_pred(FilterHelper.getFilterPred(self)), _op_make_unary_pred(FilterHelper.getFilterPred(other)))
			ret = Mat._toMat(m)
			return ret

	# NEEDED: tests
	def __getitem__(self, key):
		"""
		FIX:  fix documentation

		implements indexing on the right-hand side of an assignment.
		Usually accessed through the "[]" syntax.

		Input Arguments:
			self:  a Mat instance
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
				- a Vec object containing a contiguous range
				  of monotonically increasing integers 
		
		Output Argument:
			ret:  a Mat instance, containing the indicated vertices
			    and their incident edges from the input Mat.

		SEE ALSO:  subgraph
		"""
		
		inPlace = False
		
		#ToDo:  accept slices for key0/key1 besides ParVecs
		if type(key)==tuple:
			if len(key)==1:
				[key0] = key; key1 = -1
			elif len(key)==2:
				[key0, key1] = key
			elif len(key)==3:
				[key0, key1, inPlace] = key
			else:
				raise KeyError, 'Too many indices'
		else:
			key0 = key;  key1 = key
		if type(key0) == int or type(key0) == long or type(key0) == float:
			tmp = Vec(1)
			tmp[0] = key0
			key0 = tmp
		if type(key1) == int or type(key0) == long or type(key0) == float:
			tmp = Vec(1)
			tmp[0] = key1
			key1 = tmp
		if type(inPlace) != bool:
			raise KeyError, 'inPlace argument must be a boolean!'
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
		

		# the key vectors passed to SubsRef cannot be filtered
		if key0._hasFilter():
			key0 = key0.copy();
		if key1._hasFilter():
			key1 = key0.copy();
		
		# CombBLAS SubsRef only takes dense vectors
		key0 = key0.dense()
		key1 = key1.dense()
		
		if inPlace:
			self._m_.SubsRef(key0._v_, key1._v_, inPlace, _op_make_unary_pred(FilterHelper.getFilterPred(self)))
			self._dirty()
		else:
			if self._hasMaterializedFilter():
				return self._materialized.__getitem__(key)

			ret = Mat._toMat(self._m_.SubsRef(key0._v_, key1._v_, inPlace, _op_make_unary_pred(FilterHelper.getFilterPred(self))))
			return ret
		
	# TODO: make a _keep() which reverses the predicate
	def _prune(self, pred, inPlace=True, ignoreFilter=False):
		"""
		only keep elements for which pred(e) == false.
		"""
		if not ignoreFilter and self._hasFilter():
			raise NotImplementedError,"_prune() doesn't do filters"
			
		return Mat._toMat(self._m_.Prune(_op_make_unary_pred(pred), inPlace))

	def reduce(self, dir, op, uniOp=None, init=None):
		"""
		Accumulate matrix elements along the specified dimension.
		This function is equivalent to this (for rows if dir=Mat.Row):
		accumulator = init
		for all elements e in row:
			accumulator = op(uniOp(e), accumulator)
			
		similarly for columns if dir=Mat.Column.
		
		ToDo:  write doc
		NOTE:  need to doc clearly that the 2nd arg to the reduction
		fn is the sum;  the first is the current addend and the second
		is the running sum
		"""
		if self._hasMaterializedFilter():
			return self._materialized.reduce(dir, op, uniOp, init)
		
		if dir != Mat.Row and dir != Mat.Column and dir != Mat.All:
			raise KeyError, 'unknown direction'
		
		if init is None:
			init = self._identity_
		
		if type(init) is not type(self._identity_) and not isinstance(init, (float, int, long)):
			raise NotImplementedError, "Reduce output type must either match the matrix type or be float."
		
		uniOpOrig = uniOp
		if self._hasFilter():
			if uniOp is None:
				uniOp = (lambda x: x)
			else:
				uniOp = _makePythonOp(uniOp)
			uniOp = FilterHelper.getFilteredUniOpOrVal(self, uniOp, init)
			uniOp = pcb.unaryObj(uniOp)

		ret = Vec(element=init, sparse=False)
		
		doall = False
		if dir == Mat.All:
			dir = Mat.Column
			doall = True

		self._m_.Reduce(dir, ret._v_, _op_make_binaryObj(op), _op_make_unary(uniOp), init)
		if doall:
			ret = ret.reduce(_op_make_binaryObj(op), None, init)
		return ret

	#in-place, so no return value
	def scale(self, other, op=op_mul, dir=Column):
		"""
		multiplies the weights of the appropriate edges of each vertex of
		the passed Mat instance in-place by a vertex-specific scale 
		factor.
		
		This operation is equivalent to:
		for all i,j:
			M[i,j] = op(M[i,j], other[i or j])
		(where [i or j] depends on the direction dir)
		
		Input Arguments:
			other: a Vec whose elements are used
			op:   the operation to perform (default is multiplication)
			dir:  a direction of edges to scale, with choices being
			    Mat.Column (default) or Mat.Row.

		Output Argument:
			None.
		
		"""
		if self._hasMaterializedFilter():
			ret = self._materialized.scale(other, op, dir)
			self._copyBackMaterialized()
			return ret

		if self.isBool():
			raise NotImplementedError, 'scale not implemented on boolean matrices do to C++ template irregularities.'
		
		if not isinstance(other, Vec):
			raise KeyError, 'Invalid type for scale vector'

		if dir == Mat.Column:
			if self.ncol() != len(other):
				raise IndexError, 'ncol != len(vec)'
		elif dir == Mat.Row:
			if self.nrow() != len(other):
				raise IndexError, 'nrow != len(vec)'
		else:
			raise KeyError, 'Invalid edge direction'

		if self._hasFilter():
			op = _makePythonOp(op)
			
			class tmpS:
				def __init__(self, myop, pred):
					self.pred = pred
					self.myop = myop
				def __call__(self, x, y):
					if self.pred(x):
						#print "got x=",x,"y=",y,"pred(x)==true,  returning op(x,y)"
						return self.myop(x, y)
					else:
						#print "got x=",x,"y=",y,"pred(x)==FALSE, returning x"
						return x
			
			op = tmpS(op, FilterHelper.getFilterPred(self))

		self._m_.DimWiseApply(dir, other.dense()._v_, _op_make_binary(op))
		return

	# NEEDED: tests
	def SpGEMM(self, other, semiring, inPlace=False):
		"""
		"multiplies" two Mat instances together as though each was
		represented by a sparse matrix, with rows representing in-edges
		and columns representing out-edges.
		"""
		
		# check input
		if not isinstance(other, Mat):
			raise ValueError, "SpGEMM needs a Mat"
		if self.ncol() != other.nrow():
			raise ValueError, "Dimension mismatch in SpGEMM: self.ncol() must equal other.nrow(), but %d != %d"%(self.ncol(),other.nrow())
		
		if self._hasMaterializedFilter() and not other._hasMaterializedFilter():
			return self._materialized.SpGEMM(other, semiring, inPlace)
		if not self._hasMaterializedFilter() and other._hasMaterializedFilter():
			return self.SpGEMM(other._materialized, semiring, inPlace)
		if self._hasMaterializedFilter() and other._hasMaterializedFilter():
			return self._materialized.SpGEMM(other._materialized, semiring, inPlace)

		clearSemiringFilters = False
		if self._hasFilter() or other._hasFilter():
			semiring.setFilters(FilterHelper.getFilterPred(self), FilterHelper.getFilterPred(other))
			clearSemiringFilters = True

		if False:		
			if self._hasFilter() or other._hasFilter():
				selfPred = FilterHelper.getFilterPred(self)
				if selfPred is None:
					selfPred = lambda x: True
	
				otherPred = FilterHelper.getFilterPred(other)
				if otherPred is None:
					otherPred = lambda x: True
				
				class tmpMul:
					filterA = selfPred
					filterB = otherPred
					nullval = self._identity_
					origMulFunc = _sr_get_python_mul(semiring)
					@staticmethod
					def fn(x, y):
						if tmpMul.filterA(x) and tmpMul.filterB(y):
							return tmpMul.origMulFunc(x, y)
						else:
							return tmpMul.nullval
				tmpMulInstance = tmpMul()
				semiring = sr(_sr_get_python_add(semiring), tmpMulInstance.fn)

			
		if self._m_ is other._m_:
			# we're squaring the matrix
			if inPlace:
				#cp = self.copy()
				#self._m_ = self._m_.SpGEMM(cp._m_, semiring)
				self._m_.Square(semiring)
				return self
			else:
				cp = self.copy()
				#return Mat._toMat(self._m_.SpGEMM(cp._m_, semiring))
				cp._m_.Square(semiring)
				return cp
		
		if inPlace:
			other._m_ = self._m_.SpGEMM(other._m_, semiring)
			other._dirty()
			ret = other
		else:
			ret = Mat._toMat(self._m_.SpGEMM(other._m_, semiring))

		if clearSemiringFilters:
			semiring.setFilters(None, None)
		return ret

	spGEMM = SpGEMM

	# possibly in-place;  if so, no return value
	def SpMV(self, other, semiring, inPlace=False):
		"""
		FIX:  add doc
		inPlace -> no return value
		"""
		# check input
		if not isinstance(other, Vec):
			raise ValueError, "SpMV needs a Vec"
		if len(other) != self.ncol():
			raise ValueError, "Dimension mismatch in SpMV. The number of elements of the vector must equal the number of columns of the matrix."
		
		# materialized filters
		if self._hasMaterializedFilter() and not other._hasMaterializedFilter():
			return self._materialized.SpMV(other, semiring, inPlace)
		if not self._hasMaterializedFilter() and other._hasMaterializedFilter():
			return self.SpMV(other._materialized, semiring, inPlace)
		if self._hasMaterializedFilter() and other._hasMaterializedFilter():
			return self._materialized.SpMV(other._materialized, semiring, inPlace)

		# setup on-the-fly filter
		clearSemiringFilters = False
		if self._hasFilter() or other._hasFilter():
			semiring.setFilters(FilterHelper.getFilterPred(self), FilterHelper.getFilterPred(other))
			clearSemiringFilters = True

		# the operation itself
		if inPlace:
			self._m_.SpMV_inplace(other._v_, semiring)
			ret = other
		else:
			ret = Vec._toVec(self._m_.SpMV(other._v_, semiring))
		
		# clear out on-the-fly filter
		if clearSemiringFilters:
			semiring.setFilters(None, None)
		
		return ret
		
	spMV = SpMV

	def transpose(self):
		"""
		performs an in-place transpose of this Mat instance
		"""
		self._m_.Transpose()
		self._dirty()
	


##########################
### Operations
##########################

	# NEEDED: Handle the init properly (i.e. don't use 0, will break a test)
	def max(self, dir=Column):
		"""
		finds the maximum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the maximum,
			    with choices being DiGraph.Out (default) or DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    maximum of the weights of the corresponding vertex.

		SEE ALSO:  degree, min 
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_max, init=self._identity_)
		return ret

	# NEEDED: Handle the init properly (i.e. don't use 0, will break a test)
	def min(self, dir=Column):
		"""
		finds the minimum weights of the appropriate edges of each vertex 
		of the passed DiGraph instance.

		Input Arguments:
			self:  a DiGraph instance
			dir:  a direction of edges over which to find the minimum,
			    with choices being DiGraph.Out (default), DiGraph.In.

		Output Argument:
			ret:  a ParVec instance with each element containing the
			    minimum of the weights of the corresponding vertex.

		SEE ALSO:  degree, max 
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_min, init=self._identity_)
		return ret

	def spOnes(self, element=1.0):
		"""
		makes every element to `element`, default 1.0.
		"""
		if not self.isObj() and not self._hasFilter():
			self.apply(op_set(element))
		else:
			self.apply(lambda x: element)
		self._dirty()
		
	def removeMainDiagonal(self):
		"""
		removes all elements whose row and column index are equal.
		Operation is in-place.

		Input Argument:
			self:  a Mat instance, modified in-place.

		"""
		self._m_.removeSelfLoops()
		self._dirty()

	
	def sum(self, dir=Column):
		"""
		adds the weights of the elements along a row or column.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction of elements to sum, with choices being
			    Mat.Column (default) or Mat.Row.

		Output Argument:
			ret:  a Vec instance with each element containing the
			    sum of the weights of the corresponding row/column.

		SEE ALSO:  degree 
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_add, init=self._identity_)
		return ret
	


##########################
### Arithmetic Operations
##########################

	def _ewise_bin_op_worker(self, other, func, intOnly=False, predicate=False):
		"""
		is an internal function used to implement elementwise arithmetic operators.
		"""
		funcUse = func
		if intOnly:
			# if other is a floating point, make it an int
			if isinstance(other, (float, int, long)):
				other = int(other)
			
			# if self is a float vector, add conversions to int. Object vectors are assumed to handle
			# conversion themselves.
			if not self.isObj():
				if isinstance(other, Mat) and not other.isObj():
					funcUse = lambda x, y: func(int(x), int(y))
				else:
					funcUse = lambda x, y: func(int(x), y)

		if not isinstance(other, Mat):
			# if other is a scalar, then only apply it to the nonnull elements of self.
			return self.eWiseApply(other, funcUse, allowANulls=False, allowBNulls=False, inPlace=False, predicate=predicate)
		else:
			return self.eWiseApply(other, funcUse, allowANulls=True, allowBNulls=True, inPlace=False, predicate=predicate)
	
	def __add__(self, other):
		"""
		adds the corresponding elements of two Mat instances into the
		result Mat instance, with a nonnull element where either of
		the two input vectors was nonnull.
		ToDo:  elucidate combinations, overloading, etc.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x + other))


	def __and__(self, other):
		"""
		performs a bitwise And between the corresponding elements of two
		Mat instances into the result Mat instance.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x & other), intOnly=True)

	# NEEDED: update docstring
	def __div__(self, other):
		"""
		divides each element of the first argument (a SpParVec instance),
		by either a scalar or the corresonding element of the second 
		SpParVec instance, with a non-null element where either of the 
		two input vectors was nonnull.
		
		Note:  ZeroDivisionException will be raised if any element of 
		the second argument is zero.

		Note:  For v0.1, the second argument may only be a scalar.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x / other))

	# NEEDED: update docstring
	def __eq__(self, other):
		"""
		calculates the Boolean equality of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the scalar and the corresponding
		element of the first argument are equal.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and equal.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x == other), predicate=True)

	# NEEDED: update docstring
	def __ge__(self, other):
		"""
		#FIX: doc
		calculates the Boolean greater-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than or equal to the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x >= other), predicate=True)

	# NEEDED: update docstring
	def __gt__(self, other):
		"""
		calculates the Boolean greater-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is greater than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is greater than the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x > other), predicate=True)

	def __invert__(self):
		"""
		bitwise inverts each nonnull element of the Vec instance.
		"""
		ret = self.copy()
		if isinstance(self._identity_, (float, int, long)):
			func = lambda x: int(x).__invert__()
		else:
			func = lambda x: x.__invert__()
		ret.apply(func)
		return ret

	# NEEDED: update docstring
	def __le__(self, other):
		"""
		calculates the Boolean less-than-or-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than or equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than or equal to the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x <= other), predicate=True)

	# NEEDED: update docstring
	def __lt__(self, other):
		"""
		calculates the Boolean less-than relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is less than the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is less than the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x < other), predicate=True)

	# NEEDED: update docstring
	def __mod__(self, other):
		"""
		calculates the modulus of each element of the first argument by the
		second argument (a scalar or a SpParVec instance), with a nonnull
		element where the input SpParVec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x % other))

	# NEEDED: update docstring
	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a SpParVec instance), with a nonnull element where 
		the input SpParVec argument(s) were nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x * other))


	# NEEDED: update docstring
	def __ne__(self, other):
		"""
		calculates the Boolean not-equal relationship of the first argument with the second argument 

	SpParVec == scalar
	SpParVec == SpParVec
		In the first form, the result is a SpParVec instance with the same
		length and nonnull elements as the first argument, with each nonnull
		element being True (1.0) only if the corresponding element of the 
		first argument is not equal to the scalar.
		In the second form, the result is a SpParVec instance with the
		same length as the two SpParVec instances (which must be of the
		same length).  The result will have nonnull elements where either
		of the input arguments are nonnull, with the value being True (1.0)
		only where the corresponding elements are both nonnull and the
		first argument is not equal to the second.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x != other), predicate=True)

	def __neg__(self):
		"""
		negates each nonnull element of the passed Vec instance.
		"""
		ret = self.copy()
		func = lambda x: -x
		ret.apply(func)
		return ret


	# NEEDED: update docstring
	def __or__(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where at least one of the input vectors is True.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x | other), intOnly=True)

	# NEEDED: update docstring
	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where the input SpParVec argument(s)
		are nonnull.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x - other))

	# NEEDED: update docstring
	def __xor__(self, other):
		"""
		performs a logical Xor between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where exactly one of the input vectors is True.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x ^ other))