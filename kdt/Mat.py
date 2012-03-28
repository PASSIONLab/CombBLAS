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
		creates a new matrix. In KDT, it is usually seen as an adjacency matrix
		of a (directed) graph, with the [i, j]'th element's being the weight of
		the edge from vertex j to vertex i in the corresponding graph. A matrix
		may be constructed in one of the following ways:

	Mat():  creates an empty matrix

	Mat(i, j, v, n)
	Mat(i, j, v, n, m, element=0)
		creates a matrix with n columns and m rows (or n columns and n rows if m
		is omitted). The matrix is initialized with 3-tuples (i[k], j[k], v[k])
		by assigning value v[k] to the [i[k], j[k]]'th element.

		Input Arguments:
			i:  a Vec containing column indices (correspond to destination vertices)
			j:  a Vec containing row indices (correspond to source vertices)
			v:  a Vec containing matrix element values (correspond to edge weights)
			n:  an integer number of columns (and rows if m is omitted)
			m:  an integer number of rows
			element:  a representative matrix element by which the matrix type is
				determined. Default is 0, which corresponds to a float-valued matrix.
				Other possible values include True (results in a Boolean-valued
				matrix), kdt.pyCombBLAS.Obj1, and kdt.pyCombBLAS.Obj2.

		Output Arguments:
			ret:  a Mat instance

		Note: If more than one 3-tuple address the same element in the matrix
			((x, y, value1), ..., (x, y, valueN)), then the value of the [x, y]'th
			element of the matrix equals the sum of the values of all these 3-tuples
			(value1 + ... + valueN).

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
				raise NotImplementedError, 'tuple v only valid for Obj element'
			if len(i) != len(j):
				raise KeyError, 'source and destination vectors must be of the same length'
			if type(v) == int or type(v) == long or type(v) == float:
				v = Vec(len(i), v, sparse=False)
#			if i.max() > n-1:
#				raise KeyError, 'at least one first index greater than #vertices'
#			if j.max() > n-1:
#				raise KeyError, 'at least one second index greater than #vertices'

			if i.isObj() or j.isObj():
				raise ValueError, "j and i cannot be objects!"
			if i.isSparse() or j.isSparse():
				raise ValueError, "j and i cannot be sparse!"

			if isinstance(v, Vec):
				element = v._identity_
				if v.isSparse():
					raise ValueError, "v cannot be sparse!"
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
		creates a deep copy of a matrix.

		Input Arguments:
			self:  a Mat instance
			element:  a representative matrix element by which the matrix type is
				determined. Default is None, which corresponds to a float-valued
				matrix. Other possible value is True (results in a Boolean-valued
				matrix).

		Output Arguments:
			ret:  a Mat instance representing a deep copy of the input matrix
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
		'''
		returns a Boolean indicating whether the matrix is a Object-valued.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: a Boolean
		'''
		return not isinstance(self._identity_, (float, int, long, bool))
		#try:
		#	ret = hasattr(self,'_elementIsObject') and self._elementIsObject
		#except AttributeError:
		#	ret = False
		#return ret

	def isBool(self):
		"""
		returns a Boolean indicating whether the matrix is Boolean-valued.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: a Boolean
		"""
		return isinstance(self._m_, (pcb.pySpParMatBool))

	def isScalar(self):
		"""
		returns a Boolean indicating whether the matrix is a general-purpose
		scalar-valued sparse matrix.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: a Boolean
		"""
		return isinstance(self._m_, (pcb.pySpParMat))

	def nrow(self):
		"""
		gets the number of rows in the matrix.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: an integer number of rows
		"""
		return self._m_.getnrow()
		
	def ncol(self):
		"""
		gets the number of columns in the matrix.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: an integer number of columns
		"""
		return self._m_.getncol()
	
	def nnn(self):
		"""
		gets the number of elements in the matrix.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			ret: an integer number of elements
		"""
		if self._hasFilter():
			if self._hasMaterializedFilter():
				return self._materialized.nnn()
			return int(self.reduce(Mat.All, (lambda x,y: x+y), uniOp=(lambda x: 1), init=0))
			
		return self._m_.getnee()
	
	def toBool(self, inPlace=True):
		"""
		converts each nonnull element in the matrix into a Boolean True. The
		conversion may be optionally performed in-place.

		Input Arguments:
			self:  a Mat instance
			inPlace:  a Boolean specifying whether to perform the convertion
				in-place. Default is True.

		Output Arguments:
			ret:  if inPlace=True, the function returns an instance of Mat
				representing the conversion result. Otherwise, nothing is
				returned.
		"""
		
		if inPlace:
			if not self.isBool():
				self._m_ = pcb.pySpParMatBool(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMatBool(self._m_))

	def toScalar(self, inPlace=True):
		"""
		converts each element of a matrix into a 64-bit scalar. The
		conversion may be optionally performed in-place.

		Input Arguments:
			self:  a Mat instance
			inPlace: a Boolean specifying whether to perform the convertion
				in-place. Default is True.

		Output Arguments:
			ret:  if inPlace=True, the function returns an instance of Mat
				representing the result of conversion. Otherwise, nothing
				is returned.
		"""
		
		if inPlace:
			if not self.isScalar():
				self._m_ = pcb.pySpParMat(self._m_)
				self._identity_ = Mat._getExampleElement(self._m_)
		else:
			return Mat._toMat(pcb.pySpParMat(self._m_))

	def toVec(self):
		"""
		converts a matrix into 3 vectors i, j, and v, where each 3-tuple
		(i[k], j[k], v[k]) corresponds to a nonnull [i[k],j[k]]'th element
		of the matrix equal to v[k]. Alternatively, i[k], j[k], and v[k]
		denote the source vertex, the destination vertex, and the edge weight
		in the corresponding graph.

		Input Arguments:
			self:  a Mat instance

		Output Arguments:
			ret: a 3-tuple containing three Vec instances denoting source
				vertices, destination vertices, and edge weights, respectively.

		SEE ALSO:  Mat
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
		
		# test the indecies
		for count in range(len(i)):
			if i[count] < 0 or i[count] >= self.nrow() or j[count] < 0 or j[count] >= self.ncol():
				raise ValueError,"Matrix structure error! Element %d is (%d,%d,%s), matrix is %s."%(count, i[count], j[count], str(v[count]), (str(self.nrow()) + "-by-" + str(self.ncol()) + " (row-by-col)"))
		

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

	@staticmethod
	def load(fname, element=0.0, par_IO=False):
		"""
		loads the contents of the matrix from a file fname in the
		Coordinate Format of the Matrix Market Exchange Formats
		(http://math.nist.gov/MatrixMarket/formats.html#MMformat).

		Input Arguments:
			fname:  the name of a file from which the matrix data
				will be loaded.
			element:  a representative matrix element by which the
				matrix type is determined. Default is 0.0, which
				corresponds to a float-valued matrix. Other possible
				values include True (results in a Boolean- valued
				matrix), , kdt.pyCombBLAS.Obj1, and kdt.pyCombBLAS.Obj2.
			par_IO:  a Boolean specifying whether to use parallel I/O.
				Default is False.
		Output Arguments:
			ret:  a Mat instance containing the matrix represented
			    by the file's contents.

		NOTE:  The Matrix Market Format numbers vertices from 1 to N,
			while Python and KDT do it from 0 to N-1. The load method
			converts indexes while reading the data and creating the
			matrix.

		SEE ALSO:  save, UFget
		"""
		# Verify file exists.
		# TODO: make pcb load throw an error if the file cannot be opened.
		file = open(fname, 'r')
		file.close()
		
		#FIX:  crashes if any out-of-bound indices in file; easy to
		#      fall into with file being 1-based and Py being 0-based
		ret = Mat(element=element)
		ret._m_.load(fname, par_IO)
		return ret

	def save(self, fname):
		"""
		saves the matrix to a file in the Coordinate Format of the
		Matrix Market Exchange Formats
		(http://math.nist.gov/MatrixMarket/formats.html#MMformat).

		Input Arguments:
			self:  a Mat instance
			fname:  the name of a file to which the matrix data will be
				saved.

		NOTE:  The Matrix Market Format numbers vertices from 1 to N,
			while Python and KDT do it from 0 to N-1. The save method
			converts indexes while writing the data.

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
		generates a Kronecker product matrix using the Graph500 RMAT graph generator
		(see http://www.graph500.org/reference.html).
		
		Input Arguments:
			scale:  a binary logarithm of the number of vertices in the graph
			fillFactor:  the ratio of the number of edges to the number of
				vertices in the graph.
			initiator:  a 4-tuple of initiators A, B, C, and D. Default value is
				[0.57, 0.19, 0.19, 0.05].
			delIsolated:  a Boolean indicating whether to delete isolated vertices
				from the graph.
			element:  a representative matrix element by which the matrix type is
				determined. Default is True, which corresponds to a matrix of
				Booleans. Other possible values include 1 (results in a float-
				valued matrix), kdt.pyCombBLAS.Obj1, and kdt.pyCombBLAS.Obj2.

		Output Arguments:
			ret: a 3-tuple containing a Mat, a Vec, and a time. The matrix is
				the generated matrix; the vector contains the degree of each
				vertex of the original graph; and the time is the time for the
				Graph500 Kernel 1.

		SEE ALSO:
			http://kdt.sourceforge.net/wiki/index.php/Generating_a_Random_Graph
		"""
		degrees = Vec(0, element=1.0, sparse=False)
		matrix = Mat(element=element)
		if matrix.isObj():
			raise NotImplementedError,"RMAT generation has not yet been supported for object matrices."
		kernel1Time = matrix._m_.GenGraph500Edges(scale, degrees._v_, fillFactor, delIsolated, initiator[0], initiator[1], initiator[2], initiator[3])
		return matrix, degrees, kernel1Time

	@staticmethod
	def eye(n, m=None, element=1.0):
		"""
		creates a matrix with n columns and m rows (or n rows if m is omitted)
		having `element` values on its main diagonal.

		Input Arguments:
			n:  an integer number of columns in the matrix
			m:  an integer number of rows in the matrix. If omitted, the
				value of n is used.
			element:  the value to put on the main diagonal. Default is 1.0

		Output Arguments:
			ret:  a Mat instance
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
		creates a matrix with n columns and m rows (or n rows if m is omitted)
		having all its elements set to `element`.

		Input Arguments:
			n:  an integer number of columns in the matrix
			m:  an integer number of rows in the matrix. If omitted, the
				value of n is used.
		element:  the value to set every matrix element to. Default is 1.0.

		Output Arguments:
			ret:  a Mat instance.
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
		adds a vertex filter to the matrix.

		A vertex filter is a unary predicate accepting a vertex and returning
		True for those vertices that should be considered and False for those
		that should not.

		Vertex filters are additive, in that each vertex must pass all filters
		in order to be considered. All vertex filters are executed before a vertex
		is considered in a computation.

		Input Arguments:
			self:  a Mat instance
			filter:  a vertex filter

		SEE ALSO:
			delFilter
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
		
	def delFilter(self, filter=None):
		"""
		deletes a vertex filter from the matrix.

		A vertex filter is a unary predicate accepting a vertex and returning
		True for those vertices that should be considered and False for those
		that should not.

		Input Arguments:
			filter:  either a vertex filter that has previoiusly been added to
				this matrix by a call to addFilter or None, which signals the
				deletion of all filters.

		SEE ALSO: addFilter
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

	def apply(self, op):#, other=None, notB=False):
		"""
		applies an operator to every element in the matrix.

		Input Arguments:
			self:  a Mat instance, modified in-place
			op:  a Python or pyCombBLAS function
		"""
		if self._hasMaterializedFilter():
			ret = self._materialized.apply(op)
			self._copyBackMaterialized()
			return ret
		
		if self._hasFilter():
			op = _makePythonOp(op)
			op = FilterHelper.getFilteredUniOpOrSelf(self, op)
		
		self._m_.Apply(_op_make_unary(op))
		self._dirty()

	def count(self, dir, pred=None):
		"""
		returns the number of elements in the matrix for which a unary
		predicate evaluates to True.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction along which counting is performed. Possible
				values are Mat.Column, Mat.Row, and Mat.All. If dir=Mat.Column,
				then counting is performed within each column and the resulting
				counts are places in a vector. If dir=Mat.Row, then the counting
				is performed similarly within rows resulting in a vector of
				counts. If dir=Mat.All, then the counting is performed by all
				the elements in the matrix resulting in one scalar value.
			pred:  a unary predicate. If not specified, then the default
				predicate is considered which accepts a vertex, converts
				it to a Boolean and returns the result.

		Output Arguments:
			ret: either a scalar count (if dir=Mat.All) or a vector of counts
				(if dir=Mat.Row or dir=Mat.Column).
		"""
		if pred is None:
			pred = lambda x: bool(x)
		
		return self.reduce(dir, (lambda x,y: x+y), uniOp=pred, init=0.0)

	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, doOp=None, inPlace=False, allowIntersect=True, predicate=False):
		"""
		applies an operation to corresponding elements of two matrices. The operation
		may be optionally performed in-place.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix
			op:  a binary operation accepting two elements and returning an element
			allowANulls:  TODO
			allowBNulls:  TODO
			doOp:  a binary predicate accepting two corresponding elements of two
				matrices and returning True if they should be processed and False
				otherwise.
			inPlace:  indicates whether to perform the operation in-place storing
				the result in the first matrix or to create a new matrix.
			allowIntersect:  TODO
			predicate:  Not Supported Yet
		See Also: Vec.eWiseApply
		"""
		
		if predicate:
			raise NotImplementedError, "predicate output not yet supported for Mat.eWiseApply"
		
		if not isinstance(other, Mat):
			raise KeyError, "eWiseApply works on two Mat objects."
			# use apply()?
			return

		if not self._eWiseIgnoreMaterializedFilter:
			if self._hasMaterializedFilter() and not other._hasMaterializedFilter():
				ret = self._materialized.eWiseApply(other, op, allowANulls, allowBNulls, doOp, inPlace, allowIntersect, predicate)
			elif not self._hasMaterializedFilter() and other._hasMaterializedFilter():
				ret = self.eWiseApply(other._materialized, op, allowANulls, allowBNulls, doOp, inPlace, allowIntersect, predicate)
			elif self._hasMaterializedFilter() and other._hasMaterializedFilter():
				ret = self._materialized.eWiseApply(other._materialized, op, allowANulls, allowBNulls, doOp, inPlace, allowIntersect, predicate)
			else:
				ret = False
			
			if ret is not False:
				if inPlace:
					self._copyBackMaterialized()
				return ret
		# else:
		#   ignoring materialized filters is used for copying data back from the materialized filter to the main Mat data structure

		ANull = self._identity_
		BNull = other._identity_
		
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
		an indexing operator that allows to retrieve a submatrix by the
		keys of the elements to include in the submatrix. In its simplest
		form, allows to retrieve an element from the matrix by this
		element's key. Usually accessed through the "matrix[key]" syntax.

		Input Arguments:
			self:  a Mat instance
			key:  one of the following
					- a non-tuple denoting the key for both dimensions
					- a 2-tuple with two keys, one for each dimension
				Each key denotes the out-/in-vertices to be addressed,
				and may be one of the following:
					- an integer
					- a ":"-slice denoting all vertices, represented as
					  slice(None,None,None)
					- a Vec containing a contiguous range of monotonically
					  increasing integers

		Output Arguments:
			ret:  a Mat instance, containing the selected vertices along with
				their incident edges from the original matrix.
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
			raise KeyError, 'inPlace argument must be a Boolean!'
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
		
	# TODO: make a keep() which reverses the predicate
	def _prune(self, pred, inPlace=True, ignoreFilter=False):
		"""
		only keep elements for which pred(e) == false.
		"""
		if not ignoreFilter and self._hasFilter():
			raise NotImplementedError,"_prune() doesn't do filters"
			
		return Mat._toMat(self._m_.Prune(_op_make_unary_pred(pred), inPlace))

	def reduce(self, dir, op, uniOp=None, init=None):
		"""
		accumulates matrix elements along the specified direction.

		Input Arguments:
			self:  a Mat instance
			dir:  the direction along which to accumulate elements. Possible
				values are Mat.Row, Mat.Column, and Mat.All. If dir=Mat.Row,
				then accumulation is performed along each row, and the result
				of the operation is a vector with its i'th element equal to
				the sum accumulated along the i'th row of the matrix. If
				dir=Mat.Column, accumulation is performed similarly along the
				columns. If dir=Mat.All, then, first, accumulation is performed
				along the rows of the original matrix and, later, along the only
				column of the result of accumulation along rows.
			op:  a binary function. Its first argument is the result of application
				of uniOp to a matrix element. Its second argument is the sum accumulated
				so far. Having incorporated the uniOp of another element into the sum,
				the function returns this new value of the sum. Example:
				lambda convertedElement, sum: sum + convertedElement
			uniOp:  a unary function that converts a matrix element before it is passed
				to op function as its first argument. In its simplest form, uniOp may
				return its input, i.e. lambda element: element.
			init:  the value with which an accumulated sum is initialized.

		Output Arguments:
			ret: a Vec instance containing the accumulated values. If the chosen reduction
				direction is either Mat.Row or Mat.Column, then the vector's length will be
				equal to one of the dimensions of the matrix (which exactly depends on dir).
				If the reduction is performed in both directions (Mat.All), then the reduction
				results in a single value.
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
		applies an operation in-place to a matrix using a vector. In its simplest
		form, this operation represents row-wise or column-wise scaling of
		a matrix, that is multiplying a matrix from the right by a column vector
		(dir=Mat.Row) or multiplying a matrix from the left by a row vector
		(dir=Mat.Column).

		Input Arguments:
			self:  a Mat instance
			other:  a Vec instance length equal to the corresponding dimension
				of the matrix (which dimension depends on the choice of dir)
			op:  a binary function that accepts a matrix' element and a
				corresponding vector's element and returns some value. If op is
				omitted, then the multiplication operation is used, which
				corresponds to row-wise or column-wise scaling of the matrix.
			dir:  the direction in which to apply the vector to the matrix. The
				possible values are Mat.Column and Mat.Row. If dir=Mat.Column,
				then the vector is applied to each column of the matrix; otherwise,
				the vector is applied to each row. Default is Mat.Column.

		Output Arguments:
			None.
		"""
		if self._hasMaterializedFilter():
			ret = self._materialized.scale(other, op, dir)
			self._copyBackMaterialized()
			return ret

		if self.isBool():
			raise NotImplementedError, 'scale not implemented on boolean matrices due to C++ template irregularities.'
		
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

	def SpGEMM(self, other, semiring, inPlace=False):
		"""
		multiples two matrices together.

		Input Arguments:
			self:  a Mat instance representing the left factor
			other:  a Mat instance representing the right factor
			semiring:  a semiring object that determines the behavior of elementwise
				addition and multiplication operations. The possible values include
				  - TimesPlusSemiringObj: a semiring where addition and multiplication
				      operations are defined naturally
				  - SecondMaxSemiringObj: a semiring where both addition and multiplication
				      operations return their second operand
			inPlace:  specifies whether to perform the multiplication in-place storing the
				multiplication result in the first matrix or to create a new matrix.
				Default is False.

		Output Arguments:
			ret:  if inPlace=False, returns a matrix representing the result of matrix
				multiplication. Otherwise, nothing is returned.
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
			
		if self._m_ is other._m_:
			# we're squaring the matrix
			if inPlace:
				self._m_.Square(semiring)
				return self
			else:
				cp = self.copy()
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
		multiplies a matrix from the right by a column vector.

		Input Arguments:
			self:  a Mat instance
			other:  a Vec instance
			semiring:  a semiring object that determines the behavior of elementwise
				addition and multiplication operations. The possible values include
				  - TimesPlusSemiringObj: a semiring where addition and multiplication
				      operations are defined naturally
				  - SecondMaxSemiringObj: a semiring where both addition and multiplication
				      operations return their second operand
			inPlace:  specifies whether to perform the multiplication in-place storing the
				multiplication result in the matrix or to create a new vector.
				Default is False.

		Output Arguments:
			ret:  if inPlace=False, returns a vector representing the multiplication result;
				otherwise, the result is stored in the matrix, and the input vector is
				returned.
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
		transposes a matrix in-place.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			None
		"""
		self._m_.Transpose()
		self._dirty()
	


##########################
### Operations
##########################

	def max(self, dir=Column, init=None):
		"""
		finds the maximum elements along the specified direction, which
		corresponds to finding the maximum edge weight among the weights of
		the edges incoming to (dir=Mat.Row) or outgoing from (dir=Mat.Column)
		each vertex of the corresponding graph.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction along which to find maximums. Possible values
				are Mat.Row and Mat.Column. If dir=Mat.Row, then the maximum
				weights are looked for among the weights of the incoming edges
				of each vertex; if dir=Mat.Column, then the search if performed
				by the weights of the outgoing edges. Default is Mat.Column.
			init:  a minimal value for each maximum. In no element along the
				chosen direction is greater than this value, then the
				corresponding maximum in the resulting vector will be equal
				to init. Default is None.

		Output Arguments:
			ret:  a Vec instance containing maximums computed along the chosen
				direction.

		SEE ALSO:  degree, min
		"""
		if init is None:
			init = self._identity_
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
			
		ret = self.reduce(dir, op_max, init=init)
		return ret

	# NEEDED: Handle the init properly (i.e. don't use 0, will break a test)
	def min(self, dir=Column):
		"""
		finds the minimum elements along the specified direction, which
		corresponds to finding the minimum edge weight among the weights of
		the edges incoming to (dir=Mat.Row) or outgoing from (dir=Mat.Column)
		each vertex of the graph.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction along which to find minimums. Possible values
				are Mat.Row and Mat.Column. If dir=Mat.Row, then the minimum
				weights are looked for among the weights of the incoming edges
				of each vertex; if dir=Mat.Column, then the search if performed
				by the weights of the outgoing edges. Default is Mat.Column.

		Output Arguments:
			ret:  a Vec instance containing maximums computed along the chosen
				direction.

		SEE ALSO:  degree, max
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid edge-direction'
		ret = self.reduce(dir, op_min, init=self._identity_)
		return ret

	def spOnes(self, element=1.0):
		"""
		assigns value `element` to each element of the matrix.

		Input Arguments:
			element: a value to assign to matrix elements. Default is 1.0
		
		Output Arguments:
			None
		"""
		if not self.isObj() and not self._hasFilter():
			self.apply(op_set(element))
		else:
			self.apply(lambda x: element)
		self._dirty()
		
	def removeMainDiagonal(self):
		"""
		removes elements residing on the main diagonal of the matrix.

		Input Arguments:
			self:  a Mat instance

		Output Arguments:
			None.
		"""
		#self._m_.removeSelfLoops()
		#self._dirty()
		
		diagonal = Mat.eye(n=self.nrow(), m=self.ncol())
		self.eWiseApply(diagonal, op=(lambda s,d: s), allowANulls=False, allowBNulls=True, allowIntersect=False, inPlace=True)

	
	def sum(self, dir=Column):
		"""
		sums in-place matrix elements along the chosen direction. This is
		a particular case of reduce operation.

		Input Arguments:
			self:  a Mat instance
			dir:  a direction along which summation is performed. Possible
				values are Mat.Column and Mat.Row. Default is Mat.Column.

		Output Arguments:
			ret:  a Vec instance with each element containing the
				sum of the elements in the corresponding row/column.

		SEE ALSO:  reduce, degree
		"""
		if dir != Mat.Row and dir != Mat.Column:
			raise KeyError, 'Invalid direction'
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
		adds corresponding elements of two matrices and stores the
		result in a new matrix. If other is not a matric, but rather
		a scalar, then this scalar is added to each element of the
		first matrix.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result of addition
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x + other))


	def __and__(self, other):
		"""
		performs bitwise And between corresponding elements of two
		matrices and stores the result in a new matrix. If other is
		not a matrix, but rather an integer, than bitwise And is
		performed between each element of the first matrix and this
		integer.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				an integer.

		Output Arguments:
			ret:  a Mat instance containing the result of bitwise And
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x & other), intOnly=True)

	def __div__(self, other):
		"""
		divides elements of the first matrix by the corresponding
		elements of the second matrix and stores the result in a new
		matrix. If other is not a matrix, but rather a scalar, then
		every element of the first matrix is divided by this scalar.
		The function raises a ZeroDivisionException if a division
		by zero occures.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result of division

		NOTE:  for KDT v0.1, the second argument may be only a scalar
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x / other))

	def __eq__(self, other):
		"""
		checks corresponding elements of two matrices for equality
		and stores the Boolean results of these checks in a new matrix.
		If other is not a matrix, but rather a scalar, then every element
		of the first matrix is compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and equal.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x == other), predicate=True)

	def __ge__(self, other):
		"""
		compares corresponding elements of two matrices using `greater
		or equal` as a binary predicate for comparison. If other is not
		a matrix, but rather a scalar, then every element of the first
		matrix is compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and the left one is greater
				than or equal to the right one.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x >= other), predicate=True)

	def __gt__(self, other):
		"""
		compares corresponding elements of two matrices using `greater`
		as a binary predicate for comparison. If other is not a matrix,
		but rather a scalar, then every element of the first matrix is
		compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and the left one is greater
				than the right one.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x > other), predicate=True)

	def __invert__(self):
		"""
		replaces each nonnull element in the matrix with its bitwise inverse.
		
		Input Arguments:
			self: a Mat instance

		Output Arguments:
			None
		"""
		ret = self.copy()
		if isinstance(self._identity_, (float, int, long)):
			func = lambda x: int(x).__invert__()
		else:
			func = lambda x: x.__invert__()
		ret.apply(func)
		return ret

	def __le__(self, other):
		"""
		compares corresponding elements of two matrices using `less
		or equal` as a binary predicate for comparison. If other is not
		a matrix, but rather a scalar, then every element of the first
		matrix is compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and the left one is less than
				or equal to the right one.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x <= other), predicate=True)

	def __lt__(self, other):
		"""
		compares corresponding elements of two matrices using `less`
		as a binary predicate for comparison. If other is not a matrix,
		but rather a scalar, then every element of the first matrix is
		compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and the left one is less
				than the right one.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x < other), predicate=True)

	def __mod__(self, other):
		"""
		computes each element of the first matrix modulo the
		corresponding element of the second matrix. If other is not
		a matrix, but rather a scalar, then every element of the first
		matrix is taken modulo this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x % other))

	def __mul__(self, other):
		"""
		multiplies elements of the first matrix by the corresponding
		elements of the second matrix and stores the result in a new
		matrix. If other is not a matrix, but rather a scalar, then
		every element of the first matrix is multiplied by this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result of multiplication
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x * other))


	def __ne__(self, other):
		"""
		checks corresponding elements of two matrices for inequality
		and stores the Boolean results of these checks in a new matrix.
		If other is not a matrix, but rather a scalar, then every element
		of the first matrix is compared to this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the Boolean results of
				comparisons. The resulting matrix will have nonnull
				elements where either of comparands is nonnull, with
				the value being True only where the corresponding
				elements are both nonnull and not equal.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x != other), predicate=True)

	def __neg__(self):
		"""
		replaces each element in the matrix with its bitwise negation.

		Input Arguments:
			self: a Mat instance

		Output Arguments:
			None
		"""
		ret = self.copy()
		func = lambda x: -x
		ret.apply(func)
		return ret


	def __or__(self, other):
		"""
		computes the bitwise logical Or between the corresponding elements
		of two matrices and stores the result in a new matrix. If other is
		not a matrix, but rather a scalar, then the logical bitwise Or is
		computed between every element of the first matrix and this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x | other), intOnly=True)

	def __sub__(self, other):
		"""
		subtracts each element of the second matrix from the corresponding
		elements of the first matrix and stores the result in a new matrix.
		If other is not a matrix, but rather a scalar, then this scalar is
		subtracted from every element of the first matrix.

		Input Arguments:
			self:  a Mat instance representing the first matrix (minuend)
			other:  a Mat instance representing the second matrix or
				a scalar (subtrahend)

		Output Arguments:
			ret:  a Mat instance containing the result of subtraction.
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x - other))

	def __xor__(self, other):
		"""
		computes bitwise Xor between the corresponding elements of two
		matrices and stores the result in a new matrix. If other is not
		a matrix, but rather a scalar, then bitwise Xor is computed
		between every element of the first matrix and this scalar.

		Input Arguments:
			self:  a Mat instance representing the first matrix
			other:  a Mat instance representing the second matrix or
				a scalar

		Output Arguments:
			ret:  a Mat instance containing the result
		"""
		return self._ewise_bin_op_worker(other, (lambda x, other: x ^ other))