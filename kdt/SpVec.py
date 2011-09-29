import math
#import numpy as np # Adam: TRY TO AVOID THIS IF AT ALL POSSIBLE.
#from Graph import SpParVec 
import pyCombBLAS as pcb
import feedback
import UFget as uf

class info:
	@staticmethod
	def eps():
		"""
		Return IEEE floating point machine epsilon.
		The problem with this operation is that Python only provides a portable way to get this
		value in v2.6 and NumPy isn't always available. This function attempts to use whatever
		knows this value or returns a reasonable default otherwise.
		"""
		# try Python v2.6+ float_info
		try:
			from sys import float_info as fi
			return fi.epsilon
		except ImportError:
			pass
			
		# try Numpy
		try:
			import numpy as np
			return float(np.finfo(np.float).eps)
		except ImportError:
			pass
		except AttributeError:
			pass
			
		# return a reasonable value
		return 2.220446049250313e-16;

	def minInt():
		return -(2**62)

class SpVec:
	#Note:  all comparison ops (__ne__, __gt__, etc.) only compare against
	#   the non-null elements

	def __init__(self, length=0, element=0):
		#self._identity = element
		#HACK setting of null values should be done more generally
		if isinstance(element, (float, int, long)):
			self._elementIsAnObject = False
			#HACK
			self._identity = 0
		else:
			self._elementIsAnObject = True
			#HACK
			self._identity = element
			self._identity.weight = 0
			self._identity.category = 0
		if length > 0:
			if isinstance(element, (float, int, long)):
				self._v = pcb.pySpParVec(length)
			elif isinstance(element, pcb.Obj1):
				self._v = pcb.pySpParVecObj1(length)
			elif isinstance(element, pcb.Obj2):
				self._v = pcb.pySpParVecObj2(length)
			else:
				raise TypeError

	@staticmethod
	def _isObj(self):
		try:
			ret = hasattr(self,'_elementIsAnObject') and self._elementIsAnObject
		except AttributeError:
			ret = False
		return ret

	@staticmethod
	def _hasFilter(self):
		try:
			ret = hasattr(self,'_vFilter') and len(self._vFilter)>0
		except AttributeError:
			ret = False
		return ret

#	@staticmethod
#	def _buildSuperLambda(fnList):
#		"""
#		builds one lambda function that calls all the functions in the
#		passed function list, creating a logical And from all the 
#		results.  The return value from _buildSuperLambda is intended
#		to be used as a vertex filter.
#		"""
#		#importPrefix = "x."
#		#superFnStr = "lambda x:"
#		#for i in range(len(fnList)-1):
#		#	superFnStr += "bool(%s%s()) and " % (importPrefix, fnList[i].func_name)
#		#superFnStr += "bool(%s%s())" % (importPrefix, fnList[len(fnList)-1].func_name)
#		superFnStr = """
#def fn(x):
#	for i in range(len(fnList)):
#		if not fnList[i](x):
#			return x
#	return func(x)
#"""
#		return superFnStr

	def __abs__(self):
		ret = self.copy()
		if not SpVec._isObj(self):
			f = pcb.abs()
			noWrap = True
		else:
			f = lambda x: x.__abs__()
			noWrap = False
		ret._apply(f, noWrap=noWrap)
		return ret

	def __add__(self, other):
		"""
		adds the corresponding elements of two SpVec instances into the
		result SpVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		ToDo:  elucidate combinations, overloading, etc.
		"""
		# if no filters for self and other and self has doubleint elements
		if not SpVec._hasFilter(self) and not SpVec._hasFilter(other) and isinstance(self._identity, (float, int, long)) and isinstance(other._identity, (float, int, long)):
			ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.plus(),other)
				ret._apply(func, noWrap=True)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				ret._v = self._v + other._v
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(SpVec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if SpVec._isObj(self):
		 		func = lambda x, y: x.__iadd__(y)
			else:
		 		func = lambda x, y: y.__radd__(x)
		 	ret = self._eWiseApply(other, func, True,True)		
		return ret

	def __and__(self, other):
		"""
		performs a logical And between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where both of the input vectors are True.
		"""
#		if len(self) != len(other):
#			raise IndexError, 'arguments must be of same length'
#		ret = self.copy()
#		func = lambda x, other: x.__and__(other)
#		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)
#		return ret
		if not SpVec._hasFilter(self) and not SpVec._hasFilter(other) and isinstance(self._identity, (float, int, long)) and (isinstance(other, (float, int, long)) or isinstance(other._identity, (float, int, long))):
			ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.bitwise_and(),other)
				ret._apply(func, noWrap=True)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				ret = self._eWiseApply(other, pcb.bitwise_and(), True, True, noWrap=True)
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(SpVec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if SpVec._isObj(self):
		 		func = lambda x, y: x.__iand__(y)
			else:
		 		func = lambda x, y: y.__rand__(x)
		 	ret = self._eWiseApply(other, func, True,True)		
		return ret


	def __delitem__(self, key):
		if isinstance(other, (float, int, long)):
			del self._spv[key]
		else:
			del self._spv[key._dpv];	
		return

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__div__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			raise NotImplementedError, 'no SpVecObj/SpVecObj yet'
		 	func = lambda x, other: x.__div__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__eqPy__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__eqPy__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def __getitem__(self, key):
		"""
		performs indexing of a SpParVec instance on the right-hand side
		of an equation.  The following forms are supported:
	scalar = spparvec[integer scalar]
	spparvec = spparvec[non-boolean parvec]

		The first form takes as the index an integer scalar and returns
		the corresponding element of the SpParVec instance.  This form is
		for convenience and is not highly performing.

		The second form takes as the index a non-Boolean SpParVec instance
		and returns an SpParVec instance of the same length with the 
		elements of the result corresponding to the nonnull values of the
		index set to the values of the base SpParVec instance. 
		"""
		if isinstance(key, (int, long, float)):
			if key < 0 or key > len(self)-1:
				raise IndexError
			ret = self._v[key]
		elif isinstance(key,ParVec):
			if key.isBool():
				raise KeyError, "Boolean indexing on right-hand side for SpParVec not supported"
			ret = ParVec(-1)
			ret._dpv = self._spv[key._dpv]
		else:
			raise KeyError, 'SpParVec indexing only by ParVec or integer scalar'
		return ret

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__ge__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__ge__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__gt__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__ge__(other)
			ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)
		return ret

	def __iadd__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		if isinstance(other, (float, int, long)):
			func = lambda x: x.__iadd__(other)
			self._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj += VecObj yet'
		 	func = lambda x, other: x.__iadd__(other)
		 	self = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return self
		
	def __invert__(self):
		"""
		negates each nonnull element of the passed SpParVec instance.
		"""
		ret = self.copy()
		func = lambda x: x.__invert__()
		ret._apply(func)
		return ret

	def __isub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where either of the two input 
		arguments was nonnull.
		"""
		if isinstance(other, (float, int, long)):
			func = lambda x: x.__isub__(other)
			self._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj += VecObj yet'
		 	func = lambda x, other: x.__isub__(other)
		 	self = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return self
		
	def __len__(self):
		"""
		returns the length (the maximum number of potential nonnull elements
		that could exist) of a SpVecObj instance.
		"""
		return len(self._v)

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__le__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__le__(other)
			#ToDo:  should __le__ return a SpVec instead of a SpVecObj?
		 	#ret = self._eWisePredApply(other, pcb.binaryObjPred(func), True,True)		
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			#HACK:  note __ltPy__ called in 2 spots here, to avoid
			#	conflict with built-in C++ fn in __lt__
			func = lambda x: x.__ltPy__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__ltPy__(other)
			#ToDo:  should __lt__ return a SpVec instead of a SpVecObj?
		 	#ret = self._eWisePredApply(other, pcb.binaryObjPred(func), True,True)		
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def __mod__(self, other):
		"""
		calculates the modulus of each element of the first argument by the
		second argument (a scalar or a SpParVec instance), with a nonnull
		element where the input SpParVec argument(s) were nonnull.

		Note:  for v0.1, only a scalar divisor is supported.
		"""
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__mod__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj%VecObj yet'
			raise NotImplementedError, 'no SpVecObj%SpVecObj yet'
		 	func = lambda x, other: x.__mod__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def __mul__(self, other):
		"""
		multiplies each element of the first argument by the second argument 
		(a scalar or a SpParVec instance), with a nonnull element where 
		the input SpParVec argument(s) were nonnull.
		"""
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__mul__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj*VecObj yet'
		 	func = lambda x, other: x.__mul__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__nePy__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj >= VecObj yet'
		 	func = lambda x, other: x.__nePy__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def __neg__(self):
		"""
		negates each nonnull element of the passed SpParVec instance.
		"""
		ret = self.copy()
		func = lambda x: x.__neg__()
		ret._apply(func)
		return ret


	def __or__(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where at least one of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		func = lambda x, other: x.__or__(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)
		return ret

	_REPR_MAX = 30;
	_REPR_WARN = 0
	def __repr__(self):
		"""
		prints the first N elements of the SpParVec instance, where N
		is roughly equal to the value of self._REPR_MAX.

		SEE ALSO:  printAll
		"""
		self._v.printall()
		return ' '
		#TODO:  limit amount of printout?
		nPrinted = 0
		i = 0
		while i < len(self) and nPrinted < self._REPR_MAX:
			#HACK check for nonnull
			#ToDo: return string instead of printing here
			print "__repr__ loop,", self[i]
			if self[i].weight > info.eps or self[i].type!=0:
				print self[i]
				nPrinted += 1
			i += 1
		if i < len(self)-1 and master():
			print "Limiting print-out to first %d elements" % self._REPR_MAX
		return ' '


	def __setitem__(self, key, value):
		"""
		performs assignment of an SpParVec instance on the left-hand side of
		an equation.  The following forms are supported.
	spparvec[integer scalar] = scalar
	spparvec[Boolean parvec] = scalar
	spparvec[Boolean parvec] = parvec
	spparvec[non-Boolean spparvec] = scalar
	spparvec[non-Boolean spparvec] = spparvec

		For the first form, the element of the SpParVec instance indicated 
		by the index to the is set to the scalar value.

		For the second form, the elements of the SpParVec instance
		corresponding to the True elements of the index ParVec instance
		are set to the scalar value.

		For the third form, the elements of the SpParVec instance 
		corresponding to the True elements of the index ParVec instance
		are set to the corresponding element of the value ParVec instance.

		For the fourth form, the elements of the SpParVec instance
		corresponding to the nonnull elements of the index SpParVec
		instance are set to the scalar value.

		For the fifth form, the elements of the SpParVec instance 
		corresponding to the nonnull elements of the index SpParVec
		instance are set to the corresponding value of the value
		SpParVec instance.  Note that the base, key, and value SpParVec
		instances must all be of the same length, though the base may
		have a different number of nonzeros from the key and value. 
		"""
		if isinstance(key, (float, int, long)):
			if key > len(self)-1:
				raise IndexError
			self._v[key] = value
		elif isinstance(key,ParVec):
			if not key.isBool():
				raise KeyError, 'only Boolean ParVec indexing of SpParVecs supported'
			if isinstance(value,ParVec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = ParVec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self._spv) != len(key._dpv) or len(self._spv) != len(value._dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._spv[key._dpv] = value._dpv
		elif isinstance(key,SpParVec):
			if key.isBool():
				raise KeyError, 'Boolean SpParVec indexing of SpParVecs not supported'
			if isinstance(value,ParVec):
				pass
			elif isinstance(value,SpParVec):
				value = value.toParVec()
			elif type(value) == float or type(value) == long or type(value) == int:
				tmp = value
				value = key.copy()
				value.set(tmp)
				value = value.toParVec()
			else:
				raise KeyError, 'Unknown value type'
			key = key.toParVec()
			if len(self._spv) != len(key._dpv) or len(self._spv) != len(value._dpv):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._spv[key._dpv] = value._dpv
		elif type(key) == str and key == 'nonnull':
			self._apply(pcb.set(value), noWrap=True)
		else:
			raise KeyError, 'Unknown key type'
		return
		

	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where the input SpParVec argument(s)
		are nonnull.
		"""
		if isinstance(other, (float, int, long)):
			ret = self.copy()
			func = lambda x: x.__sub__(other)
			ret._apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj-VecObj yet'
		 	func = lambda x, other: x.__sub__(other)
		 	ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def __xor__(self, other):
		"""
		performs a logical Xor between the corresponding elements of two
		SpParVec instances into the result SpParVec instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where exactly one of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		func = lambda x, other: x.__xor__(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)
		return ret


	# in-place, so no return value
	def _apply(self, op, noWrap=False):
		"""
		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
		"""
		
		if hasattr(self, '_vFilter') and len(self._vFilter) > 0:
			class tmpU:
				_vFilter = self._vFilter
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._vFilter)):
						if not tmpU._vFilter[i](x):
							return x
					return op(x)
			tmpInstance = tmpU()
			self._v.Apply(pcb.unaryObj(tmpInstance.fn))
		else:
			if noWrap:
				self._v.Apply(op)
			else:
				self._v.Apply(pcb.unaryObj(op))
		return

	def _eWiseApply(self, other, op, allowANulls, allowBNulls, noWrap=False):
		"""
		ToDo:  write doc
		"""
		if hasattr(self, '_vFilter') or hasattr(other, '_vFilter'):
			class tmpB:
				if hasattr(self,'_vFilter') and len(self._vFilter) > 0:
					selfVFLen = len(self._vFilter)
					vFilter1 = self._vFilter
				else:
					selfVFLen = 0
				if hasattr(other,'_vFilter') and len(other._vFilter) > 0:
					otherVFLen = len(other._vFilter)
					vFilter2 = other._vFilter
				else:
					otherVFLen = 0
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfVFLen):
						if not tmpB.vFilter1[i](x):
							x = type(self._identity)()
							break
					for i in range(tmpB.otherVFLen):
						if not tmpB.vFilter2[i](y):
							y = type(other._identity)()
							break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op
		if noWrap:
			if isinstance(other, (float, int, long)):
				v = pcb.EWiseApply(self._v, other   , superOp, allowANulls, allowBNulls)
			else:
				v = pcb.EWiseApply(self._v, other._v, superOp, allowANulls, allowBNulls)
		else:
			if isinstance(other, (float, int, long)):
				v = pcb.EWiseApply(self._v, other   , pcb.binaryObj(superOp), allowANulls, allowBNulls)
			else:
				v = pcb.EWiseApply(self._v, other._v, pcb.binaryObj(superOp), allowANulls, allowBNulls)
		ret = SpVec(element=v[0])
		ret._v = v
		return ret

	# as of 2011sep27, probably don't need this function
	def _eWisePredApply(self, other, op, allowANulls, allowBNulls):
		"""
		ToDo:  write doc
		"""
		#filterPred = pcb.ifthenelse(pred, pcb.identity(), pcb.set(0))
		#if not isinstance(op, pcb.UnaryFunctionObj):
		#	self._v.Apply(pcb.unaryObj(op))
		#else:
		#if isinstance(op, pcb.BinaryFunctionObj):
			#ret = SpVecObj()
			#ret._v = pcb.EWiseApply(self._v, other._v, op, allowANulls, allowBNulls)
		#elif isinstance(op, pcb.BinaryPredicateObj):
		ret = SpVec(0)
		ret._v = pcb.EWiseApply(self._v, other._v, op, allowANulls, allowBNulls)
		return ret

#	op_add = pcb.plus()
#	op_sub = pcb.minus()
#	op_mul = pcb.multiplies()
#	op_div = pcb.divides()
#	op_mod = pcb.modulus()
#	op_fmod = pcb.fmod()
#	op_pow = pcb.pow()
#	op_max  = pcb.max()
#	op_min = pcb.min()
#	op_bitAnd = pcb.bitwise_and()
#	op_bitOr = pcb.bitwise_or()
#	op_bitXor = pcb.bitwise_xor()
#	op_and = pcb.logical_and()
#	op_or = pcb.logical_or()
#	op_xor = pcb.logical_xor()
#	op_eq = pcb.equal_to()
#	op_ne = pcb.not_equal_to()
#	op_gt = pcb.greater()
#	op_lt = pcb.less()
#	op_ge = pcb.greater_equal()
#	op_le = pcb.less_equal()
	def _reduce(self, op, pred=None):
		"""
		ToDo:  write doc
		"""
		#ToDo:  error-check on op?
#		if not isinstance(op, pcb.BinaryFunction):
#			realOp = pcb.binary(op)
#		else:
#			realOp = op
		if not pred is None:
			if SpVec._isObj(self):
				realPred = pcb.unaryObj(pred)
			else:
				realPred = pred
			ret = self._v.Reduce(op, realPred)
		else:
			ret = self._v.Reduce(op)
		return ret

	# in-place, so no return value
	def _setIdentity(self, val):
		if isinstance(val, (float, int, long)):
			self._identity = 0
		elif isinstance(val, pcb.Obj1):
			self._identity = pcb.Obj1()
			self._identity.weight = 0
			self._identity.category = 0
		elif isinstance(val, pcb.Obj2):
			self._identity = pcb.Obj2()
			self._identity.weight = 0
			self._identity.category = 0
		return
	
	# in-place, so no return value
	def addVFilter(self, filter):
		"""
		adds a vertex filter to the SpVec instance.  

		A vertex filter is a Python function that is applied elementally
		to each vertex in the SpVec, with a Boolean True return value
		causing the vertex to be considered and a False return value
		causing it not to be considered.

		Vertex filters are additive, in that each vertex must pass all
		filters to be considered.  All vertex filters are executed before
		a vertex is considered in a computation.

		Input Arguments:
			self:  a DiGraph instance
			filter:  a Python function

		SEE ALSO:
			delVFilter  
		"""
		if not SpVec._isObj(self):
			raise NotImplementedError, 'No filter support on doubleint SpVec instances'
		if hasattr(self, '_vFilter'):
			self._vFilter.append(filter)
		else:
			self._vFilter = [filter]
		return
		

	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpVecObj instance are True (nonzero), and False otherwise.
		"""
		tmp = self.copy()
		identity = pcb.Obj1()
		identity.weight = tmp[0].weight
		identity.type = 99
		tmp[0] = identity
		func = lambda x, other: x.all(other)
		ret = tmp._reduce(pcb.binaryObj(func)).weight > 0
		return ret

	def allCloseToInt(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		SpParVec instance have values within epsilon of an integer,
		and False otherwise.
		"""
		if self.nnn() == 0:
			return True;
		eps = info.eps()
		ret = (((self % 1.0) < eps) | (((-(self%1.0))+1.0)< eps)).all()
		return ret

	def any(self):
		"""
		returns a Boolean True if any of the nonnull elements of the
		SpVecObj instance is True (nonzero), and False otherwise.
		"""
		func = lambda x, other: x.any(other)
		ret = self._reduce(pcb.binaryObj(func)).weight > 0
		return ret

	def copy(self):
		"""
		creates a deep copy of the input argument.
		"""
		ret = SpVec()
		ret._v = self._v.copy()
		ret._setIdentity(self._identity)
		ret._elementIsAnObject = self._elementIsAnObject
		# filter the new vector; note generic issue of distinguishing
		#   zero from null
		if hasattr(self,'_vFilter'):
			class tmpU:
				_vFilter = self._vFilter
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._vFilter)):
						if not tmpU._vFilter[i](x):
							return type(self._identity)()
					return x
			tmpInstance = tmpU()
			ret._v.Apply(pcb.unaryObj(tmpInstance.fn))
		return ret

	# in-place, so no return value
	def delVFilter(self, filter=None):
		"""
		deletes a vertex filter from the SpVec instance.  

		Input Arguments:
			self:  a SpVec instance
			filter:  a Python function, which can be either a function
			    previously added to this DiGraph instance by a call to
			    addVFilter or None, which signals the deletion of all
			    vertex filters.

		SEE ALSO:
			addVFilter  
		"""
		if not hasattr(self, '_vFilter'):
			raise KeyError, "no vertex filters previously created"
		if filter is None:
			del self._vFilter	# remove all filters
		else:
			self._vFilter.remove(filter)
			if len(self._vFilter) == 0:
				del self._vFilter
		return

	#ToDo:  implement find/findInds when problem of any zero elements
	#         in the sparse vector getting stripped out is solved
	#ToDO:  simplfy to avoid dense() when pySpParVec.Find available
	def find(self):
		"""
		returns the elements of a Boolean SpParVec instance that are both
		nonnull and nonzero.

		Input Argument:
			self:  a SpParVec instance

		Output Argument:
			ret:  a SpParVec instance

		SEE ALSO:  findInds
		"""
		if not self.isBool():
			raise NotImplementedError, 'only implemented for Boolean vectors'
		ret = SpParVec(-1)
		ret._spv = self._spv.dense().Find(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret


	#ToDO:  simplfy to avoid dense() when pySpParVec.FindInds available
	def findInds(self):
		"""
		returns the indices of the elements of a Boolean SpParVec instance
		that are both nonnull and nonzero.

		Input Argument:
			self:  a SpParVec instance

		Output Argument:
			ret:  a ParVec instance of length equal to the number of
			    nonnull and nonzero elements in self

		SEE ALSO:  find
		"""
		if not self.isBool():
			raise NotImplementedError, 'only implemented for Boolean vectors'
		ret = ParVec(-1)
		ret._dpv = self._spv.dense().FindInds(pcb.bind2nd(pcb.not_equal_to(),0.0))
		return ret


	def isBool(self):
		"""
		returns a Boolean scalar denoting whether all elements of the input 
		SpParVec instance are equal to either True (1) or False (0).
		"""
		if self.nnn() == 0:
			return True
		eps = info.eps()
		ret = ((abs(self) < eps) | (abs(self-1.0) < eps)).all()
		return ret

#SPR  Don't know how to do this yet
	@staticmethod
	def load(fname):
                file = open(fname, 'r')
                file.close()

		ret = SpVecObj(1)
		ret._v.load(fname)
		return ret

	def logicalAnd(self, other):
		"""
		performs a logical And between the corresponding elements of two
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where both of the input vectors are True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		func = lambda x, other: x.logicalAnd(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret
	logical_and = logicalAnd	# for NumPy compatibility

	def logicalOr(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where either of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		func = lambda x, other: x.logicalOr(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret
	logical_or = logicalOr 		# for NumPy compatibility

	def logicalXor(self, other):
		"""
		performs a logical Or between the corresponding elements of two
		SpVecObj instances into the result SpVecObj instance, with a non-
		null element where either of the two input vectors is nonnull,
		and a True value where either of the input vectors is True.
		"""
		if len(self) != len(other):
			raise IndexError, 'arguments must be of same length'
		ret = self.copy()
		func = lambda x, other: x.logicalXor(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret
	logical_xor = logicalXor	# for NumPy compatibility

	def max(self):
		"""
		returns the maximum value of the nonnull elements in the SpParVec 
		instance.
		"""
		func = lambda x, other: x.max(other)
		ret = self._reduce(pcb.binaryObj(func))
		return ret

	def min(self):
		"""
		returns the minimum value of the nonnull elements in the SpParVec 
		instance.
		"""
		func = lambda x, other: x.min(other)
		ret = self._reduce(pcb.binaryObj(func))
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		SpParVec instance.

		Note:  for x a SpParVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, nnz
		"""
		return len(self) - self._spv.getnnz()

	def nnn(self):
		"""
		returns the number of non-nulls (existent entries) in the
		SpVec instance.
	
		Note:  for x a SpVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nn, nnz
		"""
		#HACK:  some better way to set initial value of redxn?
		# z = tmp.findInds()
		# if len(z) > 0:
		#   tmp[z[0]] = identity
		def f(x,y):
			if isinstance(x, (float, int, long)):
				x = x + 1
			else:
				if y.weight != 0 or y.type != 0:
					x.weight = x.weight + 1
			return x
		tmp = self.copy()
		if isinstance(self._identity,(float, int, long)):
			identity = 0
			ret = int(tmp._reduce(pcb.plus(), pred=pcb.set(1)))
		elif isinstance(self._identity,(pcb.Obj1)):
			identity = type(self._identity)()
			#HACK: referred to above
			if self[0].weight or self[0].type:
				identity.weight = 1
			tmp[0] = identity
			ret = int(tmp._reduce(pcb.binaryObj(f)).weight)
		elif isinstance(self._identity,pcb.Obj2):
			identity = type(self._identity)()
			#HACK: referred to above
			if self[0].weight or self[0].type:
				identity.weight = 1
			tmp[0] = identity
			ret = int(tmp._reduce(pcb.binaryObj(f)).weight)
		return ret

	def nnz(self):
		"""
		returns the number of non-zero entries in the SpParVec
		instance.

		Note:  for x a SpParVec instance, x.nnz() is always less than or
		equal to x.nnn().

		SEE ALSO:  nn, nnn
		"""
		ret = self._reduce(SpParVec.op_add, pcb.ifthenelse(pcb.bind2nd.not_equal_to(),0), pcb.set(1), pcb.set(0))
		return int(ret)

	#FIX:  delete, since unused
	@staticmethod
	def ones(sz, element=0):
		"""
		creates a SpParVec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		ret = SpVec.range(sz, element=element)
		#HACK:  scalar set loop for now; needs something better
		#Obj1 = pcb.Obj1()
		#Obj1.weight = 1
		#for i in range(sz):
		#	ret[i] = Obj1
		if not SpVec._isObj(ret):
			ret._apply(pcb.set(1), noWrap=True)
		else:
			ret._apply(lambda x: x.spOnes())
		return ret

	def printAll(self):
		"""
		prints all elements of a SpParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self._v.printall()
		return ' '

	@staticmethod
	def range(arg1, stop=None, element=0):
		"""
		FIX:  update doc
		creates a SpParVec instance with consecutive integer values.

	range(stop)
	range(start, stop)
		The first form creates a SpParVec instance of length stop whose
		values are all nonnull, starting at 0 and stopping at stop-1.
		The second form creates a SpParVec instance of length stop-start
		whose values are all nonnull, starting at start and stopping at
		stop-1.
		"""
		if stop == None:
			start = 0
			stop = arg1
		else:
			start = arg1
		if start > stop:
			raise ValueError, "start > stop"
		ret = SpVec(stop-start, element=element)
		#HACK:  serial set is not practical for large sizes
		if isinstance(element, (float, int, long)):
			ret._v = pcb.pySpParVec.range(stop-start,start)
		else:
			Obj1 = pcb.Obj1()
			for i in range(stop-start):
				Obj1.weight = start + i
				ret[i] = Obj1
		return ret
	
	#in-place, so no return value
	def set(self, value):
		"""
		sets every non-null value in the SpParVec instance to the second
		argument, in-place.
		"""
		def f(x):
			x.weight = value
			return x
		self._apply(f)
		return

	# in-place, so no return value
	def sort(self):
		"""
		sorts the non-null values in the passed SpParVec instance in-place
		in ascending order.

		Input Arguments:
			self:  a SpParVec instance.

		Output Argument:
			None
		"""
		self._v.Sort()
		return

	def sorted(self):
		"""
		returns a new SpParVec instance with the sorted values (in ascending
		order) from the input SpParVec instance and a SpParVec permutation 
		vector.

		Input Arguments:
			self:  a SpParVec instance.

		Output Argument:
			ret:  a tuple containing as its first element a SpParVec 
			    instance of the same length and same number and position
			    of non-nulls containing the sorted non-null values and
			    as its second element the indices of the sorted values
			    in the input vector.

		See Also:  sort
		"""
		ret1 = self.copy();
		ret2 = SpParVec(-1)
		ret2._spv = ret1._v.Sort()
		return (ret1, ret2)

	#in-place, so no return value
	def spOnes(self):
		"""
		sets every non-null value in the SpParVec instance to 1, in-place.
		"""
		if not SpVec._isObj(self):
			self._apply(pcb.set(1), noWrap=True)
		else:
			self._apply(lambda x: x.spOnes())
		return

	#in-place, so no return value
	def spRange(self):
		"""
		sets every non-null value in the SpParVec instance to its position
		(offset) in the vector, in-place.
		"""
		raise NotImplementedError
		self._v.setNumToInd()

	def sum(self):
		"""
		returns the sum of all the non-null values in the SpParVec instance.
		"""
		if self.nnn() == 0:
			ret = 0
		else:
		 	func = lambda x, other: x.__iadd__(other)
			ret = self._reduce(pcb.binaryObj(func))
		return ret

	#in-place, so no return value
	def toBool(self):
		"""
		converts the input SpParVec instance, in-place, into Boolean
		values (1.0 for True, 0.0 for False) according to whether the
		initial values are nonzero or zero, respectively.
		"""
		def f(x):
			x.weight = bool(x.weight)
			return x
		self._apply(f)
		return

	def topK(self, k):
		"""
		returns the largest k non-null values in the passed SpParVec instance.

		Input Arguments:
			self:  a SpParVec instance.
			k:  a scalar integer denoting how many values to return.

		Output Argument:
			ret:  a ParVec instance of length k containing the k largest
			    values from the input vector, in ascending order.
		"""
		raise NotImplementedError
		ret = Vec(0)
		ret._v = self._v.TopK(k)
		return ret

	def toParVec(self):	
		"""
		converts a SpParVec instance into a ParVec instance of the same
		length with the non-null elements of the SpParVec instance placed 
		in their corresonding positions in the ParVec instance.
		"""
		ret = ParVec(-1)
		ret._dpv = self._spv.dense()
		return ret

	@staticmethod
	def toSpParVec(SPV):
		#if not isinstance(SPV, pcb.pySpParVec):
		if SPV.__class__.__name__ != 'pySpParVec':
			raise TypeError, 'Only accepts pySpParVec instances'
		ret = SpParVec(-1)
		ret._spv = SPV
		return ret
	
def master():
	"""
	Return Boolean value denoting whether calling process is the 
	master process or a slave process in a parallel program.
	"""
	return pcb.root()

def version():
	"""
	Return KDT version number, as a string.
	"""
	return "0.2.x"

def revision():
	"""
	Return KDT revision number, as a string.
	"""
	return "r7xx"
