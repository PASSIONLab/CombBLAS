import math
import kdt.pyCombBLAS as pcb
import feedback
import UFget as uf
from Util import *
from Util import _op_make_unary
from Util import _op_make_unary_pred
from Util import _op_make_binary
from Util import _op_make_binary_pred

#	naming convention:
#	names that start with a single underscore and have no final underscore
#		are functions
#	names that start and end with a single underscore are fields

class Vec(object):
	#Note:  all comparison ops (__ne__, __gt__, etc.) only compare against
	#   the non-null elements

#	def __init__(self, length=0, element=0, sparse=False):
#		if not sparse:
#			self = DeVec(length, element)
#		else:
#			self = SpVec(length, element)

#		#self._identity_ = element
#		#HACK setting of null values should be done more generally
#		if isinstance(element, (float, int, long)):
#			#HACK
#			self._identity_ = 0
#		else:
#			#HACK
#			self._identity_ = element
#			self._identity_.weight = 0
#			self._identity_.category = 0
#		if length > 0:
#			if isinstance(element, (float, int, long)):
#				self._v_ = pcb.pySpParVec(length)
#			elif isinstance(element, pcb.Obj1):
#				self._v_ = pcb.pySpParVecObj1(length)
#			elif isinstance(element, pcb.Obj2):
#				self._v_ = pcb.pySpParVecObj2(length)
#			else:
#				raise TypeError

	def __new__(cls, length=None, init=None, element=None, sparse=None):
		if sparse is None or not sparse:
			self = object.__new__(DeVec,  length, init, element)
		else: 
			self = object.__new__(SpVec,  length, element)

		return self


	@staticmethod
	def isObj(self):
		return not isinstance(self._identity_, (float, int, long))

	@staticmethod
	def _hasFilter(self):
		try:
			ret = hasattr(self,'_vFilter_') and len(self._vFilter_)>0
		except AttributeError:
			ret = False
		return ret

	def __abs__(self):
		ret = self.copy()
		if not Vec.isObj(self):
			f = pcb.abs()
		else:
			f = lambda x: x.__abs__()
		ret.apply(f)
		return ret

	def __add__(self, other):
		"""
		adds the corresponding elements of two SpVec instances into the
		result SpVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		ToDo:  elucidate combinations, overloading, etc.
		"""
		# if no filters for self and other and self has doubleint elements
		if not Vec._hasFilter(self) and not Vec._hasFilter(other) and isinstance(self._identity_, (float, int, long)) and (isinstance(other, (float, int, long)) or isinstance(other._identity_, (float, int, long))):
			ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.plus(),other)
				ret.apply(func)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				ret._v_ = self._v_ + other._v_
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(Vec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if Vec.isObj(self):
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
		if not Vec._hasFilter(self) and not Vec._hasFilter(other) and isinstance(self._identity_, (float, int, long)) and (isinstance(other, (float, int, long)) or isinstance(other._identity_, (float, int, long))):
			ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.bitwise_and(),other)
				ret.apply(func)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				ret = self._eWiseApply(other, pcb.bitwise_and(), True, True, noWrap=True)
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(Vec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if Vec.isObj(self):
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
			ret.apply(func)
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
			ret.apply(func)
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
			return self._v_[key]
		else:
			return Vec._toVec(self._v_[key._v_])
		#elif isinstance(key,ParVec):
		#	if key.isBool():
		#		raise KeyError, "Boolean indexing on right-hand side for SpParVec not supported"
		#	ret = ParVec(-1)
		#	ret._dpv = self._spv[key._dpv]
		#else:
		#	raise KeyError, 'SpParVec indexing only by ParVec or integer scalar'
		#return ret

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
			ret.apply(func)
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
			ret.apply(func)
		else:
			if len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,SpVecObj):
				raise NotImplementedError, 'no SpVecObj > VecObj yet'
		 	func = lambda x, other: x.__gt__(other)
			ret = self.eWiseApply(other, pcb.binaryObj(func), True,True)
		return ret

	def __iadd__(self, other):
		"""
		adds the corresponding elements of two SpParVec instances into the
		result SpParVec instance, with a nonnull element where either of
		the two input vectors was nonnull.
		"""
		if not Vec._hasFilter(self) and not Vec._hasFilter(other) and isinstance(self._identity_, (float, int, long)) and (isinstance(other, (float, int, long)) or isinstance(other._identity_, (float, int, long))):
			#ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.plus(),other)
				self.apply(func)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				self._v_ += other._v_
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(Vec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if Vec.isObj(self):
		 		func = lambda x, y: x.__iadd__(y)
			else:
		 		func = lambda x, y: y.__radd__(x)
		 	self = self._eWiseApply(other, func, True,True)		
		return self
		
	def __invert__(self):
		"""
		negates each nonnull element of the passed SpParVec instance.
		"""
		ret = self.copy()
		if isinstance(self._identity_, (float, int, long)):
			func = lambda x: int(x).__invert__()
		else:
			func = lambda x: x.__invert__()
		ret.apply(func)
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
			self.apply(func)
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
		return len(self._v_)

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
			ret.apply(func)
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
			ret.apply(func)
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
			ret.apply(func)
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
			ret.apply(func)
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
			ret.apply(func)
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
		negates each nonnull element of the passed SpVec instance.
		"""
		ret = self.copy()
		func = lambda x: x.__neg__()
		ret.apply(func)
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
		if hasattr(self,'_v_'):
			self._v_.printall()
		else:
			return "Vec with no _v_"
		return ' '
		#TODO:  limit amount of printout?
		nPrinted = 0
		i = 0
		while i < len(self) and nPrinted < self._REPR_MAX:
			#HACK check for nonnull
			#ToDo: return string instead of printing here
			print "__repr__ loop,", self[i]
			if self[i].weight > info.eps or self[i].category!=0:
				print self[i]
				nPrinted += 1
			i += 1
		if i < len(self)-1 and master():
			print "Limiting print-out to first %d elements" % self._REPR_MAX
		return ' '


#	def __setitem__(self, key, value):
#		"""
#		performs assignment of an SpParVec instance on the left-hand side of
#		an equation.  The following forms are supported.
#	spparvec[integer scalar] = scalar
#	spparvec[Boolean parvec] = scalar
#	spparvec[Boolean parvec] = parvec
#	spparvec[non-Boolean spparvec] = scalar
#	spparvec[non-Boolean spparvec] = spparvec
#
#		For the first form, the element of the SpParVec instance indicated 
#		by the index to the is set to the scalar value.
#
#		For the second form, the elements of the SpParVec instance
#		corresponding to the True elements of the index ParVec instance
#		are set to the scalar value.
#
#		For the third form, the elements of the SpParVec instance 
#		corresponding to the True elements of the index ParVec instance
#		are set to the corresponding element of the value ParVec instance.
#
#		For the fourth form, the elements of the SpParVec instance
#		corresponding to the nonnull elements of the index SpParVec
#		instance are set to the scalar value.
#
#		For the fifth form, the elements of the SpParVec instance 
#		corresponding to the nonnull elements of the index SpParVec
#		instance are set to the corresponding value of the value
#		SpParVec instance.  Note that the base, key, and value SpParVec
#		instances must all be of the same length, though the base may
#		have a different number of nonzeros from the key and value. 
#		"""
#		if isinstance(key, (float, int, long)):
#			if key > len(self)-1:
#				raise IndexError
#			self._v_[key] = value
#		elif isinstance(key,Vec):
#			if not key.isBool():
#				raise KeyError, 'only Boolean ParVec indexing of SpParVecs supported'
#			if isinstance(value,Vec):
#				pass
#			elif type(value) == float or type(value) == long or type(value) == int:
#				value = Vec(len(key),value)
#			else:
#				raise KeyError, 'Unknown value type'
#			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
#				raise IndexError, 'Key and Value must be same length as SpParVec'
#			self._v_[key._v_] = value._v_
#		elif isinstance(key,SpVec):
#			if key.isBool():
#				raise KeyError, 'Boolean SpVec indexing of SpParVecs not supported'
#			if isinstance(value,Vec):
#				pass
#			elif isinstance(value,SpVec):
#				value = value.toDeVec()
#			elif type(value) == float or type(value) == long or type(value) == int:
#				tmp = value
#				value = key.copy()
#				value.set(tmp)
#				value = value.toDeVec()
#			else:
#				raise KeyError, 'Unknown value type'
#			key = key.toDeVec()
#			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
#				raise IndexError, 'Key and Value must be same length as SpVec'
#			self._v_[key._v_] = value._v_
#		elif type(key) == str and key == 'nonnull':
#			self.apply(pcb.set(value), noWrap=True)
#		else:
#			raise KeyError, 'Unknown key type'
#		return
		

	def __sub__(self, other):
		"""
		subtracts the corresponding elements of the second argument (a
		scalar or a SpParVec instance) from the first argument (a SpParVec
		instance), with a nonnull element where the input SpParVec argument(s)
		are nonnull.
		"""
		if not Vec._hasFilter(self) and not Vec._hasFilter(other) and isinstance(self._identity_, (float, int, long)) and (isinstance(other, (float, int, long)) or isinstance(other._identity_, (float, int, long))):
			ret = self.copy()
			# if other is scalar
			if isinstance(other, (float, int, long)):
				func = pcb.bind2nd(pcb.minus(),other)
				ret.apply(func)
			else:	# other is doubleint (Sp)Vec
				if len(self) != len(other):
					raise IndexError, 'arguments must be of same length'
				ret._v_ = self._v_ - other._v_
		else:
			if not isinstance(other, (float, int, long)) and len(self) != len(other):
				raise IndexError, 'arguments must be of same length'
			if not isinstance(other,(Vec, float, int, long)):
				raise NotImplementedError, 'no SpVecObj+VecObj yet'
			if Vec.isObj(self):
		 		func = lambda x, y: x.__isub__(y)
			else:
		 		func = lambda x, y: y.__rsub__(x)
		 	ret = self._eWiseApply(other, func, True,True)		
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
	def apply(self, op):
		"""
		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
		FIX:  doesn't look like this supports noWrap with filters
		"""
		
		if hasattr(self, '_vFilter_') and len(self._vFilter_) > 0:
			class tmpU:
				_vFilter_ = self._vFilter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._vFilter_)):
						if not tmpU._vFilter_[i](x):
							return x
					return op(x)
			tmpInstance = tmpU()
			self._v_.Apply(pcb.unaryObj(tmpInstance.fn))
		else:
			#if noWrap:
			#	self._v_.Apply(op)
			#else:
			#	self._v_.Apply(pcb.unaryObj(op))
			self._v_.Apply(_op_make_unary(op))
		return

#	# in-place, so no return value
#	def _applyInd(self, op, noWrap=False):
#		"""
#		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
#		"""
#		
#		if hasattr(self, '_vFilter_') and len(self._vFilter_) > 0:
#			class tmpU:
#				_vFilter_ = self._vFilter_
#				@staticmethod
#				def fn(x):
#					for i in range(len(tmpU._vFilter_)):
#						if not tmpU._vFilter_[i](x):
#							return x
#					return op(x)
#			tmpInstance = tmpU()
#			self._v_.ApplyInd(pcb.unaryObj(tmpInstance.fn))
#		else:
#			if noWrap:
#				self._v_.Apply(op)
#			else:
#				self._v_.Apply(pcb.unaryObj(op))
#		return
#		if hasattr(self, '_vFilter_') or hasattr(other, '_vFilter_'):
#			class tmpB:
#				if hasattr(self,'_vFilter_') and len(self._vFilter_) > 0:
#					selfVFLen = len(self._vFilter_)
#					vFilter1 = self._vFilter_
#				else:
#					selfVFLen = 0
#				if hasattr(other,'_vFilter_') and len(other._vFilter_) > 0:
#					otherVFLen = len(other._vFilter_)
#					vFilter2 = other._vFilter_
#				else:
#					otherVFLen = 0
#				@staticmethod
#				def fn(x, y):
#					for i in range(tmpB.selfVFLen):
#						if not tmpB.vFilter1[i](x):
#							x = type(self._identity_)()
#							break
#					for i in range(tmpB.otherVFLen):
#						if not tmpB.vFilter2[i](y):
#							y = type(other._identity_)()
#							break
#					return op(x, y)
#			superOp = tmpB().fn
#		else:
#			superOp = op
#		if noWrap:
#			if isinstance(other, (float, int, long)):
#				self._v_.ApplyInd(self._v_, other   , superOp, allowANulls, allowBNulls)
#			else:
#				self._v_.ApplyInd(self._v_, other._v_, superOp, allowANulls, allowBNulls)
#		else:
#			if isinstance(other, (float, int, long)):
#				self._v_.ApplyInd(self._v_, other   , pcb.binaryObj(superOp), allowANulls, allowBNulls)
#			else:
#				self._v_.ApplyInd(self._v_, other._v_, pcb.binaryObj(superOp), allowANulls, allowBNulls)
#		ret = Vec._toVec(self,self._v_)
#		return ret
#
#	def _eWiseApply(self, other, op, allowANulls, allowBNulls, noWrap=False):
#		"""
#		ToDo:  write doc
#		"""
#		if hasattr(self, '_vFilter_') or hasattr(other, '_vFilter_'):
#			class tmpB:
#				if hasattr(self,'_vFilter_') and len(self._vFilter_) > 0:
#					selfVFLen = len(self._vFilter_)
#					vFilter1 = self._vFilter_
#				else:
#					selfVFLen = 0
#				if hasattr(other,'_vFilter_') and len(other._vFilter_) > 0:
#					otherVFLen = len(other._vFilter_)
#					vFilter2 = other._vFilter_
#				else:
#					otherVFLen = 0
#				@staticmethod
#				def fn(x, y):
#					for i in range(tmpB.selfVFLen):
#						if not tmpB.vFilter1[i](x):
#							x = type(self._identity_)()
#							break
#					for i in range(tmpB.otherVFLen):
#						if not tmpB.vFilter2[i](y):
#							y = type(other._identity_)()
#							break
#					return op(x, y)
#			superOp = tmpB().fn
#		else:
#			superOp = op
#		if noWrap:
#			if isinstance(other, (float, int, long)):
#				v = pcb.EWiseApply(self._v_, other   , superOp, allowANulls, allowBNulls)
#			else:
#				v = pcb.EWiseApply(self._v_, other._v_, superOp, allowANulls, allowBNulls)
#		else:
#			if isinstance(other, (float, int, long)):
#				v = pcb.EWiseApply(self._v_, other   , pcb.binaryObj(superOp), allowANulls, allowBNulls)
#			else:
#				v = pcb.EWiseApply(self._v_, other._v_, pcb.binaryObj(superOp), allowANulls, allowBNulls)
#		ret = Vec._toVec(self,v)
#		return ret

	# as of 2011sep27, probably don't need this function
	def _eWisePredApply(self, other, op, allowANulls, allowBNulls):
		"""
		ToDo:  write doc
		"""
		#filterPred = pcb.ifthenelse(pred, pcb.identity(), pcb.set(0))
		#if not isinstance(op, pcb.UnaryFunctionObj):
		#	self._v_.Apply(pcb.unaryObj(op))
		#else:
		#if isinstance(op, pcb.BinaryFunctionObj):
			#ret = SpVecObj()
			#ret._v_ = pcb.EWiseApply(self._v_, other._v_, op, allowANulls, allowBNulls)
		#elif isinstance(op, pcb.BinaryPredicateObj):
		v = pcb.EWiseApply(self._v_, other._v_, op, allowANulls, allowBNulls)
		ret = self._toVec(v)
		return ret

	def reduce(self, op, pred=None):
		"""
		ToDo:  write doc
			return is a scalar
		"""
		ret = self._v_.Reduce(op, _op_make_binary_pred(pred))
		return ret

	# in-place, so no return value
	def _setIdentity(self, val):
		if isinstance(val, (float, int, long)):
			self._identity_ = 0
		elif isinstance(val, pcb.Obj1):
			self._identity_ = pcb.Obj1()
			self._identity_.weight = 0
			self._identity_.category = 0
		elif isinstance(val, pcb.Obj2):
			self._identity_ = pcb.Obj2()
			self._identity_.weight = 0
			self._identity_.category = 0
		return
	
	@staticmethod
	def _getExampleElement(pcbVec):
		if isinstance(pcbVec, (pcb.pyDenseParVec, pcb.pySpParVec)):
			return 0.0
		if isinstance(pcbVec, (pcb.pyDenseParVecObj1, pcb.pySpParVecObj1)):
			return pcb.Obj1()
		if isinstance(pcbVec, (pcb.pyDenseParVecObj2, pcb.pySpParVecObj2)):
			return pcb.Obj2()
		raise NotImplementedError, 'Unknown vector type!'
	
	@staticmethod
	def _isPCBVecSparse(pcbVec):
		if isinstance(pcbVec, (pcb.pyDenseParVec, pcb.pyDenseParVecObj1, pcb.pyDenseParVecObj2)):
			return False
		if isinstance(pcbVec, (pcb.pySpParVec, pcb.pySpParVecObj1, pcb.pySpParVecObj2)):
			return True
		raise NotImplementedError, 'Unknown vector type!'
	
	@staticmethod
	def _toVec(kdtVec, pcbVec = None):
		if pcbVec is None:
			pcbVec = kdtVec
			
			if Vec._isPCBVecSparse(pcbVec):
				ret = SpVec(0, Vec._getExampleElement(pcbVec))
				ret._v_ = pcbVec
				return ret
			else:
				ret = DeVec(0, Vec._getExampleElement(pcbVec))
				ret._v_ = pcbVec
				return ret
		else:
			ret = kdtVec._newLike(0, Vec._getExampleElement(pcbVec))
			ret._v_ = pcbVec
			return ret

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
		if not Vec.isObj(self):
			raise NotImplementedError, 'No filter support on doubleint SpVec instances'
		if hasattr(self, '_vFilter_'):
			self._vFilter_.append(filter)
		else:
			self._vFilter_ = [filter]
		return
	

	def all(self):
		"""
		returns a Boolean True if all the nonnull elements of the
		Vec instance are True (nonzero), and False otherwise.
		"""
		tmp = self.copy()
	# only because have to set tmp[0]
					# because of element0 snafu
		if isinstance(self._identity_, (float, int, long)):
			return self.reduce(pcb.logical_and(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		else:
			identity = pcb.Obj1()
			identity.weight = tmp[0].weight
	#FIX: "=  bool(...)"?
			identity.category = 99
			tmp[0] = identity
			func = lambda x, other: x.all(other)
			ret = tmp.reduce(pcb.binaryObj(func)).weight > 0
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
		SpVec instance is True (nonzero), and False otherwise.
		"""
		tmp = self.copy()
	# only because have to set tmp[0]
					# because of element0 snafu
		if isinstance(self._identity_, (float, int, long)):
			return self.reduce(pcb.logical_or(), pcb.ifthenelse(pcb.bind2nd(pcb.not_equal_to(),0), pcb.set(1), pcb.set(0)))
		else:
			identity = pcb.Obj1()
			identity.weight = tmp[0].weight
	#FIX: "=  bool(...)"?
			identity.category = 99
			tmp[0] = identity
			func = lambda x, other: x.any(other)
			ret = tmp.reduce(func).weight > 0
			return ret

	def copy(self, element=None):
		"""
		creates a deep copy of the input argument.
		FIX:  doc 'element' arg that converts element of result
		ToDo:  add a doFilter=True arg at some point?
		"""
		ret = Vec(element=self._identity_, sparse=self._isSparse_)
		ret._v_ = self._v_.copy()
		ret._isSparse_ = self._isSparse_
		# filter the new vector; note generic issue of distinguishing
		#   zero from null
		if hasattr(self,'_vFilter_'):
			class tmpU:
				_vFilter_ = self._vFilter_
				@staticmethod
				def fn(x):
					for i in range(len(tmpU._vFilter_)):
						if not tmpU._vFilter_[i](x):
							return type(self._identity_)()
					return x
			tmpInstance = tmpU()
			ret._v_.Apply(pcb.unaryObj(tmpInstance.fn))
			pass
		if element is not None and type(self._identity_) is not type(element):
			tmp = Vec(len(self), element=element, sparse=self.isSparse())
			def func(x, y): 
				#ToDo:  assumes that at least x or y is an ObjX
				if isinstance(x,(float,int,long)):
					ret = y.coerce(x, False)
				else:
					ret = x.coerce(y, True)
				return ret
			tmp2 = tmp._eWiseApply(ret, func, True, True)
			ret = tmp2
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
		if not hasattr(self, '_vFilter_'):
			raise KeyError, "no vertex filters previously created"
		if filter is None:
			del self._vFilter_	# remove all filters
		else:
			self._vFilter_.remove(filter)
			if len(self._vFilter_) == 0:
				del self._vFilter_
		return

	#ToDo:  implement find/findInds when problem of any zero elements
	#         in the sparse vector getting stripped out is solved
	#ToDO:  simplfy to avoid dense() when pySpParVec.Find available
	def find(self, pred=None):
		"""
		returns the elements of a Boolean SpParVec instance that are both
		nonnull and nonzero.

		Input Argument:
			self:  a SpParVec instance

		Output Argument:
			ret:  a SpParVec instance

		SEE ALSO:  findInds
		"""

		# provide a default predicate		
		if pred is None:
			if Vec.isObj(self):
				pred = lambda x: True
			else:
				pred = op_bind2nd(op_ne, 0.0)
			
		ret = self._v_.Find(_op_make_unary_pred(pred))
		return Vec._toVec(ret)


	#ToDO:  simplfy to avoid dense() when pySpParVec.FindInds available
	def findInds(self, pred=None):
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
		# provide a default predicate		
		if pred is None:
			if Vec.isObj(self):
				pred = lambda x: True
			else:
				pred = op_bind2nd(op_ne, 0.0)
			
		ret = self._v_.FindInds(_op_make_unary_pred(pred))
		return Vec._toVec(ret)


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

	def isSparse(self):
		return self._isSparse_

#SPR  Don't know how to do this yet; needs element argument
	@staticmethod
	def load(fname):
                file = open(fname, 'r')
                file.close()

		ret = SpVecObj(1)
		ret._v_.load(fname)
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
	#FIX: spurious? given 2L later?
		func = lambda x, other: x.logicalAnd(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
 #FIX:  spurious? given 2L later?
		func = lambda x, other: x.logicalOr(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

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
	#FIX: spurious? given 2L later?
		func = lambda x, other: x.logicalXor(other)
		ret = self._eWiseApply(other, pcb.binaryObj(func), True,True)		
		return ret

	def max(self):
		"""
		returns the maximum value of the nonnull elements in the SpParVec 
		instance.
		"""
		if self.nnn() == 0:
			if isinstance(self._identity_, (float, int, long)):
				ret = 0
			elif isinstance(self._identity_, pcb.Obj1):
				ret = pcb.Obj1()
				ret.weight = 0; ret.category = 0
		else:
			if isinstance(self._identity_, (float, int, long)):
				ret = self.reduce(pcb.max())
			elif isinstance(self._identity_, (pcb.Obj1, pcb.Obj2)):
				func = lambda x, other: x.max(other)
				ret = self.reduce(pcb.binaryObj(func))
		return ret

	def min(self):
		"""
		returns the minimum value of the nonnull elements in the SpParVec 
		instance.
		"""
		if self.nnn() == 0:
			if isinstance(self._identity_, (float, int, long)):
				ret = 0
			elif isinstance(self._identity_, pcb.Obj1):
				ret = pcb.Obj1()
				ret.weight = 0; ret.category = 0
		else:
			if isinstance(self._identity_, (float, int, long)):
				ret = self.reduce(pcb.min())
			elif isinstance(self._identity_, (pcb.Obj1, pcb.Obj2)):
				func = lambda x, other: x.min(other)
				ret = self.reduce(pcb.binaryObj(func))
		return ret

	def nn(self):
		"""
		returns the number of nulls (non-existent entries) in the 
		SpVec instance.

		Note:  for x a SpVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nnn, nnz
		"""
		if self.isDense():
			return 0
		return len(self) - self.nnn()

	def nnn(self):
		"""
		returns the number of non-nulls (existent entries) in the
		SpVec instance.
	
		Note:  for x a SpVec instance, x.nnn()+x.nn() always equals 
		len(x).

		SEE ALSO:  nn, nnz
		"""
		if self.isDense():
			return len(self)
		
		if not self._hasFilter(self):
			return self._v_.getnee()
			
		# Adam:
		# implement the rest with a single reduce that uses a double
		# as its return value. (That Reduce flavor needs to be added to pcb)
		# This entire function then becomes a one-line call.

		#HACK:  some better way to set initial value of redxn?
		# z = tmp.findInds()
		# if len(z) > 0:
		#   tmp[z[0]] = identity
		def f(x,y):
			if isinstance(x, (float, int, long)):
				x = x + 1
			else:
				if y.weight != 0 or y.category != 0:
					x.weight = x.weight + 1
			return x
		tmp = self.copy()
	#FIX: spurious? 
		if isinstance(self._identity_,(float, int, long)):
			identity = 0
			ret = int(tmp.reduce(pcb.plus(), pred=pcb.set(1)))
		elif isinstance(self._identity_,(pcb.Obj1)):
			identity = type(self._identity_)()
			#HACK: referred to above
			if self[0].weight or self[0].category:
				identity.weight = 1
			tmp[0] = identity
			ret = int(tmp.reduce(pcb.binaryObj(f)).weight)
		elif isinstance(self._identity_,pcb.Obj2):
			identity = type(self._identity_)()
			#HACK: referred to above
			if self[0].weight or self[0].category:
				identity.weight = 1
			tmp[0] = identity
			ret = int(tmp.reduce(pcb.binaryObj(f)).weight)
		return ret

	def nnz(self):
		"""
		returns the number of non-zero entries in the SpParVec
		instance.

		Note:  for x a SpParVec instance, x.nnz() is always less than or
		equal to x.nnn().

		SEE ALSO:  nn, nnn
		"""
		ret = self.reduce(SpParVec.op_add, pcb.ifthenelse(pcb.bind2nd.not_equal_to(),0), pcb.set(1), pcb.set(0))
		return int(ret)

#	#FIX:  delete, since unused
#	@staticmethod
#	def ones(sz, element=0):
#		"""
#		creates a SpVec instance of the specified size whose elements
#		are all nonnull with the value 1.
#		"""
#		ret = Vec(sz)._rangeLike(sz, element=element)
#		#HACK:  scalar set loop for now; needs something better
#		#Obj1 = pcb.Obj1()
#		#Obj1.weight = 1
#		#for i in range(sz):
#		#	ret[i] = Obj1
#		if not Vec.isObj(ret):
#			ret.apply(pcb.set(1), noWrap=True)
#		else:
#			ret.apply(lambda x: x.spOnes())
#		return ret

	def printAll(self):
		"""
		prints all elements of a SpParVec instance (which may number millions
		or billions).

		SEE ALSO:  print, __repr__
		"""
		self._v_.printall()
		return ' '

	
#	@staticmethod
#	def range(*args, sparse=False, **kwargs):
#		"""
#		FIX:  not working yet
#		FIX:  add doc
#		"""
#		raise NotImplementedError
#		if not sparse:
#			return DeVec.range(*args, **kwargs)
#		else:
#			return SpVec.range(*args, **kwargs)
	@staticmethod
	def range(arg1, stop=None, element=0, sparse=False):
		if sparse:
			return SpVec.range(arg1, stop=stop, element=element)
		else:
			return DeVec.range(arg1, stop=stop, element=element)


	#in-place, so no return value
	def set(self, value):
		"""
		sets every non-null value in the SpParVec instance to the second
		argument, in-place.
		"""
		def f(x):
			x.weight = value
			return x
		self.apply(f)
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
		self._v_.Sort()
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
		tmp = ret1._v_.Sort()
		ret2 = Vec._toVec(ret1, tmp)
		return (ret1, ret2)

	#in-place, so no return value
	def spOnes(self):
		"""
		sets every non-null value in the SpParVec instance to 1, in-place.
		"""
		if not Vec.isObj(self):
			self.apply(pcb.set(1))
		else:
			self.apply(lambda x: x.spOnes())
		return

	def sum(self):
		"""
		returns the sum of all the non-null values in the SpVec instance.
		"""
		if self.nnn() == 0:
			if isinstance(self._identity_, (float, int, long)):
				ret = 0
			elif isinstance(self._identity_, pcb.Obj1):
				ret = pcb.Obj1()
				ret.weight = 0; ret.category = 0
		else:
			if isinstance(self._identity_, (float, int, long)):
				ret = self.reduce(pcb.plus())
			elif isinstance(self._identity_, (pcb.Obj1, pcb.Obj2)):
		 		func = lambda x, other: x.__iadd__(other)
				ret = self.reduce(pcb.binaryObj(func))
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
		self.apply(f)
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
		if isinstance(self._identity_, (float, int, long)):
			ret = Vec(0)
			ret._v_ = self._v_.TopK(k)
		else:
			raise NotImplementedError
		return ret

	def toDeVec(self):	
		"""
		converts a SpParVec instance into a ParVec instance of the same
		length with the non-null elements of the SpParVec instance placed 
		in their corresonding positions in the ParVec instance.
		"""
		ret = Vec()
		ret._v_ = self._v_.dense()
		return ret

	@staticmethod
	def toSpVec(SPV):
		#if not isinstance(SPV, pcb.pySpParVec):
		if SPV.__class__.__name__ != 'pySpParVec':
			raise TypeError, 'Only accepts pySpParVec instances'
		ret = SpParVec(-1)
		ret._spv = SPV
		return ret
	
class DeVec(Vec):
	def isSparse(self):
		return False
	
	def isDense(self):
		return True

	def __init__(self, length=0, init=0, element=0, sparse=False):
		#HACK setting of null values should be done more generally
		self._isSparse_ = False
		if isinstance(element, (float, int, long)):
			#self._elementIsObject = False
			#HACK
			self._identity_ = 0
		else:
			#self._elementIsObject = True
			#HACK
			self._identity_ = element
			self._identity_.weight = 0
			self._identity_.category = 0
		if length is not None:
			if isinstance(element, (float, int, long)):
				self._v_ = pcb.pyDenseParVec(length, init)
			elif isinstance(element, pcb.Obj1):
				self._v_ = pcb.pyDenseParVecObj1(length, self._identity_)
			elif isinstance(element, pcb.Obj2):
				self._v_ = pcb.pyDenseParVecObj2(length, self._identity_)
			else:
				raise TypeError

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
			self._v_[key] = value
		elif isinstance(key,DeVec):
			if not key.isBool():
				raise KeyError, 'only Boolean ParVec indexing of SpParVecs supported'
			if isinstance(value,Vec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = Vec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._v_[key._v_] = value._v_
		elif isinstance(key,SpVec):
			#FIX:  get isBool() working
			#if key.isBool():
			#	raise KeyError, 'Boolean SpVec indexing of SpParVecs not supported'
			if isinstance(value,Vec):
				pass
			elif isinstance(value,SpVec):
				value = value.toDeVec()
			elif type(value) == float or type(value) == long or type(value) == int:
				tmp = value
				value = key.copy()
				value.set(tmp)
				value = value.toDeVec()
			else:
				raise KeyError, 'Unknown value type'
			#key = key.toDeVec()
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as SpVec'
			self._v_[key._v_] = value._v_
		elif type(key) == str and key == 'nonnull':
			self.apply(pcb.set(value))
		else:
			raise KeyError, 'Unknown key type'
		return

	# NOTE: this function is DeVec-specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, noWrap=False):
		"""
		ToDo:  write doc
		"""
		if hasattr(self, '_vFilter_') or hasattr(other, '_vFilter_'):
			class tmpB:
				if hasattr(self,'_vFilter_') and len(self._vFilter_) > 0:
					selfVFLen = len(self._vFilter_)
					vFilter1 = self._vFilter_
				else:
					selfVFLen = 0
				if hasattr(other,'_vFilter_') and len(other._vFilter_) > 0:
					otherVFLen = len(other._vFilter_)
					vFilter2 = other._vFilter_
				else:
					otherVFLen = 0
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfVFLen):
						if not tmpB.vFilter1[i](x):
							x = type(self._identity_)()
							break
					for i in range(tmpB.otherVFLen):
						if not tmpB.vFilter2[i](y):
							y = type(other._identity_)()
							break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op
#		if noWrap:
#			if isinstance(other, (float, int, long)):
#				self._v_.EWiseApply(other   , superOp)
#			else:
#				self._v_.EWiseApply(other._v_, superOp)
#		else:
#			if isinstance(other, (float, int, long)):
#				self._v_.EWiseApply(other   , pcb.binaryObj(superOp), pcb.binaryObj(lambda x,y: x._true_(y)), allowANulls, allowBNulls)
#			else:
#				self._v_.EWiseApply(other._v_, pcb.binaryObj(superOp), pcb.binaryObj(lambda x,y: x._true_(y)))
#		ret = Vec._toVec(self,self._v_)
		if noWrap:
			if isinstance(other, (float, int, long)):
				self._v_.EWiseApply(other   , superOp)
			else:
				self._v_.EWiseApply(other._v_, superOp)
		else:
			if isinstance(other, (float, int, long)):
				self._v_.EWiseApply(other   , pcb.binaryObj(superOp))
			else:
				self._v_.EWiseApply(other._v_, pcb.binaryObj(superOp))
		ret = Vec._toVec(self,self._v_)
		return ret
	
	def randPerm(self):
		self._v_.RandPerm()

	def _newLike(self, length, element):
		ret = DeVec(length,0,element,False)
		return ret

	def _rangeLike(self, length=0, element=None):
		ret = DeVec.range(length,element=element)
		return ret

	@staticmethod
	def ones(sz, element=0):
		"""
		creates a SpVec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		ret = DeVec(sz)._rangeLike(sz, element=element)
		#HACK:  scalar set loop for now; needs something better
		#Obj1 = pcb.Obj1()
		#Obj1.weight = 1
		#for i in range(sz):
		#	ret[i] = Obj1
		if not Vec.isObj(ret):
			ret.apply(pcb.set(1))
		else:
			ret.apply(lambda x: x.spOnes())
		return ret

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
		if stop is None:
			start = 0
			stop = arg1
		else:
			start = arg1
		if start > stop:
			raise ValueError, "start > stop"
		ret = Vec(stop-start, element=element)
		#HACK:  serial set is not practical for large sizes
		if isinstance(element, (float, int, long)):
			ret._v_ = pcb.pyDenseParVec.range(stop-start,start)
		else:
			Obj1 = pcb.Obj1()
			for i in range(stop-start):
				Obj1.weight = start + i
				ret[i] = Obj1
		return ret
	
class SpVec(Vec):
	def isSparse(self):
		return True
	
	def isDense(self):
		return False
	
	def __init__(self, length=0, ignoreInit=None, element=0, sparse=True):
		#NOTE:  ignoreInit kept to same #args as DeVec.__init__
		#self._identity_ = element
		#HACK setting of null values should be done more generally
		self._isSparse_ = True
		if isinstance(element, (float, int, long)):
			#self._elementIsObject = False
			#HACK
			self._identity_ = 0
		else:
			#self._elementIsObject = True
			#HACK
			self._identity_ = element
			self._identity_.weight = 0
			self._identity_.category = 0
		if length is not None:
			if isinstance(element, (float, int, long)):
				self._v_ = pcb.pySpParVec(length)
			elif isinstance(element, pcb.Obj1):
				self._v_ = pcb.pySpParVecObj1(length)
			elif isinstance(element, pcb.Obj2):
				self._v_ = pcb.pySpParVecObj2(length)
			else:
				raise TypeError
		return

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
			self._v_[key] = value
		elif isinstance(key,Vec):
			if not key.isBool():
				raise KeyError, 'only Boolean ParVec indexing of SpParVecs supported'
			if isinstance(value,Vec):
				pass
			elif type(value) == float or type(value) == long or type(value) == int:
				value = Vec(len(key),value)
			else:
				raise KeyError, 'Unknown value type'
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as SpParVec'
			self._v_[key._v_] = value._v_
		elif isinstance(key,SpVec):
			if key.isBool():
				raise KeyError, 'Boolean SpVec indexing of SpParVecs not supported'
			if isinstance(value,Vec):
				pass
			elif isinstance(value,SpVec):
				value = value.toDeVec()
			elif type(value) == float or type(value) == long or type(value) == int:
				tmp = value
				value = key.copy()
				value.set(tmp)
				value = value.toDeVec()
			else:
				raise KeyError, 'Unknown value type'
			key = key.toDeVec()
			if len(self._v_) != len(key._v_) or len(self._v_) != len(value._v_):
				raise IndexError, 'Key and Value must be same length as SpVec'
			self._v_[key._v_] = value._v_
		elif type(key) == str and key == 'nonnull':
			self.apply(pcb.set(value))
		else:
			raise KeyError, 'Unknown key type'
		return
	# in-place, so no return value
	def applyInd(self, op, noWrap=False):
		"""
		ToDo:  write doc;  note pcb built-ins cannot be used as filters.
		"""
		
		if hasattr(self, '_vFilter_') and len(self._vFilter_) > 0:
			class tmpB:
				selfVFLen = len(self._vFilter_)
				vFilter1 = self._vFilter_
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfVFLen):
						if not tmpB.vFilter1[i](x):
							x = type(self._identity_)()
							return x
							#break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op
		if noWrap:
			self._v_.ApplyInd(superOp)
		else:
			self._v_.ApplyInd(pcb.binaryObj(superOp))
		#NEEDED?  ret = Vec._toVec(self,self._v_)
		return

	# NOTE: this function is SpVec-specific because pyCombBLAS calling
	#  sequences are different for EWiseApply on sparse/dense vectors
	def eWiseApply(self, other, op, allowANulls=False, allowBNulls=False, noWrap=False):
		"""
		ToDo:  write doc
		"""
		if hasattr(self, '_vFilter_') or hasattr(other, '_vFilter_'):
			class tmpB:
				if hasattr(self,'_vFilter_') and len(self._vFilter_) > 0:
					selfVFLen = len(self._vFilter_)
					vFilter1 = self._vFilter_
				else:
					selfVFLen = 0
				if hasattr(other,'_vFilter_') and len(other._vFilter_) > 0:
					otherVFLen = len(other._vFilter_)
					vFilter2 = other._vFilter_
				else:
					otherVFLen = 0
				@staticmethod
				def fn(x, y):
					for i in range(tmpB.selfVFLen):
						if not tmpB.vFilter1[i](x):
							x = type(self._identity_)()
							break
					for i in range(tmpB.otherVFLen):
						if not tmpB.vFilter2[i](y):
							y = type(other._identity_)()
							break
					return op(x, y)
			superOp = tmpB().fn
		else:
			superOp = op
		if noWrap:
			if isinstance(other, (float, int, long)):
				v = pcb.EWiseApply(self._v_, other   , superOp, allowANulls, allowBNulls)
			else:
				v = pcb.EWiseApply(self._v_, other._v_, superOp, allowANulls, allowBNulls)
		else:
			if isinstance(other, (float, int, long)):
				v = pcb.EWiseApply(self._v_, other   , pcb.binaryObj(superOp), None, allowANulls, allowBNulls)
			else:
				v = pcb.EWiseApply(self._v_, other._v_, pcb.binaryObj(superOp), None, allowANulls, allowBNulls)
		ret = Vec._toVec(self,v)
		return ret

	def _newLike(self, length=0, element=None):
		ret = SpVec(length, None, element, True)
		return ret

	def _rangeLike(self, length=0, element=None):
		ret = SpVec.range(length,element=element)
		return ret

	@staticmethod
	def ones(sz, element=0):
		"""
		creates a SpVec instance of the specified size whose elements
		are all nonnull with the value 1.
		"""
		ret = Vec(sz, sparse=True)._rangeLike(sz, element=element)
		#HACK:  scalar set loop for now; needs something better
		#Obj1 = pcb.Obj1()
		#Obj1.weight = 1
		#for i in range(sz):
		#	ret[i] = Obj1
		if not Vec.isObj(ret):
			ret.apply(pcb.set(1))
		else:
			ret.apply(lambda x: x.spOnes())
		return ret

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
		ret = Vec(stop-start, element=element, sparse=True)
		#HACK:  serial set is not practical for large sizes
		if isinstance(element, (float, int, long)):
			ret._v_ = pcb.pySpParVec.range(stop-start,start)
		else:
			Obj1 = pcb.Obj1()
			for i in range(stop-start):
				Obj1.weight = start + i
				ret[i] = Obj1
		return ret
	
	#in-place, so no return value
	def spRange(self):
		"""
		sets every non-null value in the SpParVec instance to its position
		(offset) in the vector, in-place.
		"""
		if isinstance(self._identity_, (float, int, long)):
			self._v_.setNumToInd()
		else:
			func = lambda x,y: x.spRange(y)
			self.applyInd(func)
		return
